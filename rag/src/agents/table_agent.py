"""Table Reasoning Agent: thin bridge to rLLM-FinQA.

Uses rLLM's FinQAAgent + FinQAEnvironment directly for the ReAct loop,
tool execution, and answer extraction. Our code is just the bridge
between the Self-Improving RAG orchestrator and rLLM's agent system.

Setup:
    pip install -e ./rllm && pip install asteval
    python -m projects.finqa.prepare_finqa_data   # downloads company tables
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .base import AgentDecision, BaseAgent

# Add rllm to path
_RLLM_ROOT = Path(__file__).resolve().parents[4] / "rllm"
if str(_RLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_RLLM_ROOT))


class TableAgent(BaseAgent):
    """Bridge between our orchestrator and rLLM's FinQAAgent.

    Uses rLLM's full agent loop: FinQAAgent handles ReAct reasoning + tool
    call parsing, FinQAEnvironment handles tool execution against pre-loaded
    SQLite tables. We just format the task, run the loop, and return results
    as an AgentDecision.

    Two modes:
    - vLLM mode (recommended): Serves rLLM-FinQA-4B via vLLM, agent calls it
    - API mode (fallback): Uses your existing LLM provider (GPT-4o-mini etc.)
    """

    def __init__(
        self,
        model_name: str = None,
        vllm_base_url: str = None,
        max_steps: int = 20,
    ):
        super().__init__("TableAgent")
        from src.config import DEFAULTS
        self.model_name = model_name or DEFAULTS.llm_model
        self.vllm_base_url = vllm_base_url
        self.max_steps = max_steps

    def decide(self, context: Dict[str, Any]) -> AgentDecision:
        """Run rLLM's FinQA agent on a numeric computation question."""
        question = context["question"]
        company = context.get("company", "")
        attempt = context.get("attempt", self._attempt)

        answer, trace, error = self._run_finqa_agent(question, company)

        decision = AgentDecision(
            agent_name=self.name,
            decision_type="table_reasoning",
            decision_value={
                "answer": answer,
                "computation_trace": trace,
                "error": error,
            },
            confidence=0.8 if error is None else 0.3,
            reasoning=f"rLLM-FinQA: {len(trace)} steps. {'OK' if not error else error}",
            metadata={"attempt": attempt},
        )
        self.log_decision(decision)
        return decision

    def _run_finqa_agent(self, question: str, company: str = ""):
        """Run the rLLM FinQA agent+environment loop for a single question.

        Returns: (answer, trace_list, error_or_None)
        """
        try:
            from projects.finqa.fin_qa_agent import FinQAAgent
            from projects.finqa.fin_qa_environment import FinQAEnvironment
        except ImportError as e:
            return None, [], f"rLLM not installed: {e}. Run: pip install -e ./rllm"

        # Build the task dict (same format as rLLM's dataset)
        task = {"question": question}
        if company:
            task["company"] = company

        # Create agent and environment
        agent = FinQAAgent()
        env = FinQAEnvironment(task=task)

        # Run the agent loop (mirrors AgentExecutionEngine but synchronous)
        obs, info = env.reset()
        agent.reset()

        # Feed initial observation (the question)
        agent.update_from_env(
            observation={"question": question},
            reward=0.0, done=False, info={}
        )

        trace = []
        answer = None
        error = None

        for step in range(self.max_steps):
            try:
                # Get model response
                messages = agent.chat_completions
                response_text = self._call_model(messages)

                # Let agent parse the response into actions (tool calls)
                action = agent.update_from_model(response_text)

                # Step the environment (executes tools or finishes)
                next_obs, reward, done, step_info = env.step(action.action)

                # Record trace
                trace.append({
                    "step": step,
                    "response": response_text[:300],
                    "action": str(action.action)[:200] if action.action else None,
                    "reward": reward,
                    "done": done,
                })

                if done:
                    # Extract final answer from the agent's last response
                    answer = self._extract_answer(response_text)
                    if reward > 0:
                        trace[-1]["correct"] = True
                    break

                # Feed tool outputs back to agent
                agent.update_from_env(
                    observation=next_obs,
                    reward=reward, done=done, info=step_info
                )

            except Exception as e:
                error = f"Step {step}: {str(e)}"
                break

        if answer is None and not error:
            error = "Max steps reached without final answer"

        return answer, trace, error

    def _call_model(self, messages: list) -> str:
        """Call the LLM with the current conversation."""
        # Format messages for our provider
        system_parts = []
        user_parts = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "tool":
                user_parts.append(f"Tool result: {content}")
            elif role == "assistant":
                user_parts.append(f"Assistant: {content}")
            elif role == "user":
                user_parts.append(content)

        system_prompt = "\n".join(system_parts) or "You are a financial analyst."
        user_prompt = "\n\n".join(user_parts[-5:])  # Keep recent context

        if self.vllm_base_url:
            from src.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(
                model_name="rLLM/rLLM-FinQA-4B",
                api_key="not-needed",
                base_url=self.vllm_base_url,
                provider_name_override="rllm-vllm",
            )
        else:
            from src.providers import get_provider
            provider = get_provider(self.model_name)

        response = provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024,
            temperature=0.0,
        )
        return response.content

    @staticmethod
    def _extract_answer(text: str) -> Optional[str]:
        """Extract FINAL ANSWER from agent response."""
        import re
        patterns = [
            r'```\s*FINAL ANSWER:\s*(.*?)\s*```',
            r'FINAL ANSWER:\s*(.+?)(?:\n\n|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def escalate_strategy(self) -> None:
        self._attempt += 1
