"""Orchestrator: Coordinates all agents with retry logic."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import AgentDecision
from .retrieval_agent import RetrievalAgent
from .reasoning_agent import ReasoningAgent
from .judge_agent import JudgeAgent
from .logger import AgentLogger
from src.providers.base import get_usage_tracker
from src.config import calculate_cost


@dataclass
class AgenticRAGConfig:
    """Configuration for the agentic RAG orchestrator."""
    # Agent settings
    max_retries: int = 1  # Maximum retry attempts (0 = no retries)
    retry_threshold: float = 0.5  # Score below which to retry
    blind_judge: bool = False  # If True, Judge uses self-evaluation without gold answer

    # Model settings
    llm_model: str = None  # Will use config default
    judge_model: str = None  # Will use config default
    reranker_model: str = None  # Will use config default

    # Retrieval settings
    use_rule_router: bool = True  # Use free rule-based routing
    use_rse: bool = False  # Use Relevant Segment Extraction

    # Logging settings
    log_dir: str = "agent_logs"
    enable_logging: bool = True

    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 512

    # Ablation study settings
    # These flags disable specific components to measure their contribution
    ablation_no_retrieval_escalation: bool = False  # Fix top_k=10 on all attempts
    ablation_no_prompt_escalation: bool = False     # Fix "standard" prompt on all attempts
    ablation_no_hyde: bool = False                   # Never enable HyDE
    ablation_no_deterministic_verify: bool = False   # LLM judge only (no citation check)


@dataclass
class AgenticRAGResult:
    """Result from processing a question through the agentic system."""
    question_id: str
    question: str
    final_answer: str
    final_score: float
    correct: bool
    attempts: int
    improvement_from_retry: bool
    decision_log: List[Dict[str, Any]]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    error: Optional[str] = None
    cost_usd: float = 0.0


class AgenticRAGOrchestrator:
    """Coordinates all agents to process questions with self-correction.

    The orchestrator implements the core agentic RAG loop:

    ```
    while attempt <= max_retries:
        1. RetrievalAgent decides strategy and retrieves documents
        2. ReasoningAgent generates answer from documents
        3. JudgeAgent evaluates and decides: accept or retry?

        if accepted:
            break
        else:
            escalate all agent strategies
            attempt += 1
    ```

    All decisions are logged for analysis and interpretability.
    """

    def __init__(self, config: AgenticRAGConfig = None, db=None, embedding_fn=None):
        """Initialize the orchestrator.

        Args:
            config: Configuration for the agentic system
            db: ChromaDB instance
            embedding_fn: Embedding function for retrieval
        """
        self.config = config or AgenticRAGConfig()
        self.db = db
        self.embedding_fn = embedding_fn

        # Initialize agents with ablation settings
        self.retrieval_agent = RetrievalAgent(
            db=db,
            embedding_fn=embedding_fn,
            reranker_model=self.config.reranker_model,
            use_rule_router=self.config.use_rule_router,
            use_rse=self.config.use_rse,
            disable_escalation=self.config.ablation_no_retrieval_escalation,
            disable_hyde=self.config.ablation_no_hyde,
        )

        self.reasoning_agent = ReasoningAgent(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            disable_escalation=self.config.ablation_no_prompt_escalation,
        )

        self.judge_agent = JudgeAgent(
            judge_model=self.config.judge_model,
            retry_threshold=self.config.retry_threshold,
            enable_deterministic_gate=not self.config.ablation_no_deterministic_verify,
        )

        # Initialize logger
        if self.config.enable_logging:
            self.logger = AgentLogger(output_dir=self.config.log_dir)
        else:
            self.logger = None

        # Track overall statistics
        self.total_questions = 0
        self.total_retries = 0
        self.successful_retries = 0

        # Cost tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

    def reset_agents(self) -> None:
        """Reset all agents for a new question."""
        self.retrieval_agent.reset()
        self.reasoning_agent.reset()
        self.judge_agent.reset()

    def process_question(
        self,
        question: str,
        gold_answer: str = None,
        question_id: str = None,
    ) -> AgenticRAGResult:
        """Process a single question through the agentic pipeline.

        Args:
            question: The question to answer
            gold_answer: Reference answer for evaluation (optional)
            question_id: Unique identifier for logging

        Returns:
            AgenticRAGResult with answer and decision log
        """
        start_time = time.time()

        # Snapshot token counts before this question for cost calculation
        tracker = get_usage_tracker()
        prompt_tokens_before = tracker.total_prompt_tokens
        completion_tokens_before = tracker.total_completion_tokens

        # Generate question ID if not provided
        if question_id is None:
            question_id = f"Q{self.total_questions:04d}"

        self.total_questions += 1

        # Reset agents for new question
        self.reset_agents()

        # Start logging
        if self.logger:
            self.logger.start_question(question_id, question, gold_answer)

        # Initialize tracking variables
        decision_log = []
        attempt = 0
        final_answer = None
        final_score = 0.0
        best_answer = None
        best_score = 0.0
        retrieval_time_ms = 0.0
        generation_time_ms = 0.0
        error = None

        # Main retry loop
        while attempt <= self.config.max_retries:
            attempt_start = time.time()

            if self.logger:
                self.logger.start_attempt(attempt)

            try:
                # === Agent A: Retrieval ===
                retrieval_start = time.time()

                retrieval_decision = self.retrieval_agent.decide({
                    "question": question,
                    "attempt": attempt,
                })

                docs = self.retrieval_agent.retrieve(question, retrieval_decision)

                retrieval_time_ms += (time.time() - retrieval_start) * 1000

                if self.logger:
                    self.logger.log_decision("retrieval_agent", retrieval_decision)

                if not docs:
                    error = "No documents retrieved"
                    break

                # === Agent B: Reasoning ===
                generation_start = time.time()

                reasoning_decision = self.reasoning_agent.decide({
                    "question": question,
                    "documents": docs,
                    "attempt": attempt,
                })

                generation_time_ms += (time.time() - generation_start) * 1000

                if self.logger:
                    self.logger.log_decision("reasoning_agent", reasoning_decision)

                answer = reasoning_decision.decision_value.get("answer")

                if not answer:
                    error = reasoning_decision.decision_value.get("error", "No answer generated")
                    break

                # === Agent C: Judge ===
                # In blind mode, Judge uses self-evaluation without gold answer.
                # Documents are passed for blind numeric grounding verification.
                judge_gold = None if self.config.blind_judge else gold_answer
                judge_decision = self.judge_agent.decide({
                    "question": question,
                    "predicted_answer": answer,
                    "gold_answer": judge_gold,
                    "attempt": attempt,
                    "max_retries": self.config.max_retries,
                    "documents": docs,
                })

                if self.logger:
                    self.logger.log_decision("judge_agent", judge_decision)

                score = judge_decision.decision_value.get("score", 0.0)
                should_retry = judge_decision.decision_value.get("retry", False)

                # Track best answer
                if score > best_score:
                    best_score = score
                    best_answer = answer

                # Log this attempt
                decision_log.append({
                    "attempt": attempt,
                    "retrieval": retrieval_decision.to_dict(),
                    "reasoning": reasoning_decision.to_dict(),
                    "judge": judge_decision.to_dict(),
                    "attempt_time_ms": (time.time() - attempt_start) * 1000,
                })

                # Check if we should retry
                if not should_retry:
                    final_answer = answer
                    final_score = score
                    break

                # Escalate strategies for retry
                self.total_retries += 1
                self.retrieval_agent.escalate_strategy()
                self.reasoning_agent.escalate_strategy()

                attempt += 1

            except Exception as e:
                error = f"Error in attempt {attempt}: {str(e)}"
                import traceback
                traceback.print_exc()
                break

        # Use best answer if final answer not set
        if final_answer is None:
            final_answer = best_answer
            final_score = best_score

        # Determine correctness and improvement
        correct = final_score >= 0.5

        # Check if retry improved the result
        improvement_from_retry = False
        if len(decision_log) > 1:
            first_score = decision_log[0]["judge"]["decision_value"]["score"]
            if final_score > first_score:
                improvement_from_retry = True
                self.successful_retries += 1

        total_time_ms = (time.time() - start_time) * 1000

        # Calculate cost for this question from token delta
        question_prompt_tokens = tracker.total_prompt_tokens - prompt_tokens_before
        question_completion_tokens = tracker.total_completion_tokens - completion_tokens_before
        model_for_cost = self.config.llm_model or "gpt-4o-mini"
        question_cost = calculate_cost(model_for_cost, {
            "prompt_tokens": question_prompt_tokens,
            "completion_tokens": question_completion_tokens,
        }) if (question_prompt_tokens + question_completion_tokens) > 0 else 0.0
        self.total_cost_usd += question_cost
        self.total_prompt_tokens = tracker.total_prompt_tokens
        self.total_completion_tokens = tracker.total_completion_tokens

        # Finish logging
        if self.logger:
            self.logger.finish_question(final_answer, correct, improvement_from_retry)

        return AgenticRAGResult(
            question_id=question_id,
            question=question,
            final_answer=final_answer,
            final_score=final_score,
            correct=correct,
            attempts=min(attempt + 1, self.config.max_retries + 1),
            improvement_from_retry=improvement_from_retry,
            decision_log=decision_log,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            error=error,
            cost_usd=question_cost,
        )

    def process_batch(
        self,
        questions: List[str],
        gold_answers: List[str] = None,
        question_ids: List[str] = None,
        show_progress: bool = True,
    ) -> List[AgenticRAGResult]:
        """Process multiple questions.

        Args:
            questions: List of questions
            gold_answers: List of reference answers (optional)
            question_ids: List of question IDs (optional)
            show_progress: Whether to show progress bar

        Returns:
            List of AgenticRAGResult objects
        """
        results = []

        if gold_answers is None:
            gold_answers = [None] * len(questions)
        if question_ids is None:
            question_ids = [f"Q{i:04d}" for i in range(len(questions))]

        iterator = zip(questions, gold_answers, question_ids)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Processing questions")

        for question, gold, qid in iterator:
            result = self.process_question(question, gold, qid)
            results.append(result)

        return results

    def export_decisions(self, filename: str = None) -> Dict[str, str]:
        """Export all logged decisions.

        Args:
            filename: Base filename (without extension)

        Returns:
            Dictionary with paths to exported files
        """
        if self.logger is None:
            return {}

        paths = {}
        paths["json"] = self.logger.export_json(filename)
        paths["csv"] = self.logger.export_csv(filename)

        return paths

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics for the session.

        Returns:
            Dictionary with session statistics
        """
        stats = {
            "total_questions": self.total_questions,
            "total_retries": self.total_retries,
            "successful_retries": self.successful_retries,
            "retry_rate": self.total_retries / max(1, self.total_questions),
            "retry_success_rate": self.successful_retries / max(1, self.total_retries),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_question": self.total_cost_usd / max(1, self.total_questions),
        }

        if self.logger:
            stats.update(self.logger.summary())

        return stats

    def print_summary(self) -> None:
        """Print session summary to console."""
        if self.logger:
            self.logger.print_summary()
        else:
            stats = self.get_statistics()
            print(f"\nProcessed {stats['total_questions']} questions")
            print(f"Retries: {stats['total_retries']} ({stats['retry_rate']:.1%})")
            print(f"Successful retries: {stats['successful_retries']}")


def build_agentic_orchestrator(
    db,
    embedding_fn,
    config: AgenticRAGConfig = None,
    **kwargs
) -> AgenticRAGOrchestrator:
    """Factory function to build an orchestrator with default settings.

    Args:
        db: ChromaDB instance
        embedding_fn: Embedding function
        config: Optional configuration
        **kwargs: Override config values

    Returns:
        Configured AgenticRAGOrchestrator
    """
    if config is None:
        config = AgenticRAGConfig(**kwargs)
    elif kwargs:
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return AgenticRAGOrchestrator(config=config, db=db, embedding_fn=embedding_fn)
