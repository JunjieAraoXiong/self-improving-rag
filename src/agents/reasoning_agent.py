"""Reasoning Agent: Generates answers from retrieved documents."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from .base import AgentDecision, BaseAgent


# Evidence-first citation format requirement
EVIDENCE_FIRST_REQUIREMENT = """
CRITICAL - EVIDENCE-FIRST FORMAT:
For EVERY numerical claim, financial metric, or factual assertion, you MUST include an inline citation
with the exact source quote in this format:
  "[DocX: 'exact quote from document']"

Examples:
- "Revenue was $383.3B [Doc2: 'Net sales were $383,285 million']"
- "The gross margin was 43.3% [Doc1: 'Gross margin: 43.3%']"
- "Apple had 164,000 employees [Doc3: 'The Company had approximately 164,000 full-time equivalent employees']"

If you cannot find a supporting quote for a claim, DO NOT make that claim.
Every number in your answer must have a corresponding [DocX: 'quote'] citation.
"""

# Answer generation prompts that can be varied on retry
PROMPT_STRATEGIES = {
    "standard": {
        "system": """You are a precise financial analysis assistant who approaches every question methodically.
ALWAYS enter PLAN MODE before answering: first analyze what information is needed,
identify relevant data points in the context, then formulate your answer.
Be accurate with numbers, dates, and company names.
ALWAYS provide your best answer based on the available context -
never refuse to answer or say you cannot find the information.

""" + EVIDENCE_FIRST_REQUIREMENT,
        "instruction": """Answer the following question using the information from the provided context.

PLAN MODE REQUIRED - Before answering, you MUST:
1. IDENTIFY: What specific information does this question ask for?
2. LOCATE: Find the relevant data points in the context
3. VERIFY: Check that data matches the correct company, time period, and fiscal year
4. CALCULATE: If math is needed, show your work step-by-step
5. CITE: Include [DocX: 'exact quote'] for every numerical claim
6. ANSWER: Provide your final answer with inline citations

IMPORTANT:
- ALWAYS provide an answer - even if context seems incomplete
- Use precise numbers, dates, and company names from the context
- For numerical questions, provide ONLY the numerical value with units
- Every number MUST have a [DocX: 'quote'] citation
- NEVER say "The provided context does not contain sufficient information"
"""
    },
    "conservative": {
        "system": """You are a careful financial analyst who prioritizes accuracy over confidence.
When evidence is ambiguous, acknowledge uncertainty rather than guessing.
For yes/no or categorical questions, you may answer "maybe" or "uncertain" if the evidence is mixed.
Always cite specific passages that support your answer.

""" + EVIDENCE_FIRST_REQUIREMENT,
        "instruction": """Answer the following question based on the provided context.

Before answering:
1. Identify ALL relevant passages in the context
2. Evaluate whether the evidence clearly supports an answer
3. If evidence is mixed or incomplete, express appropriate uncertainty
4. Include [DocX: 'exact quote'] citations for all factual claims

For yes/no questions:
- Answer "yes" only if evidence strongly supports it
- Answer "no" only if evidence strongly contradicts it
- Answer "maybe" if evidence is ambiguous or insufficient

Always cite the specific passages that informed your answer using [DocX: 'quote'] format.
"""
    },
    "detailed": {
        "system": """You are a thorough financial analyst who provides comprehensive answers.
Your answers should include relevant context, supporting evidence, and careful reasoning.
Always reference specific documents and data points from the provided context.

""" + EVIDENCE_FIRST_REQUIREMENT,
        "instruction": """Provide a detailed answer to the following question using the context.

Your answer should:
1. Directly address the question
2. Include specific numbers, dates, and facts from the context
3. Cite each claim with [DocX: 'exact quote'] format
4. Note any relevant caveats or limitations

Be thorough but focused on what the question actually asks.
Every numerical claim must have a supporting citation.
"""
    }
}


class ReasoningAgent(BaseAgent):
    """Agent B: Generates answers from retrieved context.

    This agent:
    1. Takes retrieved documents as context
    2. Selects an appropriate prompting strategy
    3. Generates an answer with citations
    4. Can adjust its approach on retry (more conservative, more detailed, etc.)

    The agent tracks:
    - Answer content
    - Confidence in the answer
    - Which documents were cited
    """

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        disable_escalation: bool = False,
    ):
        """Initialize the reasoning agent.

        Args:
            model_name: LLM model to use for generation
            temperature: Generation temperature
            max_tokens: Maximum tokens for response
            disable_escalation: Ablation flag - always use "standard" prompt
        """
        super().__init__("ReasoningAgent")

        from src.config import DEFAULTS
        self.model_name = model_name or DEFAULTS.llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_escalation = disable_escalation

        # Lazy-loaded provider
        self._provider = None

        # Strategy progression for retries
        self._prompt_strategy = "standard"

    @property
    def provider(self):
        """Lazy-load the LLM provider."""
        if self._provider is None:
            from src.providers import get_provider
            self._provider = get_provider(self.model_name)
        return self._provider

    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into context string.

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted context string with source attribution
        """
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(f"[Document {i}] (Source: {source}, Page: {page})\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def extract_citations(self, answer: str, docs: List[Document]) -> List[str]:
        """Extract which documents were likely cited in the answer.

        Args:
            answer: The generated answer
            docs: The context documents

        Returns:
            List of document sources that appear to be cited
        """
        citations = []
        for doc in docs:
            source = doc.metadata.get("source", "")
            # Check if any significant overlap between doc content and answer
            # (Simple heuristic - could be improved with more sophisticated matching)
            doc_words = set(doc.page_content.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(doc_words & answer_words) / max(len(doc_words), 1)
            if overlap > 0.1:  # At least 10% word overlap
                citations.append(source)

        return list(set(citations))

    def estimate_confidence(self, answer: str, docs: List[Document]) -> float:
        """Estimate confidence in the answer.

        Args:
            answer: The generated answer
            docs: The context documents

        Returns:
            Confidence score between 0 and 1
        """
        # Heuristics for confidence estimation
        confidence = 0.5  # Base confidence

        # Higher confidence if answer is concise and specific
        word_count = len(answer.split())
        if 10 <= word_count <= 100:
            confidence += 0.1

        # Lower confidence if answer contains hedging language
        hedging_words = ["maybe", "possibly", "uncertain", "might", "could be", "unclear"]
        if any(word in answer.lower() for word in hedging_words):
            confidence -= 0.2

        # Lower confidence if answer says it can't find information
        refusal_phrases = ["cannot find", "not mentioned", "no information", "insufficient"]
        if any(phrase in answer.lower() for phrase in refusal_phrases):
            confidence -= 0.3

        # Higher confidence if answer contains specific numbers/dates
        import re
        if re.search(r'\$[\d,]+|\d+%|\d{4}', answer):
            confidence += 0.15

        return max(0.0, min(1.0, confidence))

    def get_prompt_strategy(self, attempt: int) -> str:
        """Get prompt strategy based on attempt number.

        Args:
            attempt: The attempt number

        Returns:
            Strategy name to use
        """
        # Ablation: disable escalation - always use "standard" prompt
        if self.disable_escalation:
            return "standard"

        strategies = ["standard", "conservative", "detailed"]
        return strategies[min(attempt, len(strategies) - 1)]

    def decide(self, context: Dict[str, Any]) -> AgentDecision:
        """Generate an answer based on question and retrieved documents.

        Args:
            context: Must contain 'question' and 'documents' keys

        Returns:
            AgentDecision with the generated answer
        """
        question = context["question"]
        docs = context["documents"]
        attempt = context.get("attempt", self._attempt)

        # Select prompt strategy
        strategy_name = self.get_prompt_strategy(attempt)
        strategy = PROMPT_STRATEGIES[strategy_name]

        # Format context
        formatted_context = self.format_context(docs)

        # Build prompt
        system_prompt = strategy["system"]
        user_prompt = f"""{strategy["instruction"]}

Context:
{formatted_context}

Question: {question}

Answer:"""

        # Generate answer
        try:
            response = self.provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            answer = response.content
            error = None
        except Exception as e:
            answer = None
            error = str(e)

        # Extract citations and estimate confidence
        if answer:
            citations = self.extract_citations(answer, docs)
            confidence = self.estimate_confidence(answer, docs)
        else:
            citations = []
            confidence = 0.0

        # Build reasoning
        reasoning = (
            f"Generated answer using '{strategy_name}' strategy. "
            f"Context: {len(docs)} documents. "
            f"Citations: {len(citations)} documents referenced."
        )
        if attempt > 0:
            reasoning = f"Retry #{attempt}: {reasoning}"

        decision = AgentDecision(
            agent_name=self.name,
            decision_type="answer_generation",
            decision_value={
                "answer": answer,
                "citations": citations,
                "strategy": strategy_name,
                "error": error,
            },
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "attempt": attempt,
                "num_docs": len(docs),
                "model": self.model_name,
            }
        )

        self.log_decision(decision)
        return decision

    def escalate_strategy(self) -> None:
        """Move to a different prompt strategy for retry."""
        self._attempt += 1
