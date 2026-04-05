"""Base class for baseline RAG implementations."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


@dataclass
class BaselineResult:
    """Result from a baseline RAG method."""
    question: str
    answer: str
    score: float  # Self-assessed or computed score
    latency_ms: float
    metadata: Dict[str, Any]
    error: Optional[str] = None


class BaselineRAG(ABC):
    """Abstract base class for baseline RAG methods.

    All baselines implement the same interface for fair comparison:
    - process(question, docs) -> BaselineResult
    """

    def __init__(
        self,
        model_name: str = None,
        judge_model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        """Initialize baseline.

        Args:
            model_name: Generation model
            judge_model: Model for evaluation/reflection
            temperature: Generation temperature
            max_tokens: Max tokens for response
        """
        from src.config import DEFAULTS

        self.model_name = model_name or DEFAULTS.llm_model
        self.judge_model = judge_model or DEFAULTS.judge_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Lazy-loaded providers
        self._gen_provider = None
        self._judge_provider = None

    @property
    def gen_provider(self):
        """Lazy-load generation provider."""
        if self._gen_provider is None:
            from src.providers import get_provider
            self._gen_provider = get_provider(self.model_name)
        return self._gen_provider

    @property
    def judge_provider(self):
        """Lazy-load judge provider."""
        if self._judge_provider is None:
            from src.providers import get_provider
            self._judge_provider = get_provider(self.judge_model)
        return self._judge_provider

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this baseline method."""
        pass

    @abstractmethod
    def process(
        self,
        question: str,
        docs: List[Document],
        gold_answer: str = None,
    ) -> BaselineResult:
        """Process a question through this baseline method.

        Args:
            question: The question to answer
            docs: Retrieved documents
            gold_answer: Optional reference answer

        Returns:
            BaselineResult with answer and metadata
        """
        pass

    def format_context(self, docs: List[Document]) -> str:
        """Format documents into context string."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(
                f"[Document {i}] (Source: {source}, Page: {page})\n{doc.page_content}"
            )
        return "\n\n".join(context_parts)

    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the standard prompt."""
        system_prompt = """You are a precise financial analysis assistant.
Be accurate with numbers, dates, and company names.
Always provide your best answer based on the available context."""

        user_prompt = f"""Answer the following question using the provided context.

Context:
{context}

Question: {question}

Answer:"""

        response = self.gen_provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.content
