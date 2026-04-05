"""HyDE (Hypothetical Document Embeddings) for improved retrieval.

HyDE generates a hypothetical answer to the query, then uses THAT
for embedding-based retrieval instead of the original query.

Why it works:
- Queries are often short and vague ("revenue 2022")
- Documents are detailed and specific ("Total revenue for FY2022 was $514B...")
- Embedding the hypothetical answer brings the query closer to document space

Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
https://arxiv.org/abs/2212.10496
"""

from typing import Callable, List, Optional

from langchain_core.documents import Document


# Re-export from router for backwards compatibility
from .router import HyDEGenerator, HyDERetriever, get_hyde_generator


class HyDE:
    """Standalone HyDE wrapper for easy integration with any retriever.

    Usage:
        hyde = HyDE(retriever_fn=my_retriever.invoke)
        docs = hyde.retrieve("What was the revenue?")
    """

    FINANCIAL_PROMPT = """You are a financial document passage generator.
Given a question about a financial document (10-K, 10-Q filing), write a
hypothetical passage that would contain the answer to this question.

Write as if you are quoting directly from a financial report. Include:
- Realistic financial terminology
- Placeholder numbers that fit the context
- Company-neutral language

The passage should be 2-3 sentences and sound like an actual excerpt."""

    GENERAL_PROMPT = """Given a question, write a hypothetical passage that
would contain the answer. Write as if quoting from a document.
Be specific and include realistic details. 2-3 sentences."""

    def __init__(
        self,
        retriever_fn: Callable[[str], List[Document]],
        llm_provider=None,
        model_name: str = None,
        domain: str = "financial",
    ):
        """Initialize HyDE.

        Args:
            retriever_fn: Function that takes a query string and returns documents
            llm_provider: LLM provider for generating hypothetical docs (optional)
            model_name: Model name if not using llm_provider directly
            domain: "financial" or "general" for prompt selection
        """
        self.retriever_fn = retriever_fn
        self._llm_provider = llm_provider
        self._model_name = model_name
        self.domain = domain
        self._hyde_generator: Optional[HyDEGenerator] = None

    @property
    def llm_provider(self):
        """Lazy-load LLM provider."""
        if self._llm_provider is None:
            from src.providers.factory import get_provider
            from src.config import DEFAULTS
            model = self._model_name or DEFAULTS.router_hyde_model
            self._llm_provider = get_provider(model)
        return self._llm_provider

    @property
    def system_prompt(self) -> str:
        """Get domain-appropriate system prompt."""
        return self.FINANCIAL_PROMPT if self.domain == "financial" else self.GENERAL_PROMPT

    def generate_hypothetical(self, question: str) -> str:
        """Generate a hypothetical document passage for the question."""
        try:
            response = self.llm_provider.generate(
                system_prompt=self.system_prompt,
                user_prompt=f"Generate a hypothetical passage answering: {question}",
                max_tokens=200,
                temperature=0.3,  # Slight creativity for realistic passages
            )
            return response.content
        except Exception:
            # Fallback to original question
            return question

    def retrieve(self, question: str) -> List[Document]:
        """Retrieve using hypothetical document embedding.

        1. Generate hypothetical answer passage
        2. Use that passage for retrieval instead of original query
        3. Return retrieved documents
        """
        hypothetical = self.generate_hypothetical(question)
        return self.retriever_fn(hypothetical)

    def retrieve_with_both(
        self,
        question: str,
        merge_strategy: str = "interleave"
    ) -> List[Document]:
        """Retrieve using both original query AND hypothetical.

        Args:
            question: The original question
            merge_strategy: "interleave" or "concat"

        Returns:
            Merged results from both retrieval approaches
        """
        # Get results from both approaches
        original_docs = self.retriever_fn(question)
        hypothetical = self.generate_hypothetical(question)
        hyde_docs = self.retriever_fn(hypothetical)

        # Merge results
        if merge_strategy == "interleave":
            return _interleave_unique(original_docs, hyde_docs)
        else:
            return _concat_unique(original_docs, hyde_docs)


def _interleave_unique(list1: List[Document], list2: List[Document]) -> List[Document]:
    """Interleave two lists, removing duplicates."""
    seen = set()
    result = []

    for doc1, doc2 in zip(list1, list2):
        for doc in [doc1, doc2]:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                result.append(doc)

    # Add remaining from longer list
    longer = list1 if len(list1) > len(list2) else list2
    for doc in longer[len(min(list1, list2, key=len)):]:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            result.append(doc)

    return result


def _concat_unique(list1: List[Document], list2: List[Document]) -> List[Document]:
    """Concatenate two lists, removing duplicates."""
    seen = set()
    result = []

    for doc in list1 + list2:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            result.append(doc)

    return result
