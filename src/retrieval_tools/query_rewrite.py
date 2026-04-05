"""Query Rewriting for improved retrieval.

Transforms user queries into more specific, retrieval-friendly forms.
This helps when users ask vague or ambiguous questions.

Techniques:
1. Query Expansion: Add synonyms and related terms
2. Query Decomposition: Break complex questions into sub-queries
3. Query Reformulation: Make queries more specific for document retrieval
"""

from typing import List, Optional

from langchain_core.documents import Document


class QueryRewriter:
    """Rewrites queries to be more specific for retrieval."""

    FINANCIAL_SYSTEM_PROMPT = """You are a query rewriter for financial document search.
Your task is to rewrite user questions to be more specific and retrieval-friendly.

Guidelines:
- Add specific financial terminology (e.g., "revenue" → "total revenue", "net revenue")
- Expand abbreviations (e.g., "Q3" → "third quarter", "YoY" → "year-over-year")
- Make implicit context explicit (e.g., add "fiscal year", "as reported in 10-K")
- Keep the same meaning but make it more specific
- Output ONLY the rewritten query, nothing else"""

    GENERAL_SYSTEM_PROMPT = """You are a query rewriter for document search.
Rewrite user questions to be more specific and retrieval-friendly.
Add relevant terms and make implicit context explicit.
Output ONLY the rewritten query, nothing else."""

    def __init__(
        self,
        llm_provider=None,
        model_name: str = None,
        domain: str = "financial",
    ):
        """Initialize QueryRewriter.

        Args:
            llm_provider: LLM provider for rewriting
            model_name: Model name if not using llm_provider
            domain: "financial" or "general"
        """
        self._llm_provider = llm_provider
        self._model_name = model_name
        self.domain = domain

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
        if self.domain == "financial":
            return self.FINANCIAL_SYSTEM_PROMPT
        return self.GENERAL_SYSTEM_PROMPT

    def rewrite(self, question: str) -> str:
        """Rewrite a question to be more retrieval-friendly.

        Args:
            question: Original user question

        Returns:
            Rewritten question (or original if rewriting fails)
        """
        try:
            response = self.llm_provider.generate(
                system_prompt=self.system_prompt,
                user_prompt=f"Rewrite this query: {question}",
                max_tokens=150,
                temperature=0.0,  # Deterministic
            )
            rewritten = response.content.strip()
            # Sanity check: don't return empty or very short
            if len(rewritten) < 5:
                return question
            return rewritten
        except Exception:
            return question


class QueryDecomposer:
    """Breaks complex questions into simpler sub-queries."""

    SYSTEM_PROMPT = """You are a query decomposer for document search.
Break complex questions into 2-4 simpler sub-queries that together answer the original.

Rules:
- Each sub-query should be self-contained
- Sub-queries should be answerable from a single document passage
- Output one sub-query per line, no numbering or bullets
- If the question is already simple, output it unchanged"""

    def __init__(
        self,
        llm_provider=None,
        model_name: str = None,
    ):
        self._llm_provider = llm_provider
        self._model_name = model_name

    @property
    def llm_provider(self):
        """Lazy-load LLM provider."""
        if self._llm_provider is None:
            from src.providers.factory import get_provider
            from src.config import DEFAULTS
            model = self._model_name or DEFAULTS.router_hyde_model
            self._llm_provider = get_provider(model)
        return self._llm_provider

    def decompose(self, question: str) -> List[str]:
        """Decompose a complex question into sub-queries.

        Args:
            question: Complex question

        Returns:
            List of simpler sub-queries
        """
        try:
            response = self.llm_provider.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=f"Decompose this question: {question}",
                max_tokens=300,
                temperature=0.0,
            )

            # Parse response into list of queries
            lines = response.content.strip().split("\n")
            sub_queries = [
                line.strip().lstrip("0123456789.-) ")
                for line in lines
                if line.strip() and len(line.strip()) > 5
            ]

            return sub_queries if sub_queries else [question]
        except Exception:
            return [question]


class QueryExpander:
    """Expands queries with synonyms and related terms."""

    # Common financial term expansions
    FINANCIAL_EXPANSIONS = {
        "revenue": ["total revenue", "net revenue", "sales", "top-line"],
        "profit": ["net income", "earnings", "bottom-line", "net profit"],
        "cost": ["expenses", "operating costs", "cost of goods sold", "COGS"],
        "debt": ["liabilities", "borrowings", "long-term debt", "total debt"],
        "margin": ["profit margin", "gross margin", "operating margin"],
        "growth": ["increase", "year-over-year growth", "YoY change"],
        "q1": ["first quarter", "Q1", "quarter ending March"],
        "q2": ["second quarter", "Q2", "quarter ending June"],
        "q3": ["third quarter", "Q3", "quarter ending September"],
        "q4": ["fourth quarter", "Q4", "quarter ending December"],
        "fy": ["fiscal year", "FY", "annual"],
    }

    def __init__(self, domain: str = "financial"):
        self.domain = domain
        self.expansions = (
            self.FINANCIAL_EXPANSIONS if domain == "financial" else {}
        )

    def expand(self, question: str) -> str:
        """Expand query with synonyms.

        Simple rule-based expansion that adds related terms.
        """
        question_lower = question.lower()
        additions = []

        for term, synonyms in self.expansions.items():
            if term in question_lower:
                # Add first synonym not already in query
                for syn in synonyms:
                    if syn.lower() not in question_lower:
                        additions.append(syn)
                        break

        if additions:
            return f"{question} ({', '.join(additions)})"
        return question


class MultiQueryRetriever:
    """Retrieves using multiple query variants and combines results."""

    def __init__(
        self,
        retriever_fn,
        query_rewriter: Optional[QueryRewriter] = None,
        query_decomposer: Optional[QueryDecomposer] = None,
        query_expander: Optional[QueryExpander] = None,
    ):
        """Initialize with retriever and optional query transformers.

        Args:
            retriever_fn: Function that takes query and returns documents
            query_rewriter: Optional rewriter
            query_decomposer: Optional decomposer
            query_expander: Optional expander
        """
        self.retriever_fn = retriever_fn
        self.rewriter = query_rewriter
        self.decomposer = query_decomposer
        self.expander = query_expander

    def retrieve(self, question: str, top_k: int = 10) -> List[Document]:
        """Retrieve using multiple query variants.

        Strategy:
        1. Generate query variants (rewrite, expand, decompose)
        2. Retrieve for each variant
        3. Combine results using RRF
        """
        from .multi_query import reciprocal_rank_fusion

        queries = [question]  # Always include original

        # Add rewritten query
        if self.rewriter:
            rewritten = self.rewriter.rewrite(question)
            if rewritten != question:
                queries.append(rewritten)

        # Add expanded query
        if self.expander:
            expanded = self.expander.expand(question)
            if expanded != question:
                queries.append(expanded)

        # Add decomposed sub-queries
        if self.decomposer:
            sub_queries = self.decomposer.decompose(question)
            if len(sub_queries) > 1:
                queries.extend(sub_queries)

        # Retrieve for each query
        all_results = [self.retriever_fn(q) for q in queries]

        # Combine with RRF
        fused = reciprocal_rank_fusion(all_results)

        return fused[:top_k]
