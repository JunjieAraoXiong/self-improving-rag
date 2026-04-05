"""Multi-query expansion and RAG-Fusion for improved retrieval recall.

This module provides:
1. Rule-based query expansion (NO API calls) - fast, free
2. LLM-based query generation - higher quality, uses API
3. RAG-Fusion with Reciprocal Rank Fusion (RRF) - combines results from multiple queries

References:
- RAG-Fusion paper: https://arxiv.org/abs/2402.03367
- RRF: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
"""

import re
from collections import defaultdict
from typing import List, Set, Dict, Any, Optional, Callable
from langchain_core.documents import Document


# Financial domain synonyms
FINANCIAL_SYNONYMS = {
    'revenue': ['sales', 'total sales', 'net sales', 'turnover'],
    'profit': ['earnings', 'net income', 'income'],
    'ebitda': ['operating income', 'operating profit'],
    'capex': ['capital expenditure', 'capital expenditures', 'pp&e purchases'],
    'capital expenditure': ['capex', 'pp&e purchases'],
    'assets': ['total assets', 'asset base'],
    'liabilities': ['total liabilities', 'debt'],
    'margin': ['profit margin', 'gross margin'],
    'growth': ['increase', 'change', 'yoy change'],
    'decline': ['decrease', 'reduction', 'drop'],
    'fy': ['fiscal year', 'fiscal'],
    'q1': ['first quarter', 'quarter 1'],
    'q2': ['second quarter', 'quarter 2'],
    'q3': ['third quarter', 'quarter 3'],
    'q4': ['fourth quarter', 'quarter 4'],
}

# Question type rephrasing patterns
QUESTION_PATTERNS = [
    (r'^What is the ', ['Find the ', 'Show the ', 'What was the ']),
    (r'^What was the ', ['What is the ', 'Find the ']),
    (r'^How much ', ['What is the amount of ', 'What is the total ']),
    (r'^Is ', ['Was ', 'Does ']),
]


class MultiQueryExpander:
    """Generates query variants using rule-based expansion (no API calls).

    Expansion strategies:
    1. Remove temporal references (years) to broaden search
    2. Replace financial terms with synonyms
    3. Rephrase question structure
    4. Extract core entities for focused search
    """

    def __init__(self, max_queries: int = 4):
        """Initialize expander.

        Args:
            max_queries: Maximum number of query variants to generate
        """
        self.max_queries = max_queries
        self.synonyms = FINANCIAL_SYNONYMS

    def expand(self, question: str) -> List[str]:
        """Generate query variants from the original question.

        Args:
            question: Original question text

        Returns:
            List of unique query variants (including original)
        """
        variants: Set[str] = {question}

        # Strategy 1: Remove year references for broader search
        no_year = self._remove_years(question)
        if no_year != question and len(no_year) > 10:
            variants.add(no_year)

        # Strategy 2: Apply financial synonyms
        for original, replacements in self.synonyms.items():
            if original.lower() in question.lower():
                for replacement in replacements[:2]:  # Limit replacements
                    variant = re.sub(
                        rf'\b{re.escape(original)}\b',
                        replacement,
                        question,
                        flags=re.IGNORECASE
                    )
                    if variant != question:
                        variants.add(variant)

        # Strategy 3: Rephrase question structure
        for pattern, replacements in QUESTION_PATTERNS:
            if re.match(pattern, question, re.IGNORECASE):
                for replacement in replacements[:1]:  # Just one rephrasing
                    variant = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
                    if variant != question:
                        variants.add(variant)
                break

        # Strategy 4: Extract focused entity query
        entity_query = self._extract_entity_query(question)
        if entity_query and len(entity_query) > 5:
            variants.add(entity_query)

        # Return up to max_queries variants, original first
        result = [question]
        for v in variants:
            if v != question and len(result) < self.max_queries:
                result.append(v)

        return result

    def _remove_years(self, text: str) -> str:
        """Remove year references to broaden search."""
        # Remove FY20XX patterns
        text = re.sub(r'\bFY\s*20\d{2}\b', '', text, flags=re.IGNORECASE)
        # Remove standalone years
        text = re.sub(r'\b20\d{2}\b', '', text)
        # Remove fiscal year phrases
        text = re.sub(r'\bfiscal year\s*20\d{2}\b', '', text, flags=re.IGNORECASE)
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_entity_query(self, question: str) -> str:
        """Extract key entities for focused retrieval."""
        # Common question words to exclude
        exclude_words = {'what', 'how', 'is', 'was', 'does', 'did', 'the', 'a', 'an',
                         'for', 'in', 'on', 'of', 'to', 'based', 'according', 'much', 'many'}

        # Extract potential company names (capitalized sequences, excluding question starters)
        companies = [
            match for match in re.findall(r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b', question)
            if match.lower() not in exclude_words and len(match) > 1
        ]

        # Extract financial metrics mentioned
        metrics = []
        for term in self.synonyms.keys():
            if term.lower() in question.lower():
                metrics.append(term)

        # Only return if we have meaningful entities
        if companies and metrics and len(companies[0]) > 2:
            return f"{companies[0]} {' '.join(metrics[:2])}"

        return ""


# Singleton instance for convenience
_expander = None


def get_expander(max_queries: int = 4) -> MultiQueryExpander:
    """Get or create the multi-query expander instance."""
    global _expander
    if _expander is None or _expander.max_queries != max_queries:
        _expander = MultiQueryExpander(max_queries=max_queries)
    return _expander


def expand_query(question: str, max_queries: int = 4) -> List[str]:
    """Convenience function to expand a query.

    Args:
        question: Original question
        max_queries: Maximum variants to generate

    Returns:
        List of query variants
    """
    return get_expander(max_queries).expand(question)


# =============================================================================
# RAG-Fusion with Reciprocal Rank Fusion
# =============================================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = Σ 1 / (k + rank(d, list_i))

    Args:
        ranked_lists: List of document lists, each sorted by relevance
        k: RRF constant (default 60, as per original paper)

    Returns:
        Combined list sorted by RRF score
    """
    # Track scores by document content hash
    doc_scores: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {"doc": None, "score": 0.0, "sources": []}
    )

    for list_idx, doc_list in enumerate(ranked_lists):
        for rank, doc in enumerate(doc_list):
            # Use content hash as document ID
            doc_id = hash(doc.page_content)

            # RRF score contribution from this list
            rrf_score = 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed

            doc_scores[doc_id]["doc"] = doc
            doc_scores[doc_id]["score"] += rrf_score
            doc_scores[doc_id]["sources"].append(list_idx)

    # Sort by combined RRF score
    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # Return documents with RRF score in metadata
    result = []
    for item in sorted_docs:
        doc = item["doc"]
        doc.metadata["rrf_score"] = item["score"]
        doc.metadata["rrf_sources"] = item["sources"]
        result.append(doc)

    return result


class RAGFusion:
    """RAG-Fusion: Generate multiple queries and combine results with RRF.

    This improves recall by:
    1. Generating semantically diverse query variants
    2. Retrieving documents for each variant
    3. Combining results using Reciprocal Rank Fusion

    Can use either rule-based expansion (free) or LLM-based generation (paid).
    """

    def __init__(
        self,
        retriever_fn: Callable[[str], List[Document]],
        num_variants: int = 4,
        use_llm: bool = False,
        llm_provider: Optional[Any] = None,
        rrf_k: int = 60,
    ):
        """Initialize RAG-Fusion.

        Args:
            retriever_fn: Function that takes a query and returns documents
            num_variants: Number of query variants to generate
            use_llm: If True, use LLM for query generation (higher quality)
            llm_provider: LLM provider instance (required if use_llm=True)
            rrf_k: RRF constant (default 60)
        """
        self.retriever_fn = retriever_fn
        self.num_variants = num_variants
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.rrf_k = rrf_k
        self.rule_expander = MultiQueryExpander(max_queries=num_variants)

    def generate_queries(self, question: str) -> List[str]:
        """Generate query variants for the question.

        Uses LLM if configured, otherwise falls back to rule-based expansion.
        """
        if self.use_llm and self.llm_provider:
            return self._generate_with_llm(question)
        return self.rule_expander.expand(question)

    def _generate_with_llm(self, question: str) -> List[str]:
        """Generate query variants using LLM."""
        prompt = f"""Generate {self.num_variants - 1} alternative search queries for the following question.
Each query should capture a different aspect or phrasing that might help find relevant documents.

Original question: {question}

Output only the queries, one per line, without numbering or explanation."""

        try:
            response = self.llm_provider.generate(
                system_prompt="You are a search query generator. Generate diverse search queries.",
                user_prompt=prompt,
                max_tokens=200,
                temperature=0.7,  # Some creativity for diverse queries
            )

            variants = [question]  # Always include original
            if response.content:
                for line in response.content.strip().split("\n"):
                    line = line.strip()
                    if line and len(line) > 5 and len(variants) < self.num_variants:
                        variants.append(line)

            return variants

        except Exception as e:
            print(f"  LLM query generation failed: {e}, falling back to rules")
            return self.rule_expander.expand(question)

    def retrieve(self, question: str, top_k: int = 10) -> List[Document]:
        """Retrieve documents using RAG-Fusion.

        Args:
            question: Original question
            top_k: Number of documents to return

        Returns:
            Documents ranked by RRF score
        """
        # Generate query variants
        queries = self.generate_queries(question)

        # Retrieve for each query
        all_results = []
        for query in queries:
            try:
                docs = self.retriever_fn(query)
                all_results.append(docs)
            except Exception as e:
                print(f"  Retrieval failed for query '{query[:50]}...': {e}")
                continue

        if not all_results:
            # Fallback to original query only
            return self.retriever_fn(question)[:top_k]

        # Combine with RRF
        fused = reciprocal_rank_fusion(all_results, k=self.rrf_k)

        return fused[:top_k]


def create_rag_fusion_retriever(
    base_retriever: Any,
    num_variants: int = 4,
    use_llm: bool = False,
    llm_provider: Optional[Any] = None,
) -> RAGFusion:
    """Factory function to create a RAG-Fusion retriever.

    Args:
        base_retriever: LangChain retriever or similar with invoke() method
        num_variants: Number of query variants
        use_llm: Use LLM for query generation
        llm_provider: LLM provider (from src.providers)

    Returns:
        RAGFusion instance
    """
    def retriever_fn(query: str) -> List[Document]:
        return base_retriever.invoke(query)

    return RAGFusion(
        retriever_fn=retriever_fn,
        num_variants=num_variants,
        use_llm=use_llm,
        llm_provider=llm_provider,
    )


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    test_questions = [
        "What is the FY2018 capital expenditure amount for 3M?",
        "What was Apple's revenue in 2022?",
        "Is 3M a capital-intensive business based on FY2022 data?",
        "How much profit did Microsoft make in Q4 2021?",
    ]

    print("=" * 60)
    print("RULE-BASED EXPANSION")
    print("=" * 60)
    expander = MultiQueryExpander()
    for q in test_questions:
        print(f"\nOriginal: {q}")
        variants = expander.expand(q)
        for i, v in enumerate(variants):
            print(f"  [{i}] {v}")

    print("\n" + "=" * 60)
    print("RRF EXAMPLE")
    print("=" * 60)

    # Simulate ranked lists from different queries
    class MockDoc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {}

    list1 = [MockDoc("Doc A"), MockDoc("Doc B"), MockDoc("Doc C")]
    list2 = [MockDoc("Doc B"), MockDoc("Doc A"), MockDoc("Doc D")]
    list3 = [MockDoc("Doc C"), MockDoc("Doc D"), MockDoc("Doc A")]

    fused = reciprocal_rank_fusion([list1, list2, list3])
    print("\nFused results:")
    for i, doc in enumerate(fused[:5]):
        print(f"  [{i}] {doc.page_content} (RRF: {doc.metadata['rrf_score']:.4f})")
