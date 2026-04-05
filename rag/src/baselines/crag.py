"""CRAG: Corrective RAG (Yan et al. 2024).

This is a simplified approximation that:
1. Scores retrieved documents for relevance
2. If relevance is low, triggers "web search" augmentation
   (we simulate this by re-querying with an expanded query)
3. Uses knowledge refinement before generation

The original CRAG has three actions: CORRECT, INCORRECT, AMBIGUOUS.
We approximate this with a relevance threshold.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .base import BaselineRAG, BaselineResult


class CRAG(BaselineRAG):
    """Corrective RAG baseline.

    Key idea: Evaluate retrieval quality and correct if needed.
    """

    def __init__(
        self,
        model_name: str = None,
        judge_model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        relevance_threshold: float = 0.5,
        db=None,  # ChromaDB for re-retrieval
    ):
        """Initialize CRAG.

        Args:
            model_name: Generation model
            judge_model: Model for relevance scoring
            temperature: Generation temperature
            max_tokens: Max tokens for response
            relevance_threshold: Score below which to trigger correction
            db: ChromaDB instance for re-retrieval
        """
        super().__init__(model_name, judge_model, temperature, max_tokens)
        self.relevance_threshold = relevance_threshold
        self.db = db

    @property
    def name(self) -> str:
        return "CRAG"

    def score_retrieval_relevance(
        self,
        question: str,
        docs: List[Document],
    ) -> Tuple[float, List[float]]:
        """Score how relevant the retrieved documents are to the question.

        Returns:
            Tuple of (overall_score, per_doc_scores)
        """
        if not docs:
            return 0.0, []

        # Score each document
        doc_scores = []
        for doc in docs[:5]:  # Only score top 5 for efficiency
            score = self._score_single_doc(question, doc)
            doc_scores.append(score)

        overall_score = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
        return overall_score, doc_scores

    def _score_single_doc(self, question: str, doc: Document) -> float:
        """Score a single document's relevance."""
        relevance_prompt = f"""Rate how relevant this document is to answering the question.

Question: {question}

Document excerpt:
{doc.page_content[:1000]}

Rate relevance from 0.0 to 1.0:
- 1.0: Directly answers the question
- 0.7: Contains highly relevant information
- 0.5: Partially relevant
- 0.3: Tangentially related
- 0.0: Completely irrelevant

Output only a number between 0.0 and 1.0:"""

        response = self.judge_provider.generate(
            system_prompt="You are a relevance scorer. Output only a number.",
            user_prompt=relevance_prompt,
            max_tokens=10,
            temperature=0.0,
        )

        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

    def expand_query(self, question: str) -> str:
        """Generate an expanded query for re-retrieval.

        This simulates the "web search" action in CRAG by
        generating query expansions.
        """
        expansion_prompt = f"""The original question didn't retrieve good results.
Generate 2-3 alternative search queries that might find better information.

Original question: {question}

Output format (one query per line):
Query 1: <expanded query>
Query 2: <alternative phrasing>
Query 3: <related terms>"""

        response = self.gen_provider.generate(
            system_prompt="You are a query expansion expert.",
            user_prompt=expansion_prompt,
            max_tokens=150,
            temperature=0.3,
        )

        # Combine original with expansions
        expansions = response.content.strip()
        return f"{question}\n\nRelated queries:\n{expansions}"

    def refine_knowledge(
        self,
        question: str,
        docs: List[Document],
        doc_scores: List[float],
    ) -> str:
        """Refine knowledge by filtering and summarizing relevant parts.

        This implements CRAG's knowledge refinement step.
        """
        # Filter to only include sufficiently relevant docs
        relevant_docs = []
        for doc, score in zip(docs, doc_scores):
            if score >= self.relevance_threshold:
                relevant_docs.append(doc)

        if not relevant_docs:
            # Fall back to all docs if none pass threshold
            relevant_docs = docs[:3]

        # Extract relevant passages
        context = self.format_context(relevant_docs)

        refinement_prompt = f"""Extract the specific information needed to answer this question.
Remove irrelevant details and focus on key facts.

Question: {question}

Context:
{context}

Refined knowledge (key facts only):"""

        response = self.gen_provider.generate(
            system_prompt="You are a knowledge refiner. Extract relevant facts concisely.",
            user_prompt=refinement_prompt,
            max_tokens=300,
            temperature=0.0,
        )

        return response.content

    def process(
        self,
        question: str,
        docs: List[Document],
        gold_answer: str = None,
    ) -> BaselineResult:
        """Process question with corrective retrieval."""
        start_time = time.time()

        metadata = {
            "retrieval_action": "CORRECT",  # CORRECT, INCORRECT, or AMBIGUOUS
            "relevance_score": 0.0,
            "doc_scores": [],
            "correction_triggered": False,
            "query_expanded": False,
        }

        # Step 1: Score retrieval relevance
        relevance_score, doc_scores = self.score_retrieval_relevance(question, docs)
        metadata["relevance_score"] = relevance_score
        metadata["doc_scores"] = doc_scores[:5]

        # Step 2: Decide action based on relevance
        if relevance_score >= 0.7:
            metadata["retrieval_action"] = "CORRECT"
        elif relevance_score >= 0.4:
            metadata["retrieval_action"] = "AMBIGUOUS"
            metadata["correction_triggered"] = True
        else:
            metadata["retrieval_action"] = "INCORRECT"
            metadata["correction_triggered"] = True

        # Step 3: Correction if needed
        if metadata["correction_triggered"] and self.db is not None:
            # Expand query and re-retrieve
            expanded_query = self.expand_query(question)
            metadata["query_expanded"] = True

            # Re-retrieve with expanded query
            new_docs = self.db.similarity_search(expanded_query, k=10)
            if new_docs:
                # Combine with original docs, prioritizing new ones
                docs = new_docs[:5] + docs[:5]
                # Re-score
                relevance_score, doc_scores = self.score_retrieval_relevance(
                    question, docs[:10]
                )
                metadata["relevance_score_after_correction"] = relevance_score

        # Step 4: Knowledge refinement
        refined_knowledge = self.refine_knowledge(question, docs, doc_scores)

        # Step 5: Generate answer from refined knowledge
        system_prompt = """You are a precise financial analysis assistant.
Answer based only on the provided refined knowledge."""

        user_prompt = f"""Answer the question using the refined knowledge below.

Refined Knowledge:
{refined_knowledge}

Question: {question}

Answer:"""

        response = self.gen_provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        latency_ms = (time.time() - start_time) * 1000

        return BaselineResult(
            question=question,
            answer=response.content,
            score=relevance_score,
            latency_ms=latency_ms,
            metadata=metadata,
        )
