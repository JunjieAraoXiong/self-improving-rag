"""Cross-encoder reranking tool.

Supports both local cross-encoders and API-based rerankers (Cohere, Jina).
"""

import os
from typing import List, Optional, Protocol
from langchain_core.documents import Document


# Default to stronger BGE reranker
DEFAULT_RERANKER = "BAAI/bge-reranker-large"

# Available reranker models (in order of quality/speed tradeoff)
RERANKER_MODELS = {
    # Local models (free)
    "BAAI/bge-reranker-large": "High quality, slower (FREE)",
    "BAAI/bge-reranker-base": "Good quality, medium speed (FREE)",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "Fast, lower quality (FREE)",
    # API models (paid but SOTA quality)
    "cohere": "Cohere rerank-v3 - SOTA quality, +28% NDCG ($1/1K queries)",
}


class Reranker:
    """Wraps a cross-encoder reranker."""

    def __init__(self, model_name: str = DEFAULT_RERANKER):
        from sentence_transformers import CrossEncoder
        import torch

        self.model_name = model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading reranker: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        question: str,
        docs: List[Document],
        top_k: int,
        score_threshold: float = 0.0
    ) -> List[Document]:
        """Rerank docs and return top_k that exceed the threshold.

        Args:
            question: The query to rerank against
            docs: List of documents to rerank
            top_k: Maximum number of documents to return
            score_threshold: Minimum reranker score to include (0.0 = no filtering).
                            Typical thresholds: 0.1 (loose), 0.3 (moderate), 0.5 (strict)

        Returns:
            List of top_k documents that pass the threshold, sorted by score
        """
        if not docs:
            return docs

        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)

        # Filter by threshold and sort
        scored_docs = [
            (doc, score) for doc, score in zip(docs, scores)
            if score >= score_threshold
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs[:top_k]]


class CohereReranker:
    """Cohere API-based reranker - SOTA quality.

    Requires COHERE_API_KEY environment variable.
    Pricing: ~$1 per 1,000 queries.
    """

    def __init__(self, model_name: str = "rerank-english-v3.0"):
        import cohere

        self.model_name = model_name
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")
        self.client = cohere.Client(api_key)
        print(f"  Initialized Cohere reranker: {model_name}")

    def rerank(
        self,
        question: str,
        docs: List[Document],
        top_k: int,
        score_threshold: float = 0.0
    ) -> List[Document]:
        """Rerank docs using Cohere API.

        Args:
            question: The query to rerank against
            docs: List of documents to rerank
            top_k: Maximum number of documents to return
            score_threshold: Minimum relevance score (0-1 scale)

        Returns:
            List of top_k documents that pass the threshold, sorted by relevance
        """
        if not docs:
            return docs

        # Call Cohere API
        results = self.client.rerank(
            model=self.model_name,
            query=question,
            documents=[doc.page_content for doc in docs],
            top_n=min(top_k * 2, len(docs)),  # Get extra for threshold filtering
            return_documents=False,
        )

        # Filter by threshold and return top_k
        reranked = []
        for result in results.results:
            if result.relevance_score >= score_threshold:
                doc = docs[result.index]
                # Store score in metadata for CRAG confidence checking
                doc.metadata["rerank_score"] = result.relevance_score
                reranked.append(doc)
            if len(reranked) >= top_k:
                break

        return reranked


# Global reranker cache to avoid reloading
_reranker_cache: dict = {}


def get_reranker(
    existing: Optional["Reranker"] = None,
    model_name: str = DEFAULT_RERANKER
) -> "Reranker":
    """Return an existing reranker or create/cache a new one.

    Supports:
    - Local cross-encoders: BAAI/bge-reranker-*, cross-encoder/*
    - API rerankers: "cohere" (SOTA, +28% NDCG)
    """
    if existing and existing.model_name == model_name:
        return existing

    # Check cache
    if model_name in _reranker_cache:
        return _reranker_cache[model_name]

    # Create appropriate reranker based on model name
    if model_name == "cohere":
        reranker = CohereReranker()
    else:
        # Default to local cross-encoder
        reranker = Reranker(model_name=model_name)

    _reranker_cache[model_name] = reranker
    return reranker
