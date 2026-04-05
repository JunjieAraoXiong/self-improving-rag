"""Hybrid BM25 + semantic retriever builder."""

from typing import Any, Dict, List, Tuple, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Default ensemble weights: (BM25_weight, Semantic_weight)
DEFAULT_ENSEMBLE_WEIGHTS = (0.5, 0.5)


def build_hybrid_retriever(
    db: Chroma,
    top_k: int,
    weights: Optional[Tuple[float, float]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create an ensemble of BM25 + semantic retrievers.

    Args:
        db: ChromaDB instance
        top_k: Number of documents to retrieve
        weights: Tuple of (bm25_weight, semantic_weight). Defaults to (0.5, 0.5).
                 Higher BM25 weight = better for exact keyword matches.
                 Higher semantic weight = better for meaning-based similarity.
        metadata_filter: Optional ChromaDB where filter for pre-filtering documents.
                        Example: {"company": "3M"} or {"$and": [{"company": "3M"}, {"year": 2018}]}

    Returns:
        EnsembleRetriever combining BM25 and semantic search
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers.ensemble import EnsembleRetriever

    if weights is None:
        weights = DEFAULT_ENSEMBLE_WEIGHTS

    # Get documents - with optional pre-filtering
    if metadata_filter:
        all_docs = db.get(where=metadata_filter)
    else:
        all_docs = db.get()

    from langchain_core.documents import Document as LCDocument

    documents: List[LCDocument] = [
        LCDocument(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]

    # If filter returned no docs, fall back to unfiltered
    if not documents:
        all_docs = db.get()
        documents = [
            LCDocument(page_content=text, metadata=meta)
            for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = top_k

    # Semantic retriever also needs the filter
    search_kwargs = {"k": top_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    semantic_retriever = db.as_retriever(search_kwargs=search_kwargs)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=list(weights),
    )


def set_retriever_k(retriever: Any, k: int) -> None:
    """Update k for an ensemble retriever."""
    if hasattr(retriever, "retrievers") and len(retriever.retrievers) >= 2:
        # BM25
        if hasattr(retriever.retrievers[0], "k"):
            retriever.retrievers[0].k = k
        # Semantic
        if hasattr(retriever.retrievers[1], "search_kwargs"):
            retriever.retrievers[1].search_kwargs["k"] = k


def take_top_k(docs: list[Document], k: int) -> list[Document]:
    """Return the first k documents."""
    return docs[:k]
