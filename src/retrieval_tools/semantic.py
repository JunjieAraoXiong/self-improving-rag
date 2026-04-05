"""Semantic-only retriever builder."""

from typing import Any
from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_semantic_retriever(db: Chroma, top_k: int) -> Any:
    """Return a Chroma retriever configured for semantic search."""
    return db.as_retriever(search_kwargs={"k": top_k})


def set_retriever_k(retriever: Any, k: int) -> None:
    """Update k for a semantic-only retriever."""
    if hasattr(retriever, "search_kwargs"):
        retriever.search_kwargs["k"] = k


def take_top_k(docs: list[Document], k: int) -> list[Document]:
    """Return the first k documents."""
    return docs[:k]
