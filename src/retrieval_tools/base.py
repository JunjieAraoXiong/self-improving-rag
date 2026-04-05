"""Lightweight interfaces for retrieval tools and pipelines."""

from typing import Protocol, List
from langchain_core.documents import Document


class RetrieverProtocol(Protocol):
    """Minimal protocol for retrievers used in pipelines."""

    def invoke(self, query: str) -> List[Document]:  # pragma: no cover - interface
        ...


class RetrievalPipeline(Protocol):
    """Protocol for retrieval pipelines that may compose multiple tools."""

    def retrieve(self, question: str) -> List[Document]:  # pragma: no cover - interface
        ...
