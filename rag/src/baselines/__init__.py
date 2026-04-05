"""Baseline RAG implementations for comparison.

These are simplified approximations of published RAG methods:
- Self-RAG: Self-reflection after generation (Asai et al. 2023)
- CRAG: Corrective retrieval with relevance scoring (Yan et al. 2024)
- Adaptive RAG: Query complexity-based routing (Jeong et al. 2024)
"""

from .base import BaselineRAG
from .self_rag import SelfRAG
from .crag import CRAG
from .adaptive_rag import AdaptiveRAG

__all__ = [
    "BaselineRAG",
    "SelfRAG",
    "CRAG",
    "AdaptiveRAG",
]
