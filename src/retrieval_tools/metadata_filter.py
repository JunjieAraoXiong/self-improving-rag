"""Metadata filtering as a reusable tool."""

from typing import List, Tuple
from langchain_core.documents import Document


def filter_with_question_metadata(question: str, docs: List[Document]) -> Tuple[List[Document], bool]:
    """Filter docs by company/year metadata extracted from the question."""
    from src.metadata_utils import extract_metadata_from_question, filter_chunks_by_metadata

    question_metadata = extract_metadata_from_question(question)
    filtered_docs = filter_chunks_by_metadata(docs, question_metadata)
    used_metadata = bool(question_metadata["years"] or question_metadata["companies"])

    if used_metadata:
        return filtered_docs, True

    return docs, False
