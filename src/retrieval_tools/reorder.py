"""Lost-in-Middle Reordering for LLM context.

LLMs tend to pay more attention to content at the beginning and end of
their context window, while ignoring content in the middle. This module
reorders retrieved documents to place the most relevant ones at positions
where they're more likely to be attended to.

Reference: "Lost in the Middle: How Language Models Use Long Contexts"
https://arxiv.org/abs/2307.03172
"""

from typing import List
from langchain_core.documents import Document


def reorder_for_llm(docs: List[Document]) -> List[Document]:
    """Reorder documents so best are at start AND end.

    The "Lost in the Middle" paper shows LLMs attend most to:
    1. Beginning of context (primacy effect)
    2. End of context (recency effect)

    This function interleaves documents: best at start, 2nd best at end,
    3rd best after first, 4th best before last, etc.

    Input order (by relevance):  [1, 2, 3, 4, 5, 6, 7, 8]
    Output order (for attention): [1, 3, 5, 7, 8, 6, 4, 2]

    Args:
        docs: Documents sorted by relevance (best first)

    Returns:
        Reordered documents with best at start/end
    """
    if len(docs) <= 2:
        return docs

    # Split into two halves
    # First half: odd indices (0, 2, 4...) - goes to start
    # Second half: even indices (1, 3, 5...) reversed - goes to end
    start_docs = docs[::2]  # 1st, 3rd, 5th...
    end_docs = docs[1::2][::-1]  # 2nd, 4th, 6th... reversed

    return start_docs + end_docs


def reorder_alternating(docs: List[Document]) -> List[Document]:
    """Alternative reordering: alternate best/worst placement.

    Places documents in order: 1st, last, 2nd, 2nd-last, 3rd, 3rd-last...
    This ensures both primacy and recency positions get high-relevance docs.

    Input:  [1, 2, 3, 4, 5, 6]
    Output: [1, 6, 2, 5, 3, 4]
    """
    if len(docs) <= 2:
        return docs

    result = []
    left, right = 0, len(docs) - 1

    while left <= right:
        result.append(docs[left])
        if left != right:
            result.append(docs[right])
        left += 1
        right -= 1

    return result


def reorder_bookend(docs: List[Document], n_bookend: int = 2) -> List[Document]:
    """Put top N docs at start AND duplicate at end.

    For critical information, place top docs at both start and end
    of context to maximize attention.

    Args:
        docs: Documents sorted by relevance
        n_bookend: Number of top docs to duplicate at end

    Returns:
        Documents with top N appearing at both start and end
    """
    if len(docs) <= n_bookend:
        return docs

    # Top N at start, middle docs, then top N again at end
    top_docs = docs[:n_bookend]
    middle_docs = docs[n_bookend:]

    return top_docs + middle_docs + top_docs
