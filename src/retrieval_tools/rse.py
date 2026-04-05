"""Relevant Segment Extraction (RSE) - dsRAG technique.

Combines adjacent high-scoring chunks into coherent segments for richer context.
This is a query-time technique that doesn't require re-ingestion.

Reference: https://github.com/D-Star-AI/dsRAG

Key insight: Simple questions need single chunks, but complex questions benefit
from combining multiple adjacent chunks into segments that provide fuller context.

Adjacency Detection:
- Primary: Uses 'page' metadata when available
- Fallback: Uses 'chunk_index' metadata (sequential chunk order in document)
- Last resort: Uses retrieval rank within document group

This allows RSE to work with any chunking strategy, not just page-based.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document


def get_chunk_position(chunk: Document, fallback_rank: int = 0) -> int:
    """Get chunk position for adjacency detection.

    Priority:
    1. 'page' metadata (traditional page-based chunking)
    2. 'chunk_index' metadata (sequential order from ingestion)
    3. fallback_rank (position in current retrieval results)

    Args:
        chunk: Document with metadata
        fallback_rank: Rank to use if no position metadata exists

    Returns:
        Integer position for adjacency comparison
    """
    # Try page first (original behavior)
    page = chunk.metadata.get("page")
    if page is not None:
        return int(page)

    # Try chunk_index (sequential order from ingestion)
    chunk_index = chunk.metadata.get("chunk_index")
    if chunk_index is not None:
        return int(chunk_index)

    # Fallback to retrieval rank within document
    return fallback_rank


# =============================================================================
# RSE Configuration Presets
# =============================================================================

RSE_PRESETS = {
    "balanced": {
        "irrelevant_chunk_penalty": 0.18,
        "minimum_value": 0.5,
        "max_segment_length": 15,
        "overall_max_length": 30,
        "decay_rate": 20,
    },
    "precision": {
        "irrelevant_chunk_penalty": 0.2,
        "minimum_value": 0.7,
        "max_segment_length": 15,
        "overall_max_length": 30,
        "decay_rate": 20,
    },
    "find_all": {
        "irrelevant_chunk_penalty": 0.18,
        "minimum_value": 0.4,
        "max_segment_length": 40,
        "overall_max_length": 100,
        "decay_rate": 20,
    },
}


def score_chunk(
    rank: int,
    relevance: float,
    decay_rate: float = 20,
    penalty: float = 0.18,
) -> float:
    """Score a chunk using dsRAG's formula.

    Formula: value = exp(-rank / decay_rate) * relevance - penalty

    Args:
        rank: Position in the ranked list (0-indexed)
        relevance: Absolute relevance score (e.g., from reranker, 0-1)
        decay_rate: Controls how fast rank penalty decays (higher = slower decay)
        penalty: Fixed penalty for including irrelevant chunks

    Returns:
        Chunk value score (can be negative if chunk is low-ranked/irrelevant)
    """
    return math.exp(-rank / decay_rate) * relevance - penalty


def extract_relevant_segments(
    chunks: List[Document],
    relevance_scores: Optional[List[float]] = None,
    max_segment_length: int = 15,
    overall_max_length: int = 30,
    minimum_value: float = 0.5,
    irrelevant_penalty: float = 0.18,
    decay_rate: float = 20,
    preset: Optional[str] = None,
) -> List[str]:
    """dsRAG-style Relevant Segment Extraction.

    Combines adjacent high-scoring chunks from the same document into
    coherent segments. This provides richer context than individual chunks.

    Args:
        chunks: Retrieved documents (should have 'source' and 'page' in metadata)
        relevance_scores: Per-chunk relevance scores (0-1). If None, uses
            metadata['rerank_score'] or falls back to position-based scoring.
        max_segment_length: Max chunks per segment
        overall_max_length: Max total chunks across all segments
        minimum_value: Minimum segment value to include
        irrelevant_penalty: Penalty for including low-relevance chunks
        decay_rate: Rank decay rate for scoring formula
        preset: Use a preset config ("balanced", "precision", "find_all")

    Returns:
        List of segment texts (merged chunk contents)
    """
    if not chunks:
        return []

    # Apply preset if specified
    if preset and preset in RSE_PRESETS:
        config = RSE_PRESETS[preset]
        max_segment_length = config["max_segment_length"]
        overall_max_length = config["overall_max_length"]
        minimum_value = config["minimum_value"]
        irrelevant_penalty = config["irrelevant_chunk_penalty"]
        decay_rate = config["decay_rate"]

    # Get relevance scores
    if relevance_scores is None:
        # Try to get from metadata, otherwise use position-based
        relevance_scores = []
        for i, chunk in enumerate(chunks):
            score = chunk.metadata.get("rerank_score")
            if score is not None:
                relevance_scores.append(score)
            else:
                # Fallback: exponential decay based on position
                relevance_scores.append(math.exp(-i / 10))

    # Score each chunk
    scored_chunks: List[Dict[str, Any]] = []
    for rank, (chunk, rel_score) in enumerate(zip(chunks, relevance_scores)):
        value = score_chunk(rank, rel_score, decay_rate, irrelevant_penalty)
        scored_chunks.append({
            "chunk": chunk,
            "value": value,
            "rank": rank,
            "relevance": rel_score,
            "source": chunk.metadata.get("source", "unknown"),
            "position": get_chunk_position(chunk, fallback_rank=rank),
            "used": False,
        })

    # Group by document source
    doc_groups: Dict[str, List[Dict]] = defaultdict(list)
    for item in scored_chunks:
        doc_groups[item["source"]].append(item)

    # Sort each group by position for adjacency detection
    # Position can be: page number, chunk_index, or retrieval rank
    for source in doc_groups:
        doc_groups[source].sort(key=lambda x: (x["position"], x["rank"]))

    # Greedy segment extraction
    segments: List[str] = []
    total_chunks_used = 0

    while total_chunks_used < overall_max_length:
        best_segment: Optional[Tuple[str, int, int, List[Dict]]] = None
        best_value = minimum_value

        # Find best contiguous segment across all documents
        for source, items in doc_groups.items():
            # Only consider unused chunks
            available = [i for i, item in enumerate(items) if not item["used"]]

            for start_idx in available:
                segment_value = 0.0
                segment_items: List[Dict] = []
                prev_position = items[start_idx]["position"]

                # Extend segment while chunks are adjacent and positive
                for idx in range(start_idx, len(items)):
                    item = items[idx]
                    if item["used"]:
                        continue

                    # Check adjacency (same position or next position)
                    # Works with page numbers, chunk indices, or ranks
                    current_position = item["position"]
                    if segment_items and (current_position - prev_position) > 1:
                        # Gap detected, stop extending
                        break

                    # Only include if adds positive value
                    if item["value"] > 0 or not segment_items:
                        segment_value += item["value"]
                        segment_items.append(item)
                        prev_position = current_position

                        if len(segment_items) >= max_segment_length:
                            break
                    else:
                        # Negative value chunk, stop extending
                        break

                # Check if this is the best segment so far
                if segment_value > best_value and segment_items:
                    best_value = segment_value
                    best_segment = (source, start_idx, len(segment_items), segment_items)

        # No more valid segments
        if best_segment is None:
            break

        # Add the best segment
        source, start_idx, length, items = best_segment
        segment_text = "\n\n".join([item["chunk"].page_content for item in items])
        segments.append(segment_text)
        total_chunks_used += len(items)

        # Mark chunks as used
        for item in items:
            item["used"] = True

    return segments


def extract_segments_with_metadata(
    chunks: List[Document],
    relevance_scores: Optional[List[float]] = None,
    preset: str = "balanced",
) -> List[Dict[str, Any]]:
    """Extract segments with full metadata for debugging/analysis.

    Returns list of dicts with:
        - text: Merged segment text
        - source: Document source
        - positions: List of chunk positions (page numbers or chunk indices)
        - num_chunks: Number of chunks in segment
        - total_value: Sum of chunk values
        - chunks: Original chunk documents
    """
    if not chunks:
        return []

    config = RSE_PRESETS.get(preset, RSE_PRESETS["balanced"])

    # Get relevance scores
    if relevance_scores is None:
        relevance_scores = []
        for i, chunk in enumerate(chunks):
            score = chunk.metadata.get("rerank_score")
            if score is not None:
                relevance_scores.append(score)
            else:
                relevance_scores.append(math.exp(-i / 10))

    # Score chunks
    scored_chunks = []
    for rank, (chunk, rel_score) in enumerate(zip(chunks, relevance_scores)):
        value = score_chunk(
            rank, rel_score,
            config["decay_rate"],
            config["irrelevant_chunk_penalty"]
        )
        scored_chunks.append({
            "chunk": chunk,
            "value": value,
            "source": chunk.metadata.get("source", "unknown"),
            "position": get_chunk_position(chunk, fallback_rank=rank),
            "used": False,
        })

    # Group by source
    doc_groups = defaultdict(list)
    for item in scored_chunks:
        doc_groups[item["source"]].append(item)

    for source in doc_groups:
        doc_groups[source].sort(key=lambda x: x["position"])

    # Extract segments
    segments_with_meta = []
    total_used = 0

    while total_used < config["overall_max_length"]:
        best = None
        best_value = config["minimum_value"]

        for source, items in doc_groups.items():
            available = [i for i, item in enumerate(items) if not item["used"]]

            for start_idx in available:
                seg_value = 0.0
                seg_items = []
                prev_position = items[start_idx]["position"]

                for idx in range(start_idx, len(items)):
                    item = items[idx]
                    if item["used"]:
                        continue

                    if seg_items and (item["position"] - prev_position) > 1:
                        break

                    if item["value"] > 0 or not seg_items:
                        seg_value += item["value"]
                        seg_items.append(item)
                        prev_position = item["position"]

                        if len(seg_items) >= config["max_segment_length"]:
                            break
                    else:
                        break

                if seg_value > best_value and seg_items:
                    best_value = seg_value
                    best = (source, seg_items, seg_value)

        if best is None:
            break

        source, items, value = best
        segments_with_meta.append({
            "text": "\n\n".join([item["chunk"].page_content for item in items]),
            "source": source,
            "positions": sorted(set(item["position"] for item in items)),
            "num_chunks": len(items),
            "total_value": value,
            "chunks": [item["chunk"] for item in items],
        })
        total_used += len(items)

        for item in items:
            item["used"] = True

    return segments_with_meta


# =============================================================================
# Simple Adjacency-Based Merging (Fallback)
# =============================================================================

def merge_adjacent_chunks(
    chunks: List[Document],
    max_gap: int = 1,
    max_segment_size: int = 5,
) -> List[str]:
    """Simple fallback: merge chunks from same document at adjacent positions.

    Simpler than full RSE but still provides some segment merging benefit.
    Uses page numbers when available, otherwise falls back to chunk_index.

    Args:
        chunks: List of retrieved documents
        max_gap: Maximum position gap to consider "adjacent"
        max_segment_size: Maximum chunks per segment

    Returns:
        List of merged segment texts
    """
    if not chunks:
        return []

    # Sort by source and position
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            c.metadata.get("source", ""),
            get_chunk_position(c, fallback_rank=0)
        )
    )

    segments = []
    current_segment = []
    current_source = None
    current_position = -999

    for i, chunk in enumerate(sorted_chunks):
        source = chunk.metadata.get("source", "")
        position = get_chunk_position(chunk, fallback_rank=i)

        # Check if should start new segment
        if (source != current_source or
            (position - current_position) > max_gap or
            len(current_segment) >= max_segment_size):

            if current_segment:
                segments.append("\n\n".join(c.page_content for c in current_segment))
            current_segment = [chunk]
            current_source = source
        else:
            current_segment.append(chunk)

        current_position = position

    # Don't forget last segment
    if current_segment:
        segments.append("\n\n".join(c.page_content for c in current_segment))

    return segments
