"""Source-grounding checks backed by the canonical numeric parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple

from evaluation.numeric_check import (
    NumericValue,
    SCALE_MULTIPLIERS,
    apply_context_scale,
    extract_numbers as parse_numbers,
    find_numeric_match,
    is_likely_year,
    strip_evidence_citations,
)


@dataclass
class NumericVerificationResult:
    """Result of checking answer quantities against retrieved evidence."""

    score: float
    answer_numbers: List[str] = field(default_factory=list)
    source_numbers: Set[float] = field(default_factory=set)
    verified_numbers: List[str] = field(default_factory=list)
    flagged_numbers: List[str] = field(default_factory=list)
    details: str = ""


# Backward-compatible public alias.
MULTIPLIERS = SCALE_MULTIPLIERS


def extract_numbers(text: str) -> List[Tuple[str, float]]:
    """Return legacy ``(raw, normalized_value)`` tuples."""

    return [(number.raw, number.value) for number in parse_numbers(text)]


def numbers_match(value1: float, value2: float, tolerance: float = 0.001) -> bool:
    """Backward-compatible scalar comparison helper."""

    if value1 == value2:
        return True
    if value1 == 0 or value2 == 0:
        return abs(value1 - value2) <= tolerance
    return abs(value1 - value2) / max(abs(value1), abs(value2)) <= tolerance


def _source_values(chunks: List) -> List[NumericValue]:
    values: List[NumericValue] = []
    for chunk in chunks:
        if hasattr(chunk, "page_content"):
            text = chunk.page_content
        elif isinstance(chunk, str):
            text = chunk
        else:
            continue

        parsed = parse_numbers(text)
        values.extend(parsed)

        # Import locally to keep this compatibility module lightweight.
        from evaluation.numeric_check import infer_scale

        scale = infer_scale(text)
        if scale:
            values.extend(
                apply_context_scale(number, scale)
                for number in parsed
                if not number.explicit_scale and not is_likely_year(number)
            )
    return values


def verify_numeric_answer(
    predicted_answer: str,
    retrieved_chunks: List,
    tolerance: float = 0.001,
) -> NumericVerificationResult:
    """Check typed answer quantities against retrieved source quantities.

    Values embedded inside evidence citations are excluded from the answer side
    so a quote cannot verify itself. Currency, percentage, ratio, and scale
    compatibility are enforced by the shared parser.
    """

    answer_values = parse_numbers(strip_evidence_citations(predicted_answer))
    source_values = _source_values(retrieved_chunks)

    verified: List[str] = []
    flagged: List[str] = []
    for answer_value in answer_values:
        match = find_numeric_match(
            answer_value,
            source_values,
            relative_tolerance=tolerance,
            absolute_tolerance=tolerance,
        )
        if match is None:
            flagged.append(answer_value.raw)
        else:
            verified.append(answer_value.raw)

    total = len(answer_values)
    if total == 0:
        score = 1.0
        details = "No numeric claims found in answer."
    else:
        score = len(verified) / total
        details = (
            f"Verified {len(verified)}/{total} typed numeric claims."
            if not flagged
            else f"Verified {len(verified)}/{total}; unsupported: {flagged}"
        )

    return NumericVerificationResult(
        score=score,
        answer_numbers=[number.raw for number in answer_values],
        source_numbers={number.value for number in source_values},
        verified_numbers=verified,
        flagged_numbers=flagged,
        details=details,
    )


def get_verification_summary(result: NumericVerificationResult) -> str:
    """Generate a human-readable verification summary."""

    return "\n".join(
        [
            f"Numeric Verification Score: {result.score:.2%}",
            f"Numbers in answer: {result.answer_numbers}",
            f"Verified: {result.verified_numbers}",
            f"Flagged (potential hallucination): {result.flagged_numbers}",
            f"Details: {result.details}",
        ]
    )
