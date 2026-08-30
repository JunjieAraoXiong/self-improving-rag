"""Unit-aware numeric parsing and comparison for financial QA.

The evaluator previously used four overlapping regular expressions and then
compared only the first extracted number. That made evaluation sensitive to
formatting and allowed numbers inside evidence citations to leak into the
predicted answer. This module provides one canonical representation shared by
gold-answer evaluation, blind grounding, and citation verification.

Values are normalized to base units for scaled quantities and to percentage
points for percentages (for example, ``150 bps == 1.5%``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NumericValue:
    """A parsed numeric quantity with normalized value and source span."""

    raw: str
    value: float
    unit: str
    confidence: float
    start: int = -1
    end: int = -1
    kind: str = "number"  # number, currency, percent, or ratio
    currency: str = ""
    explicit_scale: bool = False


SCALE_MULTIPLIERS = {
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "trillions": 1e12,
    "tn": 1e12,
    "t": 1e12,
}

_CANONICAL_SCALE = {
    1e3: "thousand",
    1e6: "million",
    1e9: "billion",
    1e12: "trillion",
}

_CURRENCY_CODES = {
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}

_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    (?P<open>\()?\s*
    (?P<sign_before>[+-])?\s*
    (?P<currency>US\$|USD|EUR|GBP|[$€£])?\s*
    (?P<sign_after>[+-])?\s*
    (?P<number>
        (?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)
        (?P<exponent>[eE][+-]?\d+)?
    )
    \s*(?P<close_before>\))?\s*
    (?P<unit>
        basis\s+points?|bps|percentage\s+points?|percent(?:age)?|%|
        trillions?|billions?|millions?|thousands?|tn|bn|mn|mm|[tbmk]|x
    )?
    \s*(?P<close_after>\))?
    (?!\w|\.\d)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_EVIDENCE_CITATION_RE = re.compile(
    r"\[Doc\d+\s*:\s*"
    r"(?:'.*?'|\".*?\"|‘.*?’|“.*?”)\]",
    re.IGNORECASE | re.DOTALL,
)

_DOC_LABEL_RE = re.compile(r"\bDoc(?:ument)?\s*\d+\b", re.IGNORECASE)
_NUMERIC_CHARACTER_TRANSLATION = str.maketrans(
    {
        "−": "-",  # U+2212 MINUS SIGN
        "﹣": "-",  # U+FE63 SMALL HYPHEN-MINUS
        "－": "-",  # U+FF0D FULLWIDTH HYPHEN-MINUS
    }
)


def strip_evidence_citations(text: str) -> str:
    """Remove inline evidence blocks while preserving surrounding prose."""

    return _EVIDENCE_CITATION_RE.sub(" ", text or "")


def _canonical_unit(unit: str) -> Tuple[str, float, str]:
    normalized = re.sub(r"\s+", " ", (unit or "").strip().lower())
    if normalized in {
        "percent",
        "percentage",
        "%",
        "percentage point",
        "percentage points",
    }:
        return "percent", 1.0, "percent"
    if normalized in {"bps", "basis point", "basis points"}:
        return "percent", 0.01, "percent"
    if normalized == "x":
        return "x", 1.0, "ratio"

    multiplier = SCALE_MULTIPLIERS.get(normalized, 1.0)
    if normalized in SCALE_MULTIPLIERS:
        return _CANONICAL_SCALE[multiplier], multiplier, "number"
    return "", 1.0, "number"


def _is_numbered_list_marker(
    text: str,
    match: re.Match,
    start: int,
    end: int,
) -> bool:
    """Return whether an otherwise bare integer is a line-list marker."""

    if (
        match.group("open")
        or match.group("sign_before")
        or match.group("sign_after")
        or match.group("currency")
        or match.group("unit")
        or match.group("exponent")
    ):
        return False

    number = match.group("number")
    if not number.isdigit() or int(number) > 100:
        return False

    line_start = text.rfind("\n", 0, start) + 1
    if text[line_start:start].strip():
        return False

    suffix = text[end:]
    has_consumed_paren = bool(
        match.group("close_before") or match.group("close_after")
    )
    if has_consumed_paren:
        return bool(re.match(r"\s+\S", suffix, flags=re.DOTALL))
    return bool(re.match(r"\.\s+\S", suffix, flags=re.DOTALL))


def extract_numbers(text: str) -> List[NumericValue]:
    """Extract non-overlapping financial quantities from ``text``.

    Supported forms include currencies, comma separators, scale aliases,
    accounting negatives, percentages, basis points, and ratios. Duplicate
    occurrences are retained so citation coverage can be measured per claim.
    """

    if not text:
        return []

    # Translate sign glyphs one-for-one so match spans still index the original
    # text and callers receive the exact source spelling in ``raw``.
    normalized_text = text.translate(_NUMERIC_CHARACTER_TRANSLATION)

    results: List[NumericValue] = []
    for match in _NUMBER_RE.finditer(normalized_text):
        matched_text = match.group(0)
        leading_whitespace = len(matched_text) - len(matched_text.lstrip())
        trailing_whitespace = len(matched_text) - len(matched_text.rstrip())
        start = match.start() + leading_whitespace
        end = match.end() - trailing_whitespace
        raw = text[start:end]

        if _is_numbered_list_marker(normalized_text, match, start, end):
            continue

        try:
            value = float(match.group("number").replace(",", ""))
        except (TypeError, ValueError):
            continue

        unit, multiplier, unit_kind = _canonical_unit(match.group("unit") or "")
        currency_token = (match.group("currency") or "").lower()
        currency = _CURRENCY_CODES.get(currency_token, "")

        is_parenthesized = bool(match.group("open")) and bool(
            match.group("close_before") or match.group("close_after")
        )
        sign = match.group("sign_before") or match.group("sign_after")
        if sign == "-" or is_parenthesized:
            value = -abs(value)

        value *= multiplier
        kind = "currency" if currency else unit_kind
        confidence = 0.5
        if currency:
            confidence += 0.2
        if unit:
            confidence += 0.2
        if "." in match.group("number") or match.group("exponent"):
            confidence += 0.1
        if sign or is_parenthesized:
            confidence += 0.05

        results.append(
            NumericValue(
                raw=raw,
                value=value,
                unit=unit,
                confidence=min(confidence, 1.0),
                start=start,
                end=end,
                kind=kind,
                currency=currency,
                explicit_scale=multiplier != 1.0 or bool(match.group("exponent")),
            )
        )

    return results


def infer_scale(text: str) -> Optional[float]:
    """Infer a single table-wide scale declaration from financial text.

    SEC tables commonly put ``(Dollars in millions)`` in a header and leave
    individual cells unscaled. A scale is returned only for one unambiguous
    declaration.
    """

    declarations = re.findall(
        r"(?:dollars?|[$])?\s*(?:are\s+)?in\s+"
        r"(thousands?|millions?|billions?|trillions?)\b|"
        r"\(\s*(thousands?|millions?|billions?|trillions?)\s*\)",
        text or "",
        flags=re.IGNORECASE,
    )
    units = {
        (first or second).lower()
        for first, second in declarations
        if first or second
    }
    multipliers = {SCALE_MULTIPLIERS[unit] for unit in units}
    if len(multipliers) == 1:
        return next(iter(multipliers))
    return None


def apply_context_scale(value: NumericValue, scale: Optional[float]) -> NumericValue:
    """Apply a table-wide scale to an otherwise unscaled quantity."""

    if not scale or value.explicit_scale or value.kind in {"percent", "ratio"}:
        return value
    return replace(
        value,
        value=value.value * scale,
        unit=_CANONICAL_SCALE[scale],
        explicit_scale=True,
    )


def is_likely_year(value: NumericValue) -> bool:
    """Return whether an untyped integer is probably a calendar/fiscal year."""

    return (
        not value.unit
        and not value.currency
        and not value.explicit_scale
        and value.value.is_integer()
        and 1900 <= value.value <= 2100
    )


def _kinds_compatible(left: NumericValue, right: NumericValue) -> bool:
    if "percent" in {left.kind, right.kind}:
        return left.kind == right.kind == "percent"
    if "ratio" in {left.kind, right.kind}:
        return left.kind == right.kind == "ratio"
    if left.currency and right.currency and left.currency != right.currency:
        return False
    return True


def numbers_match(
    left: NumericValue,
    right: NumericValue,
    relative_tolerance: float = 0.005,
    absolute_tolerance: float = 0.01,
    allow_absolute_value: bool = False,
) -> bool:
    """Return whether two typed quantities are numerically equivalent."""

    if not _kinds_compatible(left, right):
        return False
    left_is_year = is_likely_year(left)
    right_is_year = is_likely_year(right)
    if left_is_year or right_is_year:
        return left_is_year and right_is_year and left.value == right.value

    left_value = abs(left.value) if allow_absolute_value else left.value
    right_value = abs(right.value) if allow_absolute_value else right.value
    return math.isclose(
        left_value,
        right_value,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def find_numeric_match(
    target: NumericValue,
    candidates: Iterable[NumericValue],
    relative_tolerance: float = 0.005,
    absolute_tolerance: float = 0.01,
    allow_absolute_value: bool = False,
) -> Optional[NumericValue]:
    """Return the first compatible candidate for ``target``."""

    for candidate in candidates:
        if numbers_match(
            target,
            candidate,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            allow_absolute_value=allow_absolute_value,
        ):
            return candidate
    return None


def _meaningful_numbers(text: str) -> List[NumericValue]:
    cleaned = strip_evidence_citations(text)
    cleaned = _DOC_LABEL_RE.sub(" ", cleaned)
    return [number for number in extract_numbers(cleaned) if number.confidence >= 0.5]


def _magnitude_error(
    targets: Sequence[NumericValue], candidates: Sequence[NumericValue]
) -> Optional[str]:
    for target in targets:
        for candidate in candidates:
            if not _kinds_compatible(target, candidate) or target.value == 0:
                continue
            ratio = abs(candidate.value / target.value)
            for magnitude, label in (
                (1e3, "1,000x"),
                (1e6, "1,000,000x"),
                (1e9, "1,000,000,000x"),
            ):
                if math.isclose(ratio, magnitude, rel_tol=1e-6) or math.isclose(
                    ratio, 1 / magnitude, rel_tol=1e-6
                ):
                    return (
                        f"MAGNITUDE ERROR detected ({label}). Gold: {target.raw} "
                        f"({target.value:,.6g}); predicted: {candidate.raw} "
                        f"({candidate.value:,.6g})"
                    )
    return None


def _match_reference_quantities(
    references: Sequence[NumericValue],
    predictions: Sequence[NumericValue],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> Tuple[
    List[NumericValue],
    List[Tuple[NumericValue, NumericValue]],
    List[NumericValue],
]:
    """Match references one-to-one and return unmatched values on both sides."""

    available = list(predictions)
    unmatched: List[NumericValue] = []
    matched_pairs: List[Tuple[NumericValue, NumericValue]] = []
    for reference in references:
        matched_index = next(
            (
                index
                for index, prediction in enumerate(available)
                if numbers_match(
                    reference,
                    prediction,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                )
            ),
            None,
        )
        if matched_index is None:
            unmatched.append(reference)
        else:
            matched_pairs.append((reference, available.pop(matched_index)))

    return unmatched, matched_pairs, available


def _same_quantity_dimension(left: NumericValue, right: NumericValue) -> bool:
    """Return whether two quantities could be competing answer values."""

    if is_likely_year(left) or is_likely_year(right):
        return is_likely_year(left) and is_likely_year(right)
    if "percent" in {left.kind, right.kind}:
        return left.kind == right.kind == "percent"
    if "ratio" in {left.kind, right.kind}:
        return left.kind == right.kind == "ratio"
    return True


def _conflicting_extra_predictions(
    references: Sequence[NumericValue],
    predictions: Sequence[NumericValue],
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.01,
) -> List[NumericValue]:
    """Find unmatched prediction values that compete with a reference value."""

    unmatched, _, extras = _match_reference_quantities(
        references,
        predictions,
        relative_tolerance,
        absolute_tolerance,
    )
    if unmatched:
        return []

    conflicting = []
    for extra in extras:
        comparable_references = [
            reference
            for reference in references
            if _same_quantity_dimension(reference, extra)
        ]
        if not comparable_references:
            continue
        if any(
            numbers_match(
                reference,
                extra,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            for reference in comparable_references
        ):
            # Repeated equivalent values are redundant, not contradictory.
            continue
        conflicting.append(extra)
    return conflicting


def numeric_match(
    gold: str,
    predicted: str,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.01,
    reject_conflicting_extras: bool = False,
) -> Tuple[Optional[bool], str]:
    """Compare every numeric quantity in a reference answer to a prediction.

    By default, extra prediction numbers are allowed because explanations may
    show intermediate operands. Set ``reject_conflicting_extras`` for a strict
    reported metric. Every reference quantity must have a distinct,
    type-compatible match. Values copied inside evidence citations are excluded.
    """

    gold_numbers = _meaningful_numbers(gold)
    predicted_numbers = _meaningful_numbers(predicted)
    if not gold_numbers:
        return None, "No numeric values found in gold answer"
    if not predicted_numbers:
        return False, "Gold answer is numeric but prediction contains no numeric value"

    unmatched, matched_pairs, _ = _match_reference_quantities(
        gold_numbers,
        predicted_numbers,
        relative_tolerance,
        absolute_tolerance,
    )

    if not unmatched:
        if reject_conflicting_extras:
            conflicting = _conflicting_extra_predictions(
                gold_numbers,
                predicted_numbers,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            if conflicting:
                extras = ", ".join(value.raw for value in conflicting[:5])
                return False, f"Conflicting extra prediction quantities: {extras}"
        pairs = ", ".join(f"{gold.raw} ↔ {pred.raw}" for gold, pred in matched_pairs)
        return True, f"All {len(gold_numbers)} reference quantities matched: {pairs}"

    magnitude_error = _magnitude_error(unmatched, predicted_numbers)
    if magnitude_error:
        return False, magnitude_error

    missing = ", ".join(value.raw for value in unmatched)
    observed = ", ".join(value.raw for value in predicted_numbers[:5])
    return False, f"Unmatched reference quantities: {missing}. Prediction contained: {observed}"


def augmented_judge(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    llm_score: float,
    llm_justification: str,
) -> Tuple[float, str]:
    """Combine semantic judging with deterministic numeric evidence.

    A numeric match no longer forces a perfect score: matching a value alone
    cannot prove that the entity, period, or metric is correct. Conversely, a
    clear mismatch caps rather than silently replaces the semantic score.
    """

    del question  # Kept in the public API for callers and future typed checks.
    numeric_result, explanation = numeric_match(gold_answer, predicted_answer)
    if numeric_result is True:
        conflicting = _conflicting_extra_predictions(
            _meaningful_numbers(gold_answer),
            _meaningful_numbers(predicted_answer),
        )
        if conflicting:
            extras = ", ".join(value.raw for value in conflicting[:5])
            return (
                llm_score,
                f"{llm_justification}. [NUMERIC MATCH WITH CONFLICTING "
                f"EXTRAS: {extras}; score not boosted]",
            )
        score = max(llm_score, 0.8)
        return score, f"{llm_justification}. [NUMERIC VERIFIED: {explanation}]"
    if numeric_result is False:
        score = min(llm_score, 0.2)
        return score, f"{llm_justification}. [NUMERIC MISMATCH: {explanation}]"
    return llm_score, f"{llm_justification}. [No reference quantity to verify]"
