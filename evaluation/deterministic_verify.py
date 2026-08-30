"""Deterministic, unit-aware verification of inline evidence citations.

The verifier treats an inline citation as a claim-evidence link. It checks
that the cited document exists, that the quoted text occurs in that specific
document, and that each numeric claim is supported by an equivalent quantity
in its nearby quote. This catches a failure the former proximity-only check
missed: ``$1.5B [Doc1: '$1.5 million']``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from evaluation.numeric_check import (
    NumericValue,
    apply_context_scale,
    extract_numbers as extract_numeric_values,
    find_numeric_match,
    infer_scale,
    is_likely_year,
)


@dataclass(frozen=True)
class EvidenceCitation:
    """A parsed ``[DocX: 'quote']`` citation and its answer span."""

    doc_ref: str
    doc_number: int
    quote: str
    start: int
    end: int


@dataclass
class VerificationResult:
    """Result of deterministic evidence verification."""

    passed: bool
    ungrounded_claims: List[str]
    evidence_quotes: List[Tuple[str, str]]
    coverage_ratio: float
    message: str
    invalid_citations: List[str] = field(default_factory=list)
    mismatched_claims: List[str] = field(default_factory=list)
    supported_claims: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)


_CITATION_RE = re.compile(
    r"\[Doc(?P<number>\d+)\s*:\s*"
    r"(?:'(?P<single_quote>.*?)'|"
    r'"(?P<double_quote>.*?)"|'
    r"‘(?P<curly_single_quote>.*?)’|"
    r"“(?P<curly_double_quote>.*?)”)\]",
    re.IGNORECASE | re.DOTALL,
)

_SENTENCE_BOUNDARY_RE = re.compile(r"[!?;\n]|\.(?:\s|$)")
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.!?;,\n]|\b(?:and|but|while|whereas)\b",
    re.IGNORECASE,
)
_ABSOLUTE_VALUE_METRICS = re.compile(
    r"\b(?:capex|capital expenditures?|cash paid|cash outflows?|outflows?)\b",
    re.IGNORECASE,
)
_ATTRIBUTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "in", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were",
    "what", "when", "which", "with", "yes", "no", "company", "companies",
    "usd", "cad", "aud", "nzd", "sgd", "hkd", "eur", "gbp", "jpy", "cny",
    "dollar", "dollars", "thousand", "thousands", "million", "millions",
    "billion", "billions", "trillion", "trillions", "percent", "percentage",
    "bps", "basis", "points", "ratio", "days", "x",
}
_METRIC_ALIAS_GROUPS = (
    (
        frozenset({"capital", "expenditure"}),
        frozenset({"capital", "expenditures"}),
        frozenset({"capex"}),
        frozenset({"purchases", "property", "plant", "equipment"}),
    ),
    (
        frozenset({"operating", "cash", "flow"}),
        frozenset({"operating", "cash", "flows"}),
        frozenset({"net", "cash", "provided", "operating", "activities"}),
    ),
    (
        frozenset({"revenue"}),
        frozenset({"net", "sales"}),
    ),
)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|neither|nor|without|cannot|can't|didn't|doesn't|"
    r"isn't|wasn't|weren't|failed|fails|lack(?:ed|s|ing)?)\b",
    re.IGNORECASE,
)
_UPWARD_DIRECTION_RE = re.compile(
    r"\b(?:increase[ds]?|increasing|grew|grown|growth|rose|risen|rising|"
    r"improve[ds]?|improving|higher|gain(?:ed|s)?|expanded?)\b",
    re.IGNORECASE,
)
_DOWNWARD_DIRECTION_RE = re.compile(
    r"\b(?:decrease[ds]?|decreasing|decline[ds]?|declining|fell|fallen|"
    r"falling|deteriorate[ds]?|deteriorating|lower|loss|lost|contracted?)\b",
    re.IGNORECASE,
)


def _polarity_signature(text: str) -> Tuple[bool, bool, bool]:
    """Return (negated, upward, downward) cues for a local claim or quote."""

    return (
        bool(_NEGATION_RE.search(text or "")),
        bool(_UPWARD_DIRECTION_RE.search(text or "")),
        bool(_DOWNWARD_DIRECTION_RE.search(text or "")),
    )


def _polarity_compatible(claim: str, quote: str) -> bool:
    """Reject obvious negation or directional contradictions.

    This remains a conservative lexical gate rather than an entailment model.
    It deliberately fails closed on polarity mismatches so an exact affirmative
    quote cannot verify a negated claim (or vice versa).
    """

    claim_negated, claim_up, claim_down = _polarity_signature(claim)
    quote_negated, quote_up, quote_down = _polarity_signature(quote)
    if claim_negated != quote_negated:
        return False
    if (claim_up and quote_down) or (claim_down and quote_up):
        return False
    return True


def extract_evidence_citations(answer: str) -> List[EvidenceCitation]:
    """Parse structured citations while retaining their positions."""

    citations = []
    for match in _CITATION_RE.finditer(answer or ""):
        number = int(match.group("number"))
        quote = next(
            value
            for value in (
                match.group("single_quote"),
                match.group("double_quote"),
                match.group("curly_single_quote"),
                match.group("curly_double_quote"),
            )
            if value is not None
        )
        citations.append(
            EvidenceCitation(
                doc_ref=f"Doc{number}",
                doc_number=number,
                quote=quote,
                start=match.start(),
                end=match.end(),
            )
        )
    return citations


def extract_evidence_quotes(answer: str) -> List[Tuple[str, str]]:
    """Return backward-compatible ``(DocX, quote)`` tuples."""

    return [
        (citation.doc_ref, citation.quote)
        for citation in extract_evidence_citations(answer)
    ]


def _mask_citations(answer: str, citations: List[EvidenceCitation]) -> str:
    characters = list(answer)
    for citation in citations:
        characters[citation.start : citation.end] = " " * (citation.end - citation.start)
    return "".join(characters)


def extract_numbers(text: str) -> List[str]:
    """Return numeric claim strings, excluding values inside citations."""

    citations = extract_evidence_citations(text)
    claims_only = _mask_citations(text or "", citations)
    return [value.raw for value in extract_numeric_values(claims_only)]


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return " ".join(re.findall(r"\w+|[$%€£()+,.\-]", text))


def _attribution_tokens(text: str) -> set[str]:
    """Return conservative content tokens for a claim↔quote sanity check.

    This is deliberately described as lexical attribution, not entailment. It
    blocks empty/unrelated citations without pretending that token overlap can
    establish the truth of a qualitative claim.
    """

    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{1,}", normalized)
        if token not in _ATTRIBUTION_STOPWORDS
    }


def _qualitative_citations_align(
    claims_text: str,
    question: str,
    citations: List[EvidenceCitation],
    citation_validity: Dict[EvidenceCitation, bool],
) -> bool:
    """Require valid quotes to share substantive anchors with the claim.

    Very short answers such as ``yes`` inherit anchors from the question. The
    check is intentionally fail-closed and may reject legitimate paraphrases;
    semantic attribution must still be evaluated independently.
    """

    claim_tokens = _attribution_tokens(claims_text)
    if len(claim_tokens) < 2:
        claim_tokens |= _attribution_tokens(question)
    quote_tokens: set[str] = set()
    for citation in citations:
        if (
            citation_validity.get(citation, False)
            and _polarity_compatible(claims_text, citation.quote)
        ):
            quote_tokens |= _attribution_tokens(citation.quote)
    if not claim_tokens or not quote_tokens:
        return False
    overlap = claim_tokens & quote_tokens
    required = 1 if len(claim_tokens) == 1 else 2
    if len(overlap) >= required and len(overlap) / len(claim_tokens) >= 0.2:
        return True

    for variants in _METRIC_ALIAS_GROUPS:
        vocabulary = frozenset().union(*variants)
        claim_has_metric = any(variant <= claim_tokens for variant in variants)
        quote_has_metric = any(variant <= quote_tokens for variant in variants)
        if not (claim_has_metric and quote_has_metric):
            continue
        claim_residual = claim_tokens - vocabulary
        quote_residual = quote_tokens - vocabulary
        if not claim_residual or claim_residual & quote_residual:
            return True
    return False


def verify_quote_in_docs(
    quote: str,
    docs: List[Document],
    similarity_threshold: float = 0.88,
    doc_index: Optional[int] = None,
) -> bool:
    """Check that ``quote`` occurs exactly after safe text normalization.

    ``doc_index`` is zero-based. When omitted, all documents are searched for
    backward compatibility. ``similarity_threshold`` remains in the signature
    for API compatibility but fuzzy acceptance is intentionally disabled: a
    near-match can change a number, period, entity, or negation.
    """

    del similarity_threshold

    if doc_index is not None:
        if doc_index < 0 or doc_index >= len(docs):
            return False
        candidate_docs = [docs[doc_index]]
    else:
        candidate_docs = docs

    normalized_quote = _normalize_text(quote)
    for doc in candidate_docs:
        content = getattr(doc, "page_content", "")
        normalized_content = _normalize_text(content)
        if normalized_quote and normalized_quote in normalized_content:
            return True
    return False


def _has_sentence_boundary(text: str) -> bool:
    # A citation is often placed immediately after terminal punctuation. Treat
    # punctuation-only gaps as attached, but reject gaps containing another
    # sentence's words.
    if re.fullmatch(r"\s*[,.:;)]*\s*", text):
        return False
    return bool(_SENTENCE_BOUNDARY_RE.search(text))


def _citations_for_claim(
    answer: str,
    claim: NumericValue,
    citations: List[EvidenceCitation],
) -> List[EvidenceCitation]:
    """Find citations attached to a claim in the same sentence."""

    following = [
        citation
        for citation in citations
        if 0 <= citation.start - claim.end <= 320
        and not _has_sentence_boundary(answer[claim.end : citation.start])
    ]
    if following:
        nearest_distance = min(citation.start - claim.end for citation in following)
        return [
            citation
            for citation in following
            if citation.start - claim.end <= nearest_distance + 40
        ]

    preceding = [
        citation
        for citation in citations
        if 0 <= claim.start - citation.end <= 180
        and not _has_sentence_boundary(answer[citation.end : claim.start])
    ]
    if preceding:
        nearest_distance = min(claim.start - citation.end for citation in preceding)
        return [
            citation
            for citation in preceding
            if claim.start - citation.end <= nearest_distance + 40
        ]
    return []


def _citation_numbers(
    citation: EvidenceCitation,
) -> List[NumericValue]:
    values = extract_numeric_values(citation.quote)
    scale = infer_scale(citation.quote)
    if not scale:
        return values

    # A quoted scale declaration applies deterministically to eligible values.
    # Do not retain a raw alternative: doing so lets either magnitude pass.
    return [
        value if is_likely_year(value) else apply_context_scale(value, scale)
        for value in values
    ]


def _local_claim_context(text: str, claim: NumericValue) -> str:
    """Return the sentence/clause containing one numeric claim."""

    before = text[: claim.start]
    previous_boundaries = list(_CLAUSE_BOUNDARY_RE.finditer(before))
    left = previous_boundaries[-1].end() if previous_boundaries else 0

    after = text[claim.end :]
    next_boundary = _CLAUSE_BOUNDARY_RE.search(after)
    right = claim.end + (next_boundary.start() if next_boundary else len(after))
    return text[left:right]


def _allow_absolute_value_for_claim(
    claims_text: str,
    question: str,
    claim: NumericValue,
    claim_count: int,
) -> bool:
    """Allow accounting-sign equivalence only for a local outflow metric."""

    local_context = _local_claim_context(claims_text, claim)
    if _ABSOLUTE_VALUE_METRICS.search(local_context):
        return True
    # A concise numeric-only answer gets its metric name from the question.
    return claim_count == 1 and bool(_ABSOLUTE_VALUE_METRICS.search(question or ""))


def deterministic_verify(
    answer: str,
    docs: List[Document],
    require_all_numbers_cited: bool = True,
    min_coverage: float = 0.8,
    question: str = "",
) -> VerificationResult:
    """Verify document identity, quote fidelity, and numeric entailment."""

    if not answer:
        return VerificationResult(
            passed=False,
            ungrounded_claims=[],
            evidence_quotes=[],
            coverage_ratio=0.0,
            message="Empty answer",
            reason_codes=["empty_answer"],
        )

    citations = extract_evidence_citations(answer)
    evidence_quotes = [(citation.doc_ref, citation.quote) for citation in citations]
    claims_text = _mask_citations(answer, citations)
    numeric_claims = extract_numeric_values(claims_text)

    citation_validity: Dict[EvidenceCitation, bool] = {}
    invalid_citations: List[str] = []
    for citation in citations:
        valid = verify_quote_in_docs(
            citation.quote,
            docs,
            doc_index=citation.doc_number - 1,
        )
        citation_validity[citation] = valid
        if not valid:
            if citation.doc_number < 1 or citation.doc_number > len(docs):
                invalid_citations.append(f"{citation.doc_ref}: document index out of range")
            else:
                invalid_citations.append(
                    f"{citation.doc_ref}: quote not found in cited document"
                )

    if not numeric_claims:
        if not citations:
            return VerificationResult(
                passed=False,
                ungrounded_claims=[],
                evidence_quotes=[],
                coverage_ratio=0.0,
                message="A factual nonnumeric answer requires at least one exact citation",
                reason_codes=["missing_citation"],
            )
        aligned = _qualitative_citations_align(
            claims_text,
            question,
            citations,
            citation_validity,
        )
        passed = not invalid_citations and aligned
        reason_codes = []
        if invalid_citations:
            reason_codes.append("invalid_citation")
        if not aligned:
            reason_codes.append("citation_claim_mismatch")
        return VerificationResult(
            passed=passed,
            ungrounded_claims=[],
            evidence_quotes=evidence_quotes,
            coverage_ratio=1.0 if passed else 0.0,
            message=(
                "Exact citations are lexically aligned; semantic attribution "
                "still requires independent evaluation"
                if passed
                else (
                    f"Invalid citations detected: {invalid_citations}"
                    if invalid_citations
                    else "Citations do not share enough substantive anchors with the claim"
                )
            ),
            invalid_citations=invalid_citations,
            reason_codes=reason_codes,
        )

    ungrounded: List[str] = []
    mismatched: List[str] = []
    attribution_mismatched: List[str] = []
    supported: List[str] = []
    for claim in numeric_claims:
        attached = _citations_for_claim(answer, claim, citations)
        valid_attached = [
            citation for citation in attached if citation_validity.get(citation, False)
        ]
        if not valid_attached:
            ungrounded.append(claim.raw)
            continue

        local_claim = _local_claim_context(claims_text, claim)
        aligned_attached = [
            citation
            for citation in valid_attached
            if _qualitative_citations_align(
                local_claim,
                question,
                [citation],
                citation_validity,
            )
        ]
        if not aligned_attached:
            attribution_mismatched.append(claim.raw)
            continue

        matched = False
        for citation in aligned_attached:
            evidence_values = _citation_numbers(citation)
            if find_numeric_match(
                claim,
                evidence_values,
                relative_tolerance=0.005,
                absolute_tolerance=0.01,
                allow_absolute_value=_allow_absolute_value_for_claim(
                    claims_text,
                    question,
                    claim,
                    len(numeric_claims),
                ),
            ):
                matched = True
                break

        if matched:
            supported.append(claim.raw)
        else:
            mismatched.append(claim.raw)

    coverage = len(supported) / len(numeric_claims)
    reason_codes = []
    if invalid_citations:
        reason_codes.append("invalid_citation")
    if ungrounded:
        reason_codes.append("missing_citation")
    if mismatched:
        reason_codes.append("numeric_evidence_mismatch")
    if attribution_mismatched:
        reason_codes.append("citation_claim_mismatch")
    if coverage < min_coverage:
        reason_codes.append("insufficient_coverage")

    if invalid_citations:
        passed = False
        message = f"Invalid citations detected: {invalid_citations}"
    elif attribution_mismatched:
        passed = False
        message = (
            "Citations match a value but not the claim's substantive anchors: "
            f"{attribution_mismatched}"
        )
    elif mismatched:
        passed = False
        message = f"Citations do not support numeric claims: {mismatched}"
    elif require_all_numbers_cited and ungrounded:
        passed = False
        message = f"Ungrounded numerical claims: {ungrounded}"
    elif coverage < min_coverage:
        passed = False
        message = f"Insufficient citation coverage: {coverage:.1%} < {min_coverage:.1%}"
    else:
        passed = True
        message = f"Verification passed. Numeric evidence coverage: {coverage:.1%}"

    return VerificationResult(
        passed=passed,
        ungrounded_claims=ungrounded,
        evidence_quotes=evidence_quotes,
        coverage_ratio=coverage,
        message=message,
        invalid_citations=invalid_citations,
        mismatched_claims=mismatched + attribution_mismatched,
        supported_claims=supported,
        reason_codes=reason_codes,
    )


def format_verification_feedback(result: VerificationResult) -> str:
    """Format actionable, gold-free correction feedback for a retry."""

    if result.passed:
        return ""

    feedback = ["VERIFICATION FAILED - correct the previous answer:", result.message]
    if result.ungrounded_claims:
        feedback.append(
            "Add a source quote for these claims: "
            + ", ".join(result.ungrounded_claims)
        )
    if result.mismatched_claims:
        feedback.append(
            "Re-check the value, unit, sign, and table scale for: "
            + ", ".join(result.mismatched_claims)
        )
    if result.invalid_citations:
        feedback.append(
            "Use the correct Doc number and copy a quote that actually occurs there."
        )
    if "citation_claim_mismatch" in result.reason_codes:
        feedback.append(
            "Choose an exact quote that shares the claim's entity and substantive "
            "metric/event anchors; semantic attribution is evaluated separately."
        )
    feedback.append("Use [DocX: 'exact quote'] after each supported numeric claim.")
    return "\n".join(feedback)
