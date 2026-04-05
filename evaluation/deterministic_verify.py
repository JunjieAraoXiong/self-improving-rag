"""Deterministic Verifier: Rule-based verification of evidence citations.

This module provides a hard gate before LLM-based evaluation by checking
that all numerical claims in an answer have corresponding source citations.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.documents import Document


@dataclass
class VerificationResult:
    """Result of deterministic verification."""
    passed: bool
    ungrounded_claims: List[str]
    evidence_quotes: List[Tuple[str, str]]  # (doc_ref, quote)
    coverage_ratio: float  # Ratio of numbers with citations
    message: str


def extract_evidence_quotes(answer: str) -> List[Tuple[str, str]]:
    """Parse [DocX: 'quote'] patterns from an answer.

    Args:
        answer: The generated answer containing inline citations

    Returns:
        List of (doc_reference, quote) tuples

    Examples:
        >>> extract_evidence_quotes("Revenue was $383.3B [Doc2: 'Net sales were $383,285 million']")
        [('Doc2', 'Net sales were $383,285 million')]
    """
    # Pattern matches [DocX: 'quote'] or [DocX: "quote"]
    pattern = r'\[Doc(\d+):\s*[\'"]([^\'\"]+)[\'"]\]'
    matches = re.findall(pattern, answer)
    return [(f"Doc{num}", quote) for num, quote in matches]


def extract_numbers(text: str) -> List[str]:
    """Extract numerical values from text.

    Captures:
    - Dollar amounts: $383.3B, $1,234.56
    - Percentages: 43.3%, 5%
    - Plain numbers: 164,000, 2024
    - Ratios: 1.5x, 2:1

    Args:
        text: Text to extract numbers from

    Returns:
        List of number strings found
    """
    patterns = [
        r'\$[\d,]+(?:\.\d+)?[BMK]?',  # Dollar amounts
        r'[\d,]+(?:\.\d+)?%',          # Percentages
        r'[\d,]+(?:\.\d+)?x',          # Multipliers
        r'\b\d{1,3}(?:,\d{3})+\b',     # Large numbers with commas
        r'\b\d+\.\d+\b',               # Decimal numbers
    ]

    numbers = []
    for pattern in patterns:
        numbers.extend(re.findall(pattern, text))

    return list(set(numbers))  # Remove duplicates


def verify_quote_in_docs(
    quote: str,
    docs: List[Document],
    similarity_threshold: float = 0.8
) -> bool:
    """Check if a quote exists in the provided documents.

    Uses fuzzy matching to account for minor differences in whitespace
    or formatting.

    Args:
        quote: The quoted text to verify
        docs: List of documents to search
        similarity_threshold: Minimum similarity for a match (0-1)

    Returns:
        True if quote found in documents
    """
    quote_normalized = quote.lower().strip()
    quote_words = set(quote_normalized.split())

    for doc in docs:
        content_normalized = doc.page_content.lower()

        # Exact substring match
        if quote_normalized in content_normalized:
            return True

        # Fuzzy match based on word overlap
        content_words = set(content_normalized.split())
        if len(quote_words) > 0:
            overlap = len(quote_words & content_words) / len(quote_words)
            if overlap >= similarity_threshold:
                return True

    return False


def deterministic_verify(
    answer: str,
    docs: List[Document],
    require_all_numbers_cited: bool = True,
    min_coverage: float = 0.8,
) -> VerificationResult:
    """Check that all numerical claims have grounded source citations.

    This is a deterministic, rule-based check that runs BEFORE the LLM judge.
    It ensures that the answer includes explicit evidence for numerical claims.

    Args:
        answer: The generated answer to verify
        docs: The source documents used to generate the answer
        require_all_numbers_cited: If True, all numbers must have citations
        min_coverage: Minimum ratio of cited numbers (if not requiring all)

    Returns:
        VerificationResult with pass/fail status and details
    """
    if not answer:
        return VerificationResult(
            passed=False,
            ungrounded_claims=[],
            evidence_quotes=[],
            coverage_ratio=0.0,
            message="Empty answer"
        )

    # Extract citations and numbers
    evidence_quotes = extract_evidence_quotes(answer)
    numbers_in_answer = extract_numbers(answer)

    # Special case: no numbers in answer (e.g., yes/no question)
    if not numbers_in_answer:
        return VerificationResult(
            passed=True,
            ungrounded_claims=[],
            evidence_quotes=evidence_quotes,
            coverage_ratio=1.0,
            message="No numerical claims to verify"
        )

    # Check which numbers have citations
    # We look for numbers that appear near citations
    ungrounded = []
    grounded_count = 0

    for number in numbers_in_answer:
        # Check if this number appears within a citation context
        # Look for pattern: number followed by [DocX: ...]
        number_escaped = re.escape(number)
        citation_pattern = rf'{number_escaped}[^[]*\[Doc\d+:'

        if re.search(citation_pattern, answer):
            grounded_count += 1
        else:
            ungrounded.append(number)

    coverage = grounded_count / len(numbers_in_answer) if numbers_in_answer else 1.0

    # Verify that cited quotes actually exist in documents
    invalid_citations = []
    for doc_ref, quote in evidence_quotes:
        if not verify_quote_in_docs(quote, docs):
            invalid_citations.append(f"{doc_ref}: '{quote}'")

    # Determine pass/fail
    if invalid_citations:
        passed = False
        message = f"Invalid citations detected: {invalid_citations}"
    elif require_all_numbers_cited and ungrounded:
        passed = False
        message = f"Ungrounded numerical claims: {ungrounded}"
    elif coverage < min_coverage:
        passed = False
        message = f"Insufficient citation coverage: {coverage:.1%} < {min_coverage:.1%}"
    else:
        passed = True
        message = f"Verification passed. Coverage: {coverage:.1%}"

    return VerificationResult(
        passed=passed,
        ungrounded_claims=ungrounded,
        evidence_quotes=evidence_quotes,
        coverage_ratio=coverage,
        message=message
    )


def format_verification_feedback(result: VerificationResult) -> str:
    """Format verification result as feedback for retry.

    This feedback can be included in the retry prompt to help the model
    correct its response.

    Args:
        result: The verification result

    Returns:
        Formatted feedback string
    """
    if result.passed:
        return ""

    feedback_parts = [
        "VERIFICATION FAILED - Please correct your response:",
        result.message,
    ]

    if result.ungrounded_claims:
        feedback_parts.append(
            f"Missing citations for: {', '.join(result.ungrounded_claims)}"
        )
        feedback_parts.append(
            "Remember: Every number must have [DocX: 'exact quote'] citation."
        )

    return "\n".join(feedback_parts)
