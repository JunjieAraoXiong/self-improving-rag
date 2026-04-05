"""Numeric verification for RAG-generated answers.

This module extracts numbers from generated answers and retrieved chunks,
then verifies that numbers in answers exist in source documents.
Helps catch numerical hallucinations common in metric-related questions.
"""

import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional

try:
    from langchain_core.documents import Document
except ImportError:
    Document = None  # Type hint only


@dataclass
class NumericVerificationResult:
    """Result of numeric verification between answer and source chunks.

    Attributes:
        score: Verification score from 0.0 to 1.0 (1.0 = all numbers verified)
        answer_numbers: Numbers found in the answer (original format)
        source_numbers: Normalized numbers found in source chunks
        verified_numbers: Answer numbers that exist in sources
        flagged_numbers: Potentially hallucinated numbers not in sources
        details: Human-readable summary of verification
    """
    score: float
    answer_numbers: List[str] = field(default_factory=list)
    source_numbers: Set[float] = field(default_factory=set)
    verified_numbers: List[str] = field(default_factory=list)
    flagged_numbers: List[str] = field(default_factory=list)
    details: str = ""


# Multiplier mappings for text-based numbers
MULTIPLIERS = {
    'thousand': 1e3,
    'k': 1e3,
    'million': 1e6,
    'm': 1e6,
    'billion': 1e9,
    'b': 1e9,
    'trillion': 1e12,
    't': 1e12,
}

# Pattern for numbers with optional multipliers
# Matches: $1,234.56, 1.5 billion, 15%, -5.2, (1,000), 1.5e6, etc.
NUMBER_PATTERN = re.compile(
    r"""
    (?P<negative>-|\()?                    # Optional negative sign or opening paren
    \$?                                     # Optional dollar sign
    (?P<number>                            # The number itself
        \d{1,3}(?:,\d{3})*(?:\.\d+)?       # Comma-separated (1,234.56)
        |
        \d+(?:\.\d+)?                       # Plain number (1234.56)
    )
    (?:                                    # Optional multiplier/suffix
        \s*(?P<multiplier>thousand|million|billion|trillion|[kmbt])\b
        |
        (?P<scientific>[eE][+-]?\d+)       # Scientific notation (1.5e6)
        |
        (?P<percent>%)?                    # Percentage
    )?
    (?P<close_paren>\))?                   # Closing paren for negative
    """,
    re.IGNORECASE | re.VERBOSE
)


def normalize_number(match: re.Match) -> Tuple[str, float]:
    """Convert a regex match to normalized float value.

    Args:
        match: Regex match object from NUMBER_PATTERN

    Returns:
        Tuple of (original_text, normalized_value)
    """
    original = match.group(0)
    number_str = match.group('number').replace(',', '')
    value = float(number_str)

    # Check for negative (either - prefix or parentheses)
    is_negative = bool(match.group('negative')) or bool(match.group('close_paren'))
    if is_negative:
        value = -value

    # Apply multiplier if present
    multiplier = match.group('multiplier')
    if multiplier:
        mult_key = multiplier.lower()
        if mult_key in MULTIPLIERS:
            value *= MULTIPLIERS[mult_key]

    # Apply scientific notation if present
    scientific = match.group('scientific')
    if scientific:
        # Already handled by float() if we reconstruct
        value = float(number_str + scientific)
        if is_negative:
            value = -value

    return original, value


def extract_numbers(text: str) -> List[Tuple[str, float]]:
    """Extract all numbers from text with their normalized values.

    Handles various formats:
    - Currency: $1,577, $1.5M
    - Percentages: 15%, 15.5%
    - Multipliers: 1.5 billion, 2.3 million
    - Plain decimals: 3.14159
    - Negative: -5.2, ($1,000)
    - Scientific: 1.5e6

    Args:
        text: Input text to extract numbers from

    Returns:
        List of tuples (original_format, normalized_value)
    """
    results = []
    for match in NUMBER_PATTERN.finditer(text):
        original, value = normalize_number(match)
        # Skip very small numbers that might be noise (like single digits in text)
        # but keep percentages and explicit numbers
        if abs(value) >= 1 or '%' in original or '.' in match.group('number'):
            results.append((original.strip(), value))
    return results


def numbers_match(value1: float, value2: float, tolerance: float = 0.001) -> bool:
    """Check if two numbers match within tolerance.

    Uses relative tolerance for large numbers, absolute for small.

    Args:
        value1: First number
        value2: Second number
        tolerance: Relative tolerance (default 0.1%)

    Returns:
        True if numbers match within tolerance
    """
    if value1 == value2:
        return True
    if value1 == 0 or value2 == 0:
        return abs(value1 - value2) < tolerance

    # Relative tolerance for non-zero values
    relative_diff = abs(value1 - value2) / max(abs(value1), abs(value2))
    return relative_diff <= tolerance


def verify_numeric_answer(
    predicted_answer: str,
    retrieved_chunks: List,
    tolerance: float = 0.001
) -> NumericVerificationResult:
    """Verify that numbers in a generated answer exist in source chunks.

    Extracts numbers from both the answer and retrieved chunks, then
    checks that each answer number can be found in at least one source.
    Numbers not found in sources are flagged as potentially hallucinated.

    Args:
        predicted_answer: The generated answer text to verify
        retrieved_chunks: List of Document objects (or objects with page_content)
        tolerance: Relative tolerance for number matching (default 0.1%)

    Returns:
        NumericVerificationResult with score, verified/flagged numbers, and details
    """
    # Extract numbers from answer
    answer_extractions = extract_numbers(predicted_answer)
    answer_numbers = [orig for orig, _ in answer_extractions]
    answer_values = [val for _, val in answer_extractions]

    # Extract numbers from all source chunks
    source_values: Set[float] = set()
    for chunk in retrieved_chunks:
        # Handle both Document objects and raw strings
        if hasattr(chunk, 'page_content'):
            chunk_text = chunk.page_content
        elif isinstance(chunk, str):
            chunk_text = chunk
        else:
            continue

        chunk_extractions = extract_numbers(chunk_text)
        for _, value in chunk_extractions:
            source_values.add(value)

    # Verify each answer number against sources
    verified = []
    flagged = []

    for original, value in answer_extractions:
        found = False
        for source_val in source_values:
            if numbers_match(value, source_val, tolerance):
                found = True
                break

        if found:
            verified.append(original)
        else:
            flagged.append(original)

    # Calculate score
    total = len(answer_numbers)
    if total == 0:
        score = 1.0  # No numbers to verify = perfect score
        details = "No numbers found in answer to verify."
    else:
        score = len(verified) / total
        if len(flagged) == 0:
            details = f"All {total} number(s) verified in source documents."
        else:
            details = (
                f"Verified {len(verified)}/{total} numbers. "
                f"Flagged {len(flagged)} potentially hallucinated: {flagged}"
            )

    return NumericVerificationResult(
        score=score,
        answer_numbers=answer_numbers,
        source_numbers=source_values,
        verified_numbers=verified,
        flagged_numbers=flagged,
        details=details
    )


def get_verification_summary(result: NumericVerificationResult) -> str:
    """Generate a formatted summary of verification results.

    Args:
        result: NumericVerificationResult from verify_numeric_answer

    Returns:
        Formatted multi-line summary string
    """
    lines = [
        f"Numeric Verification Score: {result.score:.2%}",
        f"Numbers in answer: {result.answer_numbers}",
        f"Verified: {result.verified_numbers}",
        f"Flagged (potential hallucination): {result.flagged_numbers}",
        f"Details: {result.details}"
    ]
    return "\n".join(lines)
