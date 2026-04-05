"""Numeric verification for financial QA evaluation.

This module provides tool-augmented evaluation to catch magnitude errors
that LLM judges often miss (e.g., $1.5M vs $1.5B = 1000x error).

Per GPT-5.2 recommendation: "66% of FinanceBench questions involve numerical
calculation. A vanilla LLM-as-Judge may not consistently catch magnitude errors."
"""

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class NumericValue:
    """Represents an extracted numeric value with its scale."""
    raw: str           # Original text matched
    value: float       # Normalized numeric value
    unit: str          # Detected unit (e.g., "million", "billion", "percent")
    confidence: float  # Extraction confidence (0-1)


# Scale multipliers for financial terms
SCALE_MULTIPLIERS = {
    # Billions
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,
    # Millions
    "million": 1e6,
    "millions": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "mm": 1e6,  # Common in finance
    # Thousands
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
    # Percentages (keep as-is but flag)
    "percent": 1,
    "%": 1,
    "percentage": 1,
    "bps": 0.01,  # Basis points
    "basis points": 0.01,
}


def extract_numbers(text: str) -> List[NumericValue]:
    """Extract all numeric values from text with their scales.

    Handles common financial formats:
    - $1,234.56 billion
    - 1.5M
    - (1,234) negative in parentheses
    - 45.3%
    - 150 bps

    Args:
        text: Text to extract numbers from

    Returns:
        List of NumericValue objects
    """
    results = []
    text_lower = text.lower()

    # Pattern for numbers with optional currency, commas, decimals, and scale words
    # Captures: ($1,234.56) billion, -$1.5M, 45.3%, etc.
    patterns = [
        # Currency with scale word: $1.5 billion, $1,234 million
        r'\$?\s*\(?\s*([\d,]+\.?\d*)\s*\)?\s*(billion|million|thousand|bn|mn|mm|b|m|k)\b',
        # Percentage: 45.3%, 12.5 percent
        r'([\d,]+\.?\d*)\s*(%|percent|percentage|bps|basis\s*points)',
        # Plain currency: $1,234.56 (without scale word)
        r'\$\s*\(?\s*([\d,]+\.?\d*)\s*\)?(?!\s*(?:billion|million|thousand|bn|mn|mm|b|m|k)\b)',
        # Plain number (last resort)
        r'(?<![.\d])([\d,]+\.?\d*)(?![.\d])',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text_lower):
            try:
                # Extract the number part
                num_str = match.group(1).replace(',', '')
                value = float(num_str)

                # Check if there's a scale/unit
                unit = ""
                multiplier = 1.0

                if len(match.groups()) > 1 and match.group(2):
                    unit_raw = match.group(2).strip()
                    unit = unit_raw
                    multiplier = SCALE_MULTIPLIERS.get(unit_raw, 1.0)

                # Handle parentheses as negative (common in finance)
                if '(' in match.group(0) and ')' in match.group(0):
                    value = -abs(value)

                # Apply multiplier
                normalized_value = value * multiplier

                # Calculate confidence based on how specific the match is
                confidence = 0.5  # Base confidence
                if '$' in match.group(0):
                    confidence += 0.2  # Currency symbol increases confidence
                if unit:
                    confidence += 0.2  # Scale word increases confidence
                if '.' in num_str:
                    confidence += 0.1  # Decimal point suggests precision

                results.append(NumericValue(
                    raw=match.group(0),
                    value=normalized_value,
                    unit=unit,
                    confidence=min(confidence, 1.0)
                ))
            except (ValueError, IndexError):
                continue

    # Sort by confidence (highest first)
    results.sort(key=lambda x: x.confidence, reverse=True)

    # Remove duplicates (same value)
    seen_values = set()
    unique_results = []
    for r in results:
        # Round to avoid floating point issues
        rounded = round(r.value, 6)
        if rounded not in seen_values:
            seen_values.add(rounded)
            unique_results.append(r)

    return unique_results


def numeric_match(
    gold: str,
    predicted: str,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.01,
) -> Tuple[Optional[bool], str]:
    """Check if predicted answer matches gold answer numerically.

    This is a deterministic check that can override LLM judge for numeric answers.
    Catches magnitude errors (1000x) that LLM judges often miss.

    Args:
        gold: Gold/reference answer
        predicted: Predicted answer
        relative_tolerance: Acceptable relative difference (default 5%)
        absolute_tolerance: Acceptable absolute difference for small numbers

    Returns:
        Tuple of (match_result, explanation)
        - match_result: True (match), False (mismatch), None (can't determine)
        - explanation: Human-readable explanation
    """
    gold_nums = extract_numbers(gold)
    pred_nums = extract_numbers(predicted)

    # If no numbers found in either, fall back to LLM judge
    if not gold_nums:
        return None, "No numeric values found in gold answer"

    if not pred_nums:
        return None, "No numeric values found in predicted answer"

    # Get the highest-confidence number from each
    gold_primary = gold_nums[0]
    pred_primary = pred_nums[0]

    gold_val = gold_primary.value
    pred_val = pred_primary.value

    # Handle zero edge case
    if gold_val == 0:
        if pred_val == 0:
            return True, f"Both answers are zero"
        return False, f"Gold is 0, predicted is {pred_val}"

    # Calculate relative difference
    rel_diff = abs(gold_val - pred_val) / abs(gold_val)

    # Check for magnitude errors (1000x, 1000000x - common in finance)
    ratio = pred_val / gold_val if gold_val != 0 else float('inf')
    magnitude_error = None
    if abs(ratio - 1000) < 0.1 or abs(ratio - 0.001) < 0.0001:
        magnitude_error = "1000x (millions vs billions or thousands vs millions)"
    elif abs(ratio - 1000000) < 100 or abs(ratio - 0.000001) < 0.0000001:
        magnitude_error = "1,000,000x (thousands vs billions)"

    if magnitude_error:
        return False, (
            f"MAGNITUDE ERROR detected: {magnitude_error}. "
            f"Gold: {gold_primary.raw} ({gold_val:,.2f}), "
            f"Predicted: {pred_primary.raw} ({pred_val:,.2f})"
        )

    # Check if within tolerance
    if rel_diff <= relative_tolerance:
        return True, (
            f"Numeric match within {relative_tolerance*100}% tolerance. "
            f"Gold: {gold_primary.raw}, Predicted: {pred_primary.raw} "
            f"(diff: {rel_diff*100:.1f}%)"
        )

    # For small numbers, also check absolute tolerance
    abs_diff = abs(gold_val - pred_val)
    if abs_diff <= absolute_tolerance:
        return True, (
            f"Numeric match within absolute tolerance. "
            f"Gold: {gold_val}, Predicted: {pred_val} (diff: {abs_diff:.4f})"
        )

    # Mismatch
    return False, (
        f"Numeric mismatch: Gold: {gold_primary.raw} ({gold_val:,.2f}), "
        f"Predicted: {pred_primary.raw} ({pred_val:,.2f}). "
        f"Relative difference: {rel_diff*100:.1f}%"
    )


def augmented_judge(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    llm_score: float,
    llm_justification: str,
) -> Tuple[float, str]:
    """Augment LLM judge score with numeric verification.

    This implements the "tool-augmented judge" recommended by GPT-5.2:
    - First run LLM judge for semantic evaluation
    - Then run numeric check for deterministic verification
    - Override LLM if numeric check finds clear mismatch/match

    Args:
        question: The original question
        gold_answer: Gold/reference answer
        predicted_answer: Predicted answer
        llm_score: Score from LLM judge (0-1)
        llm_justification: Justification from LLM judge

    Returns:
        Tuple of (final_score, augmented_justification)
    """
    # Run numeric check
    numeric_result, numeric_explanation = numeric_match(gold_answer, predicted_answer)

    # If numeric check is conclusive, it can override LLM
    if numeric_result is True:
        # Numeric match - boost score if LLM was unsure
        if llm_score < 0.8:
            return 1.0, f"[NUMERIC VERIFIED] {numeric_explanation}. LLM said: {llm_justification}"
        return llm_score, f"{llm_justification}. [NUMERIC VERIFIED: {numeric_explanation}]"

    elif numeric_result is False:
        # Numeric mismatch - override LLM if it scored too high
        if llm_score > 0.3:
            return 0.0, f"[NUMERIC OVERRIDE] {numeric_explanation}. LLM originally scored {llm_score}: {llm_justification}"
        return llm_score, f"{llm_justification}. [NUMERIC CHECK: {numeric_explanation}]"

    else:
        # Can't determine numerically - trust LLM
        return llm_score, f"{llm_justification}. [No numeric values to verify]"


# Test cases for validation
if __name__ == "__main__":
    # Test number extraction
    test_cases = [
        "$1.5 billion",
        "1,577 million",
        "$1,577",
        "45.3%",
        "(1,234) million",  # Negative
        "Revenue was $2.3B in FY2023",
        "The margin improved by 150 bps",
    ]

    print("Number Extraction Tests:")
    print("-" * 60)
    for text in test_cases:
        nums = extract_numbers(text)
        print(f"'{text}' -> {[(n.raw, n.value, n.unit) for n in nums]}")

    # Test numeric matching
    print("\nNumeric Matching Tests:")
    print("-" * 60)
    match_tests = [
        ("$1.5 billion", "$1,500 million", True),  # Same value, different format
        ("$1.5 billion", "$1.5 million", False),   # 1000x error
        ("$1,577", "$1.577 billion", False),       # Magnitude error
        ("45.3%", "45.3 percent", True),           # Same percentage
        ("Revenue was $2.3B", "The revenue is 2.3 billion dollars", True),
    ]

    for gold, pred, expected in match_tests:
        result, explanation = numeric_match(gold, pred)
        status = "✓" if result == expected else "✗"
        print(f"{status} Gold: '{gold}' vs Pred: '{pred}'")
        print(f"   Result: {result}, Expected: {expected}")
        print(f"   {explanation}\n")
