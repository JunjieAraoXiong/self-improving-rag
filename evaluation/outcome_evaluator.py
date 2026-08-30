"""Shared post-selection outcome evaluation for every QA pipeline.

Selection and evaluation are separate data paths: the policy may never see the
gold answer in blind mode, while this module receives only the already-final
answer. This prevents a controller score from being relabeled as accuracy.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from evaluation.llm_judge import llm_as_judge
from evaluation.numeric_check import (
    extract_numbers,
    numeric_match,
    strip_evidence_citations,
)


JudgeFn = Callable[[str, str, str, str], Tuple[float, str]]
DEFAULT_JUDGE_CORRECTNESS_THRESHOLD = 0.99

# Accuracy labels should capture formatting-equivalent quantities, not grant a
# broad finance-domain error band.  The legacy numeric diagnostic uses 5%; a
# 0.1% tolerance here accommodates floating-point/unit normalization while
# rejecting materially different answers such as 100 versus 95.
_LABEL_RELATIVE_TOLERANCE = 0.001
_LABEL_ABSOLUTE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class OutcomeEvaluation:
    """Gold-based metrics computed only after answer selection terminates."""

    correct: Optional[bool]
    mode: str
    exact_match: bool
    numeric_correct: Optional[bool]
    numeric_explanation: str
    judge_score: Optional[float]
    judge_justification: Optional[str]
    judge_correctness_threshold: Optional[float]
    evaluated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_answer(text: str) -> str:
    text = strip_evidence_citations(text or "")
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+|[$%€£.+()-]", text))


def _is_quantity_only(text: str) -> bool:
    """Return whether text contains only parsed quantities and punctuation."""

    cleaned = strip_evidence_citations(text or "")
    quantities = extract_numbers(cleaned)
    if not quantities:
        return False
    characters = list(cleaned)
    for quantity in quantities:
        characters[quantity.start : quantity.end] = " " * (
            quantity.end - quantity.start
        )
    residual = "".join(characters)
    return re.search(r"\w", residual, flags=re.UNICODE) is None


def evaluate_post_selection(
    *,
    question: str,
    gold_answer: str,
    predicted_answer: Optional[str],
    use_llm_judge: bool,
    judge_model: str,
    judge_threshold: float = DEFAULT_JUDGE_CORRECTNESS_THRESHOLD,
    terminal_error: Optional[str] = None,
    abstained: bool = False,
    judge_fn: JudgeFn = llm_as_judge,
) -> OutcomeEvaluation:
    """Evaluate a frozen final answer without feeding results back to policy.

    Without an LLM evaluator, numeric questions receive a deterministic
    quantity-aware label and exact textual matches receive an exact label.
    Nonnumeric paraphrases otherwise remain unlabeled rather than being guessed
    from embedding similarity.
    """

    if not 0.0 <= judge_threshold <= 1.0:
        raise ValueError("judge_threshold must be between 0 and 1")

    predicted = predicted_answer or ""
    gold = "" if gold_answer is None else str(gold_answer)
    exact = bool(predicted and _normalize_answer(predicted) == _normalize_answer(gold))
    numeric_correct, numeric_explanation = numeric_match(
        gold=gold,
        predicted=predicted,
        relative_tolerance=_LABEL_RELATIVE_TOLERANCE,
        absolute_tolerance=_LABEL_ABSOLUTE_TOLERANCE,
        reject_conflicting_extras=True,
    )

    if terminal_error:
        return OutcomeEvaluation(
            correct=False,
            mode="terminal_error",
            exact_match=False,
            numeric_correct=numeric_correct,
            numeric_explanation=numeric_explanation,
            judge_score=None,
            judge_justification=terminal_error,
            judge_correctness_threshold=(
                judge_threshold if use_llm_judge else None
            ),
            evaluated=True,
        )

    if abstained or not predicted:
        return OutcomeEvaluation(
            correct=False,
            mode="post_selection_abstention",
            exact_match=False,
            numeric_correct=numeric_correct,
            numeric_explanation=numeric_explanation,
            judge_score=0.0 if use_llm_judge else None,
            judge_justification="No final answer; evaluator call skipped.",
            judge_correctness_threshold=(
                judge_threshold if use_llm_judge else None
            ),
            evaluated=True,
        )

    if use_llm_judge:
        try:
            score, justification = judge_fn(
                question,
                gold,
                predicted,
                judge_model,
            )
        except Exception as exc:
            return OutcomeEvaluation(
                correct=None,
                mode="evaluator_error",
                exact_match=exact,
                numeric_correct=numeric_correct,
                numeric_explanation=numeric_explanation,
                judge_score=None,
                judge_justification=str(exc),
                judge_correctness_threshold=judge_threshold,
                evaluated=False,
            )
        return OutcomeEvaluation(
            correct=bool(score >= judge_threshold),
            mode="post_selection_llm_judge",
            exact_match=exact,
            numeric_correct=numeric_correct,
            numeric_explanation=numeric_explanation,
            judge_score=float(score),
            judge_justification=justification,
            judge_correctness_threshold=judge_threshold,
            evaluated=True,
        )

    if exact:
        return OutcomeEvaluation(
            correct=True,
            mode="post_selection_exact",
            exact_match=True,
            numeric_correct=numeric_correct,
            numeric_explanation=numeric_explanation,
            judge_score=None,
            judge_justification=None,
            judge_correctness_threshold=None,
            evaluated=True,
        )

    if numeric_correct is False:
        return OutcomeEvaluation(
            correct=False,
            mode="post_selection_numeric_mismatch",
            exact_match=False,
            numeric_correct=False,
            numeric_explanation=numeric_explanation,
            judge_score=None,
            judge_justification=None,
            judge_correctness_threshold=None,
            evaluated=True,
        )

    if (
        numeric_correct is True
        and _is_quantity_only(gold)
        and _is_quantity_only(predicted)
    ):
        return OutcomeEvaluation(
            correct=True,
            mode="post_selection_quantity",
            exact_match=False,
            numeric_correct=True,
            numeric_explanation=numeric_explanation,
            judge_score=None,
            judge_justification=None,
            judge_correctness_threshold=None,
            evaluated=True,
        )

    if numeric_correct is True:
        return OutcomeEvaluation(
            correct=None,
            mode="numeric_component_only",
            exact_match=False,
            numeric_correct=True,
            numeric_explanation=numeric_explanation,
            judge_score=None,
            judge_justification=(
                "The reference quantity matches, but metric/entity/period semantics "
                "require an independent evaluator."
            ),
            judge_correctness_threshold=None,
            evaluated=False,
        )

    return OutcomeEvaluation(
        correct=None,
        mode="not_evaluated",
        exact_match=False,
        numeric_correct=None,
        numeric_explanation=numeric_explanation,
        judge_score=None,
        judge_justification=(
            "Enable --use-llm-judge or run an external/human evaluator for "
            "nonnumeric paraphrases."
        ),
        judge_correctness_threshold=None,
        evaluated=False,
    )


__all__ = [
    "DEFAULT_JUDGE_CORRECTNESS_THRESHOLD",
    "OutcomeEvaluation",
    "evaluate_post_selection",
]
