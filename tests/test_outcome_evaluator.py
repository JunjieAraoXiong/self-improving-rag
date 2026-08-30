"""Tests for the policy-independent, post-selection evaluator."""

import pytest

from evaluation.llm_judge import JudgeEvaluationError, parse_judge_response
from evaluation.outcome_evaluator import evaluate_post_selection


def test_numeric_outcome_is_evaluated_without_an_llm():
    result = evaluate_post_selection(
        question="What was revenue?",
        gold_answer="$1.5 billion",
        predicted_answer="$1,500 million",
        use_llm_judge=False,
        judge_model="unused",
    )

    assert result.correct is True
    assert result.numeric_correct is True
    assert result.mode == "post_selection_quantity"


def test_matching_number_does_not_prove_metric_semantics():
    result = evaluate_post_selection(
        question="What was revenue?",
        gold_answer="Revenue was $100 million.",
        predicted_answer="Operating expenses were $100 million.",
        use_llm_judge=False,
        judge_model="unused",
    )

    assert result.numeric_correct is True
    assert result.correct is None
    assert result.evaluated is False
    assert result.mode == "numeric_component_only"


@pytest.mark.parametrize(
    ("gold", "predicted"),
    [
        ("$100 million", "$95 million"),
        ("20.0%", "19.1%"),
        ("10.0x", "9.51x"),
    ],
)
def test_quantity_accuracy_does_not_inherit_loose_five_percent_tolerance(
    gold, predicted
):
    result = evaluate_post_selection(
        question="What was the reported value?",
        gold_answer=gold,
        predicted_answer=predicted,
        use_llm_judge=False,
        judge_model="unused",
    )

    assert result.numeric_correct is False
    assert result.correct is False
    assert result.mode == "post_selection_numeric_mismatch"


def test_nonnumeric_paraphrase_is_not_mislabeled_without_evaluator():
    result = evaluate_post_selection(
        question="When is revenue recognized?",
        gold_answer="When control transfers to the customer.",
        predicted_answer="Upon transfer of control.",
        use_llm_judge=False,
        judge_model="unused",
    )

    assert result.correct is None
    assert result.evaluated is False
    assert result.mode == "not_evaluated"


def test_llm_outcome_judge_runs_after_selection_and_sets_separate_label():
    calls = []

    def fake_judge(question, gold, predicted, model):
        calls.append((question, gold, predicted, model))
        return 1.0, "Equivalent answer"

    result = evaluate_post_selection(
        question="When is revenue recognized?",
        gold_answer="When control transfers.",
        predicted_answer="At the transfer of control.",
        use_llm_judge=True,
        judge_model="independent-evaluator",
        judge_fn=fake_judge,
    )

    assert calls == [
        (
            "When is revenue recognized?",
            "When control transfers.",
            "At the transfer of control.",
            "independent-evaluator",
        )
    ]
    assert result.correct is True
    assert result.judge_score == 1.0
    assert result.mode == "post_selection_llm_judge"


def test_partial_credit_is_not_labeled_fully_correct():
    result = evaluate_post_selection(
        question="Explain the change in revenue and margin.",
        gold_answer="Revenue increased and margin declined.",
        predicted_answer="Revenue increased.",
        use_llm_judge=True,
        judge_model="independent-evaluator",
        judge_fn=lambda *args: (0.5, "Only one of two requested facts is correct"),
    )

    assert result.judge_score == 0.5
    assert result.judge_correctness_threshold == 0.99
    assert result.correct is False
    assert result.mode == "post_selection_llm_judge"


def test_aggregate_judge_pass_rate_uses_persisted_full_credit_threshold():
    import pandas as pd

    from evaluation.metrics import calculate_aggregate_metrics

    frame = pd.DataFrame(
        {
            "judge_score": [0.5, 1.0],
            "outcome_judge_threshold": [0.99, 0.99],
            "evaluation_mode": [
                "post_selection_llm_judge",
                "post_selection_llm_judge",
            ],
        }
    )

    judge = calculate_aggregate_metrics(frame)["judge_score"]

    assert judge["threshold_pass_rate"] == 0.5
    assert judge["correctness_thresholds"] == [0.99]
    assert judge["partial_credit_rate_at_0_5"] == 1.0


def test_abstention_counts_as_unanswered_without_spending_evaluator_call():
    def should_not_run(*args):
        raise AssertionError("evaluator must not run for a terminal abstention")

    result = evaluate_post_selection(
        question="When is revenue recognized?",
        gold_answer="When control transfers.",
        predicted_answer=None,
        use_llm_judge=True,
        judge_model="independent-evaluator",
        abstained=True,
        judge_fn=should_not_run,
    )

    assert result.correct is False
    assert result.judge_score == 0.0
    assert result.mode == "post_selection_abstention"


def test_provider_error_is_not_counted_as_selective_abstention():
    result = evaluate_post_selection(
        question="What was revenue?",
        gold_answer="$10 million",
        predicted_answer=None,
        use_llm_judge=False,
        judge_model="unused",
        terminal_error="provider timeout",
        abstained=False,
    )

    assert result.correct is False
    assert result.mode == "terminal_error"
    assert result.judge_justification == "provider timeout"


def test_outcome_judge_failure_is_not_converted_to_zero_score():
    def failing_judge(*args):
        raise RuntimeError("evaluator unavailable")

    result = evaluate_post_selection(
        question="When is revenue recognized?",
        gold_answer="When control transfers.",
        predicted_answer="At transfer of control.",
        use_llm_judge=True,
        judge_model="independent-evaluator",
        judge_fn=failing_judge,
    )

    assert result.correct is None
    assert result.judge_score is None
    assert result.evaluated is False
    assert result.mode == "evaluator_error"


def test_judge_parser_requires_one_in_range_score_and_justification():
    assert parse_judge_response(
        "SCORE: 1.0\nJUSTIFICATION: Fully correct.\nThe units also match."
    ) == (1.0, "Fully correct.\nThe units also match.")

    malformed = [
        "JUSTIFICATION: Missing score",
        "SCORE: 999\nJUSTIFICATION: Out of range",
        "SCORE: 0.5\nSCORE: 1.0\nJUSTIFICATION: Duplicate",
        "SCORE: 1.0\nJUSTIFICATION:",
        "SCORE: NaN\nJUSTIFICATION: Not finite",
    ]
    for response in malformed:
        with pytest.raises(JudgeEvaluationError):
            parse_judge_response(response)
