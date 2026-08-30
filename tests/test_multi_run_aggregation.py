"""Offline tests for question-aligned multi-run aggregation."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from evaluation.metrics import bootstrap_ci
from src.bulk_testing import BulkTestConfig, aggregate_and_save_results


def _config(tmp_path):
    return BulkTestConfig(
        dataset_name="fixture",
        output_dir=str(tmp_path),
        timestamp="2026-08-29_00-00-00",
        seed=11,
    )


def _runs():
    run_one = pd.DataFrame(
        {
            "question_id": ["q1", "q2", "q3"],
            "question": ["one", "two", "three"],
            "run_id": [0, 0, 0],
            "local_seed": [11, 11, 11],
            "generation_seed": [101, 101, 101],
            "correct": [True, False, True],
            "semantic_similarity": [0.8, 0.2, 0.6],
            "judge_score": [0.9, 0.1, 0.7],
            "numeric_accuracy": [1.0, 0.0, None],
            "policy_accepted": [True, False, True],
            "abstained": [False, True, False],
            "error": [None, None, "timeout"],
            "prompt_tokens": [100, 200, 300],
            "completion_tokens": [10, 20, 30],
            "evaluation_prompt_tokens": [5, 5, 5],
            "evaluation_completion_tokens": [2, 2, 2],
            "llm_calls": [1, 2, 3],
            "evaluation_llm_calls": [1, 1, 1],
            "cost_usd": [0.1, 0.2, 0.3],
            "evaluation_estimated_cost_usd": [0.01, 0.02, 0.03],
            "retrieval_time_ms": [10, 20, 30],
            "generation_time_ms": [100, 200, 300],
        }
    )
    # Deliberately shuffled: aggregation must align by question_id, not row.
    run_two = pd.DataFrame(
        {
            "question_id": ["q3", "q1", "q2"],
            "question": ["three", "one", "two"],
            "run_id": [1, 1, 1],
            "local_seed": [12, 12, 12],
            "generation_seed": [102, 102, 102],
            "correct": [True, False, True],
            "semantic_similarity": [0.8, 0.6, 0.4],
            "judge_score": [0.9, 0.7, 0.3],
            "numeric_accuracy": [None, 0.0, 1.0],
            "policy_accepted": [True, True, False],
            "abstained": [True, False, False],
            "error": [None, None, None],
            "prompt_tokens": [320, 120, 220],
            "completion_tokens": [32, 12, 22],
            "evaluation_prompt_tokens": [6, 6, 6],
            "evaluation_completion_tokens": [3, 3, 3],
            "llm_calls": [3, 1, 2],
            "evaluation_llm_calls": [1, 1, 1],
            "cost_usd": [0.32, 0.12, 0.22],
            "evaluation_estimated_cost_usd": [0.03, 0.01, 0.02],
            "retrieval_time_ms": [32, 12, 22],
            "generation_time_ms": [320, 120, 220],
        }
    )
    return run_one, run_two


def test_multi_run_aggregation_aligns_questions_and_bootstraps_question_means(
    tmp_path,
):
    artifacts = aggregate_and_save_results(
        list(_runs()),
        _config(tmp_path),
        SimpleNamespace(name="toy"),
        [],
    )

    per_question = pd.read_csv(artifacts["per_question_csv"]).set_index(
        "question_id"
    )
    assert per_question.loc["q1", "correct_mean"] == 0.5
    assert per_question.loc["q2", "correct_mean"] == 0.5
    assert per_question.loc["q3", "correct_mean"] == 1.0
    assert per_question.loc["q1", "coverage_mean"] == 1.0
    assert per_question.loc["q2", "coverage_mean"] == 0.5
    assert per_question.loc["q3", "coverage_mean"] == 0.0
    assert (per_question["run_observation_count"] == 2).all()

    summary = json.loads(artifacts["summary_json"].read_text())
    assert summary["aggregation"]["local_seeds"] == [11, 12]
    assert summary["aggregation"]["generation_seeds_requested"] == [101, 102]
    assert summary["per_run_metrics"][0]["local_seed"] == 11
    assert summary["per_run_metrics"][0]["generation_seed_requested"] == 101
    assert len(summary["per_run_metrics"]) == 2
    assert summary["per_run_metrics"][0]["metrics"][
        "selective_prediction"
    ]["coverage"] == pytest.approx(1 / 3)

    expected_mean, expected_low, expected_high = bootstrap_ci(
        [0.5, 0.5, 1.0], n_bootstrap=1000, seed=11
    )
    correctness = summary["question_cluster_stats"]["correct"]
    assert correctness["mean"] == expected_mean
    assert correctness["ci_95"] == [expected_low, expected_high]
    assert correctness["question_count"] == 3
    assert correctness["bootstrap_unit"] == "question_id"

    required_metrics = {
        "correct",
        "semantic_similarity",
        "judge_score",
        "numeric_accuracy",
        "policy_accepted",
        "abstained",
        "error",
        "coverage",
        "total_tokens",
        "total_cost_usd",
        "total_time_ms",
    }
    assert required_metrics <= set(summary["question_cluster_stats"])
    assert artifacts["combined_csv"].exists()
    assert not list(tmp_path.glob("*_latex_row.tex"))


def test_multi_run_aggregation_requires_question_id(tmp_path):
    run_one, run_two = _runs()
    with pytest.raises(ValueError, match="missing required question_id"):
        aggregate_and_save_results(
            [run_one.drop(columns="question_id"), run_two],
            _config(tmp_path),
            SimpleNamespace(name="toy"),
            [],
        )


def test_multi_run_aggregation_rejects_duplicate_or_mismatched_ids(tmp_path):
    run_one, run_two = _runs()
    duplicated = pd.concat([run_one, run_one.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate question_id"):
        aggregate_and_save_results(
            [duplicated, run_two],
            _config(tmp_path),
            SimpleNamespace(name="toy"),
            [],
        )

    mismatched = run_two.copy()
    mismatched.loc[mismatched["question_id"] == "q3", "question_id"] = "q4"
    with pytest.raises(ValueError, match="question_id set differs"):
        aggregate_and_save_results(
            [run_one, mismatched],
            _config(tmp_path),
            SimpleNamespace(name="toy"),
            [],
        )
