"""Regression tests for per-question, model-attributed usage accounting."""

import pytest

from src.agents.orchestrator import _summarize_usage_records
from src.config import calculate_cost
from src.providers.base import UsageTracker


def test_usage_tracker_cursor_isolates_completed_calls():
    tracker = UsageTracker()
    tracker.record(
        {"prompt_tokens": 10, "completion_tokens": 2},
        model="gpt-4o-mini",
        provider="openai",
    )
    cursor = tracker.cursor()
    tracker.record(
        {"prompt_tokens": 20, "completion_tokens": 4},
        model="gpt-4o",
        provider="openai",
    )

    assert tracker.records_since(cursor) == [
        {
            "model": "gpt-4o",
            "provider": "openai",
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "total_tokens": 24,
        }
    ]


def test_usage_summary_prices_each_model_with_its_own_rate():
    records = [
        {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "prompt_tokens": 1000,
            "completion_tokens": 100,
        },
        {
            "model": "gpt-4o",
            "provider": "openai",
            "prompt_tokens": 200,
            "completion_tokens": 50,
        },
    ]

    summary = _summarize_usage_records(records, fallback_model="unused")
    expected = calculate_cost("gpt-4o-mini", records[0]) + calculate_cost(
        "gpt-4o", records[1]
    )

    assert summary["prompt_tokens"] == 1200
    assert summary["completion_tokens"] == 150
    assert summary["calls"] == 2
    assert summary["estimated_cost_usd"] == pytest.approx(expected)
    assert set(summary["by_model"]) == {"gpt-4o-mini", "gpt-4o"}


def test_usage_tracker_rejects_negative_provider_counts():
    tracker = UsageTracker()

    with pytest.raises(ValueError, match="non-negative"):
        tracker.record(
            {"prompt_tokens": -1, "completion_tokens": 0},
            model="broken",
            provider="test",
        )
