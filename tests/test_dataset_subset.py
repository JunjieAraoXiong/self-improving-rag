"""Fail-closed and reproducible benchmark-subset selection tests."""

import hashlib
import json

import pandas as pd
import pytest

from dataset_adapters.base import BaseDatasetAdapter
from src.bulk_testing import BulkTestConfig, BulkTestRunner


class DummyAdapter(BaseDatasetAdapter):
    def __init__(self, frame: pd.DataFrame, subset_csv=None):
        super().__init__(subset_csv=subset_csv)
        self.frame = frame

    def load_dataset(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._apply_subset_filter(self.frame.copy())
        return self._df

    def get_question_column(self) -> str:
        return "question"

    def get_answer_column(self) -> str:
        return "answer"


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "question_id": ["q1", "q2", "q3"],
            "question": ["one", "two", "three"],
            "answer": ["a", "b", "c"],
        }
    )


def test_subset_ids_are_validated_ordered_and_hashed(tmp_path, frame):
    subset = tmp_path / "subset.csv"
    subset.write_text("question_id\nq3\nq1\n", encoding="utf-8")
    adapter = DummyAdapter(frame, subset_csv=str(subset))

    selected = adapter.load_dataset()
    metadata = adapter.get_selection_metadata()

    assert selected["question_id"].tolist() == ["q3", "q1"]
    assert metadata["selection_mode"] == "id"
    assert metadata["source_row_count"] == 3
    assert metadata["requested_row_count"] == 2
    assert metadata["selected_row_count"] == 2
    assert metadata["subset_file_sha256"] == hashlib.sha256(
        subset.read_bytes()
    ).hexdigest()
    assert len(metadata["selected_ids_sha256"]) == 64


@pytest.mark.parametrize(
    "csv_text,match",
    [
        ("question_id\nq4\n", "absent from the dataset"),
        ("question_id\nq1\nq1\n", "duplicate IDs"),
        ("unexpected\nq1\n", "no supported selection column"),
        ("question_id\n", "empty"),
    ],
)
def test_invalid_subset_never_falls_back_to_full_dataset(
    tmp_path, frame, csv_text, match
):
    subset = tmp_path / "bad.csv"
    subset.write_text(csv_text, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        DummyAdapter(frame, subset_csv=str(subset)).load_dataset()


def test_out_of_range_subset_index_fails_closed(tmp_path, frame):
    subset = tmp_path / "index.csv"
    subset.write_text("index\n0\n99\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-range"):
        DummyAdapter(frame, subset_csv=str(subset)).load_dataset()


def test_saved_artifacts_persist_subset_hash_and_count(tmp_path, frame):
    subset = tmp_path / "subset.csv"
    subset.write_text("question_id\nq2\n", encoding="utf-8")
    adapter = DummyAdapter(frame, subset_csv=str(subset))
    adapter.load_dataset()
    config = BulkTestConfig(
        dataset_name="dummy",
        output_dir=str(tmp_path),
        timestamp="fixed",
    )
    runner = BulkTestRunner(config)
    results = pd.DataFrame(
        {
            "question_id": ["q2"],
            "correct": [True],
            "evaluation_mode": ["post_selection_exact"],
            "abstained": [False],
            "error": [None],
        }
    )

    csv_path = runner.save_results(results, adapter)
    summary = json.loads(csv_path.with_suffix(".json").read_text())
    saved = pd.read_csv(csv_path)

    assert summary["dataset_selection"]["selected_row_count"] == 1
    assert summary["dataset_selection"]["subset_file_sha256"] == (
        hashlib.sha256(subset.read_bytes()).hexdigest()
    )
    assert saved.loc[0, "dataset_selected_row_count"] == 1
    assert saved.loc[0, "dataset_subset_file_sha256"] == summary[
        "dataset_selection"
    ]["subset_file_sha256"]
