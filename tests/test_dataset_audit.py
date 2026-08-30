"""Tests for benchmark snapshot auditing."""

import json

from scripts.audit_dataset import audit_questions


def test_audit_reports_counts_hashes_and_duplicates(tmp_path):
    path = tmp_path / "questions.jsonl"
    rows = [
        {
            "financebench_id": "q1",
            "question": "Question one?",
            "answer": "One",
            "question_type": "metrics-generated",
            "dataset_subset_label": "OPEN_SOURCE",
            "company": "A",
            "doc_name": "A_10K",
        },
        {
            "financebench_id": "q2",
            "question": "Question two?",
            "answer": "Two",
            "question_type": "domain-relevant",
            "dataset_subset_label": "OPEN_SOURCE",
            "company": "B",
            "doc_name": "B_10K",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = audit_questions(path)

    assert report["rows"] == 2
    assert report["unique_ids"] == 2
    assert report["duplicate_ids"] == []
    assert report["question_type_counts"] == {
        "domain-relevant": 1,
        "metrics-generated": 1,
    }
    assert len(report["sha256"]) == 64


def test_audit_surfaces_duplicate_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    row = {
        "financebench_id": "q1",
        "question": "Question?",
        "answer": "Answer",
        "question_type": "metrics-generated",
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    report = audit_questions(path)

    assert report["duplicate_ids"] == ["q1"]
