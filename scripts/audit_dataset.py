#!/usr/bin/env python3
"""Audit a JSONL benchmark artifact before running experiments.

The report is deliberately generated from the source file rather than copied
into paper tables by hand. It records hashes, duplicate IDs, category counts,
and missing fields so a run can pin the exact benchmark snapshot it used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_QUESTIONS = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "question_sets"
    / "financebench_open_source.jsonl"
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file with line-specific validation errors."""

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "<missing>")) for row in rows).items()))


def audit_questions(path: Path, id_field: str = "financebench_id") -> Dict[str, Any]:
    """Build a deterministic dataset audit report."""

    rows = read_jsonl(path)
    ids = [row.get(id_field) for row in rows]
    duplicate_ids = sorted(
        str(value) for value, count in Counter(ids).items() if count > 1
    )
    required_fields = (id_field, "question", "answer", "question_type")
    missing_fields = {
        field: sum(row.get(field) in (None, "") for row in rows)
        for field in required_fields
    }

    return {
        "file": path.name,
        "sha256": sha256(path),
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": duplicate_ids,
        "question_type_counts": _counts(rows, "question_type"),
        "subset_counts": _counts(rows, "dataset_subset_label"),
        "company_count": len({row.get("company") for row in rows}),
        "document_count": len({row.get("doc_name") for row in rows}),
        "missing_required_fields": missing_fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Question JSONL to audit",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    report = audit_questions(args.path.resolve())
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")

    has_missing = any(report["missing_required_fields"].values())
    return 1 if report["duplicate_ids"] or has_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
