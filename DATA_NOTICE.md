# Data notice

The repository's MIT license applies to the project code. It does not replace
the licenses of third-party datasets or source filings.

## FinanceBench

The files under `data/question_sets/` are snapshots derived from FinanceBench,
published by Patronus AI:

- Upstream repository: <https://github.com/patronus-ai/financebench>
- Dataset card: <https://huggingface.co/datasets/PatronusAI/financebench>
- Upstream dataset license: **CC BY-NC 4.0**
- Paper: *FinanceBench: A New Benchmark for Financial Question Answering*
  (<https://arxiv.org/abs/2311.11944>)

Snapshot checksums:

| File | Rows | SHA-256 |
|---|---:|---|
| `financebench_open_source.jsonl` | 150 | `a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877` |
| `financebench_document_information.jsonl` | 361 | `1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575` |

The bundled open-source question file contains 50 `metrics-generated`, 50
`domain-relevant`, and 50 `novel-generated` examples. Generate these counts
from the artifact rather than copying them manually:

```bash
python scripts/audit_dataset.py
```

SEC filings linked by FinanceBench may have their own terms and are not
redistributed in this repository.
