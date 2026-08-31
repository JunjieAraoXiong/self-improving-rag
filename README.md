# Evidence-Gap-Driven Financial QA

> **V3 proposal:** a verifiable financial research runtime with an immutable
> question contract, typed evidence, deterministic execution, independent
> verification, localized recovery, and a replayable run ledger.

[![V3 verifiable financial research runtime system design](docs/assets/v3-system-design.png)](docs/assets/v3-system-design.png)

[Read the v3 design](docs/architecture-v3.md) ·
[Open the full-resolution diagram](docs/assets/v3-system-design.png) ·
[Edit the Excalidraw source](docs/assets/v3-system-design.excalidraw)

This repository contains two related systems:

1. **`gap_driven_v2` (default):** an experimental financial-QA runtime that
   compiles a question contract, retrieves only for diagnosed evidence gaps,
   verifies calculated answers with a typed program, and abstains when the
   contract cannot be verified.
2. **`paper_fixed`:** the historical fixed-retry system described in
   *Towards Expert Financial QA via Self-Improving RAG*, accepted at the AFA
   Workshop @ ICLR 2026.

Junjie Xiong (UC Berkeley), Shawheen Ghezavat (Cal Poly), Aum Hirpara (Hofstra
University), Sean Wu (Pepperdine University)

[[Current arXiv paper]](https://arxiv.org/abs/2608.26706) ·
[[v2 architecture]](docs/architecture-v2.md) ·
[[v3 system proposal]](docs/architecture-v3.md) ·
[[data notice]](DATA_NOTICE.md)

> The list above mirrors the original repository/workshop record. arXiv v1
> currently lists Junjie Xiong, Shawheen Ghezavat, and Aum Hirpara; this
> implementation work did not alter author metadata.

## What changed in v2

The original controller retried the whole pipeline with a wider `k`, a more
elaborate prompt, and a lower acceptance threshold. That is useful as a simple
baseline, but it cannot distinguish a missing source from a bad citation, an
incorrect formula, or a formatting error.

The default runtime now follows a narrower loop:

```text
question
   │
   ▼
deterministic query contract ── unresolved constraint ──► abstain
   │
   ▼
retrieval + stable evidence manifest
   │
   ├── extraction / qualitative answer ──► quote/numeric attribution gates
   │
   └── calculation ──► typed finance program ──► local Decimal executor
                                                   │
                          ┌────────────────────────┼─────────────────────┐
                          ▼                        ▼                     ▼
                       accept          local recompute/rerender   diagnosed gap
                                                                        │
                            reuse evidence ◄── correction policy ──► targeted retrieval
                                                                        │
                                                                        ▼
                                                                     abstain
```

For a calculated answer, the model may fill source-backed operand values and
quotes, but it may not choose the formula. The query compiler creates the
trusted expression tree before generation. The verifier then checks:

- the exact operand IDs and allowlisted expression tree;
- entity, fiscal period, metric, unit, currency, and scale;
- exact, line-bounded evidence quotes and locally associated values;
- deterministic `Decimal` arithmetic, dimensional consistency, and rounding;
- agreement between execution, the declared result, and the visible answer.

If only the declared or displayed result is wrong, the orchestrator uses the
executor's canonical answer without another model call. If evidence is missing,
it retrieves the affected operand need while retaining the existing evidence
snapshot. Repeated gaps over unchanged evidence terminate in abstention.

This is **self-correcting inference**, not a learned self-improving system. The
runtime does not update model weights, prompts, or policy parameters from its
own history.

## Historical paper results

The paper reports **86% oracle-guided accuracy** and a **36.4% Lazarus Rate** on
the bundled 150-question FinanceBench subset. Interpret those values narrowly:

- the 86% policy Judge could see the gold answer while selecting retries and
  candidates, so it is a best-of-budget upper bound rather than deployment
  accuracy;
- the reported Lazarus Rate is a Judge-scored improvement among selected retry
  cases, not yet an independently evaluated wrong-to-correct transition rate;
- a blind Judge pass rate is policy acceptance, not correctness.

No v2 benchmark result is claimed yet. The repository lacks the original SEC
PDF corpus, frozen Chroma index, API model snapshots, and raw paper runs needed
for exact replication.

## Quick start

```bash
git clone https://github.com/JunjieAraoXiong/self-improving-rag.git
cd self-improving-rag
python -m pip install -r requirements.txt
cp .env.example .env
```

Add the provider keys you intend to use to `.env`.

The Google backend uses the current `google-genai` client rather than the
deprecated `google-generativeai` SDK. Hosted APIs receive a best-effort seed
when supported, but a recorded seed does not make remote inference exactly
reproducible.

Audit the bundled benchmark snapshot:

```bash
python scripts/audit_dataset.py
```

Build an index from SEC PDFs you are authorized to use:

```bash
python src/ingest_docling.py \
  --input-dir data/pdfs \
  --output-dir chroma_docling
```

Indexes made before the v2 metadata changes should be rebuilt so chunks have
canonical company keys, stable IDs, and source-local ordering.

Run the default gold-free correction policy:

```bash
python src/bulk_testing.py \
  --dataset financebench \
  --pipeline hybrid_filter_rerank \
  --model gpt-4o-mini \
  --use-agentic-retry \
  --policy-mode gap_driven_v2 \
  --max-retries 2 \
  --blind-judge \
  --use-llm-judge \
  --outcome-judge-model gpt-4o \
  --outcome-judge-threshold 0.99
```

The v2 orchestrator never exposes the gold answer to its policy, even if the
dataset contains one. `--blind-judge` is included above to make experimental
intent explicit. `--use-llm-judge` invokes a separate gold-based evaluator only
after the final output is frozen. Policy utility, policy acceptance, and outcome
correctness are stored in different fields. Without that evaluator, numeric
equivalence remains a component metric and nonnumeric paraphrases remain
unlabeled unless they are exact matches; neither is silently called accuracy.
The outcome model is configured separately from the policy Judge, and only a
strictly parsed full-credit score is labeled correct; partial credit remains in
the continuous evaluator score. The threshold is persisted with every outcome
so aggregate reports apply the same definition of correctness.

If `--subset path/to/subset.csv` is supplied, the loader validates every ID or
row index, preserves the requested order, and aborts on empty, duplicate,
unknown, null, or out-of-range selections. It never silently falls back to the
full dataset.

## Reproduce the historical policy configuration

These commands recreate configurations, not the absent frozen artifacts or
the exact published numbers.

```bash
# Single-pass baseline
python src/bulk_testing.py \
  --dataset financebench \
  --pipeline hybrid_filter_rerank \
  --top-k 10 \
  --model gpt-4o-mini \
  --use-llm-judge

# Paper-style oracle-guided fixed retry
python src/bulk_testing.py \
  --dataset financebench \
  --pipeline hybrid_filter_rerank \
  --top-k 10 \
  --model gpt-4o-mini \
  --use-agentic-retry \
  --policy-mode paper_fixed \
  --max-retries 2

# Paper-style schedule without policy access to gold
python src/bulk_testing.py \
  --dataset financebench \
  --pipeline hybrid_filter_rerank \
  --model gpt-4o-mini \
  --use-agentic-retry \
  --policy-mode paper_fixed \
  --max-retries 2 \
  --blind-judge
```

In `paper_fixed`, retrieval follows `k=10 → 20 → 30`, prompt strategies follow
Standard → Conservative → Detailed, and the threshold decays from 0.5 by 0.1
per retry to a floor of 0.3. In v2, the threshold remains fixed and controller
actions depend on structured verifier issues.

## Outputs and evaluation

Agentic result rows and decision logs include:

- the compiled query and finance-question contracts;
- per-attempt evidence manifests, content hashes, source/index/parser/embedding
  versions when available, and correction actions;
- parsed finance programs, execution traces, and verification issues;
- separate policy and post-selection outcome scores;
- acceptance, abstention, error, latency, per-model token/call, and estimated
  cost fields;
- dataset and subset paths, selected-row counts, selected-ID hashes, and input
  file hashes;
- local seeds, requested provider seeds, provider/model identifiers, and request
  metadata;
- independently computed semantic and numeric outcome metrics.

For repeated runs, aggregation first validates that each run has unique and
identical question-ID sets, averages observations within each question, and
bootstraps question clusters. It does not treat repeated API calls as
independent questions.

Run offline tests with:

```bash
python -m pytest -q -m "not slow and not api"
```

For a credible new study, compare v2 against single pass, full restart,
best-of-N, Judge reranking, and self-refinement under matched token, tool-call,
dollar, and latency budgets. Report independent correctness, citation
attribution, calculation validity, issue-detection precision/recall,
risk–coverage, abstention, and paired uncertainty intervals.

## Project structure

```text
src/
├── agents/
│   ├── orchestrator.py       # controller and terminal-state invariants
│   ├── correction_policy.py  # issue → smallest corrective action
│   ├── retrieval_agent.py
│   ├── reasoning_agent.py
│   ├── judge_agent.py
│   └── logger.py
├── query_understanding/      # deterministic finance query compiler
├── finance_contract.py       # query plan → trusted formula/operand contract
├── finance_program.py        # strict schema, evidence binding, execution
├── retrieval_tools/
├── providers/
└── bulk_testing.py
evaluation/                    # independent metrics and legacy verification
dataset_adapters/             # FinanceBench loader
scripts/                      # dataset audit and experiment helpers
tests/                        # offline and adversarial regression suite
```

## Current boundaries

- Flattened table text cannot prove true row/column cell identity. Structured
  cell coordinates or XBRL facts are the next evidence-layer upgrade.
- Exact quote matching intentionally rejects some harmless OCR variants.
- The query compiler supports an allowlist of finance formulas; unresolved or
  unknown calculations abstain instead of asking the model to invent a formula.
- On the bundled 150-question snapshot, 58 questions are classified as
  calculation-like: 30 have trusted contracts across 17 formula families and
  28 deliberately fail closed. Four dataset rows tagged with numerical
  reasoning are causal “what drove the margin change?” questions and remain on
  the qualitative path.
- Qualitative and extraction questions use citation/support gates, not the
  full typed-calculation guarantee. Their deterministic gate establishes exact
  quote fidelity and conservative lexical attribution, not semantic entailment;
  correctness and citation attribution still need independent evaluation.
- Cost and latency are measured, but the controller is not yet trained or
  optimized for expected accuracy gain per dollar or second.
- Dollar fields use the repository's static rate table and are estimates;
  publish a dated, frozen rate card alongside any cost comparison.
- Logs are research traces, not a tamper-evident regulatory audit ledger.
- FrontierFinance-style web, filing, market-data, calculator, and long-document
  tools are a research roadmap, not implemented capabilities in this repo.

The proposed v3 architecture adds a typed, multi-source evidence graph,
structured XBRL/table adapters, as-of-time enforcement, contradiction
preservation, budget-aware recovery, and deterministic replay. It deliberately
uses one controller with typed tools rather than a peer-to-peer agent swarm.
Those components are a staged design, not implemented capabilities or reported
results.

See [docs/architecture-v2.md](docs/architecture-v2.md) for the implemented
assurance model and [docs/architecture-v3.md](docs/architecture-v3.md) for the
upgraded system design, contracts, migration gates, and experiment plan.

## Data, license, and citation

The code is MIT licensed. The bundled FinanceBench-derived data is licensed
separately under CC BY-NC 4.0; see [DATA_NOTICE.md](DATA_NOTICE.md).

```bibtex
@inproceedings{xiong2026selfimproving,
  title={Towards Expert Financial QA via Self-Improving RAG},
  author={Xiong, Junjie and Ghezavat, Shawheen and Hirpara, Aum and Wu, Sean},
  booktitle={AFA Workshop at the International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://github.com/JunjieAraoXiong/self-improving-rag}
}
```
