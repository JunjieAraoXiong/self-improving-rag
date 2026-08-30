# V2 architecture: evidence-gap repair with typed financial programs

## Design claim

The v2 hypothesis is deliberately narrower than “agents improve themselves”:

> Error-localized correction should repair financial QA more reliably and
> efficiently than blind full-pipeline retry, while typed execution should make
> derived numeric answers independently verifiable without a gold answer.

The implementation provides an experimentable runtime for that hypothesis. It
does not establish the hypothesis without a matched-budget benchmark.

## Assurance boundary

The pipeline has three different assurance levels. They should not be merged
into one generic confidence score.

| Path | What is checked | What is not established |
|---|---|---|
| Qualitative/extraction | Citation syntax, exact quote fidelity, answer-number support, conservative claim↔quote lexical anchors, policy coverage | Full semantic entailment, citation attribution, or correctness |
| Typed calculation, evidence/arithmetic | Operand evidence, quantity typing, expression safety, arithmetic | That the formula and requested answer semantics match the question |
| Typed calculation, full contract | All of the above plus trusted entity, period, metric, operands, formula, output type, rounding, and visible answer | True table-cell identity in flattened text or source truth beyond the corpus |

Only the full-contract path may set `fully_verified=true`. The system still
means “verified against this retrieved corpus and contract,” not “universally
true.”

## Contracts

### Query plan

`compile_finance_query` extracts entities, periods, source hints, output shape,
evidence needs, an allowlisted `formula_id`, and unresolved constraints. It is
deterministic and does not call a model or retrieval tool. Relative reporting
periods remain unresolved until a corpus-aware filing calendar exists.

Coverage is intentionally explicit. On the bundled 150-question snapshot, the
compiler identifies 58 calculation-like questions. Thirty map to 17 trusted
formula families and can build full contracts; 28 remain calculation-like but
have no allowlisted formula and therefore fail closed. Four rows labeled as
numerical reasoning in the source dataset ask what *drove* a margin change and
are intentionally handled as qualitative causal questions.

### Finance question contract

For supported calculations, `build_finance_question_spec` converts the query
plan into an exact expression tree and operand contracts. The generated answer
cannot change:

- the answer entity, period, metric, unit, scale, or rounding;
- the set and identity of operands;
- each operand's entity, period, metric, unit, currency, or scale contract;
- the expression tree or constants.

This pre-generation boundary is important. Merely checking that a model's
self-authored program is internally consistent would still allow it to choose
the wrong formula consistently.

### Finance program

The model returns a visible answer plus one JSON block inside
`<finance_program>...</finance_program>`. The schema accepts only a small
allowlist of operations: references, trusted constants, addition, subtraction,
multiplication, division, average, absolute value, negation, and percent
change. Values are finite decimal strings and execute with Python `Decimal`;
no model-authored code is evaluated.

Each operand carries an exact evidence reference:

```json
{
  "id": "need:revenue:acme:fy2024",
  "value": "1250",
  "currency": "USD",
  "scale": "million",
  "unit": "money",
  "entity": "ACME",
  "period": "FY2024",
  "metric": "revenue",
  "evidence": {
    "doc_id": "Doc2",
    "quote": "Revenue USD 1,250 million",
    "value_text": "USD 1,250 million",
    "metric_label": "Revenue",
    "period_label": "FY2024",
    "occurrence": 1
  }
}
```

## Controller semantics

The Judge emits structured issues. `CorrectionPolicy` maps them to the smallest
action that can plausibly fix them.

| Issue family | Action | Retrieval? | New generation? |
|---|---|---:|---:|
| Missing/mismatched operand evidence | `targeted_retrieval` | affected needs only | yes |
| Citation/schema/missing-program problem | `reuse_evidence_regenerate` | no | yes |
| Formula, arithmetic, or result-unit problem | `replan` | no | yes |
| Declared result differs from verified execution | `local_recompute` | no | no |
| Visible answer differs from verified result | `rerender` | no | no |
| Conflicting evidence | `reconcile` | no by default | yes |
| Repeated unchanged gap, exhausted budget, unresolved constraint | `abstain` | no | no |

The evidence fingerprint includes source identity, corpus/index version when
available, and a content hash. A reused chunk ID with changed text is therefore
new evidence rather than a false cache hit.

### Terminal invariants

For `gap_driven_v2`:

1. Gold answers are never passed to the policy Judge.
2. A result is accepted only after the relevant verification path passes or a
   deterministic local repair produces the canonical answer.
3. A rejected draft is never returned as the final answer.
4. Repeating the same verifier gap over the same evidence terminates early.
5. Unresolved relative periods terminate before retrieval.

`paper_fixed` is intentionally separate because it retains historical
threshold decay, full retrieval/generation retries, and best-candidate behavior
needed to study the published configuration.

## Reproducibility boundary

Dataset selection is fail-closed. A requested subset must contain valid,
unique IDs or valid row indices, and its requested order is preserved. Empty,
null, duplicate, unknown, and out-of-range selections raise an error rather
than reverting to the full benchmark. Result artifacts record source and subset
paths, source/subset SHA-256 hashes, requested and selected row counts, and a
hash of the ordered selected IDs.

Every evidence item records a content SHA-256 hash even when the retriever does
not expose a stable chunk ID. Corpus, index, parser, and embedding versions,
filing identifiers, ranks, and retrieval scores are recorded when the backend
provides them. Missing provenance remains visibly missing; it is not inferred.

Runs distinguish the local sampling seed from the generation seed requested of
the hosted provider. Provider, SDK, model, and request metadata are persisted.
Remote APIs may ignore seeds or change implementations, so repeated hosted
generations are not claimed to be deterministic or statistically independent.

## Evaluation protocol for a main-conference paper

A credible experiment should freeze corpora, indexes, prompts, model versions,
budgets, and per-question traces, then compare:

1. single-pass RAG;
2. full restart at the same maximum budget;
3. best-of-N plus independent reranking;
4. generic self-refinement/corrective RAG;
5. fixed paper retry;
6. evidence-gap correction;
7. evidence-gap correction plus typed execution.

Match systems on model calls, tool calls, input/output tokens, dollars, and
latency. Report paired confidence intervals for exact and numeric correctness,
citation attribution, evidence recall, calculation validity, verifier issue
detection precision/recall, correction rate, abstention, and risk–coverage.
Do not use the same evaluator to select an answer and declare it correct.

The bulk runner now records policy utility separately from a shared
post-selection evaluator used by both single-pass and agentic paths. An LLM
outcome evaluator is opt-in and sees the gold answer only after selection.
Without it, a matching quantity does not establish the requested metric,
entity, or period; nonnumeric paraphrases remain unlabeled. Provider/evaluator
failures are recorded as errors rather than zero-quality model outputs.

The default LLM outcome threshold is `0.99`, representing the evaluator's
full-credit label. That threshold is stored in each result row and reused by
aggregate reports. Scores at or above `0.5` are reported separately as partial
credit and are not counted as correct. For multi-run studies, the aggregator
requires unique and identical question-ID sets, averages repetitions within
each question, and bootstraps question means—not run means or individual API
calls.

FinanceBench alone is too small for the intended claim. Add FinQA and TAT-QA
for program/table-text reasoning, plus a modern end-to-end workflow set such as
[FrontierFinance](https://arxiv.org/abs/2608.11683). Test multiple model
families and include adversarial perturbations for wrong entity, wrong period,
swapped metrics, units/scales, duplicated values, conflicting passages, and
citation tampering.

## FrontierFinance-informed roadmap

The public FrontierFinance work emphasizes realistic workflows, specialized
tools, rubric-level evaluation, latency, and cost. Those are useful directions,
but its reported system is not a drop-in architecture or a substitute for
controlled ablations here.

Potential next adapters, clearly not implemented yet:

- SEC/EDGAR filing search with accession, filing date, and as-of controls;
- XBRL and structured table-cell lookup before flattened-text fallback;
- web search plus HTML parsing with source and publication-time provenance;
- market-price data with timestamp, exchange, adjustment, and currency fields;
- a deterministic calculator exposed through the same typed quantity schema;
- long-document retrieval with explicit tool, token, latency, and dollar budgets;
- rubric/checklist evaluation for multi-part analyst workflows.

The public references are the
[FrontierFinance overview](https://samaya.ai/blog/frontier-finance),
[benchmark page](https://research.samaya.ai/benchmarks/frontier-finance), and
[open evaluation harness](https://github.com/samaya-ai/frontier-finance).

## Known limitations

- Flattened tables do not provide cryptographic or structural proof that a
  value belongs to a particular row and column. The verifier fails closed on
  ambiguous multi-header cases, which can reduce recall.
- Entity aliases and fiscal calendars are rule-based and incomplete.
- Exact normalized quotes are robust to safe whitespace/Unicode normalization,
  not general OCR correction.
- Formula coverage is allowlisted and intentionally finite.
- Qualitative claims lack a typed entailment representation.
- There is no online learning loop, learned planner, calibrated value model, or
  cost-aware policy optimizer.
- Usage is attributed by provider/model and question, but dollar totals use a
  static repository rate table; any paper must freeze and disclose its rate-card date.
- No new end-to-end benchmark has been run for this revision.
