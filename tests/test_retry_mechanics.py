"""Offline tests for the retry loop described by the paper."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.agents.base import AgentDecision
from src.agents.orchestrator import (
    AgenticRAGConfig,
    AgenticRAGOrchestrator,
    _evidence_fingerprint,
    _merge_evidence,
    build_agentic_orchestrator,
)
from src.agents.judge_agent import JudgeAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.finance_contract import build_finance_question_spec
from src.finance_program import FinanceOperandSpec, FinanceProgram, FinanceQuestionSpec
from src.providers.base import LLMResponse
from src.query_understanding import compile_finance_query
from src.retrieval_tools.tool_registry import SimplePipeline


def _decision(agent, value, metadata=None):
    return AgentDecision(
        agent_name=agent,
        decision_type="test",
        decision_value=value,
        confidence=float(value.get("score", 1.0)),
        reasoning="test",
        metadata=metadata or {},
    )


class FakeRetrievalAgent:
    def __init__(self, batches):
        self.batches = batches
        self.index = 0
        self.queries = []

    def reset(self):
        self.index = 0
        self.queries = []

    def decide(self, context):
        return _decision(
            "RetrievalAgent",
            {"top_k": 10 * (context["attempt"] + 1)},
        )

    def retrieve(self, question, decision):
        self.queries.append(question)
        batch = self.batches[min(self.index, len(self.batches) - 1)]
        self.index += 1
        return batch

    def escalate_strategy(self):
        pass


class FakeReasoningAgent:
    def __init__(self, answers):
        self.answers = answers
        self.index = 0
        self.contexts = []

    def reset(self):
        self.index = 0
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context.copy())
        answer = self.answers[min(self.index, len(self.answers) - 1)]
        self.index += 1
        return _decision("ReasoningAgent", {"answer": answer})

    def escalate_strategy(self):
        pass


class FailingReasoningAgent(FakeReasoningAgent):
    def __init__(self, message="provider timeout"):
        super().__init__([None])
        self.message = message

    def decide(self, context):
        self.contexts.append(context.copy())
        return _decision(
            "ReasoningAgent",
            {
                "answer": None,
                "error": self.message,
                "finance_program": None,
                "finance_program_issues": [],
            },
        )


class FakeJudgeAgent:
    def __init__(self, decisions):
        self.test_decisions = decisions
        self.index = 0

    def reset(self):
        self.index = 0

    def decide(self, context):
        value, metadata = self.test_decisions[
            min(self.index, len(self.test_decisions) - 1)
        ]
        self.index += 1
        return _decision("JudgeAgent", value, metadata)


class FailingJudgeAgent(FakeJudgeAgent):
    def __init__(self, message="judge timeout"):
        super().__init__([])
        self.message = message

    def decide(self, context):
        raise RuntimeError(self.message)


def _orchestrator(retrieval, reasoning, judge):
    orchestrator = AgenticRAGOrchestrator(
        AgenticRAGConfig(
            max_retries=1,
            enable_logging=False,
            policy_mode="paper_fixed",
        ),
    )
    orchestrator.retrieval_agent = retrieval
    orchestrator.reasoning_agent = reasoning
    orchestrator.judge_agent = judge
    return orchestrator


def _v2_orchestrator(retrieval, reasoning, judge, max_retries=2):
    orchestrator = AgenticRAGOrchestrator(
        AgenticRAGConfig(
            max_retries=max_retries,
            enable_logging=False,
            policy_mode="gap_driven_v2",
        ),
    )
    orchestrator.retrieval_agent = retrieval
    orchestrator.reasoning_agent = reasoning
    orchestrator.judge_agent = judge
    return orchestrator


def _typed_ratio_fixture():
    quote = (
        "Acme FY2022 Revenue USD 20 million; "
        "Cost of revenue USD 10 million"
    )
    payload = {
        "schema_version": "1.0",
        "answer": {
            "value": "2",
            "unit": "ratio",
            "entity": "Acme",
            "period": "FY2022",
            "metric": "revenue_to_cost_ratio",
            "rounding": {"places": 2, "mode": "half_up"},
        },
        "operands": [
            {
                "id": "revenue",
                "value": "20",
                "currency": "USD",
                "scale": "million",
                "unit": "money",
                "entity": "Acme",
                "period": "FY2022",
                "metric": "Revenue",
                "evidence": {
                    "doc_id": "Doc1",
                    "quote": quote,
                    "value_text": "USD 20 million",
                    "metric_label": "Revenue",
                    "period_label": "FY2022",
                },
            },
            {
                "id": "cost",
                "value": "10",
                "currency": "USD",
                "scale": "million",
                "unit": "money",
                "entity": "Acme",
                "period": "FY2022",
                "metric": "Cost of revenue",
                "evidence": {
                    "doc_id": "Doc1",
                    "quote": quote,
                    "value_text": "USD 10 million",
                    "metric_label": "Cost of revenue",
                    "period_label": "FY2022",
                },
            },
        ],
        "expression": {
            "op": "div",
            "args": [
                {"op": "ref", "operand_id": "revenue"},
                {"op": "ref", "operand_id": "cost"},
            ],
        },
    }
    program = FinanceProgram.model_validate(payload)
    spec = FinanceQuestionSpec(
        entity="Acme",
        period="FY2022",
        metric="revenue_to_cost_ratio",
        unit="ratio",
        rounding={"places": 2},
        expression=program.expression,
        operands=tuple(
            FinanceOperandSpec(
                id=operand.id,
                entity=operand.entity,
                period=operand.period,
                metric=operand.metric,
                unit=operand.unit,
                currency=operand.currency,
                scale=operand.scale,
            )
            for operand in program.operands
        ),
    )
    doc = Document(
        page_content=quote,
        metadata={
            "company": "Acme",
            "source": "Acme_2022_10K.pdf",
            "currency": "USD",
        },
    )
    return payload, spec.model_dump(mode="json"), doc


def test_retry_feedback_and_previous_answer_reach_reasoner():
    docs = [Document(page_content="Revenue was $10 million")]
    retrieval = FakeRetrievalAgent([docs, docs])
    reasoning = FakeReasoningAgent(["$10 billion", "$10 million"])
    judge = FakeJudgeAgent(
        [
            (
                {
                    "score": 0.0,
                    "retry": True,
                    "verification_feedback": "Unit mismatch: use millions.",
                },
                {"deterministic_gate_triggered": True},
            ),
            ({"score": 0.9, "retry": False}, {}),
        ]
    )

    result = _orchestrator(retrieval, reasoning, judge).process_question("Revenue?")

    assert result.final_answer == "$10 million"
    assert reasoning.contexts[1]["previous_answer"] == "$10 billion"
    assert "Unit mismatch" in reasoning.contexts[1]["retry_feedback"]


def test_paper_fixed_retry_preserves_the_original_retrieval_query():
    docs = [Document(page_content="Revenue was $10 million")]
    retrieval = FakeRetrievalAgent([docs, docs])
    orchestrator = _orchestrator(
        retrieval,
        FakeReasoningAgent(["draft", "revised"]),
        FakeJudgeAgent(
            [
                (
                    {
                        "score": 0.0,
                        "retry": True,
                        "pass": False,
                        "verification_passed": False,
                        "verification_reason_codes": ["missing_evidence"],
                    },
                    {},
                ),
                (
                    {
                        "score": 0.9,
                        "retry": False,
                        "pass": True,
                        "verification_passed": True,
                    },
                    {},
                ),
            ]
        ),
    )

    question = "What was Apple's FY2022 revenue?"
    orchestrator.process_question(question)

    assert retrieval.queries == [question, question]


def test_v2_invalid_citation_reuses_evidence_without_retrieval_call():
    docs = [Document(page_content="Revenue was $10 million")]
    retrieval = FakeRetrievalAgent([docs])
    reasoning = FakeReasoningAgent(["bad citation", "fixed citation"])
    judge = FakeJudgeAgent(
        [
            (
                {
                    "score": 0.0,
                    "retry": True,
                    "pass": False,
                    "verification_passed": False,
                    "verification_reason_codes": ["invalid_citation"],
                },
                {"deterministic_gate_triggered": True},
            ),
            (
                {
                    "score": 0.9,
                    "retry": False,
                    "pass": True,
                    "verification_passed": True,
                },
                {},
            ),
        ]
    )

    result = _v2_orchestrator(
        retrieval,
        reasoning,
        judge,
        max_retries=1,
    ).process_question("What was Apple's FY2022 revenue?")

    assert retrieval.queries == ["What was Apple's FY2022 revenue?"]
    assert len(reasoning.contexts) == 2
    assert result.correction_history[0]["action"] == "reuse_evidence_regenerate"
    assert result.decision_log[1]["retrieval"]["decision_type"] == "evidence_reuse"


def test_changed_content_with_same_chunk_id_is_new_evidence():
    old = Document(page_content="Revenue was $10 million", metadata={"chunk_id": "c1"})
    revised = Document(
        page_content="Revenue was $12 million",
        metadata={"chunk_id": "c1"},
    )

    assert _evidence_fingerprint([old]) != _evidence_fingerprint([revised])
    assert _merge_evidence([old], [revised]) == [old, revised]


def test_v2_missing_metric_runs_targeted_retrieval_and_preserves_evidence():
    first = Document(
        page_content="Other evidence",
        metadata={"chunk_id": "first"},
    )
    second = Document(
        page_content="Revenue was $10 million",
        metadata={"chunk_id": "second"},
    )
    retrieval = FakeRetrievalAgent([[first], [second]])
    reasoning = FakeReasoningAgent(["missing", "complete"])
    judge = FakeJudgeAgent(
        [
            (
                {
                    "score": 0.0,
                    "retry": True,
                    "pass": False,
                    "verification_passed": False,
                    "verification_reason_codes": ["missing_evidence"],
                },
                {"deterministic_gate_triggered": True},
            ),
            (
                {
                    "score": 0.9,
                    "retry": False,
                    "pass": True,
                    "verification_passed": True,
                },
                {},
            ),
        ]
    )

    result = _v2_orchestrator(
        retrieval,
        reasoning,
        judge,
        max_retries=1,
    ).process_question("What was Apple's revenue in FY2022?")

    assert len(retrieval.queries) == 2
    assert "Target missing evidence" in retrieval.queries[1]
    assert "revenue" in retrieval.queries[1].lower()
    assert len(result.decision_log[1]["evidence_manifest"]) == 2
    assert result.correction_history[0]["action"] == "targeted_retrieval"


def test_v2_structured_operand_issue_targets_only_the_affected_need():
    first = Document(page_content="Incomplete filing", metadata={"chunk_id": "a"})
    second = Document(page_content="Current assets were 10", metadata={"chunk_id": "b"})
    retrieval = FakeRetrievalAgent([[first], [second]])
    judge = FakeJudgeAgent(
        [
            (
                {
                    "score": 0.0,
                    "retry": True,
                    "pass": False,
                    "verification_passed": False,
                    "verification_issues": [
                        {
                            "code": "missing_operand",
                            "operand_id": "need:current-assets:apple:fy2022",
                            "metric": "current_assets",
                            "period": "FY2022",
                            "message": "Current-assets operand is missing",
                        }
                    ],
                },
                {"deterministic_gate_triggered": True},
            ),
            (
                {
                    "score": 1.0,
                    "retry": False,
                    "pass": True,
                    "verification_passed": True,
                },
                {},
            ),
        ]
    )
    result = _v2_orchestrator(
        retrieval,
        FakeReasoningAgent(["draft", "verified"]),
        judge,
        max_retries=1,
    ).process_question("What was Apple's working capital ratio in FY2022?")

    assert len(retrieval.queries) == 2
    assert "current assets" in retrieval.queries[1].lower()
    assert "current liabilities" not in retrieval.queries[1].lower()
    assert result.correction_history[0]["affected_need_ids"] == [
        "need:current-assets:apple:fy2022"
    ]


def test_v2_applies_typed_local_repair_without_another_model_call():
    docs = [Document(page_content="Apple FY2022 balance sheet")]
    retrieval = FakeRetrievalAgent([docs])
    reasoning = FakeReasoningAgent(["9.99x"])
    judge = FakeJudgeAgent(
        [
            (
                {
                    "score": 1.0,
                    "retry": True,
                    "pass": False,
                    "verification_passed": False,
                    "verification_issues": [
                        {
                            "code": "result_value_mismatch",
                            "message": "Declared result differs from execution",
                        }
                    ],
                    "canonical_answer": "1.25x",
                    "finance_program_verification": {
                        "passed": False,
                        "evidence_coverage": 1.0,
                        "rendered_answer": "1.25x",
                    },
                    "repaired_finance_program": {"schema_version": "1.0"},
                    "repaired_finance_program_verification": {
                        "passed": True,
                        "fully_verified": True,
                        "evidence_coverage": 1.0,
                        "rendered_answer": "1.25x",
                    },
                },
                {"deterministic_gate_triggered": True},
            )
        ]
    )
    orchestrator = _v2_orchestrator(retrieval, reasoning, judge, max_retries=0)

    result = orchestrator.process_question(
        "What was Apple's working capital ratio in FY2022?"
    )

    assert result.final_answer == "1.25x"
    assert result.policy_accepted is True
    assert result.abstained is False
    assert len(reasoning.contexts) == 1
    assert retrieval.queries == ["What was Apple's working capital ratio in FY2022?"]
    assert orchestrator.total_retries == 0
    assert result.correction_history[0]["action"] == "local_recompute"
    assert result.correction_history[0]["local_repair_applied"] is True
    assert result.finance_verification["locally_repaired"] is True
    assert result.finance_verification["repaired_and_reverified"] is True


def test_v2_unresolved_relative_period_abstains_before_retrieval():
    retrieval = FakeRetrievalAgent([[Document(page_content="should not be read")]])
    reasoning = FakeReasoningAgent(["should not be generated"])
    orchestrator = _v2_orchestrator(
        retrieval,
        reasoning,
        FakeJudgeAgent([]),
        max_retries=2,
    )

    result = orchestrator.process_question(
        "What was Apple's working capital ratio last year?"
    )

    assert result.final_answer is None
    assert result.abstained is True
    assert retrieval.queries == []
    assert reasoning.contexts == []
    assert result.decision_log[0]["state"] == "preflight"
    assert result.correction_history[0]["action"] == "abstain"


def test_v2_unknown_derived_formula_abstains_before_retrieval():
    retrieval = FakeRetrievalAgent([[Document(page_content="should not be read")]])
    reasoning = FakeReasoningAgent(["should not be generated"])
    orchestrator = _v2_orchestrator(
        retrieval,
        reasoning,
        FakeJudgeAgent([]),
        max_retries=2,
    )

    result = orchestrator.process_question(
        "What was Apple's two-year revenue CAGR from FY2021 to FY2023?"
    )

    assert result.final_answer is None
    assert result.abstained is True
    assert retrieval.queries == []
    assert reasoning.contexts == []
    assert result.decision_log[0]["state"] == "preflight"
    assert any(
        issue["message"] == "formula:unresolved"
        for issue in result.decision_log[0]["issues"]
    )


def test_typed_generation_provider_failure_is_an_error_not_abstention():
    docs = [Document(page_content="Apple FY2022 balance sheet")]
    reasoning = FailingReasoningAgent("upstream timeout")
    judge = FakeJudgeAgent([])
    orchestrator = _v2_orchestrator(
        FakeRetrievalAgent([docs]),
        reasoning,
        judge,
        max_retries=2,
    )

    result = orchestrator.process_question(
        "What was Apple's working capital ratio in FY2022?"
    )

    assert result.final_answer is None
    assert result.error == "Reasoning provider failed: upstream timeout"
    assert result.abstained is False
    assert result.policy_accepted is False
    assert result.decision_log[0]["state"] == "error"
    assert judge.index == 0


def test_policy_judge_failure_is_an_error_not_a_low_score_or_abstention():
    docs = [Document(page_content="Apple revenue was stable")]
    orchestrator = _v2_orchestrator(
        FakeRetrievalAgent([docs]),
        FakeReasoningAgent(
            ["Revenue was stable [Doc1: 'Apple revenue was stable']"]
        ),
        FailingJudgeAgent("judge unavailable"),
        max_retries=2,
    )

    result = orchestrator.process_question("How did Apple's revenue change?")

    assert result.final_answer is None
    assert result.error == "Error in attempt 0: judge unavailable"
    assert result.abstained is False
    assert result.policy_accepted is False
    assert result.attempts == 1


def test_v2_repeated_gap_on_unchanged_evidence_abstains_early():
    docs = [
        Document(
            page_content="Revenue was $10 million",
            metadata={"chunk_id": "same-evidence"},
        )
    ]
    failed = (
        {
            "score": 0.0,
            "retry": True,
            "pass": False,
            "verification_passed": False,
            "verification_reason_codes": ["invalid_citation"],
        },
        {"deterministic_gate_triggered": True},
    )
    retrieval = FakeRetrievalAgent([docs])
    orchestrator = _v2_orchestrator(
        retrieval,
        FakeReasoningAgent(["bad one", "bad two", "bad three"]),
        FakeJudgeAgent([failed, failed, failed]),
        max_retries=2,
    )

    result = orchestrator.process_question("What was Apple's FY2022 revenue?")

    assert retrieval.queries == ["What was Apple's FY2022 revenue?"]
    assert result.attempts == 2
    assert result.abstained is True
    assert result.final_answer is None
    assert result.correction_history[-1]["action"] == "abstain"


def test_later_retrieval_error_does_not_return_an_unaccepted_draft():
    docs = [Document(page_content="Some evidence")]
    retrieval = FakeRetrievalAgent([docs, []])
    reasoning = FakeReasoningAgent(["best available answer"])
    judge = FakeJudgeAgent(
        [({"score": 0.0, "retry": True, "justification": "weak"}, {})]
    )

    result = _orchestrator(retrieval, reasoning, judge).process_question("Question?")

    assert result.final_answer is None
    assert result.final_score == 0.0
    assert result.error == "No documents retrieved"
    assert result.abstained is False
    assert result.policy_accepted is False


def test_blind_policy_acceptance_is_not_labeled_as_accuracy():
    docs = [Document(page_content="Evidence")]
    orchestrator = _orchestrator(
        FakeRetrievalAgent([docs]),
        FakeReasoningAgent(["Answer"]),
        FakeJudgeAgent([({"score": 0.9, "retry": False}, {})]),
    )
    orchestrator.config.blind_judge = True

    result = orchestrator.process_question("Question?", gold_answer="Gold")

    assert result.policy_accepted is True
    assert result.correct is None
    assert result.evaluation_mode == "blind_policy"


def test_stopping_returns_best_scored_candidate_and_relaxed_acceptance():
    first_docs = [
        Document(
            page_content="First evidence",
            metadata={
                "source": "first.pdf",
                "page": 2,
                "chunk_id": "first:chunk:00002",
            },
        )
    ]
    second_docs = [
        Document(
            page_content="Second evidence",
            metadata={
                "source": "second.pdf",
                "positions": [10, 11],
                "child_chunk_ids": ["second:c10", "second:c11"],
            },
        )
    ]
    orchestrator = _orchestrator(
        FakeRetrievalAgent([first_docs, second_docs]),
        FakeReasoningAgent(["better answer", "worse answer"]),
        FakeJudgeAgent(
            [
                ({"score": 0.49, "retry": True, "pass": False}, {}),
                (
                    {
                        "score": 0.45,
                        "retry": False,
                        "pass": True,
                        "acceptance_threshold": 0.4,
                    },
                    {},
                ),
            ]
        ),
    )

    result = orchestrator.process_question("Question?")

    assert result.final_answer == "better answer"
    assert result.final_score == 0.49
    assert result.policy_accepted is True
    assert result.correct is None
    assert result.evaluation_mode == "blind_policy"
    assert result.evidence_manifest == {
        "Doc1": {
            "source": "first.pdf",
            "content_sha256": hashlib.sha256(
                b"First evidence"
            ).hexdigest(),
            "page": 2,
            "chunk_id": "first:chunk:00002",
        }
    }
    assert result.decision_log[0]["evidence_manifest"] == result.evidence_manifest
    assert result.decision_log[1]["evidence_manifest"] == {
        "Doc1": {
            "source": "second.pdf",
            "content_sha256": hashlib.sha256(
                b"Second evidence"
            ).hexdigest(),
            "positions": [10, 11],
            "child_chunk_ids": ["second:c10", "second:c11"],
        }
    }


def test_oracle_correctness_uses_fixed_base_threshold_not_relaxed_policy():
    docs = [Document(page_content="Evidence")]
    orchestrator = _orchestrator(
        FakeRetrievalAgent([docs, docs]),
        FakeReasoningAgent(["first", "second"]),
        FakeJudgeAgent(
            [
                ({"score": 0.2, "retry": True, "pass": False}, {}),
                (
                    {
                        "score": 0.4,
                        "retry": False,
                        "pass": True,
                        "acceptance_threshold": 0.4,
                    },
                    {},
                ),
            ]
        ),
    )

    result = orchestrator.process_question("Question?", gold_answer="Gold")

    assert result.policy_accepted is True
    assert result.final_score == 0.4
    assert result.correct is False
    assert result.evaluation_mode == "oracle_guided"


def test_evidence_manifest_persists_in_logger_json_and_csv(tmp_path):
    docs = [
        Document(
            page_content="Evidence",
            metadata={
                "source": "report.pdf",
                "positions": [4, 5],
                "child_chunk_ids": ["report:c4", "report:c5"],
                "corpus_version": "financebench-open-v1",
                "index_version": "docling-v2",
                "reranker_score": 0.87,
            },
        )
    ]
    orchestrator = AgenticRAGOrchestrator(
        AgenticRAGConfig(
            max_retries=0,
            enable_logging=True,
            log_dir=str(tmp_path),
        )
    )
    orchestrator.retrieval_agent = FakeRetrievalAgent([docs])
    orchestrator.reasoning_agent = FakeReasoningAgent(["grounded answer"])
    orchestrator.judge_agent = FakeJudgeAgent(
        [
            (
                {
                    "score": 0.9,
                    "retry": False,
                    "pass": True,
                    "verification_passed": True,
                },
                {},
            )
        ]
    )

    result = orchestrator.process_question("Question?", question_id="q1")
    paths = orchestrator.export_decisions("manifest")

    expected = {
        "Doc1": {
            "source": "report.pdf",
            "content_sha256": hashlib.sha256(b"Evidence").hexdigest(),
            "positions": [4, 5],
            "child_chunk_ids": ["report:c4", "report:c5"],
            "corpus_version": "financebench-open-v1",
            "index_version": "docling-v2",
            "reranker_score": 0.87,
        }
    }
    assert result.evidence_manifest == expected

    exported = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert exported[0]["evidence_manifest"] == expected
    assert exported[0]["attempts"][0]["evidence_manifest"] == expected
    assert exported[0]["query_plan"]["task_type"] == "qualitative"
    assert exported[0]["correction_history"][0]["action"] == "accept"

    with Path(paths["csv"]).open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["evidence_manifest"]) == expected
    assert json.loads(row["final_evidence_manifest"]) == expected
    assert json.loads(row["query_plan"])["task_type"] == "qualitative"
    assert row["correction_action"] == "accept"


def test_final_grounding_abstention_does_not_return_unsupported_draft():
    docs = [Document(page_content="Evidence")]
    orchestrator = _orchestrator(
        FakeRetrievalAgent([docs]),
        FakeReasoningAgent(["unsupported draft"]),
        FakeJudgeAgent(
            [
                (
                    {
                        "score": 0.0,
                        "retry": False,
                        "pass": False,
                        "abstain": True,
                        "verification_passed": False,
                    },
                    {"deterministic_gate_triggered": True},
                )
            ]
        ),
    )

    result = orchestrator.process_question("Question?")

    assert result.final_answer is None
    assert result.abstained is True
    assert result.policy_accepted is False


def test_v2_budget_exhaustion_never_leaks_a_rejected_draft():
    docs = [Document(page_content="Apple FY2022 Revenue was $10 million")]
    orchestrator = _v2_orchestrator(
        FakeRetrievalAgent([docs]),
        FakeReasoningAgent(["unsupported draft"]),
        FakeJudgeAgent(
            [
                (
                    {
                        "score": 0.49,
                        "retry": False,
                        "pass": False,
                        "abstain": False,
                        "verification_passed": True,
                    },
                    {},
                )
            ]
        ),
        max_retries=0,
    )

    result = orchestrator.process_question(
        "What was Apple's revenue in FY2022?"
    )

    assert result.final_answer is None
    assert result.abstained is True
    assert result.policy_accepted is False


def test_final_abstention_falls_back_to_earlier_grounded_candidate():
    docs = [Document(page_content="Evidence")]
    orchestrator = _orchestrator(
        FakeRetrievalAgent([docs, docs]),
        FakeReasoningAgent(["grounded draft", "unsupported draft"]),
        FakeJudgeAgent(
            [
                (
                    {
                        "score": 0.3,
                        "retry": True,
                        "pass": False,
                        "verification_passed": True,
                    },
                    {},
                ),
                (
                    {
                        "score": 0.0,
                        "retry": False,
                        "pass": False,
                        "abstain": True,
                        "verification_passed": False,
                    },
                    {"deterministic_gate_triggered": True},
                ),
            ]
        ),
    )

    result = orchestrator.process_question("Question?")

    assert result.final_answer == "grounded draft"
    assert result.abstained is False
    assert result.policy_accepted is False


def test_equal_score_invalid_draft_cannot_displace_verified_candidate():
    docs = [Document(page_content="Evidence")]
    orchestrator = _orchestrator(
        FakeRetrievalAgent([docs, docs]),
        FakeReasoningAgent(["invalid draft", "verified draft"]),
        FakeJudgeAgent(
            [
                (
                    {
                        "score": 0.0,
                        "retry": True,
                        "pass": False,
                        "verification_passed": False,
                    },
                    {"deterministic_gate_triggered": True},
                ),
                (
                    {
                        "score": 0.0,
                        "retry": False,
                        "pass": False,
                        "verification_passed": True,
                        "acceptance_threshold": 0.4,
                    },
                    {},
                ),
            ]
        ),
    )

    result = orchestrator.process_question("Question?")

    assert result.final_answer == "verified draft"
    assert result.policy_accepted is False


def test_retry_prompt_does_not_carry_stale_positional_citations():
    docs = [Document(page_content="Revenue was $10 million")]
    reasoning = FakeReasoningAgent(
        [
            "$10 billion [Doc1: 'Revenue was $10 million']",
            "$10 million",
        ]
    )
    orchestrator = _orchestrator(
        FakeRetrievalAgent([docs, docs]),
        reasoning,
        FakeJudgeAgent(
            [
                ({"score": 0.0, "retry": True}, {}),
                ({"score": 0.9, "retry": False, "pass": True}, {}),
            ]
        ),
    )

    orchestrator.process_question("Revenue?")

    assert reasoning.contexts[1]["previous_answer"] == "$10 billion"


def test_blind_aggregate_reports_acceptance_not_accuracy():
    import pandas as pd

    from evaluation.metrics import calculate_aggregate_metrics, format_metrics_summary

    frame = pd.DataFrame(
        {
            "judge_score": [0.9, 0.2],
            "policy_accepted": [True, False],
            "evaluation_mode": ["blind_policy", "blind_policy"],
            "correct": [None, None],
            "sources": [["a"], ["b"]],
            "semantic_similarity": [0.0, 0.0],
        }
    )

    metrics = calculate_aggregate_metrics(frame)
    summary = format_metrics_summary(metrics)

    assert "accuracy" not in metrics["judge_score"]
    assert metrics["policy_acceptance"]["rate"] == 0.5
    assert "Full-credit Rate" in summary
    assert "Policy Acceptance" in summary


def test_agentic_evidence_manifest_is_not_classified_as_empty_retrieval():
    import pandas as pd

    from evaluation.metrics import calculate_failure_breakdown

    frame = pd.DataFrame(
        [
            {
                "evidence_manifest": {
                    "Doc1": {"source": "report.pdf", "chunk_id": "chunk-1"}
                },
                "semantic_similarity": 0.9,
            },
            {
                "evidence_manifest": json.dumps(
                    {"Doc1": {"source": "report.pdf", "chunk_id": "chunk-1"}}
                ),
                "semantic_similarity": 0.9,
            }
        ]
    )

    breakdown = calculate_failure_breakdown(frame)

    assert breakdown["counts"] == {"ok": 2}


def test_retrieved_document_count_is_not_classified_as_empty_retrieval():
    from evaluation.metrics import categorize_failure

    assert categorize_failure(
        {
            "retrieved_document_count": 2,
            "semantic_similarity": 0.2,
        }
    ) == "generation_poor"
    assert categorize_failure(
        {
            "retrieved_document_count": 0,
            "evidence_manifest": "{}",
            "semantic_similarity": 0.9,
        }
    ) == "retrieval_empty"


def test_csv_boolean_strings_are_not_all_counted_as_true():
    from io import StringIO

    import pandas as pd

    from evaluation.metrics import calculate_aggregate_metrics

    frame = pd.read_csv(
        StringIO(
            "policy_accepted,correct,evaluation_mode\n"
            "False,False,oracle_guided\n"
            "True,True,oracle_guided\n"
        ),
        dtype={"policy_accepted": "string", "correct": "string"},
    )

    metrics = calculate_aggregate_metrics(frame)

    assert metrics["policy_acceptance"]["rate"] == 0.5
    assert metrics["labeled_correctness"]["rate"] == 0.5


def test_selective_metrics_separate_abstentions_from_system_errors():
    import pandas as pd

    from evaluation.metrics import calculate_aggregate_metrics

    frame = pd.DataFrame(
        {
            "correct": [True, False, False, False],
            "evaluation_mode": [
                "post_selection_llm_judge",
                "post_selection_llm_judge",
                "post_selection_abstention",
                "terminal_error",
            ],
            "abstained": [False, False, True, False],
            "error": [None, None, None, "provider timeout"],
        }
    )

    metrics = calculate_aggregate_metrics(frame)
    selective = metrics["selective_prediction"]

    assert selective["coverage"] == 0.5
    assert selective["noncoverage_rate"] == 0.5
    assert selective["abstention_rate"] == 0.25
    assert selective["error_rate"] == 0.25
    assert selective["selective_accuracy"] == 0.5
    assert selective["selective_risk"] == 0.5
    assert selective["covered_question_count"] == 2
    assert selective["abstention_count"] == 1
    assert selective["error_count"] == 1
    assert selective["eligible_question_count"] == 4
    assert selective["total_question_count"] == 4
    assert metrics["error_rate"] == 0.25


def test_agentic_routed_pipeline_is_rejected_with_clear_error():
    with pytest.raises(ValueError, match="fixed pipeline"):
        AgenticRAGConfig(pipeline_id="routed")


def test_agentic_config_rejects_invalid_budgets_and_thresholds():
    with pytest.raises(ValueError, match="max_retries"):
        AgenticRAGConfig(max_retries=-1)
    with pytest.raises(ValueError, match="retry_threshold"):
        AgenticRAGConfig(retry_threshold=1.1)

    with pytest.raises(ValueError, match="max_tokens >= 2048"):
        AgenticRAGConfig(max_tokens=1024)

    with pytest.raises(ValueError, match="paper_fixed-only"):
        AgenticRAGConfig(ablation_no_retrieval_escalation=True)

    historical = AgenticRAGConfig(
        policy_mode="paper_fixed",
        max_tokens=512,
        ablation_no_retrieval_escalation=True,
    )
    assert historical.max_tokens == 512
    with pytest.raises(ValueError, match="top_k"):
        AgenticRAGConfig(top_k=0)
    with pytest.raises(ValueError, match="initial_k_factor"):
        AgenticRAGConfig(initial_k_factor=0.5)


def test_orchestrator_factory_revalidates_overrides_and_rejects_unknown_keys():
    config = AgenticRAGConfig(enable_logging=False)
    with pytest.raises(ValueError, match="policy_mode"):
        build_agentic_orchestrator(
            None,
            None,
            config=config,
            policy_mode="invalid",
        )
    with pytest.raises(TypeError, match="Unknown AgenticRAGConfig"):
        build_agentic_orchestrator(
            None,
            None,
            config=config,
            typo_option=True,
        )


def test_v2_policy_keeps_acceptance_threshold_constant():
    orchestrator = AgenticRAGOrchestrator(
        AgenticRAGConfig(policy_mode="gap_driven_v2", enable_logging=False)
    )
    legacy = AgenticRAGOrchestrator(
        AgenticRAGConfig(policy_mode="paper_fixed", enable_logging=False)
    )

    assert orchestrator.judge_agent.acceptance_threshold(1) == 0.5
    assert legacy.judge_agent.acceptance_threshold(1) == 0.4


def test_reasoning_prompt_contains_gold_free_retry_diagnostics():
    class RecordingProvider:
        def __init__(self):
            self.user_prompt = ""

        def generate(self, **kwargs):
            self.user_prompt = kwargs["user_prompt"]
            return LLMResponse(content="$10 million", model="fake", provider="fake")

    provider = RecordingProvider()
    agent = ReasoningAgent()
    agent._provider = provider
    decision = agent.decide(
        {
            "question": "What was revenue?",
            "documents": [Document(page_content="Revenue was $10 million")],
            "attempt": 1,
            "previous_answer": "$10 billion",
            "retry_feedback": "The cited value uses millions, not billions.",
        }
    )

    assert decision.decision_value["feedback_used"] is True
    assert "$10 billion" in provider.user_prompt
    assert "millions, not billions" in provider.user_prompt


def test_reasoning_agent_parses_typed_program_and_hides_machine_block():
    payload, spec, doc = _typed_ratio_fixture()

    class RecordingProvider:
        def __init__(self):
            self.user_prompt = ""

        def generate(self, **kwargs):
            self.user_prompt = kwargs["user_prompt"]
            return LLMResponse(
                content=(
                    "2.00x\n<finance_program>"
                    + json.dumps(payload)
                    + "</finance_program>"
                ),
                model="fake",
                provider="fake",
            )

    provider = RecordingProvider()
    agent = ReasoningAgent()
    agent._provider = provider

    decision = agent.decide(
        {
            "question": "What was Acme revenue-to-cost ratio in FY2022?",
            "documents": [doc],
            "attempt": 0,
            "require_finance_program": True,
            "finance_question_spec": spec,
        }
    )

    assert decision.decision_value["answer"] == "2.00x"
    assert decision.decision_value["finance_program"] == (
        FinanceProgram.model_validate(payload).model_dump(mode="json")
    )
    assert decision.decision_value["finance_program_issues"] == []
    assert "TRUSTED FINANCE QUESTION CONTRACT" in provider.user_prompt


def test_judge_accepts_derived_typed_result_without_gold_or_verbatim_answer():
    payload, spec, doc = _typed_ratio_fixture()
    assert "2.00x" not in doc.page_content

    decision = JudgeAgent().decide(
        {
            "question": "What was Acme revenue-to-cost ratio in FY2022?",
            "predicted_answer": "2.00x",
            "gold_answer": "intentionally wrong oracle",
            "attempt": 0,
            "max_retries": 0,
            "documents": [doc],
            "require_finance_program": True,
            "finance_program": payload,
            "finance_question_spec": spec,
        }
    )

    assert decision.decision_value["pass"] is True
    assert decision.decision_value["canonical_answer"] == "2.00x"
    assert decision.decision_value["finance_program_verification"][
        "fully_verified"
    ] is True
    assert decision.metadata["accepted_by"] == "typed_finance_full_contract"
    assert decision.metadata["has_gold_answer"] is False


def test_v2_end_to_end_typed_calculation_uses_real_reasoner_and_judge():
    question = "What was Apple's current ratio in FY2022?"
    spec = build_finance_question_spec(compile_finance_query(question))
    assert spec is not None
    asset_line = "Apple Inc. FY2022 Total current assets USD 135,405 million"
    liability_line = (
        "Apple Inc. FY2022 Total current liabilities USD 153,982 million"
    )
    docs = [
        Document(
            page_content=asset_line,
            metadata={
                "company": "Apple",
                "source": "Apple_2022_10K.pdf",
                "currency": "USD",
            },
        ),
        Document(
            page_content=liability_line,
            metadata={
                "company": "Apple",
                "source": "Apple_2022_10K.pdf",
                "currency": "USD",
            },
        ),
    ]
    program = {
        "schema_version": "1.0",
        "answer": {
            "value": "0.88",
            "unit": "ratio",
            "entity": "APPLE",
            "period": "FY2022",
            "metric": "working_capital_ratio",
            "rounding": {"places": 2, "mode": "half_up"},
        },
        "operands": [
            {
                "id": "need:current-assets:apple:fy2022",
                "value": "135405",
                "currency": "USD",
                "scale": "million",
                "unit": "money",
                "entity": "APPLE",
                "period": "FY2022",
                "metric": "current_assets",
                "evidence": {
                    "doc_id": "Doc1",
                    "quote": asset_line,
                    "value_text": "USD 135,405 million",
                    "metric_label": "Total current assets",
                    "period_label": "FY2022",
                },
            },
            {
                "id": "need:current-liabilities:apple:fy2022",
                "value": "153982",
                "currency": "USD",
                "scale": "million",
                "unit": "money",
                "entity": "APPLE",
                "period": "FY2022",
                "metric": "current_liabilities",
                "evidence": {
                    "doc_id": "Doc2",
                    "quote": liability_line,
                    "value_text": "USD 153,982 million",
                    "metric_label": "Total current liabilities",
                    "period_label": "FY2022",
                },
            },
        ],
        "expression": spec.expression.model_dump(mode="json"),
    }

    class TypedProvider:
        def generate(self, **kwargs):
            return LLMResponse(
                content=(
                    "0.88x\n<finance_program>"
                    + json.dumps(program)
                    + "</finance_program>"
                ),
                model="fake",
                provider="fake",
            )

    reasoning = ReasoningAgent()
    reasoning._provider = TypedProvider()
    orchestrator = AgenticRAGOrchestrator(
        AgenticRAGConfig(max_retries=0, enable_logging=False)
    )
    orchestrator.retrieval_agent = FakeRetrievalAgent([docs])
    orchestrator.reasoning_agent = reasoning

    result = orchestrator.process_question(question, gold_answer="wrong oracle")

    assert result.final_answer == "0.88x"
    assert result.policy_accepted is True
    assert result.correct is None
    assert result.evaluation_mode == "blind_policy"
    assert result.finance_verification["fully_verified"] is True
    assert result.decision_log[0]["judge"]["metadata"]["has_gold_answer"] is False


def test_no_prompt_escalation_ablation_forces_standard_strategy():
    class RecordingProvider:
        def generate(self, **kwargs):
            return LLMResponse(content="Answer", model="fake", provider="fake")

    agent = ReasoningAgent(disable_escalation=True)
    agent._provider = RecordingProvider()
    decision = agent.decide(
        {
            "question": "What was revenue?",
            "documents": [Document(page_content="Revenue was $10 million")],
            "attempt": 2,
            "correction_plan": {"action": "reconcile"},
        }
    )

    assert decision.decision_value["strategy"] == "standard"


def test_reasoning_prompt_allows_precise_abstention_instead_of_forced_answer():
    class RecordingProvider:
        def __init__(self):
            self.system_prompt = ""
            self.user_prompt = ""

        def generate(self, **kwargs):
            self.system_prompt = kwargs["system_prompt"]
            self.user_prompt = kwargs["user_prompt"]
            return LLMResponse(
                content="The required period is missing.",
                model="fake",
                provider="fake",
            )

    provider = RecordingProvider()
    agent = ReasoningAgent()
    agent._provider = provider

    agent.decide(
        {
            "question": "What was revenue in FY2024?",
            "documents": [Document(page_content="Revenue for FY2023 was $10 million")],
            "attempt": 0,
        }
    )

    prompt = f"{provider.system_prompt}\n{provider.user_prompt}".lower()
    assert "always provide an answer" not in prompt
    assert "never refuse" not in prompt
    assert "state exactly what is missing" in prompt


def test_typed_reasoning_prompt_has_one_nonconflicting_provenance_format():
    class RecordingProvider:
        def __init__(self):
            self.system_prompt = ""
            self.user_prompt = ""

        def generate(self, **kwargs):
            self.system_prompt = kwargs["system_prompt"]
            self.user_prompt = kwargs["user_prompt"]
            return LLMResponse(content="0.88x", model="fake", provider="fake")

    provider = RecordingProvider()
    agent = ReasoningAgent()
    agent._provider = provider
    decision = agent.decide(
        {
            "question": "What was Apple's current ratio in FY2023?",
            "documents": [Document(page_content="Apple FY2023 balance sheet")],
            "attempt": 0,
            "require_finance_program": True,
            "finance_question_spec": {"schema_version": "1.0"},
        }
    )

    assert "For EVERY numerical claim" not in provider.system_prompt
    assert "CALCULATION-SPECIFIC OUTPUT RULE" in provider.system_prompt
    assert "PLAN MODE REQUIRED" not in provider.user_prompt
    assert "provenance exclusively" in provider.system_prompt
    assert decision.decision_value["finance_program"] is None


def test_reasoning_support_score_uses_exact_citations_not_assertiveness():
    docs = [Document(page_content="Revenue was $10 million")]
    agent = ReasoningAgent()

    grounded = agent.estimate_confidence(
        "Uncertain, but revenue was $10 million "
        "[Doc1: 'Revenue was $10 million']",
        docs,
    )
    unsupported = agent.estimate_confidence(
        "Revenue definitely was $99 million.",
        docs,
    )

    assert grounded == 1.0
    assert unsupported == 0.0


def test_retrieval_escalates_exactly_to_10_20_30_with_final_rse():
    agent = RetrievalAgent(db=None)

    strategies = [agent.get_strategy(attempt) for attempt in range(3)]

    assert [strategy.top_k for strategy in strategies] == [10, 20, 30]
    assert [strategy.use_rse for strategy in strategies] == [False, False, True]


def test_retrieval_honors_configured_pipeline_and_initial_depth():
    agent = RetrievalAgent(
        db=None,
        pipeline_id="semantic",
        base_top_k=5,
        base_initial_k_factor=2.0,
    )

    strategies = [agent.get_strategy(attempt) for attempt in range(3)]

    assert [strategy.pipeline_id for strategy in strategies] == ["semantic"] * 3
    assert [strategy.top_k for strategy in strategies] == [5, 15, 25]
    assert [strategy.initial_k_factor for strategy in strategies] == [2.0, 3.0, 5.0]


def test_retrieval_applies_dynamic_pipeline_settings_and_rse():
    class FakePipeline:
        def __init__(self):
            self.top_k = 1
            self.initial_k_factor = 1.0
            self.segment_calls = 0

        def retrieve(self, question):
            return [Document(page_content="plain")]

        def retrieve_segments(self, question):
            self.segment_calls += 1
            return ["merged evidence"]

    pipeline = FakePipeline()
    agent = RetrievalAgent(db=None)
    agent.set_pipeline(pipeline)
    decision = _decision(
        "RetrievalAgent",
        {
            "pipeline_id": "hybrid_filter_rerank",
            "top_k": 30,
            "initial_k_factor": 6.0,
            "use_hyde": False,
            "use_rse": True,
            "use_rerank": True,
        },
    )

    docs = agent.retrieve("question", decision)

    assert pipeline.top_k == 30
    assert pipeline.initial_k_factor == 6.0
    assert pipeline.segment_calls == 1
    assert docs[0].page_content == "merged evidence"


def test_cached_filtered_retriever_receives_escalated_initial_k():
    class FakeRetriever:
        def __init__(self, k):
            self.k = k

    class FakeDB:
        def as_retriever(self, search_kwargs):
            return FakeRetriever(search_kwargs["k"])

    pipeline = SimplePipeline(
        retriever=FakeRetriever(1),
        top_k=10,
        use_metadata_filter=True,
        use_rerank=False,
        initial_k_factor=3.0,
        set_k_fn=lambda retriever, k: setattr(retriever, "k", k),
        take_top_k_fn=lambda docs, k: docs[:k],
        db=FakeDB(),
        use_hybrid=False,
    )
    metadata_filter = {"company": "3M", "year": 2022}

    first = pipeline._get_filtered_retriever(metadata_filter, 30)
    second = pipeline._get_filtered_retriever(metadata_filter, 180)

    assert first is second
    assert second.k == 180
