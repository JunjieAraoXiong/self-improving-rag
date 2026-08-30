"""Offline tests for the failure-conditioned v2 correction policy."""

import json
from enum import Enum

import pytest

from src.agents.correction_policy import (
    CorrectionAction,
    CorrectionPolicy,
    PolicyMode,
    plan_correction,
)
from src.query_understanding.finance_plan import compile_finance_query


def _report(*issues, passed=False):
    return {"passed": passed, "issues": list(issues)}


def test_passed_report_accepts_without_work():
    plan = plan_correction(_report(passed=True))

    assert plan.action is CorrectionAction.ACCEPT
    assert plan.terminal is True
    assert plan.should_retry is False
    assert plan.requires_retrieval is False


def test_invalid_citation_reuses_evidence_and_only_regenerates():
    plan = plan_correction(
        _report({"code": "invalid_citation", "doc_id": "chunk-7"}),
        evidence_fingerprint="evidence-A",
    )

    assert plan.action is CorrectionAction.REUSE_EVIDENCE_REGENERATE
    assert plan.reuse_evidence is True
    assert plan.requires_generation is True
    assert plan.requires_retrieval is False
    assert plan.requires_local_execution is False


def test_typed_enum_issue_code_routes_by_wire_value():
    class TypedIssueCode(str, Enum):
        INVALID_CITATION = "invalid_citation"

    class TypedIssue:
        code = TypedIssueCode.INVALID_CITATION
        message = "bad source span"

    class TypedVerificationResult:
        passed = False
        issues = (TypedIssue(),)

    plan = plan_correction(TypedVerificationResult())
    direct_code_plan = plan_correction(
        {"passed": False, "reason_codes": [TypedIssueCode.INVALID_CITATION]}
    )

    assert plan.issue_codes == ("invalid_citation",)
    assert plan.action is CorrectionAction.REUSE_EVIDENCE_REGENERATE
    assert direct_code_plan.action is CorrectionAction.REUSE_EVIDENCE_REGENERATE


def test_missing_operand_targets_only_matching_evidence_need():
    query_plan = compile_finance_query(
        "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? "
        "It is revenue divided by average PP&E between FY2018 and FY2019."
    )
    target = next(
        need
        for need in query_plan.evidence_needs
        if need.metric == "property_plant_equipment" and need.period == "FY2018"
    )

    plan = plan_correction(
        _report(
            {
                "code": "missing_operand",
                "operand_id": target.need_id,
                "metric": target.metric,
                "period": target.period,
            }
        ),
        evidence_needs=query_plan.evidence_needs,
    )

    assert plan.action is CorrectionAction.TARGETED_RETRIEVAL
    assert plan.affected_need_ids == (target.need_id,)
    assert plan.requires_retrieval is True
    assert plan.requires_generation is True


@pytest.mark.parametrize("code", ["result_value_mismatch"])
def test_math_failures_use_local_executor_without_retrieval_or_generation(code):
    plan = plan_correction(_report({"code": code}))

    assert plan.action is CorrectionAction.LOCAL_RECOMPUTE
    assert plan.requires_local_execution is True
    assert plan.requires_retrieval is False
    assert plan.requires_generation is False


@pytest.mark.parametrize("code", ["answer_result_mismatch"])
def test_render_failures_only_rerender_verified_result(code):
    plan = plan_correction(_report({"code": code}))

    assert plan.action is CorrectionAction.RERENDER
    assert plan.reuse_evidence is True
    assert plan.requires_retrieval is False
    assert plan.requires_generation is False


def test_formula_mismatch_replans_before_spending_on_retrieval():
    plan = plan_correction(_report({"code": "formula_mismatch"}))

    assert plan.action is CorrectionAction.REPLAN
    assert plan.requires_replan is True
    assert plan.requires_generation is True
    assert plan.reuse_evidence is True
    assert plan.requires_retrieval is False


def test_missing_operand_precedes_replan_for_mixed_failure_report():
    query_plan = compile_finance_query(
        "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? "
        "It is revenue divided by average PP&E between FY2018 and FY2019."
    )
    target = next(
        need
        for need in query_plan.evidence_needs
        if need.metric == "property_plant_equipment" and need.period == "FY2018"
    )

    plan = plan_correction(
        _report(
            {"code": "formula_mismatch"},
            {
                "code": "missing_operand",
                "operand_id": target.need_id,
                "metric": target.metric,
                "period": target.period,
            },
        ),
        evidence_needs=query_plan.evidence_needs,
    )

    assert plan.action is CorrectionAction.TARGETED_RETRIEVAL
    assert plan.affected_need_ids == (target.need_id,)
    assert plan.requires_retrieval is True
    assert plan.requires_generation is True
    assert plan.requires_replan is False


@pytest.mark.parametrize("code", ["arithmetic_error", "result_unit_mismatch"])
def test_invalid_math_or_dimensions_rebuild_the_program(code):
    plan = plan_correction(_report({"code": code}))

    assert plan.action is CorrectionAction.REPLAN
    assert plan.requires_generation is True
    assert plan.requires_local_execution is False


def test_conflicting_evidence_routes_to_explicit_reconciliation():
    plan = plan_correction(_report({"code": "conflicting_evidence"}))

    assert plan.action is CorrectionAction.RECONCILE
    assert plan.requires_reconciliation is True
    assert plan.reuse_evidence is True


def test_repeated_gap_over_same_evidence_abstains_early():
    first = plan_correction(
        _report({"code": "missing_evidence", "metric": "revenue", "period": "FY2022"}),
        evidence_fingerprint="snapshot-42",
    )
    repeated = plan_correction(
        _report({"code": "missing_evidence", "metric": "revenue", "period": "FY2022"}),
        evidence_fingerprint="snapshot-42",
        previous_gap_fingerprints=(first.gap_fingerprint,),
    )

    assert repeated.action is CorrectionAction.ABSTAIN
    assert repeated.terminal is True
    assert "same verifier gap" in repeated.rationale


def test_same_gap_with_changed_evidence_is_not_treated_as_a_repeat():
    first = plan_correction(
        _report({"code": "missing_evidence", "metric": "revenue"}),
        evidence_fingerprint="snapshot-A",
    )
    changed = plan_correction(
        _report({"code": "missing_evidence", "metric": "revenue"}),
        evidence_fingerprint="snapshot-B",
        previous_gap_fingerprints=(first.gap_fingerprint,),
    )

    assert changed.action is CorrectionAction.TARGETED_RETRIEVAL


def test_budget_exhaustion_abstains_before_selecting_more_work():
    plan = plan_correction(
        _report({"code": "invalid_citation"}),
        attempt=2,
        max_attempts=3,
    )

    assert plan.action is CorrectionAction.ABSTAIN
    assert plan.terminal is True


def test_final_attempt_can_apply_a_zero_call_local_repair():
    plan = plan_correction(
        _report({"code": "result_value_mismatch"}),
        attempt=2,
        max_attempts=3,
    )

    assert plan.action is CorrectionAction.LOCAL_RECOMPUTE
    assert plan.terminal is False


def test_paper_fixed_mode_preserves_full_retry_concept():
    policy = CorrectionPolicy(PolicyMode.PAPER_FIXED)
    plan = policy.plan(
        _report({"code": "arithmetic_error"}),
        attempt=0,
        max_attempts=3,
    )

    assert plan.policy_mode is PolicyMode.PAPER_FIXED
    assert plan.action is CorrectionAction.PAPER_FIXED_RETRY
    assert plan.requires_retrieval is True
    assert plan.requires_generation is True
    assert plan.requires_local_execution is False


def test_correction_plan_to_dict_is_json_serializable():
    plan = plan_correction(
        _report({"code": "missing_evidence", "operand_id": "need:revenue:any:fy2022"}),
        evidence_fingerprint="abc",
    )
    payload = plan.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["policy_mode"] == "gap_driven_v2"
    assert payload["action"] == "targeted_retrieval"


def test_policy_rejects_invalid_attempt_budgets():
    with pytest.raises(ValueError, match="max_attempts"):
        plan_correction(_report({"code": "missing_evidence"}), max_attempts=0)
