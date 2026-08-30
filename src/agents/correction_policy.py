"""Failure-conditioned correction policy for the v2 finance QA runtime.

The policy is intentionally standalone.  It translates structured verifier
issues into the smallest useful next action, while retaining an explicit
``paper_fixed`` mode for historical 10/20/30 retry experiments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class PolicyMode(str, Enum):
    """Available correction policies."""

    PAPER_FIXED = "paper_fixed"
    GAP_DRIVEN_V2 = "gap_driven_v2"


class CorrectionAction(str, Enum):
    """A single controller action selected after verification."""

    ACCEPT = "accept"
    PAPER_FIXED_RETRY = "paper_fixed_retry"
    REUSE_EVIDENCE_REGENERATE = "reuse_evidence_regenerate"
    TARGETED_RETRIEVAL = "targeted_retrieval"
    REPLAN = "replan"
    LOCAL_RECOMPUTE = "local_recompute"
    RERENDER = "rerender"
    RECONCILE = "reconcile"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class IssueDescriptor:
    """Normalized verifier issue fields used by the policy."""

    code: str
    message: str = ""
    operand_id: Optional[str] = None
    metric: Optional[str] = None
    period: Optional[str] = None
    doc_id: Optional[str] = None
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "operand_id": self.operand_id,
            "metric": self.metric,
            "period": self.period,
            "doc_id": self.doc_id,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class CorrectionPlan:
    """Serialization-safe next action returned to an orchestrator."""

    policy_mode: PolicyMode
    action: CorrectionAction
    issue_codes: Tuple[str, ...]
    affected_need_ids: Tuple[str, ...]
    gap_fingerprint: str
    rationale: str
    terminal: bool = False
    reuse_evidence: bool = False
    requires_retrieval: bool = False
    requires_generation: bool = False
    requires_local_execution: bool = False
    requires_replan: bool = False
    requires_reconciliation: bool = False

    @property
    def should_retry(self) -> bool:
        return not self.terminal and self.action is not CorrectionAction.ACCEPT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_mode": self.policy_mode.value,
            "action": self.action.value,
            "issue_codes": list(self.issue_codes),
            "affected_need_ids": list(self.affected_need_ids),
            "gap_fingerprint": self.gap_fingerprint,
            "rationale": self.rationale,
            "terminal": self.terminal,
            "should_retry": self.should_retry,
            "reuse_evidence": self.reuse_evidence,
            "requires_retrieval": self.requires_retrieval,
            "requires_generation": self.requires_generation,
            "requires_local_execution": self.requires_local_execution,
            "requires_replan": self.requires_replan,
            "requires_reconciliation": self.requires_reconciliation,
        }


_REPLAN_CODES = {
    "arithmetic_error",
    "constant_not_question_bound",
    "formula_mismatch",
    "result_unit_mismatch",
    "unsupported_operator",
}
_CONFLICT_CODES = {"conflicting_evidence"}
_RETRIEVAL_CODES = {
    "insufficient_coverage",
    "missing_evidence",
    "missing_operand",
    "numeric_evidence_mismatch",
    "operand_currency_mismatch",
    "operand_entity_mismatch",
    "operand_metric_mismatch",
    "operand_period_mismatch",
    "operand_unit_mismatch",
    "operand_value_mismatch",
}
_LOCAL_RECOMPUTE_CODES = {
    "result_value_mismatch",
}
_RERENDER_CODES = {
    "answer_result_mismatch",
}
_REGENERATE_CODES = {
    "citation_claim_mismatch",
    "empty_answer",
    "invalid_citation",
    "low_policy_score",
    "missing_citation",
    "missing_program",
    "schema_invalid",
    "unsupported_claim",
}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_issue(issue: Any) -> IssueDescriptor:
    if isinstance(issue, IssueDescriptor):
        return issue
    if isinstance(issue, str):
        return IssueDescriptor(code=str(getattr(issue, "value", issue)))
    raw_code = _field(issue, "code") or _field(issue, "reason_code")
    if not raw_code:
        raise ValueError(f"Verifier issue has no code: {issue!r}")
    code = getattr(raw_code, "value", raw_code)
    raw_severity = _field(issue, "severity", "error") or "error"
    severity = getattr(raw_severity, "value", raw_severity)
    return IssueDescriptor(
        code=str(code),
        message=str(_field(issue, "message", "") or ""),
        operand_id=_optional_string(_field(issue, "operand_id")),
        metric=_optional_string(_field(issue, "metric")),
        period=_optional_string(_field(issue, "period")),
        doc_id=_optional_string(_field(issue, "doc_id")),
        severity=str(severity),
    )


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _extract_report(report: Any) -> Tuple[bool, Tuple[IssueDescriptor, ...]]:
    if report is None:
        return False, (IssueDescriptor(code="low_policy_score"),)

    passed_value = _field(report, "passed")
    if passed_value is None:
        passed_value = _field(report, "verification_passed")
    passed = bool(passed_value)

    raw_issues = _field(report, "issues")
    if raw_issues is None:
        raw_issues = _field(report, "reason_codes", ())
    if isinstance(raw_issues, (str, Mapping)):
        raw_issues = (raw_issues,)
    issues = tuple(_normalize_issue(issue) for issue in (raw_issues or ()))
    if not passed and not issues:
        issues = (IssueDescriptor(code="low_policy_score"),)
    return passed, issues


def make_gap_fingerprint(
    issues: Sequence[IssueDescriptor],
    evidence_fingerprint: str,
) -> str:
    """Hash the actionable gap plus the evidence snapshot that produced it."""

    payload = {
        "evidence_fingerprint": evidence_fingerprint or "",
        "issues": sorted(
            (
                {
                    "code": issue.code,
                    "operand_id": issue.operand_id,
                    "metric": issue.metric,
                    "period": issue.period,
                    "doc_id": issue.doc_id,
                }
                for issue in issues
            ),
            key=lambda item: json.dumps(item, sort_keys=True),
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _need_value(need: Any, field_name: str) -> Optional[str]:
    value = _field(need, field_name)
    return _optional_string(value)


def _affected_need_ids(
    issues: Sequence[IssueDescriptor],
    evidence_needs: Sequence[Any],
) -> Tuple[str, ...]:
    affected = []
    for issue in issues:
        for need in evidence_needs:
            need_id = _need_value(need, "need_id")
            if not need_id:
                continue
            operand_match = issue.operand_id in {None, need_id}
            metric_match = issue.metric is None or issue.metric == _need_value(need, "metric")
            period_match = issue.period is None or issue.period == _need_value(need, "period")
            if operand_match and metric_match and period_match and need_id not in affected:
                affected.append(need_id)

        # Preserve a directly named operand even before a QueryPlan is attached.
        if issue.operand_id and issue.operand_id not in affected:
            affected.append(issue.operand_id)
    return tuple(affected)


def _make_plan(
    *,
    mode: PolicyMode,
    action: CorrectionAction,
    issues: Sequence[IssueDescriptor],
    affected_need_ids: Sequence[str],
    gap_fingerprint: str,
    rationale: str,
) -> CorrectionPlan:
    return CorrectionPlan(
        policy_mode=mode,
        action=action,
        issue_codes=tuple(dict.fromkeys(issue.code for issue in issues)),
        affected_need_ids=tuple(affected_need_ids),
        gap_fingerprint=gap_fingerprint,
        rationale=rationale,
        terminal=action in {CorrectionAction.ACCEPT, CorrectionAction.ABSTAIN},
        reuse_evidence=action in {
            CorrectionAction.REUSE_EVIDENCE_REGENERATE,
            CorrectionAction.REPLAN,
            CorrectionAction.LOCAL_RECOMPUTE,
            CorrectionAction.RERENDER,
            CorrectionAction.RECONCILE,
        },
        requires_retrieval=action in {
            CorrectionAction.PAPER_FIXED_RETRY,
            CorrectionAction.TARGETED_RETRIEVAL,
        },
        requires_generation=action in {
            CorrectionAction.PAPER_FIXED_RETRY,
            CorrectionAction.REUSE_EVIDENCE_REGENERATE,
            CorrectionAction.TARGETED_RETRIEVAL,
            CorrectionAction.REPLAN,
            CorrectionAction.RECONCILE,
        },
        requires_local_execution=action is CorrectionAction.LOCAL_RECOMPUTE,
        requires_replan=action is CorrectionAction.REPLAN,
        requires_reconciliation=action is CorrectionAction.RECONCILE,
    )


class CorrectionPolicy:
    """Select the smallest corrective action justified by verifier evidence."""

    def __init__(self, mode: PolicyMode | str = PolicyMode.GAP_DRIVEN_V2):
        self.mode = PolicyMode(mode)

    def plan(
        self,
        report: Any,
        *,
        attempt: int = 0,
        max_attempts: int = 3,
        evidence_fingerprint: str = "",
        previous_gap_fingerprints: Sequence[str] = (),
        evidence_needs: Sequence[Any] = (),
    ) -> CorrectionPlan:
        """Return the next action for one verification result.

        ``attempt`` is zero-indexed and ``max_attempts`` is the total allowed
        attempt count.  Repeating the same gap over the same evidence snapshot
        terminates early rather than spending the remaining budget blindly.
        """

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if attempt < 0:
            raise ValueError("attempt must be non-negative")

        passed, issues = _extract_report(report)
        gap_fingerprint = make_gap_fingerprint(issues, evidence_fingerprint)
        affected = _affected_need_ids(issues, evidence_needs)

        if passed and not issues:
            return _make_plan(
                mode=self.mode,
                action=CorrectionAction.ACCEPT,
                issues=(),
                affected_need_ids=(),
                gap_fingerprint=gap_fingerprint,
                rationale="Verification passed with no actionable gaps.",
            )

        codes = {issue.code for issue in issues}
        local_only = bool(codes) and codes <= (
            _LOCAL_RECOMPUTE_CODES | _RERENDER_CODES
        )
        budget_exhausted = attempt >= max_attempts - 1
        if budget_exhausted and not local_only:
            return _make_plan(
                mode=self.mode,
                action=CorrectionAction.ABSTAIN,
                issues=issues,
                affected_need_ids=affected,
                gap_fingerprint=gap_fingerprint,
                rationale="Correction budget exhausted before verification passed.",
            )

        if self.mode is PolicyMode.PAPER_FIXED:
            return _make_plan(
                mode=self.mode,
                action=CorrectionAction.PAPER_FIXED_RETRY,
                issues=issues,
                affected_need_ids=affected,
                gap_fingerprint=gap_fingerprint,
                rationale="Historical paper policy: rerun retrieval and generation with fixed escalation.",
            )

        if gap_fingerprint in set(previous_gap_fingerprints):
            return _make_plan(
                mode=self.mode,
                action=CorrectionAction.ABSTAIN,
                issues=issues,
                affected_need_ids=affected,
                gap_fingerprint=gap_fingerprint,
                rationale="The same verifier gap repeated over the same evidence snapshot.",
            )

        # Evidence is a prerequisite for a trustworthy program.  When a report
        # contains both a program defect and a missing/mismatched operand,
        # retrieve the affected evidence first; replanning against the same
        # incomplete snapshot only spends a model call without resolving the
        # root cause.
        if codes & _RETRIEVAL_CODES:
            action = CorrectionAction.TARGETED_RETRIEVAL
            affected = _affected_need_ids(
                tuple(issue for issue in issues if issue.code in _RETRIEVAL_CODES),
                evidence_needs,
            )
            rationale = (
                "Retrieve only the missing or mismatched evidence needs before "
                "rebuilding any affected program."
            )
        elif codes & _REPLAN_CODES:
            action = CorrectionAction.REPLAN
            rationale = "Rebuild the generated program against the trusted question contract."
        elif codes & _CONFLICT_CODES:
            action = CorrectionAction.RECONCILE
            rationale = "Conflicting evidence requires explicit source/period reconciliation."
        elif codes & _LOCAL_RECOMPUTE_CODES:
            action = CorrectionAction.LOCAL_RECOMPUTE
            rationale = "Evidence is sufficient; re-execute the typed calculation locally."
        elif codes & _RERENDER_CODES:
            action = CorrectionAction.RERENDER
            rationale = "The verified result is available; repair only answer rendering."
        elif codes & _REGENERATE_CODES:
            action = CorrectionAction.REUSE_EVIDENCE_REGENERATE
            rationale = "Reuse the evidence snapshot and repair generation/citations."
        else:
            action = CorrectionAction.REUSE_EVIDENCE_REGENERATE
            rationale = "Unknown gap type; prefer one evidence-preserving repair before broader search."

        return _make_plan(
            mode=self.mode,
            action=action,
            issues=issues,
            affected_need_ids=affected,
            gap_fingerprint=gap_fingerprint,
            rationale=rationale,
        )


def plan_correction(
    report: Any,
    *,
    mode: PolicyMode | str = PolicyMode.GAP_DRIVEN_V2,
    attempt: int = 0,
    max_attempts: int = 3,
    evidence_fingerprint: str = "",
    previous_gap_fingerprints: Sequence[str] = (),
    evidence_needs: Sequence[Any] = (),
) -> CorrectionPlan:
    """Convenience wrapper around :class:`CorrectionPolicy`."""

    return CorrectionPolicy(mode).plan(
        report,
        attempt=attempt,
        max_attempts=max_attempts,
        evidence_fingerprint=evidence_fingerprint,
        previous_gap_fingerprints=previous_gap_fingerprints,
        evidence_needs=evidence_needs,
    )


__all__ = [
    "CorrectionAction",
    "CorrectionPlan",
    "CorrectionPolicy",
    "IssueDescriptor",
    "PolicyMode",
    "make_gap_fingerprint",
    "plan_correction",
]
