"""Deterministic finance-query understanding primitives.

The v2 query compiler is intentionally independent from the legacy retrieval
pipeline.  Callers can opt into a structured plan without changing the
historical paper configuration.
"""

from .finance_plan import (
    AnswerKind,
    EntityConstraint,
    EvidenceNeed,
    FinanceQueryPlan,
    MagnitudeScale,
    OutputContract,
    PeriodConstraint,
    PeriodKind,
    TaskType,
    compile_finance_query,
)

__all__ = [
    "AnswerKind",
    "EntityConstraint",
    "EvidenceNeed",
    "FinanceQueryPlan",
    "MagnitudeScale",
    "OutputContract",
    "PeriodConstraint",
    "PeriodKind",
    "TaskType",
    "compile_finance_query",
]
