"""Compile financial questions into deterministic, serialization-safe plans.

This module is the query-understanding boundary for the v2 architecture.  It
does not call an LLM and does not retrieve documents.  Its output describes
the constraints and evidence gaps that downstream retrieval, calculation, and
verification components can consume.

The compiler is deliberately conservative.  Missing entity or time
constraints are recorded as abstentions rather than guessed.  Relative periods
such as "latest quarter" remain unresolved until a corpus-aware component can
anchor them to a filing calendar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.metadata_utils import extract_metadata_from_question, normalize_company_name


class TaskType(str, Enum):
    """High-level work required to answer a question."""

    EXTRACTION = "extraction"
    CALCULATION = "calculation"
    COMPARISON = "comparison"
    TREND = "trend"
    QUALITATIVE = "qualitative"
    SCREENING = "screening"
    MONITORING = "monitoring"
    UNKNOWN = "unknown"


class AnswerKind(str, Enum):
    """Expected semantic shape of the final answer."""

    AMOUNT = "amount"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    DAYS = "days"
    BOOLEAN = "boolean"
    LIST = "list"
    TABLE = "table"
    NARRATIVE = "narrative"
    UNKNOWN = "unknown"


class MagnitudeScale(str, Enum):
    """Requested display scale for an amount."""

    UNIT = "unit"
    THOUSAND = "thousand"
    MILLION = "million"
    BILLION = "billion"
    TRILLION = "trillion"


class PeriodKind(str, Enum):
    """How a requested time constraint should be interpreted."""

    FISCAL_YEAR = "fiscal_year"
    FISCAL_QUARTER = "fiscal_quarter"
    CALENDAR_YEAR = "calendar_year"
    DATE = "date"
    QUARTER = "quarter"
    RELATIVE = "relative"


@dataclass(frozen=True)
class EntityConstraint:
    """A surface entity mention and its canonical retrieval key."""

    surface_form: str
    canonical_name: str
    role: str = "issuer"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_form": self.surface_form,
            "canonical_name": self.canonical_name,
            "role": self.role,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class PeriodConstraint:
    """A fiscal, calendar, quarter, or relative time constraint."""

    label: str
    kind: PeriodKind
    year: Optional[int] = None
    quarter: Optional[int] = None
    relative_expression: Optional[str] = None
    anchor: Optional[str] = None
    resolved: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind.value,
            "year": self.year,
            "quarter": self.quarter,
            "relative_expression": self.relative_expression,
            "anchor": self.anchor,
            "resolved": self.resolved,
        }


@dataclass(frozen=True)
class OutputContract:
    """Machine-checkable output requirements extracted from the question."""

    answer_kind: AnswerKind
    currency: Optional[str] = None
    scale: Optional[MagnitudeScale] = None
    decimal_places: Optional[int] = None
    presentation: str = "narrative"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_kind": self.answer_kind.value,
            "currency": self.currency,
            "scale": self.scale.value if self.scale is not None else None,
            "decimal_places": self.decimal_places,
            "presentation": self.presentation,
        }


@dataclass(frozen=True)
class EvidenceNeed:
    """One independently retrievable fact required by the answer plan."""

    need_id: str
    metric: str
    query: str
    entity: Optional[str] = None
    period: Optional[str] = None
    statement_hints: Tuple[str, ...] = ()
    source_types: Tuple[str, ...] = ("authorized_corpus",)
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "need_id": self.need_id,
            "metric": self.metric,
            "query": self.query,
            "entity": self.entity,
            "period": self.period,
            "statement_hints": list(self.statement_hints),
            "source_types": list(self.source_types),
            "required": self.required,
        }


@dataclass(frozen=True)
class FinanceQueryPlan:
    """Structured contract shared by retrieval, execution, and verification."""

    question: str
    task_type: TaskType
    entities: Tuple[EntityConstraint, ...]
    periods: Tuple[PeriodConstraint, ...]
    output: OutputContract
    evidence_needs: Tuple[EvidenceNeed, ...]
    source_hints: Tuple[str, ...] = ()
    formula_id: Optional[str] = None
    formula_hint: Optional[str] = None
    answer_metric: Optional[str] = None
    constraint_abstentions: Tuple[str, ...] = ()
    unresolved_constraints: Tuple[str, ...] = ()
    as_of: Optional[str] = None
    compiler_version: str = "finance-rules-v1"
    confidence: float = 1.0

    @property
    def requires_calculation(self) -> bool:
        return self.task_type is TaskType.CALCULATION

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation with stable field names."""

        return {
            "question": self.question,
            "task_type": self.task_type.value,
            "entities": [entity.to_dict() for entity in self.entities],
            "periods": [period.to_dict() for period in self.periods],
            "output": self.output.to_dict(),
            "evidence_needs": [need.to_dict() for need in self.evidence_needs],
            "source_hints": list(self.source_hints),
            "formula_id": self.formula_id,
            "formula_hint": self.formula_hint,
            "answer_metric": self.answer_metric,
            "constraint_abstentions": list(self.constraint_abstentions),
            "unresolved_constraints": list(self.unresolved_constraints),
            "as_of": self.as_of,
            "compiler_version": self.compiler_version,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class _MetricRule:
    name: str
    aliases: Tuple[str, ...]
    statements: Tuple[str, ...]


@dataclass(frozen=True)
class _FormulaRule:
    formula_id: str
    pattern: re.Pattern[str]
    metrics: Tuple[str, ...]
    expression: str
    balance_metrics: Tuple[str, ...] = ()


_METRIC_RULES: Tuple[_MetricRule, ...] = (
    _MetricRule("depreciation_and_amortization", ("depreciation and amortization", "d&a"), ("cash_flow_statement",)),
    _MetricRule("property_plant_equipment", ("property, plant, and equipment", "property plant and equipment", "net pp&e", "net ppne", "pp&e", "ppne"), ("balance_sheet",)),
    _MetricRule("operating_cash_flow", ("cash from operations", "cash flow from operations", "operating cash flow"), ("cash_flow_statement",)),
    _MetricRule("accounts_receivable", ("accounts receivable", "account receivable", "receivables"), ("balance_sheet",)),
    _MetricRule("accounts_payable", ("accounts payable", "account payable", "payables"), ("balance_sheet",)),
    _MetricRule("current_liabilities", ("total current liabilities", "current liabilities"), ("balance_sheet",)),
    _MetricRule("current_assets", ("total current assets", "current assets", "quick assets"), ("balance_sheet",)),
    _MetricRule("operating_income", ("unadjusted operating income", "operating income"), ("income_statement",)),
    _MetricRule("operating_margin", ("operating margin",), ("income_statement",)),
    _MetricRule("total_assets", ("average total assets", "total assets"), ("balance_sheet",)),
    _MetricRule("net_income", ("net income attributable to shareholders", "net income"), ("income_statement",)),
    _MetricRule("cost_of_goods_sold", ("cost of goods sold", "cost of revenue", "cogs"), ("income_statement",)),
    _MetricRule("capital_expenditure", ("capital expenditures", "capital expenditure", "capex"), ("cash_flow_statement",)),
    _MetricRule("inventory", ("average inventory", "inventory"), ("balance_sheet",)),
    _MetricRule("cash_dividends", ("total cash dividends paid", "cash dividends", "dividends paid"), ("cash_flow_statement",)),
    _MetricRule("revenue", ("net sales", "revenue", "sales"), ("income_statement",)),
    _MetricRule("ebitda", ("unadjusted ebitda", "ebitda"), ("income_statement", "cash_flow_statement")),
    _MetricRule("debt_securities", ("debt securities",), ("balance_sheet",)),
    _MetricRule("quick_ratio", ("quick ratio",), ("balance_sheet",)),
)

_METRIC_BY_NAME = {rule.name: rule for rule in _METRIC_RULES}

_FORMULA_RULES: Tuple[_FormulaRule, ...] = (
    _FormulaRule(
        "multi_year_net_profit_margin",
        re.compile(
            r"\b(?:three|3)[- ]year average net profit margin\b",
            re.IGNORECASE,
        ),
        ("net_income", "revenue"),
        "average(net_income / revenue by fiscal year)",
    ),
    _FormulaRule(
        "multi_year_operating_income_margin",
        re.compile(
            r"\b(?:three|3)[- ]year average (?:unadjusted )?operating income\s*(?:%|percent|percentage)?\s*margin\b",
            re.IGNORECASE,
        ),
        ("operating_income", "revenue"),
        "average(operating_income / revenue by fiscal year)",
    ),
    _FormulaRule(
        "multi_year_cogs_margin",
        re.compile(
            r"\b(?:three|3)[- ]year average (?:of )?(?:cost of goods sold|cogs) as a?\s*% of revenue\b",
            re.IGNORECASE,
        ),
        ("cost_of_goods_sold", "revenue"),
        "average(cost_of_goods_sold / revenue by fiscal year)",
    ),
    _FormulaRule(
        "cash_conversion_cycle",
        re.compile(r"\bcash conversion cycle\b|\bccc\b", re.IGNORECASE),
        ("inventory", "accounts_receivable", "accounts_payable", "cost_of_goods_sold", "revenue"),
        "DIO + DSO - DPO",
        ("inventory", "accounts_receivable", "accounts_payable"),
    ),
    _FormulaRule(
        "days_payable_outstanding",
        re.compile(r"\bdays payable outstanding\b|\bdpo\b", re.IGNORECASE),
        ("accounts_payable", "cost_of_goods_sold", "inventory"),
        "365 * average(accounts_payable) / (cost_of_goods_sold + change(inventory))",
        ("accounts_payable", "inventory"),
    ),
    _FormulaRule(
        "fixed_asset_turnover",
        re.compile(r"\bfixed asset turnover\b", re.IGNORECASE),
        ("revenue", "property_plant_equipment"),
        "revenue / average(property_plant_equipment)",
        ("property_plant_equipment",),
    ),
    _FormulaRule(
        "inventory_turnover",
        re.compile(r"\binventory turnover\b|how many times .* sold (?:its )?inventory", re.IGNORECASE),
        ("cost_of_goods_sold", "inventory"),
        "cost_of_goods_sold / average(inventory)",
        ("inventory",),
    ),
    _FormulaRule(
        "return_on_assets",
        re.compile(r"\breturn on assets\b|\broa\b", re.IGNORECASE),
        ("net_income", "total_assets"),
        "net_income / average(total_assets)",
        ("total_assets",),
    ),
    _FormulaRule(
        "asset_turnover",
        re.compile(r"(?<!fixed )\basset turnover\b", re.IGNORECASE),
        ("revenue", "total_assets"),
        "revenue / average(total_assets)",
        ("total_assets",),
    ),
    _FormulaRule(
        "operating_cash_flow_ratio",
        re.compile(r"\boperating cash flow ratio\b", re.IGNORECASE),
        ("operating_cash_flow", "current_liabilities"),
        "operating_cash_flow / current_liabilities",
    ),
    _FormulaRule(
        "working_capital_ratio",
        re.compile(r"\b(?:working capital|current) ratio\b", re.IGNORECASE),
        ("current_assets", "current_liabilities"),
        "current_assets / current_liabilities",
    ),
    _FormulaRule(
        "free_cash_flow",
        re.compile(r"\bfree cash flow\b|\bfcf\b", re.IGNORECASE),
        ("operating_cash_flow", "capital_expenditure"),
        "operating_cash_flow - capital_expenditure",
    ),
    _FormulaRule(
        "retention_ratio",
        re.compile(r"\bretention ratio\b", re.IGNORECASE),
        ("net_income", "cash_dividends"),
        "1 - cash_dividends / net_income",
    ),
    _FormulaRule(
        "ebitda_margin",
        re.compile(r"\bebitda\s*(?:%|percent|percentage)?\s*margin\b", re.IGNORECASE),
        ("operating_income", "depreciation_and_amortization", "revenue"),
        "(operating_income + depreciation_and_amortization) / revenue",
    ),
    _FormulaRule(
        "cogs_margin",
        re.compile(r"\b(?:cogs|cost of goods sold)\s*(?:%|percent|percentage)\s*margin\b", re.IGNORECASE),
        ("cost_of_goods_sold", "revenue"),
        "cost_of_goods_sold / revenue",
    ),
    _FormulaRule(
        "capex_percent_revenue",
        re.compile(r"\bcapex\s+as\s+a?\s*%\s+of\s+revenue\b", re.IGNORECASE),
        ("capital_expenditure", "revenue"),
        "capital_expenditure / revenue",
    ),
    _FormulaRule(
        "year_over_year_change",
        re.compile(
            r"\b(?:year[- ]over[- ]year|yoy) (?:change|growth)\b|"
            r"\b(?:change|growth) (?:year[- ]over[- ]year|yoy)\b",
            re.IGNORECASE,
        ),
        ("revenue",),
        "(current - prior) / prior",
    ),
)

_SOURCE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("balance_sheet", re.compile(r"\bbalance sheet\b|statement of financial position", re.IGNORECASE)),
    ("income_statement", re.compile(r"\bincome statement\b|statement of income|\bp&l\b", re.IGNORECASE)),
    ("cash_flow_statement", re.compile(r"statement of cash flows?|cash flow statement", re.IGNORECASE)),
    ("earnings_transcript", re.compile(r"earnings (?:call|transcript)", re.IGNORECASE)),
    ("investor_presentation", re.compile(r"investor presentation", re.IGNORECASE)),
    ("annual_report", re.compile(r"annual report", re.IGNORECASE)),
    ("10-k", re.compile(r"\b10-?k\b", re.IGNORECASE)),
    ("10-q", re.compile(r"\b10-?q\b", re.IGNORECASE)),
    ("8-k", re.compile(r"\b8-?k\b", re.IGNORECASE)),
    ("market_data", re.compile(r"share price|stock price|market cap|trading volume", re.IGNORECASE)),
)

_RELATIVE_PERIOD_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("latest reported quarter", re.compile(r"\b(?:latest|most recent) (?:reported )?quarter\b", re.IGNORECASE)),
    ("previous reported quarter", re.compile(r"\bprevious (?:reported )?quarter\b", re.IGNORECASE)),
    ("latest fiscal year", re.compile(r"\b(?:latest|most recent) fiscal year\b", re.IGNORECASE)),
    ("last year", re.compile(r"\blast year\b", re.IGNORECASE)),
)

_ROUNDING_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}

_EXPLICIT_TICKER_PATTERNS = (
    re.compile(r"\$(?P<ticker>[A-Z]{1,5})\b"),
    re.compile(r"\b(?:ticker|symbol)\s*[:=]?\s*(?P<ticker>[A-Z]{1,5})\b", re.IGNORECASE),
    re.compile(r"\b(?:NASDAQ|NYSE)\s*:\s*(?P<ticker>[A-Z]{1,5})\b", re.IGNORECASE),
)

# These are literal instruction fragments in the bundled FinanceBench rows
# whose word "calculate" is boilerplate around a verbatim line-item lookup.
# Do not generalize this exception to arbitrary single-metric questions: doing
# so turns real arithmetic such as "calculate 10% of revenue" into extraction.
_BUNDLED_EXTRACTION_INSTRUCTIONS = (
    "calculate (or extract) the answer from the statement of income and the "
    "cash flow statement.",
    "compute or extract the answer by primarily using the details outlined "
    "in the statement of cash flows.",
    "we need to calculate a financial metric by using information only "
    "provided within the balance sheet. please answer the following question:",
)

_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _formula_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


# Explicit definitions are trusted only when they match one of these known
# semantic forms. This is intentionally an allowlist, not a best-effort parser:
# an unrecognized modifier must not be silently replaced by the conventional
# definition of a similarly named ratio.
_EXPLICIT_FORMULA_ALLOWLIST: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "fixed_asset_turnover": (
        _formula_pattern(
            r"fixed asset turnover ratio is defined as:\s*"
            r"FY\s*\d{2,4} revenue\s*/\s*\(average (?:PP&E|PPNE) between "
            r"FY\s*\d{2,4} and FY\s*\d{2,4}\)"
        ),
    ),
    "operating_cash_flow_ratio": (
        _formula_pattern(
            r"operating cash flow ratio is defined as:\s*cash from operations\s*"
            r"(?:/|divided by)\s*total current liabilities"
        ),
    ),
    "working_capital_ratio": (
        _formula_pattern(
            r"define working capital ratio as total current assets\s+divided by\s+"
            r"total current liabilities"
        ),
        _formula_pattern(
            r"(?:working capital|current) ratio is defined as:\s*"
            r"(?:total )?current assets\s*(?:/|divided by)\s*"
            r"(?:total )?current liabilities"
        ),
        _formula_pattern(
            r"(?:working capital|current) ratio\s+using\s+(?:total )?current assets\s+"
            r"(?:and|divided by|/)\s+(?:total )?current liabilities"
        ),
    ),
    "return_on_assets": (
        _formula_pattern(
            r"(?:return on assets|ROA) is defined as:\s*FY\s*\d{2,4} net income\s*/\s*"
            r"\(average total assets between FY\s*\d{2,4} and FY\s*\d{2,4}\)"
        ),
    ),
    "days_payable_outstanding": (
        _formula_pattern(
            r"DPO is defined as:\s*365\s*\*\s*\(average accounts payable between "
            r"FY\s*\d{2,4} and FY\s*\d{2,4}\)\s*/\s*\(FY\s*\d{2,4} COGS\s*\+\s*"
            r"change in inventory between FY\s*\d{2,4} and FY\s*\d{2,4}\)"
        ),
    ),
    "cash_conversion_cycle": (
        _formula_pattern(
            r"CCC is defined as:\s*DIO\s*\+\s*DSO\s*-\s*DPO\.\s*"
            r"DIO is defined as:\s*365\s*\*\s*\(average inventory between "
            r"FY\s*\d{2,4} and FY\s*\d{2,4}\)\s*/\s*\(FY\s*\d{2,4} COGS\)\.\s*"
            r"DSO is defined as:\s*365\s*\*\s*\(average accounts receivable between "
            r"FY\s*\d{2,4} and FY\s*\d{2,4}\)\s*/\s*\(FY\s*\d{2,4} Revenue\)\.\s*"
            r"DPO is defined as:\s*365\s*\*\s*\(average accounts payable between "
            r"FY\s*\d{2,4} and FY\s*\d{2,4}\)\s*/\s*\(FY\s*\d{2,4} COGS\s*\+\s*"
            r"change in inventory between FY\s*\d{2,4} and FY\s*\d{2,4}\)"
        ),
    ),
    "free_cash_flow": (
        _formula_pattern(
            r"FCF here is defined as:\s*\(cash from operations\s*-\s*capex\)"
        ),
    ),
    "retention_ratio": (
        _formula_pattern(
            r"retention ratio\s*\(using total cash dividends paid and net income "
            r"attributable to shareholders\)"
        ),
    ),
    "inventory_turnover": (
        _formula_pattern(
            r"inventory turnover ratio is defined as:\s*\(FY\s*\d{2,4} COGS\)\s*/\s*"
            r"\(average inventory between FY\s*\d{2,4} and FY\s*\d{2,4}\)"
        ),
    ),
    "asset_turnover": (
        _formula_pattern(
            r"asset turnover ratio is defined as:\s*FY\s*\d{2,4} revenue\s*/\s*"
            r"\(average total assets between FY\s*\d{2,4} and FY\s*\d{2,4}\)"
        ),
    ),
    "ebitda_margin": (
        _formula_pattern(
            r"calculate unadjusted EBITDA using unadjusted operating income and "
            r"(?:D&A|depreciation and amortization)"
        ),
        _formula_pattern(
            r"define unadjusted EBITDA as unadjusted operating income\s*\+\s*"
            r"depreciation and amortization"
        ),
    ),
    "capex_percent_revenue": (
        _formula_pattern(
            r"(?:FY\s*\d{2,4}\s*(?:-|–|—|to|through)\s*FY\s*\d{2,4}\s+)?"
            r"(?:three|3)[- ]year average of capex as a?\s*% of revenue"
        ),
    ),
}

_DERIVED_METRICS_BY_FORMULA = {
    "ebitda_margin": {"ebitda"},
}

_METRIC_ALIAS_PATTERN = "|".join(
    re.escape(alias)
    for alias in sorted(
        (alias for rule in _METRIC_RULES for alias in rule.aliases),
        key=len,
        reverse=True,
    )
)

_EXPLICIT_FORMULA_SYNTAX = re.compile(
    r"\b(?:is|here is) defined as\b|\bdefine\b.{0,80}\bas\b|"
    r"\b(?:is\s+)?(?:computed|calculated)\s+(?:as|by)\b|"
    r"\bformula\s*:|(?<![<>=])=(?!=)|[+*/]|"
    r"\b(?:equals?|is equal to|means|quotient|numerator|denominator)\b|"
    r"\b(?:divided (?:by|into)|multiplied by|subtract(?:ed)?|minus|plus|less)\b|"
    rf"(?:{_METRIC_ALIAS_PATTERN})\s+over\s+(?:{_METRIC_ALIAS_PATTERN})",
    re.IGNORECASE | re.DOTALL,
)

_FORMULA_MODIFIER_SYNTAX = re.compile(
    r"\b(?:360|365)(?:[- ]days?)?\b|"
    r"\b(?:basis|convention|ending|beginning|opening|closing|year[- ]end|"
    r"before|after|rather than|instead of|excluding|including|average|"
    r"adjusted|unadjusted)\b",
    re.IGNORECASE,
)

# Broad task-boundary cues for derived answers.  This is intentionally wider
# than the trusted formula allowlist: recognizing that a question requires a
# calculation is safe, while pretending an unknown derived metric is a direct
# extraction would bypass typed execution.  Unknown formulas therefore become
# CALCULATION plans with ``formula:unresolved`` and fail closed at preflight.
_DERIVED_ANSWER_SYNTAX = re.compile(
    r"\bwhat\s+(?:percent|percentage|proportion|share)\s+of\b|"
    r"\bas\s+(?:a\s+)?(?:%|percent(?:age)?)\s+of\b|"
    r"(?:%|percent(?:age)?)\s+margin\b|"
    r"\b(?:cagr|growth rate|working capital)\b|"
    r"\bhow much has\b.{0,120}\bchanged between\b|"
    r"\b(?:improv\w*|declin\w*|increas\w*|decreas\w*)\b.{0,80}"
    r"\b(?:margin|ratio|debt)\b|"
    r"\b(?:margin|ratio|debt)\b.{0,80}"
    r"\b(?:improv\w*|declin\w*|increas\w*|decreas\w*)\b|"
    r"\bhas\b.{0,120}\b(?:increased|decreased|improved|declined)\b"
    r".{0,120}\bbetween\b|"
    r"\bamong\b.{0,180}\b(?:most|least|largest|smallest)\b|"
    r"\b(?:represent|account for)\b.{0,80}\bmore than\b.{0,30}%|"
    r"\bchange\s+in\b.{0,100}\bmargin\b|"
    r"\bmargin\b.{0,100}\b(?:change|increase[ds]?|decrease[ds]?|"
    r"grew|growth|decline[ds]?)\b",
    re.IGNORECASE | re.DOTALL,
)
_NONCALCULATIVE_DERIVED_INTENT = re.compile(
    r"\bwhat\s+(?:did\s+management\s+attribute|drove|drives)\b|"
    r"\bwhy\b.{0,120}\b(?:growth|margin|ratio|working capital)\b|"
    r"\b(?:describe|explain)\b.{0,120}\b(?:policy|growth|margin|ratio|"
    r"working capital)\b|"
    r"\bdid\s+management\s+(?:discuss|mention)\b|"
    r"\bwhat\s+(?:is|was)\s+the\s+(?:reported|stated|disclosed)\b|"
    r"\b(?:definition of|what (?:is|does)\b.{0,40}\bmean)\b",
    re.IGNORECASE | re.DOTALL,
)


def _requests_derived_answer(question: str) -> bool:
    """Distinguish a derived value from discussion of a derived metric."""

    return bool(
        _DERIVED_ANSWER_SYNTAX.search(question)
        and not _NONCALCULATIVE_DERIVED_INTENT.search(question)
    )


def _iso_anchor(as_of: Optional[object]) -> Optional[str]:
    if as_of is None:
        return None
    if isinstance(as_of, datetime):
        return as_of.isoformat()
    if isinstance(as_of, date):
        return as_of.isoformat()
    return str(as_of)


def _find_surface_spans(question: str, surface: str) -> Iterable[Tuple[int, int]]:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(surface)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    for match in pattern.finditer(question):
        yield match.span()


def _extract_entities(question: str) -> Tuple[EntityConstraint, ...]:
    metadata = extract_metadata_from_question(question)
    candidates = list(dict.fromkeys(metadata.get("companies", [])))

    matches: List[Tuple[int, int, str]] = []
    for candidate in candidates:
        for start, end in _find_surface_spans(question, candidate):
            matches.append((start, end, question[start:end]))

    # Bare all-caps tokens are frequently financial acronyms (EPS, CAGR,
    # PP&E), not issuers.  Only treat ticker text as an entity when the user
    # marks it with a conventional ticker form.
    for pattern in _EXPLICIT_TICKER_PATTERNS:
        for match in pattern.finditer(question):
            surface = match.group("ticker").upper()
            ticker_start, ticker_end = match.span("ticker")
            matches.append((ticker_start, ticker_end, surface))

    # Longest match wins at the same/overlapping span ("CVS Health" over "CVS").
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2].lower()))
    selected: List[Tuple[int, int, str]] = []
    for candidate in matches:
        start, end, _ = candidate
        if any(start < other_end and end > other_start for other_start, other_end, _ in selected):
            continue
        selected.append(candidate)

    entities: List[EntityConstraint] = []
    seen_canonical = set()
    for _, _, surface in sorted(selected, key=lambda item: item[0]):
        canonical = normalize_company_name(surface)
        if not canonical or canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        entities.append(
            EntityConstraint(
                surface_form=surface,
                canonical_name=canonical,
                role="issuer",
                confidence=1.0,
            )
        )
    return tuple(entities)


def _normalize_year(raw: str) -> int:
    value = int(raw)
    return 2000 + value if value < 100 else value


def _add_date_period(
    located: List[Tuple[int, int, PeriodConstraint]],
    occupied: List[Tuple[int, int]],
    *,
    start: int,
    end: int,
    year: int,
    month: int,
    day: int,
) -> None:
    """Record a valid explicit date without degrading it to a bare year."""

    try:
        parsed = date(year, month, day)
    except ValueError:
        return
    located.append(
        (
            start,
            end,
            PeriodConstraint(
                label=parsed.isoformat(),
                kind=PeriodKind.DATE,
                year=parsed.year,
            ),
        )
    )
    occupied.append((start, end))


def _extract_periods(
    question: str,
    anchor: Optional[str],
) -> Tuple[Tuple[PeriodConstraint, ...], Tuple[str, ...]]:
    located: List[Tuple[int, int, PeriodConstraint]] = []
    occupied: List[Tuple[int, int]] = []

    for match in re.finditer(
        r"\b(?P<year>(?:19|20)\d{2})-(?P<month>0?[1-9]|1[0-2])-"
        r"(?P<day>0?[1-9]|[12]\d|3[01])\b",
        question,
    ):
        _add_date_period(
            located,
            occupied,
            start=match.start(),
            end=match.end(),
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
        )

    month_names = "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True))
    for match in re.finditer(
        rf"\b(?P<month>{month_names})\.?\s+(?P<day>[0-3]?\d)(?:st|nd|rd|th)?"
        r",?\s+(?P<year>(?:19|20)\d{2})\b",
        question,
        re.IGNORECASE,
    ):
        _add_date_period(
            located,
            occupied,
            start=match.start(),
            end=match.end(),
            year=int(match.group("year")),
            month=_MONTH_NUMBERS[match.group("month").lower()],
            day=int(match.group("day")),
        )

    combined_patterns = (
        re.compile(r"\bFY\s*(?P<year>\d{2,4})\s*Q(?P<quarter>[1-4])\b", re.IGNORECASE),
        re.compile(r"\bQ(?P<quarter>[1-4])\s+(?:of\s+)?FY\s*(?P<year>\d{2,4})\b", re.IGNORECASE),
    )
    for pattern in combined_patterns:
        for match in pattern.finditer(question):
            year = _normalize_year(match.group("year"))
            quarter = int(match.group("quarter"))
            located.append(
                (
                    match.start(),
                    match.end(),
                    PeriodConstraint(
                        label=f"FY{year}Q{quarter}",
                        kind=PeriodKind.FISCAL_QUARTER,
                        year=year,
                        quarter=quarter,
                    ),
                )
            )
            occupied.append(match.span())

    for match in re.finditer(r"\bFY\s*(?P<year>\d{2,4})\b", question, re.IGNORECASE):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        year = _normalize_year(match.group("year"))
        located.append(
            (
                match.start(),
                match.end(),
                PeriodConstraint(
                    label=f"FY{year}",
                    kind=PeriodKind.FISCAL_YEAR,
                    year=year,
                ),
            )
        )
        occupied.append(match.span())

    for match in re.finditer(
        r"\b(?:CY\s*|calendar year\s+)(?P<year>(?:19|20)\d{2})\b",
        question,
        re.IGNORECASE,
    ):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        year = int(match.group("year"))
        located.append(
            (
                match.start(),
                match.end(),
                PeriodConstraint(
                    label=f"CY{year}",
                    kind=PeriodKind.CALENDAR_YEAR,
                    year=year,
                ),
            )
        )
        occupied.append(match.span())

    # Expand explicit fiscal ranges so every annual operand gets its own need.
    for match in re.finditer(
        r"\bFY\s*(?P<start>\d{2,4})\s*(?:-|–|—|to|through)\s*FY?\s*(?P<end>\d{2,4})\b",
        question,
        re.IGNORECASE,
    ):
        start_year = _normalize_year(match.group("start"))
        end_year = _normalize_year(match.group("end"))
        if 0 <= end_year - start_year <= 20:
            for offset, year in enumerate(range(start_year, end_year + 1)):
                located.append(
                    (
                        match.start() + offset,
                        match.end(),
                        PeriodConstraint(
                            label=f"FY{year}",
                            kind=PeriodKind.FISCAL_YEAR,
                            year=year,
                        ),
                    )
                )

    for match in re.finditer(r"\b(?:19|20)\d{2}\b", question):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        year = int(match.group(0))
        located.append(
            (
                match.start(),
                match.end(),
                PeriodConstraint(
                    label=str(year),
                    kind=PeriodKind.CALENDAR_YEAR,
                    year=year,
                ),
            )
        )

    # A quarter without a year is a real constraint, but not yet resolvable.
    for match in re.finditer(r"\bQ(?P<quarter>[1-4])\b", question, re.IGNORECASE):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        quarter = int(match.group("quarter"))
        located.append(
            (
                match.start(),
                match.end(),
                PeriodConstraint(
                    label=f"Q{quarter}",
                    kind=PeriodKind.QUARTER,
                    quarter=quarter,
                    resolved=False,
                ),
            )
        )

    for label, pattern in _RELATIVE_PERIOD_PATTERNS:
        for match in pattern.finditer(question):
            located.append(
                (
                    match.start(),
                    match.end(),
                    PeriodConstraint(
                        label=label,
                        kind=PeriodKind.RELATIVE,
                        relative_expression=match.group(0),
                        anchor=anchor,
                        # A wall-clock anchor alone cannot identify the latest
                        # *reported* fiscal period; the corpus must resolve it.
                        resolved=False,
                    ),
                )
            )

    periods: List[PeriodConstraint] = []
    seen = set()
    for _, _, period in sorted(located, key=lambda item: (item[0], item[2].label)):
        key = (period.kind.value, period.year, period.quarter, period.label)
        if key in seen:
            continue
        seen.add(key)
        periods.append(period)

    unresolved = tuple(
        f"period:{period.label}" for period in periods if not period.resolved
    )
    return tuple(periods), unresolved


def _extract_source_hints(question: str) -> Tuple[str, ...]:
    return tuple(
        name for name, pattern in _SOURCE_PATTERNS if pattern.search(question)
    )


def _extract_decimal_places(question: str) -> Optional[int]:
    match = re.search(
        r"round(?:\s+your\s+answer)?\s+to\s+"
        r"(?P<count>\d+|zero|one|two|three|four|five|six)\s+decimal places?",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None
    raw = match.group("count").lower()
    return int(raw) if raw.isdigit() else _ROUNDING_WORDS[raw]


def _compile_output_contract(question: str) -> OutputContract:
    lower = question.lower()
    currency = "USD" if re.search(r"\busd\b|\bu\.s\. dollars?\b|\$", question, re.IGNORECASE) else None

    scale: Optional[MagnitudeScale] = None
    scale_patterns = (
        (MagnitudeScale.TRILLION, r"\btrillions?\b"),
        (MagnitudeScale.BILLION, r"\bbillions?\b"),
        (MagnitudeScale.MILLION, r"\bmillions?\b"),
        (MagnitudeScale.THOUSAND, r"\bthousands?\b"),
    )
    for candidate, pattern in scale_patterns:
        if re.search(pattern, lower):
            scale = candidate
            break

    boolean_question = bool(
        re.match(r"\s*(?:is|are|was|were|does|do|did|has|have|can|could|would|should)\b", lower)
        or "yes or no" in lower
        or "whether " in lower
    )
    explicit_percentage = bool(
        re.search(
            r"answer (?:in|using) (?:units? of )?percents?|"
            r"in units? of percents?|as a?\s*%|"
            r"(?:%|percent|percentage)\s*margin|"
            r"percentage change",
            lower,
        )
    )
    explicit_ratio = bool(
        re.search(r"\bratio\b|\bhow many times\b|\bturnover ratio\b", lower)
    )

    if re.search(r"\btable\b", lower):
        answer_kind = AnswerKind.TABLE
        presentation = "table"
    elif boolean_question:
        answer_kind = AnswerKind.BOOLEAN
        presentation = "yes_no_with_support"
    elif explicit_percentage:
        answer_kind = AnswerKind.PERCENTAGE
        presentation = "numeric"
    elif explicit_ratio:
        answer_kind = AnswerKind.RATIO
        presentation = "numeric"
    elif currency or re.search(r"\bamount\b|\bhow much\b|\bwhat (?:is|was) the (?:value|total)\b", lower):
        answer_kind = AnswerKind.AMOUNT
        presentation = "numeric"
    elif re.match(r"\s*which\b", lower) or re.search(r"\blist\b|\bname the\b", lower):
        answer_kind = AnswerKind.LIST
        presentation = "list"
    else:
        answer_kind = AnswerKind.NARRATIVE
        presentation = "narrative"

    return OutputContract(
        answer_kind=answer_kind,
        currency=currency,
        scale=scale,
        decimal_places=_extract_decimal_places(question),
        presentation=presentation,
    )


def _align_output_with_formula(
    output: OutputContract,
    formula: Optional[_FormulaRule],
) -> OutputContract:
    """Apply the known semantic result type of an allowlisted formula."""

    if formula is None:
        return output
    formula_kinds = {
        "cash_conversion_cycle": AnswerKind.DAYS,
        "days_payable_outstanding": AnswerKind.DAYS,
        "year_over_year_change": AnswerKind.PERCENTAGE,
        "ebitda_margin": AnswerKind.PERCENTAGE,
        "cogs_margin": AnswerKind.PERCENTAGE,
        "capex_percent_revenue": AnswerKind.PERCENTAGE,
        "multi_year_net_profit_margin": AnswerKind.PERCENTAGE,
        "multi_year_operating_income_margin": AnswerKind.PERCENTAGE,
        "multi_year_cogs_margin": AnswerKind.PERCENTAGE,
        "fixed_asset_turnover": AnswerKind.RATIO,
        "inventory_turnover": AnswerKind.RATIO,
        "return_on_assets": AnswerKind.RATIO,
        "asset_turnover": AnswerKind.RATIO,
        "operating_cash_flow_ratio": AnswerKind.RATIO,
        "working_capital_ratio": AnswerKind.RATIO,
        "retention_ratio": AnswerKind.RATIO,
    }
    answer_kind = formula_kinds.get(formula.formula_id, output.answer_kind)
    return OutputContract(
        answer_kind=answer_kind,
        currency=output.currency,
        scale=output.scale,
        decimal_places=output.decimal_places,
        presentation="numeric",
    )


def _select_formula(question: str) -> Optional[_FormulaRule]:
    for rule in _FORMULA_RULES:
        if rule.pattern.search(question):
            return rule
    return None


def _has_formula_instruction(question: str) -> bool:
    if _EXPLICIT_FORMULA_SYNTAX.search(question):
        return True

    # "Using" denotes a formula only when it introduces operands/operators.
    # Source directions such as "using the income statement" are not formula
    # modifications and must not be rejected.
    return bool(
        re.search(
            rf"\busing\s+(?!(?:only\s+)?(?:the\s+)?(?:information|details|data|"
            rf"line items?|balance sheet|income statement|cash flow statement))"
            rf"(?=.{{0,120}}(?:{_METRIC_ALIAS_PATTERN}|divid(?:e|ed)|"
            r"subtract|minus|less|plus|average))",
            question,
            re.IGNORECASE | re.DOTALL,
        )
    )


def _formula_definition_is_trusted(
    question: str,
    formula: _FormulaRule,
    extracted_metrics: Sequence[str],
) -> bool:
    """Reject user-supplied formula semantics outside the exact allowlist."""

    derived_metrics = _DERIVED_METRICS_BY_FORMULA.get(formula.formula_id, set())
    expected_metrics = (
        set(extracted_metrics)
        if formula.formula_id == "year_over_year_change"
        else set(formula.metrics)
    )
    unexpected_metrics = set(extracted_metrics) - expected_metrics - derived_metrics
    if unexpected_metrics:
        return False

    # Strip every occurrence of the derived metric's own name before looking
    # for operands. Thus "inventory turnover" alone is still conventional,
    # while "inventory turnover based on ending inventory" is qualified and
    # must provide a complete allowlisted definition.
    outside_formula_name = formula.pattern.sub(" ", question)
    outside_formula_name = re.sub(
        r"\bconventional inventory management\b",
        " ",
        outside_formula_name,
        flags=re.IGNORECASE,
    )
    if formula.formula_id == "year_over_year_change":
        # A generic YoY operation necessarily names its one target metric; that
        # name is a binding, not a competing formula definition.
        for metric in extracted_metrics:
            metric_rule = _METRIC_BY_NAME.get(metric)
            if metric_rule is None:
                continue
            for alias in sorted(metric_rule.aliases, key=len, reverse=True):
                outside_formula_name = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                    " ",
                    outside_formula_name,
                    flags=re.IGNORECASE,
                )
    constant_context = re.sub(
        r"\b(?:FY|CY)\s*\d{2,4}(?:Q[1-4])?\b|"
        r"\bQ[1-4]\b|\b(?:19|20)\d{2}\b|"
        r"\b(?:10|8)-?[KQ]\b|"
        r"round(?:\s+your\s+answer)?\s+to\s+"
        r"(?:\d+|zero|one|two|three|four|five|six)\s+decimal places?",
        " ",
        outside_formula_name,
        flags=re.IGNORECASE,
    )
    mentions_constant = bool(
        re.search(
            r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])",
            constant_context,
        )
    )
    mentioned_operands = (
        set(_extract_metric_names(outside_formula_name)) & set(formula.metrics)
    )
    has_semantic_detail = bool(
        _has_formula_instruction(question)
        or mentioned_operands
        or mentions_constant
        or _FORMULA_MODIFIER_SYNTAX.search(outside_formula_name)
    )
    if not has_semantic_detail:
        return True
    for pattern in _EXPLICIT_FORMULA_ALLOWLIST.get(formula.formula_id, ()):
        match = pattern.search(question)
        if match is None:
            continue
        outside_match = question[: match.start()] + question[match.end() :]
        if _EXPLICIT_FORMULA_SYNTAX.search(outside_match):
            # A valid standard definition elsewhere in the question does not
            # authorize a second equation or modifier.
            continue
        # An allowlisted expression cannot be used merely as a prefix before
        # another operation (for example, "assets / liabilities, then * 2").
        # Inspect the rest of that sentence for arithmetic modifiers.
        suffix = re.split(r"[.;?!]", question[match.end() :], maxsplit=1)[0]
        if not re.search(
            r"(?:[+*/]|(?<!\w)-(?!\w)|\b(?:plus|minus|subtract(?:ed)?|"
            r"multiply|multiplied|divide|divided|average|excluding|net of)\b)",
            suffix,
            re.IGNORECASE,
        ):
            return True
    return False


def _extract_metric_names(question: str) -> Tuple[str, ...]:
    matches: List[Tuple[int, int, str]] = []
    for rule in _METRIC_RULES:
        for alias in rule.aliases:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            for match in pattern.finditer(question):
                matches.append((match.start(), match.end(), rule.name))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    selected_spans: List[Tuple[int, int]] = []
    metrics: List[str] = []
    for start, end, metric in matches:
        if any(start < other_end and end > other_start for other_start, other_end in selected_spans):
            continue
        selected_spans.append((start, end))
        if metric not in metrics:
            metrics.append(metric)
    return tuple(metrics)


def _classify_task(
    question: str,
    output: OutputContract,
    formula: Optional[_FormulaRule],
    metrics: Sequence[str],
    *,
    force_calculation: bool = False,
) -> TaskType:
    lower = question.lower()
    # Arithmetic intent dominates screening/monitoring vocabulary.  Those
    # multi-intent workflows are not implemented, so allowing them onto a
    # lower-assurance path would bypass the trusted calculation boundary.
    if formula or force_calculation:
        return TaskType.CALCULATION
    if re.search(
        r"(?<!\w)[+*](?!\w)|(?<![<>=])=(?!=)|"
        r"\b(?:multiplied by|subtract(?:ed)?|minus|plus|less)\b",
        question,
        re.IGNORECASE,
    ):
        return TaskType.CALCULATION
    # Formula structure is a calculation cue even when the operand vocabulary
    # is not in our metric dictionary.  Requiring recognized metrics here made
    # unknown ratios fail open as extraction/qualitative questions.
    if _has_formula_instruction(question):
        return TaskType.CALCULATION
    if _requests_derived_answer(question):
        return TaskType.CALCULATION
    if re.search(
        r"\bdefined as\b|\bdefine\b.{0,80}\bas\b|"
        r"\bthree[- ]year average\b|\b3 year average\b|"
        r"\bdivide(?:d)? by\b|\bpercentage change\b",
        lower,
    ):
        return TaskType.CALCULATION
    if re.search(r"\bcalculate\b|\bcompute\b", lower):
        normalized = " ".join(lower.split())
        if not any(
            instruction in normalized
            for instruction in _BUNDLED_EXTRACTION_INSTRUCTIONS
        ):
            return TaskType.CALCULATION
    if re.search(
        r"\bwhat\s+(?:is|was)\s+the\s+(?:reported|stated|disclosed)\b"
        r".{0,100}\b(?:rate|ratio|margin|percentage|percent)\b",
        lower,
    ):
        return TaskType.EXTRACTION
    if re.search(r"\bscreen(?:ing)?\b|find (?:all )?(?:companies|stocks)", lower):
        return TaskType.SCREENING
    if re.search(r"\bmonitor(?:ing)?\b|\bcatalyst monitoring\b|keep track of", lower):
        return TaskType.MONITORING
    if re.search(r"\bstable trend\b|\bover time\b|\btrend\b", lower):
        return TaskType.TREND
    if re.search(r"\bcompare\b|\bversus\b|\bvs\.?\b|\bdifference\b|year[- ]over[- ]year|\byoy\b", lower):
        return TaskType.COMPARISON
    if re.search(r"\bwhy\b|\bwhat drove\b|\bexplain\b|\bassess\b|\banaly[sz]e\b|\bhealthy\b", lower):
        return TaskType.QUALITATIVE
    if output.answer_kind in {
        AnswerKind.AMOUNT,
        AnswerKind.PERCENTAGE,
        AnswerKind.RATIO,
        AnswerKind.LIST,
    }:
        return TaskType.EXTRACTION
    if question.strip():
        return TaskType.QUALITATIVE
    return TaskType.UNKNOWN


def _period_for_year(year: int) -> PeriodConstraint:
    return PeriodConstraint(
        label=f"FY{year}",
        kind=PeriodKind.FISCAL_YEAR,
        year=year,
    )


def _periods_for_metric(
    metric: str,
    periods: Sequence[PeriodConstraint],
    formula: Optional[_FormulaRule],
) -> Tuple[PeriodConstraint, ...]:
    resolved_periods = tuple(
        sorted(
            {
                (period.kind.value, period.year, period.quarter, period.label): period
                for period in periods
                if period.year is not None
            }.values(),
            key=lambda period: (period.year or -1, period.label),
        )
    )
    if formula is None or not resolved_periods:
        return tuple(periods)

    # Never turn a calendar year or point-in-time date into a fiscal-year
    # operand. Those semantics require issuer-specific filing-calendar
    # resolution before a full trusted contract can be constructed.
    if any(period.kind is not PeriodKind.FISCAL_YEAR for period in resolved_periods):
        if formula.formula_id in {
            "fixed_asset_turnover",
            "inventory_turnover",
            "return_on_assets",
            "asset_turnover",
            "days_payable_outstanding",
            "cash_conversion_cycle",
        } and metric not in formula.balance_metrics:
            return (resolved_periods[-1],)
        return resolved_periods

    resolved_years = [period.year for period in resolved_periods if period.year is not None]
    target_year = max(resolved_years)
    if metric in formula.balance_metrics:
        if len(resolved_years) == 1:
            resolved_years = [target_year - 1, target_year]
        return tuple(_period_for_year(year) for year in resolved_years)

    # Flow metrics in average-balance formulas normally use the target year.
    if formula.formula_id in {
        "fixed_asset_turnover",
        "inventory_turnover",
        "return_on_assets",
        "asset_turnover",
        "days_payable_outstanding",
        "cash_conversion_cycle",
    }:
        return (_period_for_year(target_year),)

    # Multi-year averages and YoY questions require each explicitly requested year.
    return tuple(_period_for_year(year) for year in resolved_years)


def _source_types_for_hints(source_hints: Sequence[str]) -> Tuple[str, ...]:
    result: List[str] = []
    if any(
        hint in {
            "balance_sheet",
            "income_statement",
            "cash_flow_statement",
            "annual_report",
            "10-k",
            "10-q",
            "8-k",
        }
        for hint in source_hints
    ):
        result.append("sec_filing")
    if "earnings_transcript" in source_hints:
        result.append("earnings_transcript")
    if "investor_presentation" in source_hints:
        result.append("investor_presentation")
    if "market_data" in source_hints:
        result.append("market_data")
    return tuple(result) or ("authorized_corpus",)


def _slug(value: Optional[str]) -> str:
    if not value:
        return "any"
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "any"


def _build_evidence_needs(
    question: str,
    entities: Sequence[EntityConstraint],
    periods: Sequence[PeriodConstraint],
    metrics: Sequence[str],
    source_hints: Sequence[str],
    formula: Optional[_FormulaRule],
) -> Tuple[EvidenceNeed, ...]:
    entity_values: Tuple[Optional[str], ...] = (
        tuple(entity.canonical_name for entity in entities) or (None,)
    )
    metric_values: Tuple[str, ...] = tuple(metrics) or ("topic",)
    source_types = _source_types_for_hints(source_hints)

    raw_needs: List[Tuple[str, Optional[str], Optional[PeriodConstraint], Tuple[str, ...]]] = []
    for metric in metric_values:
        metric_periods = _periods_for_metric(metric, periods, formula)
        period_values: Tuple[Optional[PeriodConstraint], ...] = metric_periods or (None,)
        metric_rule = _METRIC_BY_NAME.get(metric)
        default_statements = metric_rule.statements if metric_rule is not None else ()
        statement_hints = tuple(dict.fromkeys(source_hints or default_statements))
        for entity in entity_values:
            for period in period_values:
                raw_needs.append((metric, entity, period, statement_hints))

    needs: List[EvidenceNeed] = []
    seen = set()
    for metric, entity, period, statement_hints in raw_needs:
        period_label = period.label if period is not None else None
        key = (metric, entity, period_label, statement_hints)
        if key in seen:
            continue
        seen.add(key)

        need_id = ":".join(
            (
                "need",
                _slug(metric),
                _slug(entity),
                _slug(period_label),
            )
        )
        query_parts = [
            entity.replace("_", " ") if entity else "",
            period_label or "",
            metric.replace("_", " ") if metric != "topic" else question,
            " ".join(hint.replace("_", " ") for hint in statement_hints),
        ]
        query = " ".join(part for part in query_parts if part).strip()
        needs.append(
            EvidenceNeed(
                need_id=need_id,
                metric=metric,
                query=query,
                entity=entity,
                period=period_label,
                statement_hints=statement_hints,
                source_types=source_types,
            )
        )
    return tuple(needs)


def compile_finance_query(
    question: str,
    *,
    as_of: Optional[object] = None,
) -> FinanceQueryPlan:
    """Compile one question without network, model, or retrieval side effects.

    Args:
        question: Natural-language financial question.
        as_of: Optional wall-clock anchor.  Relative reporting periods remain
            unresolved until a corpus-aware filing calendar resolves them.

    Returns:
        A deterministic :class:`FinanceQueryPlan` suitable for JSON logging.
    """

    normalized_question = (question or "").strip()
    anchor = _iso_anchor(as_of)
    entities = _extract_entities(normalized_question)
    periods, unresolved = _extract_periods(normalized_question, anchor)
    selected_formula = _select_formula(normalized_question)
    extracted_metrics = _extract_metric_names(normalized_question)
    formula_conflict_reason: Optional[str] = None
    if (
        selected_formula is not None
        and selected_formula.formula_id == "year_over_year_change"
        and len(extracted_metrics) != 1
    ):
        # YoY is a generic operation. It cannot choose a metric for the user or
        # silently pick one of multiple metrics in the question.
        formula_conflict_reason = "formula:ambiguous_metric"
    elif (
        selected_formula is not None
        and not _formula_definition_is_trusted(
            normalized_question,
            selected_formula,
            extracted_metrics,
        )
    ):
        formula_conflict_reason = "formula:explicit_definition_untrusted"
    formula_conflict = formula_conflict_reason is not None
    formula = None if formula_conflict else selected_formula
    output = _align_output_with_formula(
        _compile_output_contract(normalized_question),
        formula,
    )
    if formula is None:
        metrics = extracted_metrics
    elif formula.formula_id == "year_over_year_change":
        # YoY is an operation, not a revenue-specific metric.  Bind it to the
        # metric named by the question (for example, operating income).
        metrics = extracted_metrics or formula.metrics
    else:
        metrics = formula.metrics
    source_hints = _extract_source_hints(normalized_question)
    task_type = _classify_task(
        normalized_question,
        output,
        formula,
        metrics,
        force_calculation=formula_conflict,
    )
    evidence_needs = _build_evidence_needs(
        question=normalized_question,
        entities=entities,
        periods=periods,
        metrics=metrics,
        source_hints=source_hints,
        formula=formula,
    )

    abstentions: List[str] = []
    if not entities:
        abstentions.append("entity")
    if not periods:
        abstentions.append("period")
    if formula_conflict_reason is not None:
        unresolved = (*unresolved, formula_conflict_reason)
    if task_type is TaskType.CALCULATION and formula is None:
        if formula_conflict_reason is None:
            unresolved = (*unresolved, "formula:unresolved")
    if task_type is TaskType.CALCULATION:
        unresolved_periods = tuple(
            f"period_semantics:{period.kind.value}:{period.label}"
            for period in periods
            if period.kind in {PeriodKind.CALENDAR_YEAR, PeriodKind.DATE}
        )
        unresolved = (*unresolved, *unresolved_periods)

    confidence = 1.0
    if unresolved:
        confidence -= 0.2
    if task_type is TaskType.UNKNOWN:
        confidence -= 0.3
    if not metrics:
        confidence -= 0.1

    return FinanceQueryPlan(
        question=normalized_question,
        task_type=task_type,
        entities=entities,
        periods=periods,
        output=output,
        evidence_needs=evidence_needs,
        source_hints=source_hints,
        formula_id=formula.formula_id if formula is not None else None,
        formula_hint=formula.expression if formula is not None else None,
        answer_metric=(
            formula.formula_id
            if formula is not None
            else (metrics[0] if len(metrics) == 1 else None)
        ),
        constraint_abstentions=tuple(abstentions),
        unresolved_constraints=unresolved,
        as_of=anchor,
        confidence=max(0.0, min(1.0, confidence)),
    )
