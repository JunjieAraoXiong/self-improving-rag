"""Typed, evidence-bound execution for financial calculations.

The generative model is allowed to select evidence and propose a small program;
it is never allowed to execute arbitrary code.  This module validates the
program, binds every operand to one exact quote in one retrieved document, and
recomputes the answer with :class:`~decimal.Decimal` arithmetic.

The module is intentionally independent of the agent loop.  The integration
surface is four functions:

``parse_finance_response``
    Extract and validate a ``<finance_program>...</finance_program>`` JSON
    block while returning the user-facing text separately.
``execute_program``
    Evaluate an already validated allowlisted expression tree.
``verify_program``
    Check evidence, anchors, question-bound constants, dimensions, and the
    declared result without using a gold answer.
``render_result``
    Render the executor's value according to the declared output format.

This is not a substitute for structured table-cell or XBRL alignment.  Row and
column labels are required when supplied and logged for audit, but flattened
text can prove only that those anchors and a value occur in the same quote.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from src.metadata_utils import normalize_company_name, parse_filename


MAX_PROGRAM_CHARS = 100_000
MAX_EXPRESSION_DEPTH = 12
MAX_EXPRESSION_NODES = 128
MAX_ABS_VALUE = Decimal("1e30")

_DECIMAL_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_DOC_ID_RE = re.compile(r"^Doc([1-9]\d*)$")
_PROGRAM_BLOCK_RE = re.compile(
    r"<finance_program>\s*(.*?)\s*</finance_program>",
    flags=re.IGNORECASE | re.DOTALL,
)

_SCALE_FACTORS: Dict[str, Decimal] = {
    "one": Decimal("1"),
    "thousand": Decimal("1e3"),
    "million": Decimal("1e6"),
    "billion": Decimal("1e9"),
    "trillion": Decimal("1e12"),
}
_SCALE_ALIASES = {
    "k": "thousand",
    "thousand": "thousand",
    "thousands": "thousand",
    "m": "million",
    "mm": "million",
    "mn": "million",
    "million": "million",
    "millions": "million",
    "b": "billion",
    "bn": "billion",
    "billion": "billion",
    "billions": "billion",
    "t": "trillion",
    "tn": "trillion",
    "trillion": "trillion",
    "trillions": "trillion",
}
_CURRENCY_ALIASES = {
    "us$": "USD",
    "usd": "USD",
    "cad": "CAD",
    "aud": "AUD",
    "nzd": "NZD",
    "sgd": "SGD",
    "hkd": "HKD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "jpy": "JPY",
    "cny": "CNY",
}

_EVIDENCE_VALUE_RE = re.compile(
    r"""
    ^\s*(?P<open>\()?\s*
    (?P<sign_before>[+-])?\s*
    (?P<currency>US\$|USD|CAD|AUD|NZD|SGD|HKD|EUR|GBP|JPY|CNY|[$€£])?\s*
    (?P<sign_after>[+-])?\s*
    (?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)
    (?P<exponent>[eE][+-]?\d+)?\s*
    (?P<unit>
        basis\s+points?|bps|percentage\s+points?|percent(?:age)?|%|
        trillions?|billions?|millions?|thousands?|tn|bn|mn|mm|[tbmk]|x|days?
    )?\s*
    (?P<close>\))?\s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_NUMBER_IN_TEXT_RE = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"
)
_FINANCIAL_TOKEN_RE = re.compile(
    r"""
    (?<![\w.,])
    (?:\(\s*)?
    (?:[+\-−﹣－]\s*)?
    (?:(?:US\$|USD|CAD|AUD|NZD|SGD|HKD|EUR|GBP|JPY|CNY|[$€£])\s*)?
    (?:[+\-−﹣－]\s*)?
    (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+\-]?\d+)?
    (?:\s*(?:basis\s+points?|bps|percentage\s+points?|percent(?:age)?|%|
        trillions?|billions?|millions?|thousands?|tn|bn|mn|mm|[tbmkx]|days?))?
    (?:\s*\))?
    (?![\w.,])
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_QUARTER_RE = re.compile(r"(?<![A-Za-z])Q([1-4])\b", flags=re.IGNORECASE)
_MONTH_DATE_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
    flags=re.IGNORECASE,
)


class Scale(str, Enum):
    ONE = "one"
    THOUSAND = "thousand"
    MILLION = "million"
    BILLION = "billion"
    TRILLION = "trillion"


class UnitKind(str, Enum):
    MONEY = "money"
    NUMBER = "number"
    COUNT = "count"
    SHARES = "shares"
    RATIO = "ratio"
    PERCENT = "percent"
    BASIS_POINTS = "basis_points"
    DAYS = "days"


class Operation(str, Enum):
    REF = "ref"
    CONST = "const"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    AVG = "avg"
    ABS = "abs"
    NEG = "neg"
    PERCENT_CHANGE = "percent_change"


class IssueCode(str, Enum):
    MISSING_PROGRAM = "missing_program"
    SCHEMA_INVALID = "schema_invalid"
    MISSING_OPERAND = "missing_operand"
    MISSING_EVIDENCE = "missing_evidence"
    INVALID_CITATION = "invalid_citation"
    OPERAND_VALUE_MISMATCH = "operand_value_mismatch"
    OPERAND_UNIT_MISMATCH = "operand_unit_mismatch"
    OPERAND_CURRENCY_MISMATCH = "operand_currency_mismatch"
    OPERAND_ENTITY_MISMATCH = "operand_entity_mismatch"
    OPERAND_PERIOD_MISMATCH = "operand_period_mismatch"
    OPERAND_METRIC_MISMATCH = "operand_metric_mismatch"
    CONSTANT_NOT_QUESTION_BOUND = "constant_not_question_bound"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    FORMULA_MISMATCH = "formula_mismatch"
    ARITHMETIC_ERROR = "arithmetic_error"
    RESULT_VALUE_MISMATCH = "result_value_mismatch"
    RESULT_UNIT_MISMATCH = "result_unit_mismatch"
    ANSWER_RESULT_MISMATCH = "answer_result_mismatch"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class AssuranceLevel(str, Enum):
    EVIDENCE_ARITHMETIC = "evidence_arithmetic"
    FULL_CONTRACT = "full_contract"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


def _validate_decimal_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("must be a finite decimal encoded as a JSON string")
    value = value.strip()
    if len(value) > 128 or not _DECIMAL_RE.fullmatch(value):
        raise ValueError("must be a finite decimal encoded as a JSON string")
    exponent_match = re.search(r"[eE]([+-]?\d+)$", value)
    if exponent_match and abs(int(exponent_match.group(1))) > 30:
        raise ValueError("decimal exponent must be between -30 and 30")
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("must be a valid decimal") from exc
    if not decimal_value.is_finite():
        raise ValueError("NaN and infinity are not allowed")
    if abs(decimal_value) > MAX_ABS_VALUE:
        raise ValueError(f"absolute value must not exceed {MAX_ABS_VALUE}")
    return value


class EvidenceRef(_StrictModel):
    """One exact source span supporting one operand."""

    doc_id: StrictStr = Field(pattern=r"^Doc[1-9]\d*$")
    quote: StrictStr = Field(min_length=1, max_length=20_000)
    value_text: StrictStr = Field(min_length=1, max_length=128)
    metric_label: StrictStr = Field(min_length=2, max_length=512)
    period_label: StrictStr = Field(min_length=2, max_length=512)
    row_label: Optional[StrictStr] = Field(default=None, min_length=2, max_length=512)
    column_label: Optional[StrictStr] = Field(
        default=None, min_length=2, max_length=512
    )
    occurrence: StrictInt = Field(default=1, ge=1, le=100)


class FinancialQuantity(_StrictModel):
    """A source-backed financial operand in its displayed scale."""

    id: StrictStr = Field(pattern=r"^[A-Za-z][A-Za-z0-9_:-]{0,127}$")
    value: StrictStr
    currency: Optional[StrictStr] = Field(default=None, pattern=r"^[A-Z]{3}$")
    scale: Scale = Scale.ONE
    unit: UnitKind
    entity: StrictStr = Field(min_length=1, max_length=256)
    period: StrictStr = Field(min_length=1, max_length=128)
    metric: StrictStr = Field(min_length=1, max_length=256)
    evidence: EvidenceRef

    _decimal_value = field_validator("value")(_validate_decimal_text)

    @field_validator("scale", "unit", mode="before")
    @classmethod
    def _reject_coerced_enum_values(cls, value: Any) -> Any:
        if not isinstance(value, (str, Enum)):
            raise ValueError("enum values must be JSON strings")
        return value

    @field_validator("currency")
    @classmethod
    def _canonical_currency(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value != value.upper():
            raise ValueError("currency must be an uppercase ISO-4217 code")
        return value

    @model_validator(mode="after")
    def _validate_quantity_type(self) -> "FinancialQuantity":
        if self.unit == UnitKind.MONEY and self.currency is None:
            raise ValueError("money operands require a currency")
        if self.unit != UnitKind.MONEY and self.currency is not None:
            raise ValueError("currency is only valid for money operands")
        if self.unit in {
            UnitKind.RATIO,
            UnitKind.PERCENT,
            UnitKind.BASIS_POINTS,
            UnitKind.DAYS,
        } and self.scale != Scale.ONE:
            raise ValueError(f"scale is not valid for unit {self.unit.value}")
        if abs(
            _normalized_internal_value(Decimal(self.value), self.unit, self.scale)
        ) > MAX_ABS_VALUE:
            raise ValueError(
                f"scaled absolute value must not exceed {MAX_ABS_VALUE}"
            )
        return self


class RoundingSpec(_StrictModel):
    places: StrictInt = Field(default=2, ge=0, le=12)
    mode: StrictStr = Field(default="half_up", pattern=r"^half_up$")


class AnswerSpec(_StrictModel):
    """The model's declared answer and its requested presentation."""

    value: StrictStr
    currency: Optional[StrictStr] = Field(default=None, pattern=r"^[A-Z]{3}$")
    scale: Scale = Scale.ONE
    unit: UnitKind
    entity: StrictStr = Field(min_length=1, max_length=256)
    period: StrictStr = Field(min_length=1, max_length=128)
    metric: StrictStr = Field(min_length=1, max_length=256)
    rounding: RoundingSpec = Field(default_factory=RoundingSpec)

    _decimal_value = field_validator("value")(_validate_decimal_text)

    @field_validator("scale", "unit", mode="before")
    @classmethod
    def _reject_coerced_enum_values(cls, value: Any) -> Any:
        if not isinstance(value, (str, Enum)):
            raise ValueError("enum values must be JSON strings")
        return value

    @field_validator("currency")
    @classmethod
    def _canonical_currency(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value != value.upper():
            raise ValueError("currency must be an uppercase ISO-4217 code")
        return value

    @model_validator(mode="after")
    def _validate_answer_type(self) -> "AnswerSpec":
        if self.unit == UnitKind.MONEY and self.currency is None:
            raise ValueError("money answers require a currency")
        if self.unit != UnitKind.MONEY and self.currency is not None:
            raise ValueError("currency is only valid for money answers")
        if self.unit in {
            UnitKind.RATIO,
            UnitKind.PERCENT,
            UnitKind.BASIS_POINTS,
            UnitKind.DAYS,
        } and self.scale != Scale.ONE:
            raise ValueError(f"scale is not valid for unit {self.unit.value}")
        if abs(
            _normalized_internal_value(Decimal(self.value), self.unit, self.scale)
        ) > MAX_ABS_VALUE:
            raise ValueError(
                f"scaled absolute value must not exceed {MAX_ABS_VALUE}"
            )
        return self


class Expression(_StrictModel):
    """One node in the safe arithmetic expression tree."""

    op: Operation
    args: Tuple["Expression", ...] = ()
    operand_id: Optional[StrictStr] = Field(
        default=None, pattern=r"^[A-Za-z][A-Za-z0-9_:-]{0,127}$"
    )
    value: Optional[StrictStr] = None
    unit: Optional[UnitKind] = None
    source_text: Optional[StrictStr] = Field(default=None, min_length=1, max_length=512)

    @field_validator("value")
    @classmethod
    def _validate_optional_decimal(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_decimal_text(value)

    @field_validator("op", "unit", mode="before")
    @classmethod
    def _reject_coerced_enum_values(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, (str, Enum)):
            raise ValueError("enum values must be JSON strings")
        return value

    @model_validator(mode="after")
    def _validate_shape(self) -> "Expression":
        if self.op == Operation.REF:
            if self.operand_id is None:
                raise ValueError("ref requires operand_id")
            if self.args or self.value is not None or self.unit is not None or self.source_text:
                raise ValueError("ref accepts only operand_id")
            return self

        if self.op == Operation.CONST:
            if self.value is None or self.unit is None or self.source_text is None:
                raise ValueError("const requires value, unit, and source_text")
            if self.unit not in {UnitKind.NUMBER, UnitKind.DAYS}:
                raise ValueError("constants may use only number or days units")
            if self.args or self.operand_id is not None:
                raise ValueError("const does not accept args or operand_id")
            return self

        if self.operand_id is not None or self.value is not None or self.unit is not None or self.source_text is not None:
            raise ValueError("operator nodes accept only op and args")

        required_arity = {
            Operation.ABS: 1,
            Operation.NEG: 1,
            Operation.ADD: 2,
            Operation.SUB: 2,
            Operation.MUL: 2,
            Operation.DIV: 2,
            Operation.PERCENT_CHANGE: 2,
        }
        if self.op in required_arity and len(self.args) != required_arity[self.op]:
            raise ValueError(
                f"{self.op.value} requires exactly {required_arity[self.op]} args"
            )
        if self.op == Operation.AVG and len(self.args) < 2:
            raise ValueError("avg requires at least two args")
        return self


Expression.model_rebuild()


def _walk_expression(expression: Expression) -> Iterable[Expression]:
    yield expression
    for child in expression.args:
        yield from _walk_expression(child)


def _expression_size(expression: Expression, depth: int = 1) -> Tuple[int, int]:
    count = 1
    maximum_depth = depth
    for child in expression.args:
        child_count, child_depth = _expression_size(child, depth + 1)
        count += child_count
        maximum_depth = max(maximum_depth, child_depth)
    return count, maximum_depth


class FinanceProgram(_StrictModel):
    """A complete auditable financial calculation."""

    schema_version: StrictStr = Field(default="1.0", pattern=r"^1\.0$")
    answer: AnswerSpec
    operands: Tuple[FinancialQuantity, ...] = Field(min_length=1, max_length=64)
    expression: Expression

    @model_validator(mode="after")
    def _validate_graph(self) -> "FinanceProgram":
        operand_ids = [operand.id for operand in self.operands]
        if len(operand_ids) != len(set(operand_ids)):
            raise ValueError("operand ids must be unique")

        references = [
            node.operand_id
            for node in _walk_expression(self.expression)
            if node.op == Operation.REF
        ]
        missing = sorted(set(references) - set(operand_ids))
        if missing:
            raise ValueError(f"expression references unknown operands: {missing}")
        unused = sorted(set(operand_ids) - set(references))
        if unused:
            raise ValueError(f"unused operands are not allowed: {unused}")

        node_count, depth = _expression_size(self.expression)
        if node_count > MAX_EXPRESSION_NODES:
            raise ValueError(
                f"expression exceeds {MAX_EXPRESSION_NODES} nodes ({node_count})"
            )
        if depth > MAX_EXPRESSION_DEPTH:
            raise ValueError(
                f"expression exceeds depth {MAX_EXPRESSION_DEPTH} ({depth})"
            )
        return self


class FinanceOperandSpec(_StrictModel):
    """Trusted identity of one operand required by the compiled question."""

    id: StrictStr = Field(pattern=r"^[A-Za-z][A-Za-z0-9_:-]{0,127}$")
    entity: StrictStr = Field(min_length=1, max_length=256)
    period: StrictStr = Field(min_length=1, max_length=128)
    metric: StrictStr = Field(min_length=1, max_length=256)
    unit: UnitKind
    currency: Optional[StrictStr] = Field(default=None, pattern=r"^[A-Z]{3}$")
    scale: Optional[Scale] = None

    @model_validator(mode="after")
    def _validate_quantity_contract(self) -> "FinanceOperandSpec":
        if self.unit != UnitKind.MONEY and self.currency is not None:
            raise ValueError("currency is only valid for money operand contracts")
        return self


class FinanceQuestionSpec(_StrictModel):
    """Trusted semantic contract produced before answer generation.

    Free-form question text is still used to bind explicit constants, but it is
    not a reliable source of structured identity.  A caller that has parsed the
    question should pass this contract so the verifier can reject an answer for
    the wrong entity, period, metric, or formula even when its arithmetic is
    internally consistent.
    """

    entity: StrictStr = Field(min_length=1, max_length=256)
    period: StrictStr = Field(min_length=1, max_length=128)
    metric: StrictStr = Field(min_length=1, max_length=256)
    unit: UnitKind
    currency: Optional[StrictStr] = Field(default=None, pattern=r"^[A-Z]{3}$")
    scale: Scale = Scale.ONE
    rounding: RoundingSpec = Field(default_factory=RoundingSpec)
    expression: Expression
    operands: Tuple[FinanceOperandSpec, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def _validate_operand_contract(self) -> "FinanceQuestionSpec":
        ids = [operand.id for operand in self.operands]
        if len(ids) != len(set(ids)):
            raise ValueError("trusted operand ids must be unique")
        if ids:
            references = {
                node.operand_id
                for node in _walk_expression(self.expression)
                if node.op == Operation.REF
            }
            if set(ids) != references:
                raise ValueError(
                    "trusted operand ids must exactly match expression references"
                )
        return self

    @field_validator("scale", "unit", mode="before")
    @classmethod
    def _reject_coerced_enum_values(cls, value: Any) -> Any:
        if not isinstance(value, (str, Enum)):
            raise ValueError("enum values must be JSON strings")
        return value

    @model_validator(mode="after")
    def _validate_output_contract(self) -> "FinanceQuestionSpec":
        if self.unit == UnitKind.MONEY and self.currency is None:
            raise ValueError("money output contracts require a currency")
        if self.unit != UnitKind.MONEY and self.currency is not None:
            raise ValueError("currency is only valid for money output contracts")
        if self.unit in {
            UnitKind.RATIO,
            UnitKind.PERCENT,
            UnitKind.BASIS_POINTS,
            UnitKind.DAYS,
        } and self.scale != Scale.ONE:
            raise ValueError(f"scale is not valid for unit {self.unit.value}")
        return self


@dataclass(frozen=True)
class VerificationIssue:
    code: IssueCode
    message: str
    operand_id: Optional[str] = None
    metric: Optional[str] = None
    period: Optional[str] = None
    doc_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: value.value if isinstance(value, Enum) else value
            for key, value in {
                "code": self.code,
                "message": self.message,
                "operand_id": self.operand_id,
                "metric": self.metric,
                "period": self.period,
                "doc_id": self.doc_id,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class ParsedFinanceResponse:
    answer_text: str
    program: Optional[FinanceProgram]
    issues: Tuple[VerificationIssue, ...] = ()
    raw_program: Optional[str] = None

    @property
    def passed(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class ExecutionStep:
    path: str
    op: str
    value: str
    dimensions: Tuple[Tuple[str, int], ...]
    semantic_unit: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "op": self.op,
            "value": self.value,
            "dimensions": dict(self.dimensions),
            "semantic_unit": self.semantic_unit,
        }


@dataclass(frozen=True)
class ExecutionResult:
    value: Decimal
    dimensions: Tuple[Tuple[str, int], ...]
    semantic_unit: UnitKind
    trace: Tuple[ExecutionStep, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": str(self.value),
            "dimensions": dict(self.dimensions),
            "semantic_unit": self.semantic_unit.value,
            "trace": [step.to_dict() for step in self.trace],
        }


@dataclass(frozen=True)
class ProgramVerificationResult:
    passed: bool
    issues: Tuple[VerificationIssue, ...]
    execution: Optional[ExecutionResult]
    rendered_answer: Optional[str]
    evidence_coverage: float
    assurance_level: AssuranceLevel = AssuranceLevel.EVIDENCE_ARITHMETIC

    @property
    def fully_verified(self) -> bool:
        return self.passed and self.assurance_level == AssuranceLevel.FULL_CONTRACT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "execution": self.execution.to_dict() if self.execution else None,
            "rendered_answer": self.rendered_answer,
            "evidence_coverage": self.evidence_coverage,
            "assurance_level": self.assurance_level.value,
            "fully_verified": self.fully_verified,
        }


class ProgramExecutionError(ValueError):
    """A deterministic, machine-routable execution failure."""

    def __init__(
        self,
        code: IssueCode,
        message: str,
        *,
        operand_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.operand_id = operand_id


@dataclass(frozen=True)
class _CalcQuantity:
    value: Decimal
    dimensions: Tuple[Tuple[str, int], ...]
    semantic_unit: UnitKind


@dataclass(frozen=True)
class _EvidenceValue:
    value: Decimal
    unit_token: str
    currency: Optional[str]
    scale: Scale
    currency_explicit: bool = False
    currency_ambiguous: bool = False
    scale_ambiguous: bool = False


def _issue_from_validation_error(error: ValidationError) -> Tuple[VerificationIssue, ...]:
    issues = []
    seen = set()
    for detail in error.errors(include_url=False):
        location = tuple(str(item) for item in detail.get("loc", ()))
        error_type = detail.get("type", "")
        message = detail.get("msg", "Invalid finance program")

        if location and location[-1] == "op" and error_type == "enum":
            code = IssueCode.UNSUPPORTED_OPERATOR
        elif "evidence" in location and error_type == "missing":
            code = IssueCode.MISSING_EVIDENCE
        elif "operands" in location and error_type == "missing":
            code = IssueCode.MISSING_OPERAND
        else:
            code = IssueCode.SCHEMA_INVALID

        key = (code, location, message)
        if key in seen:
            continue
        seen.add(key)
        issues.append(
            VerificationIssue(
                code=code,
                message=f"{'.'.join(location) or 'program'}: {message}",
            )
        )
    return tuple(issues) or (
        VerificationIssue(IssueCode.SCHEMA_INVALID, "Invalid finance program"),
    )


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def parse_finance_response(
    response: str,
    *,
    require_program: bool = False,
) -> ParsedFinanceResponse:
    """Extract one strict finance-program block from an LLM response.

    The returned ``answer_text`` never contains the machine-readable block.
    Zero blocks are valid unless ``require_program`` is true.  Multiple or
    malformed blocks are rejected instead of choosing one ambiguously.
    """

    response = response or ""
    matches = list(_PROGRAM_BLOCK_RE.finditer(response))
    has_any_tag = bool(
        re.search(r"</?finance_program\b", response, flags=re.IGNORECASE)
    )

    if not matches:
        issue = None
        if has_any_tag:
            issue = VerificationIssue(
                IssueCode.SCHEMA_INVALID,
                "Malformed <finance_program> block",
            )
        elif require_program:
            issue = VerificationIssue(
                IssueCode.MISSING_PROGRAM,
                "A finance program is required for this answer",
            )
        return ParsedFinanceResponse(
            answer_text=response.strip(),
            program=None,
            issues=(issue,) if issue else (),
        )

    if len(matches) != 1:
        answer_text = _PROGRAM_BLOCK_RE.sub("", response).strip()
        return ParsedFinanceResponse(
            answer_text=answer_text,
            program=None,
            issues=(
                VerificationIssue(
                    IssueCode.SCHEMA_INVALID,
                    "Exactly one <finance_program> block is allowed",
                ),
            ),
        )

    match = matches[0]
    raw_program = match.group(1).strip()
    answer_text = (response[: match.start()] + response[match.end() :]).strip()
    if re.search(r"</?finance_program\b", answer_text, flags=re.IGNORECASE):
        return ParsedFinanceResponse(
            answer_text=answer_text,
            program=None,
            raw_program=raw_program,
            issues=(
                VerificationIssue(
                    IssueCode.SCHEMA_INVALID,
                    "Unmatched or nested <finance_program> tag is not allowed",
                ),
            ),
        )
    if len(raw_program) > MAX_PROGRAM_CHARS:
        return ParsedFinanceResponse(
            answer_text=answer_text,
            program=None,
            raw_program=None,
            issues=(
                VerificationIssue(
                    IssueCode.SCHEMA_INVALID,
                    f"Finance program exceeds {MAX_PROGRAM_CHARS} characters",
                ),
            ),
        )

    try:
        payload = json.loads(
            raw_program,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, _DuplicateJSONKey, ValueError) as exc:
        return ParsedFinanceResponse(
            answer_text=answer_text,
            program=None,
            raw_program=raw_program,
            issues=(
                VerificationIssue(
                    IssueCode.SCHEMA_INVALID,
                    f"Invalid finance-program JSON: {exc}",
                ),
            ),
        )

    try:
        # The strict model rejects coercion and every undeclared field.
        program = FinanceProgram.model_validate(payload)
    except ValidationError as exc:
        return ParsedFinanceResponse(
            answer_text=answer_text,
            program=None,
            raw_program=raw_program,
            issues=_issue_from_validation_error(exc),
        )

    return ParsedFinanceResponse(
        answer_text=answer_text,
        program=program,
        raw_program=raw_program,
    )


def _dimensions_for(unit: UnitKind, currency: Optional[str]) -> Tuple[Tuple[str, int], ...]:
    if unit == UnitKind.MONEY:
        return ((f"money:{currency}", 1),)
    if unit == UnitKind.DAYS:
        return (("days", 1),)
    if unit == UnitKind.SHARES:
        return (("shares", 1),)
    if unit == UnitKind.COUNT:
        return (("count", 1),)
    return ()


def _normalized_internal_value(
    value: Decimal,
    unit: UnitKind,
    scale: Scale,
) -> Decimal:
    if unit == UnitKind.PERCENT:
        return value / Decimal("100")
    if unit == UnitKind.BASIS_POINTS:
        return value / Decimal("10000")
    return value * _SCALE_FACTORS[scale.value]


def _calc_from_operand(operand: FinancialQuantity) -> _CalcQuantity:
    value = _normalized_internal_value(
        Decimal(operand.value), operand.unit, operand.scale
    )
    semantic_unit = (
        UnitKind.PERCENT
        if operand.unit == UnitKind.BASIS_POINTS
        else operand.unit
    )
    return _CalcQuantity(
        value=value,
        dimensions=_dimensions_for(operand.unit, operand.currency),
        semantic_unit=semantic_unit,
    )


def _combine_dimensions(
    left: Tuple[Tuple[str, int], ...],
    right: Tuple[Tuple[str, int], ...],
    sign: int,
) -> Tuple[Tuple[str, int], ...]:
    combined = dict(left)
    for key, exponent in right:
        combined[key] = combined.get(key, 0) + sign * exponent
        if combined[key] == 0:
            del combined[key]
    return tuple(sorted(combined.items()))


def _semantic_for_dimensions(
    dimensions: Tuple[Tuple[str, int], ...],
    fallback: UnitKind = UnitKind.NUMBER,
) -> UnitKind:
    if not dimensions:
        return fallback
    if dimensions == (("days", 1),):
        return UnitKind.DAYS
    if dimensions == (("shares", 1),):
        return UnitKind.SHARES
    if dimensions == (("count", 1),):
        return UnitKind.COUNT
    if len(dimensions) == 1 and dimensions[0][0].startswith("money:") and dimensions[0][1] == 1:
        return UnitKind.MONEY
    return UnitKind.NUMBER


def _dimensionless_kinds_compatible(left: UnitKind, right: UnitKind) -> bool:
    if left == right:
        return True
    return {left, right} <= {UnitKind.RATIO, UnitKind.PERCENT}


def execute_program(program: FinanceProgram) -> ExecutionResult:
    """Evaluate a validated program with bounded, allowlisted Decimal math."""

    operands = {operand.id: _calc_from_operand(operand) for operand in program.operands}
    trace = []

    def evaluate(expression: Expression, path: str) -> _CalcQuantity:
        if expression.op == Operation.REF:
            try:
                result = operands[expression.operand_id]
            except KeyError as exc:  # Defensive; FinanceProgram validates refs.
                raise ProgramExecutionError(
                    IssueCode.MISSING_OPERAND,
                    f"Unknown operand {expression.operand_id!r}",
                    operand_id=expression.operand_id,
                ) from exc
        elif expression.op == Operation.CONST:
            unit = expression.unit
            result = _CalcQuantity(
                value=Decimal(expression.value),
                dimensions=_dimensions_for(unit, None),
                semantic_unit=unit,
            )
        else:
            children = [
                evaluate(child, f"{path}.{index}")
                for index, child in enumerate(expression.args)
            ]
            try:
                if expression.op in {Operation.ADD, Operation.SUB, Operation.AVG}:
                    first = children[0]
                    if any(child.dimensions != first.dimensions for child in children[1:]):
                        raise ProgramExecutionError(
                            IssueCode.OPERAND_UNIT_MISMATCH,
                            f"{expression.op.value} requires identical dimensions",
                        )
                    if not first.dimensions and any(
                        not _dimensionless_kinds_compatible(
                            first.semantic_unit, child.semantic_unit
                        )
                        for child in children[1:]
                    ):
                        raise ProgramExecutionError(
                            IssueCode.OPERAND_UNIT_MISMATCH,
                            f"{expression.op.value} mixes incompatible dimensionless units",
                        )
                    if expression.op == Operation.ADD:
                        value = first.value + children[1].value
                    elif expression.op == Operation.SUB:
                        value = first.value - children[1].value
                    else:
                        value = sum((child.value for child in children), Decimal("0")) / Decimal(len(children))
                    result = _CalcQuantity(value, first.dimensions, first.semantic_unit)
                elif expression.op in {Operation.ABS, Operation.NEG}:
                    child = children[0]
                    value = abs(child.value) if expression.op == Operation.ABS else -child.value
                    result = _CalcQuantity(value, child.dimensions, child.semantic_unit)
                elif expression.op == Operation.MUL:
                    left, right = children
                    dimensions = _combine_dimensions(left.dimensions, right.dimensions, 1)
                    if not dimensions:
                        if left.semantic_unit == UnitKind.NUMBER:
                            semantic = right.semantic_unit
                        elif right.semantic_unit == UnitKind.NUMBER:
                            semantic = left.semantic_unit
                        elif left.semantic_unit == right.semantic_unit:
                            semantic = left.semantic_unit
                        else:
                            semantic = UnitKind.RATIO
                    else:
                        semantic = _semantic_for_dimensions(dimensions)
                    result = _CalcQuantity(left.value * right.value, dimensions, semantic)
                elif expression.op == Operation.DIV:
                    left, right = children
                    if right.value == 0:
                        raise ProgramExecutionError(
                            IssueCode.ARITHMETIC_ERROR,
                            "Division by zero",
                        )
                    dimensions = _combine_dimensions(left.dimensions, right.dimensions, -1)
                    if not dimensions:
                        semantic = UnitKind.RATIO
                    elif not right.dimensions:
                        semantic = left.semantic_unit
                    else:
                        semantic = _semantic_for_dimensions(dimensions)
                    result = _CalcQuantity(left.value / right.value, dimensions, semantic)
                elif expression.op == Operation.PERCENT_CHANGE:
                    current, previous = children
                    if current.dimensions != previous.dimensions:
                        raise ProgramExecutionError(
                            IssueCode.OPERAND_UNIT_MISMATCH,
                            "percent_change requires identical dimensions",
                        )
                    if (
                        not current.dimensions
                        and current.semantic_unit != previous.semantic_unit
                    ):
                        raise ProgramExecutionError(
                            IssueCode.OPERAND_UNIT_MISMATCH,
                            "percent_change requires identical semantic units",
                        )
                    if previous.value == 0:
                        raise ProgramExecutionError(
                            IssueCode.ARITHMETIC_ERROR,
                            "Percent change from zero is undefined",
                        )
                    result = _CalcQuantity(
                        (current.value - previous.value) / abs(previous.value),
                        (),
                        UnitKind.PERCENT,
                    )
                else:  # pragma: no cover - Operation enum and schema are closed.
                    raise ProgramExecutionError(
                        IssueCode.UNSUPPORTED_OPERATOR,
                        f"Unsupported operator: {expression.op}",
                    )
            except (InvalidOperation, OverflowError) as exc:
                raise ProgramExecutionError(
                    IssueCode.ARITHMETIC_ERROR,
                    f"Decimal arithmetic failed at {path}: {exc}",
                ) from exc

        if not result.value.is_finite() or abs(result.value) > MAX_ABS_VALUE:
            raise ProgramExecutionError(
                IssueCode.ARITHMETIC_ERROR,
                f"Result at {path} is non-finite or exceeds {MAX_ABS_VALUE}",
                operand_id=expression.operand_id,
            )
        trace.append(
            ExecutionStep(
                path=path,
                op=expression.op.value,
                value=str(result.value),
                dimensions=result.dimensions,
                semantic_unit=result.semantic_unit.value,
            )
        )
        return result

    with localcontext() as context:
        context.prec = 50
        final = evaluate(program.expression, "root")

    return ExecutionResult(
        value=final.value,
        dimensions=final.dimensions,
        semantic_unit=final.semantic_unit,
        trace=tuple(trace),
    )


def _normalize_anchor(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(text.split())


def _anchor_pattern(anchor: str) -> re.Pattern[str]:
    parts = re.split(r"\s+", anchor.strip())
    body = r"\s+".join(re.escape(part) for part in parts if part)
    if not body:
        return re.compile(r"(?!)")
    prefix = r"(?<!\w)" if anchor.strip()[0].isalnum() else ""
    suffix = r"(?!\w)" if anchor.strip()[-1].isalnum() else ""
    return re.compile(f"{prefix}{body}{suffix}", flags=re.IGNORECASE)


def _anchor_matches(quote: str, anchor: str) -> Tuple[re.Match[str], ...]:
    return tuple(_anchor_pattern(anchor).finditer(quote)) if anchor else ()


def _quote_has_anchor(quote: str, anchor: str) -> bool:
    return bool(_anchor_matches(quote, anchor))


def _infer_quote_scale(quote: str) -> Tuple[Scale, bool]:
    tokens = re.findall(
        r"(?:amounts?|dollars?|[$])?\s*(?:are\s+)?in\s+"
        r"(thousands?|millions?|billions?|trillions?)\b|"
        r"\(\s*(thousands?|millions?|billions?|trillions?)\s*\)",
        quote or "",
        flags=re.IGNORECASE,
    )
    scales = {
        _SCALE_ALIASES[(first or second).casefold()]
        for first, second in tokens
        if first or second
    }
    if len(scales) == 1:
        return Scale(next(iter(scales))), False
    return Scale.ONE, len(scales) > 1


def _infer_quote_currency(quote: str) -> Tuple[Optional[str], bool]:
    currencies = set()
    lower = (quote or "").casefold()
    explicit_patterns = {
        "USD": r"\busd\b|\bu\.?s\.?\s+dollars?\b",
        "CAD": r"\bcad\b|\bcanadian\s+dollars?\b",
        "AUD": r"\baud\b|\baustralian\s+dollars?\b",
        "NZD": r"\bnzd\b|\bnew\s+zealand\s+dollars?\b",
        "SGD": r"\bsgd\b|\bsingapore\s+dollars?\b",
        "HKD": r"\bhkd\b|\bhong\s+kong\s+dollars?\b",
        "JPY": r"\bjpy\b|\bjapanese\s+yen\b",
        "CNY": r"\bcny\b|\bchinese\s+(?:yuan|renminbi)\b",
    }
    for currency, pattern in explicit_patterns.items():
        if re.search(pattern, lower, flags=re.IGNORECASE):
            currencies.add(currency)
    if "€" in quote or re.search(r"\beur\b", lower):
        currencies.add("EUR")
    if "£" in quote or re.search(r"\bgbp\b", lower):
        currencies.add("GBP")
    if len(currencies) == 1:
        return next(iter(currencies)), False
    generic_dollar = "$" in quote or bool(re.search(r"\bdollars?\b", lower))
    return None, len(currencies) > 1 or generic_dollar


def _document_currency(doc: Any) -> Optional[str]:
    metadata = getattr(doc, "metadata", {}) or {}
    value = metadata.get("currency")
    if isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value):
        return value
    return None


def _parse_evidence_value(
    value_text: str,
    quote: str,
    *,
    document_currency: Optional[str] = None,
) -> _EvidenceValue:
    normalized = unicodedata.normalize("NFKC", value_text).translate(
        str.maketrans({"−": "-", "﹣": "-", "－": "-"})
    )
    match = _EVIDENCE_VALUE_RE.fullmatch(normalized)
    if not match:
        raise ValueError(f"Unsupported evidence value format: {value_text!r}")

    number = Decimal(match.group("number").replace(",", ""))
    exponent = match.group("exponent") or ""
    if exponent:
        number *= Decimal(f"1{exponent}")
    if match.group("sign_before") == "-" or match.group("sign_after") == "-":
        number = -abs(number)
    if match.group("open") and match.group("close"):
        number = -abs(number)

    currency_token = (match.group("currency") or "").casefold()
    token_currency = _CURRENCY_ALIASES.get(currency_token)
    quote_currency, quote_ambiguous = _infer_quote_currency(quote)
    currency_candidates = {
        candidate
        for candidate in (token_currency, quote_currency, document_currency)
        if candidate
    }
    currency_ambiguous = len(currency_candidates) > 1 or (
        not currency_candidates and (quote_ambiguous or currency_token == "$")
    )
    currency = (
        next(iter(currency_candidates))
        if len(currency_candidates) == 1 and not currency_ambiguous
        else None
    )

    unit_token = " ".join((match.group("unit") or "").casefold().split())
    scale = Scale.ONE
    scale_ambiguous = False
    if unit_token in _SCALE_ALIASES:
        scale = Scale(_SCALE_ALIASES[unit_token])
    elif not unit_token:
        inferred_scale, ambiguous_scale = _infer_quote_scale(quote)
        if not ambiguous_scale:
            scale = inferred_scale
        else:
            scale_ambiguous = True

    return _EvidenceValue(
        value=number,
        unit_token=unit_token,
        currency=currency,
        scale=scale,
        currency_explicit=bool(currency_token),
        currency_ambiguous=currency_ambiguous,
        scale_ambiguous=scale_ambiguous,
    )


def _value_occurrence_spans(quote: str, value_text: str) -> Tuple[Tuple[int, int], ...]:
    """Locate complete financial tokens, not sign/unit/currency substrings."""

    spans = []
    for match in re.finditer(re.escape(value_text), quote):
        start, end = match.span()
        if start > 0 and re.match(r"[\w,.]", quote[start - 1]):
            continue
        if end < len(quote) and re.match(r"[\w,.]", quote[end]):
            continue

        prefix = quote[:start]
        suffix = quote[end:]
        if re.search(
            r"(?:US\$|USD|CAD|AUD|NZD|SGD|HKD|EUR|GBP|JPY|CNY|[$€£]|[+\-−﹣－])\s*$",
            prefix,
            flags=re.IGNORECASE,
        ):
            continue
        if prefix.rstrip().endswith("(") and suffix.lstrip().startswith(")"):
            continue
        if re.match(
            r"\s*(?:%|basis\s+points?\b|bps\b|percentage\s+points?\b|"
            r"percent(?:age)?\b|trillions?\b|billions?\b|millions?\b|"
            r"thousands?\b|tn\b|bn\b|mn\b|mm\b|[tbmkx]\b|days?\b)",
            suffix,
            flags=re.IGNORECASE,
        ):
            continue
        spans.append((start, end))
    return tuple(spans)


def _span_overlaps_label(
    quote: str,
    span: Tuple[int, int],
    label: str,
) -> bool:
    for match in _anchor_matches(quote, label):
        label_start, label_end = match.span()
        if span[0] < label_end and label_start < span[1]:
            return True
    return False


_METRIC_STOPWORDS = frozenset(
    {
        "and",
        "amount",
        "amounts",
        "balance",
        "by",
        "from",
        "of",
        "reported",
        "the",
        "total",
    }
)

_METRIC_QUALIFIERS = frozenset(
    {
        "adjusted",
        "consolidated",
        "deferred",
        "gaap",
        "organic",
        "other",
        "segment",
        "unadjusted",
        "unearned",
    }
)

_METRIC_ALIASES = {
    "revenue": "revenue",
    "netsale": "revenue",
    "operatingcashflow": "operating_cash_flow",
    "netcashprovidedoperatingactivity": "operating_cash_flow",
    "netcashoperatingactivity": "operating_cash_flow",
    "capex": "capital_expenditures",
    "capitalexpenditure": "capital_expenditures",
    "purchasepropertyplantequipment": "capital_expenditures",
    "purchasepropertyequipment": "capital_expenditures",
    "purchaseppe": "capital_expenditures",
    "netincome": "net_income_including_nci",
    "netincomeincludingnoncontrollinginterest": "net_income_including_nci",
    "netincomeincludingnci": "net_income_including_nci",
    "netincomeinclnci": "net_income_including_nci",
    "propertyplantequipment": "ppe_net",
    "propertyplantequipmentnet": "ppe_net",
    "propertyplantequipmentppenet": "ppe_net",
    "ppenet": "ppe_net",
}


def _metric_tokens(text: str) -> Tuple[str, ...]:
    tokens = re.findall(r"[a-z0-9]+", _normalize_anchor(text).replace("_", " "))
    normalized = []
    for token in tokens:
        if token in _METRIC_STOPWORDS or token.isdigit():
            continue
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            token = token[:-1]
        normalized.append(token)
    return tuple(normalized)


def _metric_signature(text: str) -> Tuple[str, frozenset[str]]:
    """Return a canonical FinanceBench metric plus semantic qualifiers."""

    raw_tokens = re.findall(
        r"[a-z0-9]+",
        _normalize_anchor(text).replace("_", " "),
    )
    qualifiers = set()
    base_tokens = []
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        if token == "non" and index + 1 < len(raw_tokens) and raw_tokens[index + 1] == "gaap":
            qualifiers.add("non_gaap")
            index += 2
            continue
        if token in _METRIC_QUALIFIERS:
            qualifiers.add(token)
            index += 1
            continue
        if token not in _METRIC_STOPWORDS and not token.isdigit():
            if token.endswith("ies") and len(token) > 4:
                token = f"{token[:-3]}y"
            elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
                token = token[:-1]
            base_tokens.append(token)
        index += 1

    compact = "".join(base_tokens)
    canonical = _METRIC_ALIASES.get(compact, " ".join(base_tokens))
    return canonical, frozenset(qualifiers)


def _metric_claims_equal(left: str, right: str) -> bool:
    left_signature = _metric_signature(left)
    right_signature = _metric_signature(right)
    return bool(left_signature[0]) and left_signature == right_signature


def _metric_binding_consistent(
    metric: str,
    metric_label: str,
    period_label: str,
) -> bool:
    def without_period(text: str) -> str:
        return re.sub(
            re.escape(period_label),
            " ",
            text,
            flags=re.IGNORECASE,
        )

    declared_text = without_period(metric)
    anchored_text = without_period(metric_label)
    # A metric anchor may not swallow a data value to move the proximity cursor.
    if re.search(r"\d", anchored_text):
        return False
    return _metric_claims_equal(declared_text, anchored_text)


def _label_occurrence_spans(quote: str, labels: Iterable[Optional[str]]) -> Tuple[Tuple[int, int], ...]:
    spans = []
    for label in labels:
        if not label:
            continue
        spans.extend(match.span() for match in _anchor_matches(quote, label))
    return tuple(spans)


def _spans_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _markdown_cells(line: str) -> Optional[Tuple[Tuple[str, Tuple[int, int]], ...]]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    delimiters = [match.start() for match in re.finditer(r"\|", line)]
    if len(delimiters) < 3:
        return None

    cells = []
    for start_delimiter, end_delimiter in zip(delimiters, delimiters[1:]):
        raw_start = start_delimiter + 1
        raw_end = end_delimiter
        raw = line[raw_start:raw_end]
        left_padding = len(raw) - len(raw.lstrip())
        right_padding = len(raw) - len(raw.rstrip())
        cell_start = raw_start + left_padding
        cell_end = raw_end - right_padding
        cells.append((raw.strip(), (cell_start, cell_end)))
    return tuple(cells)


def _markdown_table_binding(
    content: str,
    quote: str,
    evidence: EvidenceRef,
    selected_span: Tuple[int, int],
) -> Optional[bool]:
    """Bind a Markdown row value to a unique header column conservatively."""

    row_cells = _markdown_cells(quote)
    if row_cells is None:
        return None
    if not evidence.row_label or not evidence.column_label:
        return False

    quote_start = content.find(quote)
    preceding_lines = [
        line
        for line in content[:quote_start].splitlines()
        if line.strip()
    ]
    separator_index = None
    separator_cells = None
    for index in range(len(preceding_lines) - 1, 0, -1):
        candidate_cells = _markdown_cells(preceding_lines[index])
        if candidate_cells and all(
            re.fullmatch(r":?-{3,}:?", cell_text.replace(" ", ""))
            for cell_text, _ in candidate_cells
        ):
            separator_index = index
            separator_cells = candidate_cells
            break
    if separator_index is None or separator_cells is None:
        return False
    header_cells = _markdown_cells(preceding_lines[separator_index - 1])
    if header_cells is None:
        return False
    if len(header_cells) != len(row_cells) or len(separator_cells) != len(row_cells):
        return False
    intervening_rows = preceding_lines[separator_index + 1 :]
    if any(
        (cells := _markdown_cells(line)) is None or len(cells) != len(row_cells)
        for line in intervening_rows
    ):
        return False

    row_label_matches = [
        index
        for index, (cell_text, _) in enumerate(row_cells)
        if _normalize_anchor(cell_text) == _normalize_anchor(evidence.row_label)
    ]
    column_matches = [
        index
        for index, (cell_text, _) in enumerate(header_cells)
        if _normalize_anchor(cell_text) == _normalize_anchor(evidence.column_label)
    ]
    if len(row_label_matches) != 1 or len(column_matches) != 1:
        return False

    row_label_index = row_label_matches[0]
    value_index = column_matches[0]
    if row_label_index == value_index:
        return False
    value_cell, value_cell_span = row_cells[value_index]
    if not (
        value_cell_span[0] <= selected_span[0]
        and selected_span[1] <= value_cell_span[1]
    ):
        return False
    if _normalize_anchor(value_cell) != _normalize_anchor(evidence.value_text):
        return False
    return _normalize_anchor(evidence.period_label) == _normalize_anchor(
        evidence.column_label
    )


def _value_is_locally_associated(
    quote: str,
    selected_span: Tuple[int, int],
    evidence: EvidenceRef,
) -> bool:
    """Require the value to be the nearest data token after its metric anchor.

    This deliberately fails closed for flattened multi-column rows.  Such rows
    need structured cell coordinates rather than a model-selected occurrence.
    """

    # Proximity is always measured from the canonical metric label. Optional
    # row/column text is supporting context and may not move the value cursor.
    primary_anchor = evidence.metric_label
    anchor_matches = _anchor_matches(quote, primary_anchor)
    excluded_spans = _label_occurrence_spans(
        quote,
        (
            evidence.metric_label,
            evidence.period_label,
            evidence.row_label,
            evidence.column_label,
        ),
    )
    for anchor_match in anchor_matches:
        for token_match in _FINANCIAL_TOKEN_RE.finditer(quote, anchor_match.end()):
            token_span = token_match.span()
            if any(_spans_overlap(token_span, label_span) for label_span in excluded_spans):
                continue
            gap = quote[anchor_match.end() : token_span[0]]
            column_is_inline = False
            if evidence.column_label:
                if not _is_canonical_period_header(evidence.column_label):
                    return False
                column_matches = _anchor_matches(gap, evidence.column_label)
                if len(column_matches) != 1:
                    return False
                column_is_inline = True
                gap = (
                    gap[: column_matches[0].start()]
                    + " "
                    + gap[column_matches[0].end() :]
                )
            prior_context = quote[: anchor_match.start()]
            prior_years = set(_YEAR_RE.findall(prior_context))
            prior_quarters = set(_QUARTER_RE.findall(prior_context))
            prior_dates = {
                _normalize_anchor(match.group(0))
                for match in _MONTH_DATE_RE.finditer(prior_context)
            }
            has_multiple_headers = (
                len(prior_years) > 1
                or len(prior_quarters) > 1
                or len(prior_dates) > 1
            )
            period_is_inline = _quote_has_anchor(
                primary_anchor,
                evidence.period_label,
            )
            if has_multiple_headers and not (period_is_inline or column_is_inline):
                return False
            gap = re.sub(
                r"%|\bbps\b|\bbasis\s+points?\b|\bpercent(?:age)?\b|"
                r"\bratio\b|\btimes\b|\bx\b|\bdays?\b",
                " ",
                gap,
                flags=re.IGNORECASE,
            )
            if re.sub(r"[\s:|,;=\-]+", "", gap):
                return False
            return token_span == selected_span
    return False


def _entity_candidates(doc: Any) -> Tuple[str, ...]:
    metadata = getattr(doc, "metadata", {}) or {}
    candidates = []
    company = metadata.get("company")
    if company:
        candidates.append(normalize_company_name(str(company)))
    source = metadata.get("source") or metadata.get("source_file")
    if source:
        parsed = parse_filename(Path(str(source)).name)
        if parsed:
            candidates.append(parsed.company)
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _document_period_context(
    doc: Any,
) -> Tuple[frozenset[str], frozenset[str], frozenset[str]]:
    metadata = getattr(doc, "metadata", {}) or {}
    fiscal_periods = set()
    doc_types = set()
    quarters = set()

    fiscal_period = str(metadata.get("fiscal_period") or "")
    if fiscal_period:
        fiscal_periods.add(re.sub(r"[^A-Z0-9]", "", fiscal_period.upper()))
    doc_type = str(metadata.get("doc_type") or "").upper()
    if doc_type:
        doc_types.add(doc_type.replace("-", ""))
    quarter = str(metadata.get("quarter") or "").upper()
    if re.fullmatch(r"[1-4]", quarter):
        quarter = f"Q{quarter}"
    if re.fullmatch(r"Q[1-4]", quarter):
        quarters.add(quarter)

    source = metadata.get("source") or metadata.get("source_file")
    parsed = parse_filename(Path(str(source)).name) if source else None
    if parsed:
        fiscal_periods.add(
            re.sub(r"[^A-Z0-9]", "", parsed.to_dict()["fiscal_period"].upper())
        )
        doc_types.add(parsed.doc_type.upper().replace("-", ""))
        if parsed.quarter:
            quarters.add(parsed.quarter.upper())

    return (
        frozenset(fiscal_periods),
        frozenset(doc_types),
        frozenset(quarters),
    )


def _full_date_signature(value: str) -> Optional[Tuple[str, str, str]]:
    month_numbers = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }
    match = re.search(
        r"\b("
        + "|".join(month_numbers)
        + r")\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return (
        match.group(3),
        month_numbers[match.group(1).casefold()],
        match.group(2).zfill(2),
    )


def _periods_consistent(period: str, label: str, *, doc: Any = None) -> bool:
    """Bind fiscal semantics without equating fiscal and calendar quarters."""

    period_compact = re.sub(r"[^A-Z0-9]", "", period.upper())
    label_compact = re.sub(r"[^A-Z0-9]", "", label.upper())
    fiscal_periods, doc_types, metadata_quarters = _document_period_context(doc)
    annual_doc_types = {"10K", "ANNUALREPORT"}
    interim_doc_types = {"10Q"}

    relative_tokens = {"ttm", "ltm", "ytd"}
    period_relative = relative_tokens & set(re.findall(r"[a-z]+", period.casefold()))
    label_relative = relative_tokens & set(re.findall(r"[a-z]+", label.casefold()))
    if period_relative or label_relative:
        return period_relative == label_relative and period_compact == label_compact

    period_date = _full_date_signature(period)
    label_date = _full_date_signature(label)
    if period_date is not None:
        return label_date == period_date

    annual_match = re.fullmatch(r"FY((?:19|20)\d{2})", period_compact)
    quarter_match = re.fullmatch(r"FY((?:19|20)\d{2})Q([1-4])", period_compact)
    label_years = set(_YEAR_RE.findall(label))
    label_quarters = {f"Q{quarter}" for quarter in _QUARTER_RE.findall(label)}

    if annual_match:
        requested_year = annual_match.group(1)
        interim_label = bool(
            label_quarters
            or re.search(
                r"\b(?:quarter(?:ly)?|three\s+months?|six\s+months?|nine\s+months?)\b",
                label,
                flags=re.IGNORECASE,
            )
        )
        interim_document = bool(
            metadata_quarters
            or interim_doc_types & set(doc_types)
            or any(re.search(r"Q[1-4]", item) for item in fiscal_periods)
        )
        if interim_label or interim_document:
            return False
        if doc_types and not set(doc_types) <= annual_doc_types:
            return False

        explicit_fiscal_label = re.search(
            r"\bFY\s*((?:19|20)\d{2})\b",
            label,
            flags=re.IGNORECASE,
        )
        if explicit_fiscal_label:
            return explicit_fiscal_label.group(1) == requested_year
        if label_date is not None:
            # A calendar date proves an off-calendar fiscal year only when the
            # authoritative annual filing identifies the requested fiscal year.
            return period_compact in fiscal_periods and bool(
                annual_doc_types & set(doc_types)
            )
        return label_years == {requested_year}

    if quarter_match:
        requested_year, requested_quarter_number = quarter_match.groups()
        requested_quarter = f"Q{requested_quarter_number}"
        if label_quarters and label_quarters != {requested_quarter}:
            return False
        explicit_fiscal_year = re.search(
            r"\bFY\s*((?:19|20)\d{2})\b",
            label,
            flags=re.IGNORECASE,
        )
        if explicit_fiscal_year and explicit_fiscal_year.group(1) != requested_year:
            return False
        if metadata_quarters and metadata_quarters != {requested_quarter}:
            return False
        if fiscal_periods and period_compact not in fiscal_periods:
            return False
        if doc_types and not set(doc_types) <= interim_doc_types:
            return False

        if label_date is not None:
            # Fiscal Q1 may end in calendar Q4 (and vice versa). Metadata, not
            # the month number, establishes the fiscal-quarter mapping.
            return period_compact in fiscal_periods and bool(
                interim_doc_types & set(doc_types)
            )
        return bool(label_quarters) and (
            not label_years or label_years == {requested_year}
        )

    # Unknown period vocabularies are accepted only as exact canonical labels.
    return bool(period_compact) and period_compact == label_compact


def _is_canonical_period_header(label: str) -> bool:
    normalized = _normalize_anchor(label)
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    if re.fullmatch(r"(?:fy)?(?:19|20)\d{2}(?:q[1-4])?", compact):
        return True
    if re.fullmatch(r"q[1-4](?:fy)?(?:19|20)\d{2}", compact):
        return True
    months = (
        "january|february|march|april|may|june|july|august|"
        "september|october|november|december"
    )
    return bool(
        re.fullmatch(
            rf"(?:{months})\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}",
            normalized,
        )
    )


def _evidence_value_as_operand(
    parsed: _EvidenceValue,
    operand: FinancialQuantity,
) -> Tuple[Optional[_CalcQuantity], Optional[IssueCode], Optional[str]]:
    unit_token = parsed.unit_token
    percent_tokens = {
        "%",
        "percent",
        "percentage",
        "percentage point",
        "percentage points",
    }
    basis_tokens = {"bps", "basis point", "basis points"}
    ratio_tokens = {"x"}
    day_tokens = {"day", "days"}

    if parsed.scale_ambiguous and operand.unit in {
        UnitKind.MONEY,
        UnitKind.NUMBER,
        UnitKind.COUNT,
        UnitKind.SHARES,
    }:
        return (
            None,
            IssueCode.OPERAND_UNIT_MISMATCH,
            "Evidence contains conflicting scale declarations",
        )

    # A bare table cell may rely on its row/column header for a unit, but the
    # model may not simply assert that unit without an explicit source anchor.
    if not unit_token and operand.unit in {
        UnitKind.PERCENT,
        UnitKind.BASIS_POINTS,
        UnitKind.RATIO,
        UnitKind.DAYS,
    }:
        unit_context = " ".join(
            filter(
                None,
                (
                    operand.evidence.metric_label,
                    operand.evidence.row_label,
                    operand.evidence.column_label,
                ),
            )
        ).casefold()
        required_unit_anchor = {
            UnitKind.PERCENT: r"%|\bpercent(?:age)?\b",
            UnitKind.BASIS_POINTS: r"\bbps\b|\bbasis\s+points?\b",
            UnitKind.RATIO: r"\bratio\b|\btimes\b|\bx\b",
            UnitKind.DAYS: r"\bdays?\b",
        }[operand.unit]
        if not re.search(required_unit_anchor, unit_context):
            return (
                None,
                IssueCode.OPERAND_UNIT_MISMATCH,
                f"Bare evidence value has no explicit {operand.unit.value} anchor",
            )

    if unit_token in percent_tokens and operand.unit != UnitKind.PERCENT:
        return None, IssueCode.OPERAND_UNIT_MISMATCH, "Evidence is a percentage"
    if unit_token in basis_tokens and operand.unit not in {
        UnitKind.BASIS_POINTS,
        UnitKind.PERCENT,
    }:
        return None, IssueCode.OPERAND_UNIT_MISMATCH, "Evidence is in basis points"
    if unit_token in ratio_tokens and operand.unit != UnitKind.RATIO:
        return None, IssueCode.OPERAND_UNIT_MISMATCH, "Evidence is a ratio"
    if unit_token in day_tokens and operand.unit != UnitKind.DAYS:
        return None, IssueCode.OPERAND_UNIT_MISMATCH, "Evidence is in days"

    if operand.unit == UnitKind.MONEY:
        if parsed.currency_ambiguous:
            return (
                None,
                IssueCode.OPERAND_CURRENCY_MISMATCH,
                "Evidence currency is ambiguous or conflicts with document metadata",
            )
        if parsed.currency != operand.currency:
            return (
                None,
                IssueCode.OPERAND_CURRENCY_MISMATCH,
                f"Evidence currency {parsed.currency!r} does not match {operand.currency}",
            )
    elif parsed.currency_explicit:
        return None, IssueCode.OPERAND_UNIT_MISMATCH, "Currency evidence is not typed as money"

    if operand.unit in {
        UnitKind.MONEY,
        UnitKind.NUMBER,
        UnitKind.COUNT,
        UnitKind.SHARES,
    } and parsed.scale != operand.scale:
        return (
            None,
            IssueCode.OPERAND_UNIT_MISMATCH,
            f"Evidence scale {parsed.scale.value} does not match {operand.scale.value}",
        )

    if unit_token in basis_tokens:
        evidence_value = parsed.value / Decimal("10000")
        semantic = UnitKind.PERCENT
    elif unit_token in percent_tokens or operand.unit == UnitKind.PERCENT:
        evidence_value = parsed.value / Decimal("100")
        semantic = UnitKind.PERCENT
    elif operand.unit == UnitKind.BASIS_POINTS:
        evidence_value = parsed.value / Decimal("10000")
        semantic = UnitKind.PERCENT
    else:
        evidence_value = parsed.value * _SCALE_FACTORS[parsed.scale.value]
        semantic = operand.unit

    return (
        _CalcQuantity(
            evidence_value,
            _dimensions_for(operand.unit, operand.currency),
            semantic,
        ),
        None,
        None,
    )


def _verify_operand(
    operand: FinancialQuantity,
    docs: Sequence[Any],
) -> Tuple[VerificationIssue, ...]:
    evidence = operand.evidence
    issue_context = {
        "operand_id": operand.id,
        "metric": operand.metric,
        "period": operand.period,
        "doc_id": evidence.doc_id,
    }
    match = _DOC_ID_RE.fullmatch(evidence.doc_id)
    doc_index = int(match.group(1)) - 1 if match else -1
    if doc_index < 0 or doc_index >= len(docs):
        return (
            VerificationIssue(
                IssueCode.INVALID_CITATION,
                f"{evidence.doc_id} is outside the retrieved document list",
                **issue_context,
            ),
        )

    doc = docs[doc_index]
    content = getattr(doc, "page_content", "") or ""
    issues = []
    if evidence.quote not in content:
        issues.append(
            VerificationIssue(
                IssueCode.INVALID_CITATION,
                "Exact quote not found in the cited document",
                **issue_context,
            )
        )
        return tuple(issues)

    # The model may select a source line, but not crop a convenient substring
    # out of a flattened table row.  Line-bounded spans preserve neighboring
    # headers that are necessary to detect column/metric swaps.
    quote_start = content.find(evidence.quote)
    quote_end = quote_start + len(evidence.quote)
    before_line = content.rfind("\n", 0, quote_start) + 1
    after_newline = content.find("\n", quote_end)
    after_line = len(content) if after_newline < 0 else after_newline
    if content[before_line:quote_start].strip() or content[quote_end:after_line].strip():
        issues.append(
            VerificationIssue(
                IssueCode.INVALID_CITATION,
                "Evidence quote must cover complete source line boundaries",
                **issue_context,
            )
        )
        return tuple(issues)

    selected_span = None
    markdown_binding: Optional[bool] = None
    value_spans = _value_occurrence_spans(evidence.quote, evidence.value_text)
    if len(value_spans) < evidence.occurrence:
        raw_spans = tuple(
            match.span()
            for match in re.finditer(re.escape(evidence.value_text), evidence.quote)
        )
        overlaps_anchor = bool(raw_spans) and all(
            any(
                _span_overlaps_label(evidence.quote, span, label)
                for label in (
                    evidence.metric_label,
                    evidence.period_label,
                    evidence.row_label,
                    evidence.column_label,
                )
                if label
            )
            for span in raw_spans
        )
        issues.append(
            VerificationIssue(
                (
                    IssueCode.OPERAND_VALUE_MISMATCH
                    if overlaps_anchor
                    else IssueCode.MISSING_EVIDENCE
                ),
                (
                    "value_text selects numeric anchor text rather than a complete data value"
                    if overlaps_anchor
                    else "value_text is not present as a complete financial token in the exact quote"
                ),
                **issue_context,
            )
        )
    else:
        selected_span = value_spans[evidence.occurrence - 1]
        markdown_binding = _markdown_table_binding(
            content,
            evidence.quote,
            evidence,
            selected_span,
        )
        overlapping_labels = [
            name
            for label, name in (
                (evidence.metric_label, "metric"),
                (evidence.period_label, "period"),
                (evidence.row_label, "row"),
                (evidence.column_label, "column"),
            )
            if label and _span_overlaps_label(evidence.quote, selected_span, label)
        ]
        if overlapping_labels:
            issues.append(
                VerificationIssue(
                    IssueCode.OPERAND_VALUE_MISMATCH,
                    "Selected value occurrence overlaps "
                    f"{', '.join(overlapping_labels)} anchor text rather than a data value",
                    **issue_context,
                )
            )
        elif markdown_binding is not True and (
            markdown_binding is False
            or not _value_is_locally_associated(
                evidence.quote,
                selected_span,
                evidence,
            )
        ):
            issues.append(
                VerificationIssue(
                    IssueCode.OPERAND_METRIC_MISMATCH,
                    "Selected value is not the nearest data token after its metric/row anchor",
                    **issue_context,
                )
            )

    for label, code, name in (
        (evidence.metric_label, IssueCode.OPERAND_METRIC_MISMATCH, "metric"),
        (evidence.period_label, IssueCode.OPERAND_PERIOD_MISMATCH, "period"),
        (evidence.row_label, IssueCode.OPERAND_METRIC_MISMATCH, "row"),
        (evidence.column_label, IssueCode.OPERAND_PERIOD_MISMATCH, "column"),
    ):
        header_bound = markdown_binding is True and name in {"period", "column"}
        if label and not header_bound and not _quote_has_anchor(evidence.quote, label):
            issues.append(
                VerificationIssue(
                    code,
                    f"{name} anchor {label!r} is absent from the exact quote",
                    **issue_context,
                )
            )

    if not _metric_binding_consistent(
        operand.metric,
        evidence.metric_label,
        evidence.period_label,
    ):
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_METRIC_MISMATCH,
                f"Operand metric {operand.metric!r} is not bound by metric anchor {evidence.metric_label!r}",
                **issue_context,
            )
        )

    row_metric_mismatch = bool(
        evidence.row_label
        and not _metric_claims_equal(operand.metric, evidence.row_label)
    )
    if selected_span is not None:
        semantic_anchor = evidence.row_label or evidence.metric_label
        metric_matches = [
            match
            for match in _anchor_matches(evidence.quote, semantic_anchor)
            if match.end() <= selected_span[0]
        ]
        if metric_matches:
            nearest_metric = max(metric_matches, key=lambda match: match.end())
            prefix = evidence.quote[: nearest_metric.start()]
            qualifier_prefix = re.search(
                r"((?:[A-Za-z]+(?:-[A-Za-z]+)?\s+){1,4})$",
                prefix,
            )
            if qualifier_prefix:
                candidate = qualifier_prefix.group(1) + semantic_anchor
                expected_qualifiers = _metric_signature(operand.metric)[1]
                observed_qualifiers = _metric_signature(candidate)[1]
                row_metric_mismatch = row_metric_mismatch or (
                    observed_qualifiers != expected_qualifiers
                )
    if row_metric_mismatch:
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_METRIC_MISMATCH,
                "Full row metric semantics do not match the declared metric",
                **issue_context,
            )
        )

    if not _periods_consistent(operand.period, evidence.period_label, doc=doc):
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_PERIOD_MISMATCH,
                f"Operand period {operand.period!r} conflicts with period anchor {evidence.period_label!r}",
                **issue_context,
            )
        )
    if evidence.column_label and (
        not _is_canonical_period_header(evidence.column_label)
        or not _periods_consistent(operand.period, evidence.column_label, doc=doc)
    ):
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_PERIOD_MISMATCH,
                f"Operand period {operand.period!r} conflicts with column anchor {evidence.column_label!r}",
                **issue_context,
            )
        )

    expected_entity = normalize_company_name(operand.entity)
    doc_entities = _entity_candidates(doc)
    if len(doc_entities) > 1:
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_ENTITY_MISMATCH,
                f"Document metadata and source disagree on entity: {doc_entities}",
                **issue_context,
            )
        )
    elif not doc_entities or expected_entity != doc_entities[0]:
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_ENTITY_MISMATCH,
                (
                    f"Operand entity {expected_entity!r} does not match authoritative "
                    f"document entity {doc_entities[0]!r}"
                    if doc_entities
                    else "Document has no authoritative entity metadata"
                ),
                **issue_context,
            )
        )

    if selected_span is not None:
        try:
            parsed = _parse_evidence_value(
                evidence.value_text,
                evidence.quote,
                document_currency=_document_currency(doc),
            )
            evidence_quantity, issue_code, explanation = _evidence_value_as_operand(
                parsed, operand
            )
            if issue_code:
                issues.append(
                    VerificationIssue(
                        issue_code,
                        explanation or "Evidence type does not match operand type",
                        **issue_context,
                    )
                )
            elif evidence_quantity != _calc_from_operand(operand):
                issues.append(
                    VerificationIssue(
                        IssueCode.OPERAND_VALUE_MISMATCH,
                        f"Evidence value {evidence.value_text!r} does not equal declared operand value {operand.value!r}",
                        **issue_context,
                    )
                )
        except (InvalidOperation, ValueError) as exc:
            issues.append(
                VerificationIssue(
                    IssueCode.OPERAND_VALUE_MISMATCH,
                    str(exc),
                    **issue_context,
                )
            )

    return tuple(issues)


def _constant_issues(
    expression: Expression,
    question: str,
    *,
    trusted_expression: Optional[Expression] = None,
) -> Tuple[VerificationIssue, ...]:
    # An exact AST compiled before generation is itself the authority for
    # domain constants such as 365 days or the leading 1 in retention ratio.
    if trusted_expression is not None and expression == trusted_expression:
        return ()
    issues = []
    normalized_question = unicodedata.normalize("NFKC", question or "")
    for node in _walk_expression(expression):
        if node.op != Operation.CONST:
            continue
        source_text = unicodedata.normalize("NFKC", node.source_text or "")
        numeric_tokens = _NUMBER_IN_TEXT_RE.findall(source_text)
        source_pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(source_text)}(?![A-Za-z0-9])"
        )
        bound = bool(source_pattern.search(normalized_question)) and any(
            Decimal(token.replace(",", "")) == Decimal(node.value)
            for token in numeric_tokens
        )
        if not bound:
            issues.append(
                VerificationIssue(
                    IssueCode.CONSTANT_NOT_QUESTION_BOUND,
                    f"Constant {node.value} is not bound to its exact source_text in the question",
                )
            )
    return tuple(issues)


def _conflict_issues(operands: Sequence[FinancialQuantity]) -> Tuple[VerificationIssue, ...]:
    observed: Dict[Tuple[str, str, str], Tuple[_CalcQuantity, FinancialQuantity]] = {}
    issues = []
    for operand in operands:
        key = (
            normalize_company_name(operand.entity),
            _normalize_anchor(operand.period),
            _normalize_anchor(operand.metric),
        )
        calculated = _calc_from_operand(operand)
        previous = observed.get(key)
        if previous and previous[0] != calculated:
            issues.append(
                VerificationIssue(
                    IssueCode.CONFLICTING_EVIDENCE,
                    f"Operands {previous[1].id!r} and {operand.id!r} claim different values for the same entity/period/metric",
                    operand_id=operand.id,
                    metric=operand.metric,
                    period=operand.period,
                    doc_id=operand.evidence.doc_id,
                )
            )
        else:
            observed[key] = (calculated, operand)
    return tuple(issues)


def _program_context_issues(program: FinanceProgram) -> Tuple[VerificationIssue, ...]:
    """Check answer-level entity/period claims against the operand ledger."""

    issues = []
    answer_entity = normalize_company_name(program.answer.entity)
    operand_entities = {
        normalize_company_name(operand.entity) for operand in program.operands
    }
    answer_covers_entities = (
        answer_entity == next(iter(operand_entities))
        if len(operand_entities) == 1
        else all(entity in answer_entity for entity in operand_entities)
    )
    if not answer_covers_entities:
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_ENTITY_MISMATCH,
                f"Answer entity {program.answer.entity!r} is not supported by operand entities {sorted(operand_entities)}",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )

    answer_years = set(_YEAR_RE.findall(program.answer.period))
    operand_years = {
        year
        for operand in program.operands
        for year in _YEAR_RE.findall(operand.period)
    }
    if answer_years and operand_years and not answer_years <= operand_years:
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_PERIOD_MISMATCH,
                f"Answer period {program.answer.period!r} introduces a year absent from all operands",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )

    answer_quarters = set(_QUARTER_RE.findall(program.answer.period))
    operand_quarters = {
        quarter
        for operand in program.operands
        for quarter in _QUARTER_RE.findall(operand.period)
    }
    if answer_quarters and operand_quarters and not answer_quarters <= operand_quarters:
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_PERIOD_MISMATCH,
                f"Answer period {program.answer.period!r} introduces a quarter absent from all operands",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )

    currencies = {
        operand.currency
        for operand in program.operands
        if operand.unit == UnitKind.MONEY and operand.currency
    }
    if len(currencies) > 1:
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_CURRENCY_MISMATCH,
                f"Program mixes currencies without an allowlisted FX conversion: {sorted(currencies)}",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )
    return tuple(issues)


def _period_claims_equal(left: str, right: str) -> bool:
    def canonical(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", _normalize_anchor(value))

    return bool(canonical(left)) and canonical(left) == canonical(right)


def _question_spec_issues(
    program: FinanceProgram,
    expected: FinanceQuestionSpec,
) -> Tuple[VerificationIssue, ...]:
    """Bind model-controlled answer claims to a trusted pre-generation plan."""

    issues = []
    if normalize_company_name(program.answer.entity) != normalize_company_name(
        expected.entity
    ):
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_ENTITY_MISMATCH,
                f"Answer entity {program.answer.entity!r} does not match trusted question entity {expected.entity!r}",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )
    if not _period_claims_equal(program.answer.period, expected.period):
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_PERIOD_MISMATCH,
                f"Answer period {program.answer.period!r} does not match trusted question period {expected.period!r}",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )
    if not _metric_claims_equal(program.answer.metric, expected.metric):
        issues.append(
            VerificationIssue(
                IssueCode.OPERAND_METRIC_MISMATCH,
                f"Answer metric {program.answer.metric!r} does not match trusted question metric {expected.metric!r}",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )
    if (
        program.answer.unit != expected.unit
        or program.answer.currency != expected.currency
        or program.answer.scale != expected.scale
    ):
        issues.append(
            VerificationIssue(
                IssueCode.RESULT_UNIT_MISMATCH,
                "Answer unit/currency/scale does not match the trusted output contract",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )
    if program.answer.rounding != expected.rounding:
        issues.append(
            VerificationIssue(
                IssueCode.FORMULA_MISMATCH,
                "Answer rounding does not match the trusted output contract",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )
    if program.expression != expected.expression:
        issues.append(
            VerificationIssue(
                IssueCode.FORMULA_MISMATCH,
                "Program expression does not match the trusted question formula",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )
    if expected.operands:
        actual_by_id = {operand.id: operand for operand in program.operands}
        for operand_spec in expected.operands:
            actual = actual_by_id.get(operand_spec.id)
            if actual is None:
                issues.append(
                    VerificationIssue(
                        IssueCode.MISSING_OPERAND,
                        f"Trusted operand {operand_spec.id!r} is absent",
                        operand_id=operand_spec.id,
                        metric=operand_spec.metric,
                        period=operand_spec.period,
                    )
                )
                continue
            if normalize_company_name(actual.entity) != normalize_company_name(
                operand_spec.entity
            ):
                issues.append(
                    VerificationIssue(
                        IssueCode.OPERAND_ENTITY_MISMATCH,
                        f"Operand {actual.id!r} entity does not match its trusted evidence need",
                        operand_id=actual.id,
                        metric=operand_spec.metric,
                        period=operand_spec.period,
                        doc_id=actual.evidence.doc_id,
                    )
                )
            if not _period_claims_equal(actual.period, operand_spec.period):
                issues.append(
                    VerificationIssue(
                        IssueCode.OPERAND_PERIOD_MISMATCH,
                        f"Operand {actual.id!r} period does not match its trusted evidence need",
                        operand_id=actual.id,
                        metric=operand_spec.metric,
                        period=operand_spec.period,
                        doc_id=actual.evidence.doc_id,
                    )
                )
            if not _metric_claims_equal(actual.metric, operand_spec.metric):
                issues.append(
                    VerificationIssue(
                        IssueCode.OPERAND_METRIC_MISMATCH,
                        f"Operand {actual.id!r} metric does not match its trusted evidence need",
                        operand_id=actual.id,
                        metric=operand_spec.metric,
                        period=operand_spec.period,
                        doc_id=actual.evidence.doc_id,
                    )
                )
            if actual.unit != operand_spec.unit:
                issues.append(
                    VerificationIssue(
                        IssueCode.OPERAND_UNIT_MISMATCH,
                        f"Operand {actual.id!r} unit does not match its trusted evidence need",
                        operand_id=actual.id,
                        metric=operand_spec.metric,
                        period=operand_spec.period,
                        doc_id=actual.evidence.doc_id,
                    )
                )
            if (
                operand_spec.currency is not None
                and actual.currency != operand_spec.currency
            ):
                issues.append(
                    VerificationIssue(
                        IssueCode.OPERAND_CURRENCY_MISMATCH,
                        f"Operand {actual.id!r} currency does not match its trusted evidence need",
                        operand_id=actual.id,
                        metric=operand_spec.metric,
                        period=operand_spec.period,
                        doc_id=actual.evidence.doc_id,
                    )
                )
            if operand_spec.scale is not None and actual.scale != operand_spec.scale:
                issues.append(
                    VerificationIssue(
                        IssueCode.OPERAND_UNIT_MISMATCH,
                        f"Operand {actual.id!r} scale does not match its trusted evidence need",
                        operand_id=actual.id,
                        metric=operand_spec.metric,
                        period=operand_spec.period,
                        doc_id=actual.evidence.doc_id,
                    )
                )
    return tuple(issues)


def _expected_answer_dimensions(answer: AnswerSpec) -> Tuple[Tuple[str, int], ...]:
    return _dimensions_for(answer.unit, answer.currency)


def _answer_semantic_compatible(answer: AnswerSpec, execution: ExecutionResult) -> bool:
    if execution.dimensions != _expected_answer_dimensions(answer):
        return False
    if execution.dimensions:
        return True
    if answer.unit == UnitKind.NUMBER:
        return execution.semantic_unit == UnitKind.NUMBER
    if answer.unit in {UnitKind.RATIO, UnitKind.PERCENT, UnitKind.BASIS_POINTS}:
        return execution.semantic_unit in {UnitKind.RATIO, UnitKind.PERCENT}
    return False


def _display_decimal(execution: ExecutionResult, answer: AnswerSpec) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        value = execution.value
        if answer.unit == UnitKind.PERCENT:
            value *= Decimal("100")
        elif answer.unit == UnitKind.BASIS_POINTS:
            value *= Decimal("10000")
        else:
            value /= _SCALE_FACTORS[answer.scale.value]
        quantum = Decimal("1").scaleb(-answer.rounding.places)
        return value.quantize(quantum, rounding=ROUND_HALF_UP)


def render_result(execution: ExecutionResult, answer: AnswerSpec) -> str:
    """Render the deterministic result; never reuse LLM-computed arithmetic."""

    if not _answer_semantic_compatible(answer, execution):
        raise ProgramExecutionError(
            IssueCode.RESULT_UNIT_MISMATCH,
            "Execution dimensions are incompatible with the answer specification",
        )
    value = _display_decimal(execution, answer)
    rendered = f"{value:.{answer.rounding.places}f}"
    if answer.unit == UnitKind.MONEY:
        scale = "" if answer.scale == Scale.ONE else f" {answer.scale.value}"
        return f"{answer.currency} {rendered}{scale}"
    if answer.unit == UnitKind.PERCENT:
        return f"{rendered}%"
    if answer.unit == UnitKind.BASIS_POINTS:
        return f"{rendered} bps"
    if answer.unit == UnitKind.RATIO:
        return f"{rendered}x"
    if answer.unit == UnitKind.DAYS:
        return f"{rendered} days"
    if answer.unit in {UnitKind.COUNT, UnitKind.SHARES, UnitKind.NUMBER}:
        scale = "" if answer.scale == Scale.ONE else f" {answer.scale.value}"
        return f"{rendered}{scale}"
    return rendered


def _displayed_answer_matches(
    answer_text: str,
    rendered_answer: str,
) -> bool:
    """Bind a dedicated display field, never search untrusted prose for a token."""

    return _normalize_anchor(answer_text) == _normalize_anchor(rendered_answer)


def verify_program(
    program: FinanceProgram | Mapping[str, Any],
    docs: Sequence[Any],
    question: str,
    *,
    question_spec: Optional[FinanceQuestionSpec | Mapping[str, Any]] = None,
    answer_text: Optional[str] = None,
    require_full_contract: bool = False,
) -> ProgramVerificationResult:
    """Verify evidence and recompute a finance program without a gold answer.

    ``question_spec`` is the trusted semantic output of query understanding;
    callers should provide it in production to bind entity/period/metric and an
    exact AST and output contract. ``answer_text`` must be a dedicated display
    value (not prose). Set ``require_full_contract`` in production so omitting
    either binding fails closed.
    """

    if not isinstance(program, FinanceProgram):
        try:
            program = FinanceProgram.model_validate(program)
        except ValidationError as exc:
            return ProgramVerificationResult(
                passed=False,
                issues=_issue_from_validation_error(exc),
                execution=None,
                rendered_answer=None,
                evidence_coverage=0.0,
            )

    if question_spec is not None and not isinstance(
        question_spec, FinanceQuestionSpec
    ):
        try:
            question_spec = FinanceQuestionSpec.model_validate(question_spec)
        except ValidationError as exc:
            return ProgramVerificationResult(
                passed=False,
                issues=_issue_from_validation_error(exc),
                execution=None,
                rendered_answer=None,
                evidence_coverage=0.0,
            )

    issues = []
    contract_complete = bool(
        question_spec is not None
        and answer_text is not None
        and question_spec.operands
    )
    if require_full_contract and not contract_complete:
        issues.append(
            VerificationIssue(
                IssueCode.UNSUPPORTED_CLAIM,
                "Full verification requires question_spec, trusted operand specs, and answer_text",
                metric=program.answer.metric,
                period=program.answer.period,
            )
        )
    issues.extend(_program_context_issues(program))
    if question_spec is not None:
        issues.extend(_question_spec_issues(program, question_spec))
    grounded_operands = 0
    for operand in program.operands:
        operand_issues = _verify_operand(operand, docs)
        issues.extend(operand_issues)
        if not operand_issues:
            grounded_operands += 1
    issues.extend(
        _constant_issues(
            program.expression,
            question,
            trusted_expression=(
                question_spec.expression if question_spec is not None else None
            ),
        )
    )
    issues.extend(_conflict_issues(program.operands))

    evidence_coverage = grounded_operands / len(program.operands)
    if issues:
        return ProgramVerificationResult(
            passed=False,
            issues=tuple(issues),
            execution=None,
            rendered_answer=None,
            evidence_coverage=evidence_coverage,
        )

    try:
        execution = execute_program(program)
    except ProgramExecutionError as exc:
        return ProgramVerificationResult(
            passed=False,
            issues=(
                VerificationIssue(
                    exc.code,
                    str(exc),
                    operand_id=exc.operand_id,
                ),
            ),
            execution=None,
            rendered_answer=None,
            evidence_coverage=evidence_coverage,
        )

    if not _answer_semantic_compatible(program.answer, execution):
        return ProgramVerificationResult(
            passed=False,
            issues=(
                VerificationIssue(
                    IssueCode.RESULT_UNIT_MISMATCH,
                    f"Expression dimensions {dict(execution.dimensions)} and semantic unit {execution.semantic_unit.value!r} do not match answer unit {program.answer.unit.value!r}",
                    metric=program.answer.metric,
                    period=program.answer.period,
                ),
            ),
            execution=execution,
            rendered_answer=None,
            evidence_coverage=evidence_coverage,
        )

    computed_display = _display_decimal(execution, program.answer)
    declared = Decimal(program.answer.value)
    if declared != computed_display:
        rendered_answer = render_result(execution, program.answer)
        return ProgramVerificationResult(
            passed=False,
            issues=(
                VerificationIssue(
                    IssueCode.RESULT_VALUE_MISMATCH,
                    f"Declared answer {declared} does not equal recomputed and rounded value {computed_display}",
                    metric=program.answer.metric,
                    period=program.answer.period,
                ),
            ),
            execution=execution,
            rendered_answer=rendered_answer,
            evidence_coverage=evidence_coverage,
        )

    rendered_answer = render_result(execution, program.answer)
    if answer_text is not None and not _displayed_answer_matches(
        answer_text,
        rendered_answer,
    ):
        return ProgramVerificationResult(
            passed=False,
            issues=(
                VerificationIssue(
                    IssueCode.ANSWER_RESULT_MISMATCH,
                    f"Displayed answer {answer_text!r} does not equal deterministic rendering {rendered_answer!r}",
                    metric=program.answer.metric,
                    period=program.answer.period,
                ),
            ),
            execution=execution,
            rendered_answer=rendered_answer,
            evidence_coverage=evidence_coverage,
        )

    return ProgramVerificationResult(
        passed=True,
        issues=(),
        execution=execution,
        rendered_answer=rendered_answer,
        evidence_coverage=evidence_coverage,
        assurance_level=(
            AssuranceLevel.FULL_CONTRACT
            if contract_complete
            else AssuranceLevel.EVIDENCE_ARITHMETIC
        ),
    )


def repair_program_result(
    program: FinanceProgram | Mapping[str, Any],
    docs: Sequence[Any],
    question: str,
    *,
    question_spec: FinanceQuestionSpec | Mapping[str, Any],
    answer_text: Optional[str] = None,
) -> Tuple[Optional[FinanceProgram], ProgramVerificationResult]:
    """Repair only a declared/displayed result, then fully reverify it.

    Evidence, operands, expression, units, and the trusted question contract
    are immutable. The function returns ``None`` unless the sole remaining gap
    is model-supplied result/rendering and the repaired artifact passes a fresh
    full-contract verification.
    """

    parsed_program = (
        program
        if isinstance(program, FinanceProgram)
        else FinanceProgram.model_validate(program, strict=True)
    )
    parsed_spec = (
        question_spec
        if isinstance(question_spec, FinanceQuestionSpec)
        else FinanceQuestionSpec.model_validate(question_spec, strict=True)
    )
    initial = verify_program(
        parsed_program,
        docs,
        question,
        question_spec=parsed_spec,
        answer_text=(
            answer_text if answer_text is not None else parsed_program.answer.value
        ),
        require_full_contract=True,
    )
    if initial.fully_verified:
        return parsed_program, initial
    repairable = {IssueCode.RESULT_VALUE_MISMATCH, IssueCode.ANSWER_RESULT_MISMATCH}
    if (
        not initial.issues
        or {issue.code for issue in initial.issues} - repairable
        or initial.execution is None
        or initial.rendered_answer is None
    ):
        return None, initial

    repaired_value = str(_display_decimal(initial.execution, parsed_program.answer))
    repaired_answer = parsed_program.answer.model_copy(
        update={"value": repaired_value}
    )
    repaired_program = parsed_program.model_copy(update={"answer": repaired_answer})
    repaired = verify_program(
        repaired_program,
        docs,
        question,
        question_spec=parsed_spec,
        answer_text=initial.rendered_answer,
        require_full_contract=True,
    )
    if not repaired.fully_verified:
        return None, repaired
    return repaired_program, repaired


__all__ = [
    "AssuranceLevel",
    "AnswerSpec",
    "EvidenceRef",
    "ExecutionResult",
    "Expression",
    "FinanceQuestionSpec",
    "FinanceOperandSpec",
    "FinanceProgram",
    "FinancialQuantity",
    "IssueCode",
    "Operation",
    "ParsedFinanceResponse",
    "ProgramExecutionError",
    "ProgramVerificationResult",
    "RoundingSpec",
    "Scale",
    "UnitKind",
    "VerificationIssue",
    "execute_program",
    "parse_finance_response",
    "render_result",
    "repair_program_result",
    "verify_program",
]
