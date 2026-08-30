"""Bridge deterministic query plans to trusted finance-program contracts.

The model may fill operand values and evidence spans, but it does not choose
the entity, period, output type, or formula.  Those fields are compiled before
generation and expressed here as an exact :class:`FinanceQuestionSpec`.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

from src.finance_program import (
    Expression,
    FinanceOperandSpec,
    FinanceQuestionSpec,
    Operation,
    RoundingSpec,
    Scale,
    UnitKind,
)
from src.query_understanding import AnswerKind, FinanceQueryPlan


def _ref(need_id: str) -> Expression:
    return Expression(op=Operation.REF, operand_id=need_id)


def _op(operation: Operation, *args: Expression) -> Expression:
    return Expression(op=operation, args=tuple(args))


def _constant(value: str, unit: UnitKind, source_text: str) -> Expression:
    return Expression(
        op=Operation.CONST,
        value=value,
        unit=unit,
        source_text=source_text,
    )


def _period_sort_key(period: Optional[str]) -> tuple[int, str]:
    match = re.search(r"(?:19|20)\d{2}", period or "")
    return (int(match.group(0)) if match else -1, period or "")


class _NeedIndex:
    def __init__(self, plan: FinanceQueryPlan) -> None:
        self._by_metric: Dict[str, list] = {}
        for need in plan.evidence_needs:
            self._by_metric.setdefault(need.metric, []).append(need)
        for needs in self._by_metric.values():
            needs.sort(key=lambda need: _period_sort_key(need.period))

    def needs(self, metric: str) -> Sequence:
        return tuple(self._by_metric.get(metric, ()))

    def one(self, metric: str, *, latest: bool = True):
        needs = self.needs(metric)
        if not needs:
            raise ValueError(f"No evidence need for metric {metric!r}")
        return needs[-1 if latest else 0]

    def at(self, metric: str, period: str):
        for need in self.needs(metric):
            if need.period == period:
                return need
        raise ValueError(f"No evidence need for {metric!r} in {period!r}")

    def average(self, metric: str) -> Expression:
        refs = tuple(_ref(need.need_id) for need in self.needs(metric))
        if not refs:
            raise ValueError(f"No evidence needs for metric {metric!r}")
        return refs[0] if len(refs) == 1 else _op(Operation.AVG, *refs)

    def ratios_by_period(self, numerator: str, denominator: str) -> Expression:
        numerator_periods = {
            need.period for need in self.needs(numerator) if need.period
        }
        denominator_periods = {
            need.period for need in self.needs(denominator) if need.period
        }
        periods = sorted(
            numerator_periods & denominator_periods,
            key=_period_sort_key,
        )
        if not periods:
            raise ValueError(
                f"No aligned periods for {numerator!r} and {denominator!r}"
            )
        ratios = tuple(
            _op(
                Operation.DIV,
                _ref(self.at(numerator, period).need_id),
                _ref(self.at(denominator, period).need_id),
            )
            for period in periods
        )
        return ratios[0] if len(ratios) == 1 else _op(Operation.AVG, *ratios)


def _days_constant(question: str) -> Expression:
    source = next(
        (match.group(0) for match in re.finditer(r"(?<!\d)365(?!\d)", question)),
        "trusted DPO/CCC formula",
    )
    return _constant("365", UnitKind.DAYS, source)


def _one_constant(question: str) -> Expression:
    match = re.search(r"(?<![\d.])1(?:\.0+)?(?![\d.])", question)
    return _constant(
        "1",
        UnitKind.NUMBER,
        match.group(0) if match else "trusted retention-ratio formula",
    )


def _dpo_expression(index: _NeedIndex, question: str) -> Expression:
    inventory = index.needs("inventory")
    if len(inventory) < 2:
        raise ValueError("DPO requires prior and current inventory")
    inventory_change = _op(
        Operation.SUB,
        _ref(inventory[-1].need_id),
        _ref(inventory[0].need_id),
    )
    denominator = _op(
        Operation.ADD,
        _ref(index.one("cost_of_goods_sold").need_id),
        inventory_change,
    )
    return _op(
        Operation.MUL,
        _days_constant(question),
        _op(Operation.DIV, index.average("accounts_payable"), denominator),
    )


def _formula_expression(plan: FinanceQueryPlan) -> Optional[Expression]:
    formula_id = plan.formula_id
    if not formula_id:
        return None
    index = _NeedIndex(plan)

    simple_ratios = {
        "fixed_asset_turnover": ("revenue", "property_plant_equipment", True),
        "inventory_turnover": ("cost_of_goods_sold", "inventory", True),
        "return_on_assets": ("net_income", "total_assets", True),
        "asset_turnover": ("revenue", "total_assets", True),
        "operating_cash_flow_ratio": (
            "operating_cash_flow",
            "current_liabilities",
            False,
        ),
        "working_capital_ratio": ("current_assets", "current_liabilities", False),
    }
    if formula_id in simple_ratios:
        numerator, denominator, average_denominator = simple_ratios[formula_id]
        denominator_expression = (
            index.average(denominator)
            if average_denominator
            else _ref(index.one(denominator).need_id)
        )
        return _op(
            Operation.DIV,
            _ref(index.one(numerator).need_id),
            denominator_expression,
        )

    if formula_id in {
        "cogs_margin",
        "multi_year_cogs_margin",
    }:
        return index.ratios_by_period("cost_of_goods_sold", "revenue")
    if formula_id == "capex_percent_revenue":
        expression = index.ratios_by_period("capital_expenditure", "revenue")

        def make_capex_absolute(node: Expression) -> Expression:
            if node.op == Operation.REF and node.operand_id in {
                need.need_id for need in index.needs("capital_expenditure")
            }:
                return _op(Operation.ABS, node)
            if not node.args:
                return node
            return node.model_copy(
                update={"args": tuple(make_capex_absolute(arg) for arg in node.args)}
            )

        return make_capex_absolute(expression)
    if formula_id in {
        "multi_year_net_profit_margin",
    }:
        return index.ratios_by_period("net_income", "revenue")
    if formula_id == "multi_year_operating_income_margin":
        return index.ratios_by_period("operating_income", "revenue")
    if formula_id == "ebitda_margin":
        periods = sorted(
            {
                need.period
                for need in index.needs("operating_income")
                if need.period
            },
            key=_period_sort_key,
        )
        margins = tuple(
            _op(
                Operation.DIV,
                _op(
                    Operation.ADD,
                    _ref(index.at("operating_income", period).need_id),
                    _ref(index.at("depreciation_and_amortization", period).need_id),
                ),
                _ref(index.at("revenue", period).need_id),
            )
            for period in periods
        )
        if not margins:
            raise ValueError("EBITDA margin requires aligned fiscal periods")
        return margins[0] if len(margins) == 1 else _op(Operation.AVG, *margins)
    if formula_id == "year_over_year_change":
        metric = next(
            metric
            for metric in {need.metric for need in plan.evidence_needs}
            if len(index.needs(metric)) >= 2
        )
        needs = index.needs(metric)
        return _op(
            Operation.PERCENT_CHANGE,
            _ref(needs[-1].need_id),
            _ref(needs[0].need_id),
        )
    if formula_id == "free_cash_flow":
        return _op(
            Operation.SUB,
            _ref(index.one("operating_cash_flow").need_id),
            _op(
                Operation.ABS,
                _ref(index.one("capital_expenditure").need_id),
            ),
        )
    if formula_id == "retention_ratio":
        return _op(
            Operation.SUB,
            _one_constant(plan.question),
            _op(
                Operation.DIV,
                _op(
                    Operation.ABS,
                    _ref(index.one("cash_dividends").need_id),
                ),
                _ref(index.one("net_income").need_id),
            ),
        )
    if formula_id == "days_payable_outstanding":
        return _dpo_expression(index, plan.question)
    if formula_id == "cash_conversion_cycle":
        days = _days_constant(plan.question)
        dio = _op(
            Operation.MUL,
            days,
            _op(
                Operation.DIV,
                index.average("inventory"),
                _ref(index.one("cost_of_goods_sold").need_id),
            ),
        )
        dso = _op(
            Operation.MUL,
            days,
            _op(
                Operation.DIV,
                index.average("accounts_receivable"),
                _ref(index.one("revenue").need_id),
            ),
        )
        return _op(
            Operation.SUB,
            _op(Operation.ADD, dio, dso),
            _dpo_expression(index, plan.question),
        )
    return None


def _answer_unit(answer_kind: AnswerKind, currency: Optional[str]) -> UnitKind:
    mapping = {
        AnswerKind.PERCENTAGE: UnitKind.PERCENT,
        AnswerKind.RATIO: UnitKind.RATIO,
        AnswerKind.DAYS: UnitKind.DAYS,
    }
    if answer_kind in mapping:
        return mapping[answer_kind]
    if answer_kind is AnswerKind.AMOUNT and currency:
        return UnitKind.MONEY
    return UnitKind.NUMBER


def _answer_period(plan: FinanceQueryPlan) -> Optional[str]:
    labels = sorted(
        set(period.label for period in plan.periods),
        key=_period_sort_key,
    )
    if not labels:
        return None
    range_formulas = {
        "multi_year_net_profit_margin",
        "multi_year_operating_income_margin",
        "multi_year_cogs_margin",
        "year_over_year_change",
    }
    repeated_period_formulas = {
        "capex_percent_revenue",
        "ebitda_margin",
    }
    if plan.formula_id in range_formulas or (
        plan.formula_id in repeated_period_formulas and len(labels) > 1
    ):
        return "-".join(labels)
    return labels[-1]


def build_finance_question_spec(
    plan: FinanceQueryPlan,
) -> Optional[FinanceQuestionSpec]:
    """Return a strict pre-generation contract or ``None`` when unresolved.

    Returning ``None`` is intentional: callers must abstain or use the legacy
    text verifier rather than letting the generated program define its own
    formula or output semantics.
    """

    if not plan.requires_calculation or plan.unresolved_constraints:
        return None
    if len(plan.entities) != 1 or not plan.answer_metric:
        return None
    period = _answer_period(plan)
    if period is None:
        return None
    try:
        expression = _formula_expression(plan)
    except (KeyError, StopIteration, ValueError):
        return None
    if expression is None:
        return None

    answer_unit = _answer_unit(plan.output.answer_kind, plan.output.currency)
    scale_value = plan.output.scale.value if plan.output.scale is not None else "one"
    if scale_value == "unit" or answer_unit in {
        UnitKind.RATIO,
        UnitKind.PERCENT,
        UnitKind.BASIS_POINTS,
        UnitKind.DAYS,
    }:
        scale_value = "one"
    answer_currency = (
        plan.output.currency if answer_unit == UnitKind.MONEY else None
    )
    try:
        return FinanceQuestionSpec(
            entity=plan.entities[0].canonical_name,
            period=period,
            metric=plan.answer_metric,
            expression=expression,
            unit=answer_unit,
            currency=answer_currency,
            scale=Scale(scale_value),
            rounding=RoundingSpec(
                places=(
                    plan.output.decimal_places
                    if plan.output.decimal_places is not None
                    else 2
                )
            ),
            operands=tuple(
                FinanceOperandSpec(
                    id=need.need_id,
                    entity=need.entity or plan.entities[0].canonical_name,
                    period=need.period or period,
                    metric=need.metric,
                    unit=UnitKind.MONEY,
                    currency=plan.output.currency,
                )
                for need in plan.evidence_needs
            ),
        )
    except (TypeError, ValueError):
        # Missing output semantics are not permission for generation to choose
        # them. The orchestrator will fail closed when no contract is returned.
        return None


def finance_program_prompt_contract(spec: FinanceQuestionSpec) -> str:
    """Return compact JSON planning metadata for the generation prompt."""

    return spec.model_dump_json(exclude_none=True)


__all__ = ["build_finance_question_spec", "finance_program_prompt_contract"]
