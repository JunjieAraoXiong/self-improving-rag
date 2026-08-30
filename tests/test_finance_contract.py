"""Tests for binding query understanding to typed finance execution."""

import json
from pathlib import Path

from src.finance_contract import (
    build_finance_question_spec,
    finance_program_prompt_contract,
)
from src.finance_program import UnitKind
from src.query_understanding import compile_finance_query


def test_compiled_formula_becomes_a_full_trusted_contract():
    plan = compile_finance_query(
        "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? "
        "Fixed asset turnover ratio is defined as: FY2019 revenue / "
        "(average PP&E between FY2018 and FY2019). Round your answer to two "
        "decimal places."
    )

    spec = build_finance_question_spec(plan)

    assert spec is not None
    assert spec.entity == "ACTIVISIONBLIZZARD"
    assert spec.period == "FY2019"
    assert spec.metric == "fixed_asset_turnover"
    assert spec.unit is UnitKind.RATIO
    assert spec.rounding.places == 2
    assert {operand.id for operand in spec.operands} == {
        need.need_id for need in plan.evidence_needs
    }
    assert {
        (operand.metric, operand.period) for operand in spec.operands
    } == {
        (need.metric, need.period) for need in plan.evidence_needs
    }
    assert json.loads(finance_program_prompt_contract(spec))["expression"]["op"] == "div"


def test_every_allowlisted_financebench_calculation_compiles_to_a_contract():
    dataset = Path("data/question_sets/financebench_open_source.jsonl")
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    row_plans = [
        (row, compile_finance_query(row["question"])) for row in rows
    ]
    calculation_plans = [plan for _, plan in row_plans if plan.requires_calculation]
    trusted_plans = [plan for plan in calculation_plans if plan.formula_id]
    unsupported_plans = [plan for plan in calculation_plans if not plan.formula_id]

    assert len(trusted_plans) == 30
    assert all(build_finance_question_spec(plan) is not None for plan in trusted_plans)
    # Every row explicitly labeled as numerical reasoning must enter the typed
    # calculation boundary.  The small formula allowlist covers only a subset;
    # all other derived/comparison questions must fail closed rather than fall
    # through to the lower-assurance extraction path.
    numerical_plans = [
        plan
        for row, plan in row_plans
        if "Numerical reasoning" in str(row.get("question_reasoning"))
    ]
    assert len(numerical_plans) == 57
    noncalculation_numerical = [
        plan for plan in numerical_plans if not plan.requires_calculation
    ]
    assert len(noncalculation_numerical) == 4
    assert all(
        plan.task_type.value == "qualitative"
        and plan.question.lower().startswith("what drove")
        for plan in noncalculation_numerical
    )
    assert unsupported_plans
    assert all(
        "formula:unresolved" in plan.unresolved_constraints
        for plan in unsupported_plans
    )
    assert all(build_finance_question_spec(plan) is None for plan in unsupported_plans)


def test_unknown_calculation_formula_fails_closed():
    plan = compile_finance_query(
        "Compute Acme's FY2023 mystery efficiency by multiplying three unknown inputs."
    )

    assert plan.requires_calculation
    assert "formula:unresolved" in plan.unresolved_constraints
    assert build_finance_question_spec(plan) is None


def test_ratio_output_does_not_inherit_operand_currency_or_display_scale():
    plan = compile_finance_query(
        "Calculate Apple's current ratio in FY2022 using USD millions."
    )

    spec = build_finance_question_spec(plan)

    assert spec is not None
    assert spec.unit is UnitKind.RATIO
    assert spec.currency is None
    assert spec.scale.value == "one"
    assert {operand.currency for operand in spec.operands} == {"USD"}


def test_conflicting_user_formula_cannot_receive_a_trusted_contract():
    plan = compile_finance_query(
        "What was Apple's current ratio in FY2023? Current ratio is defined as: "
        "(current assets - inventory) / current liabilities."
    )

    assert plan.requires_calculation
    assert build_finance_question_spec(plan) is None


def test_calendar_and_point_in_time_periods_cannot_receive_fiscal_contracts():
    calendar_plan = compile_finance_query(
        "Calculate Apple's current ratio for calendar year 2023."
    )
    date_plan = compile_finance_query(
        "Calculate Apple's current ratio as of 2023-12-31."
    )

    assert build_finance_question_spec(calendar_plan) is None
    assert build_finance_question_spec(date_plan) is None
