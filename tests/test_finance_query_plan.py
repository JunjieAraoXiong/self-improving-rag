"""Offline tests for deterministic finance query compilation."""

import json

from src.query_understanding.finance_plan import (
    AnswerKind,
    MagnitudeScale,
    PeriodKind,
    TaskType,
    compile_finance_query,
)


def _need_keys(plan):
    return {(need.metric, need.period) for need in plan.evidence_needs}


def test_compiles_fixed_asset_turnover_into_operand_evidence_needs():
    plan = compile_finance_query(
        "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? "
        "Fixed asset turnover ratio is defined as: FY2019 revenue / "
        "(average PP&E between FY2018 and FY2019). Round your answer to two "
        "decimal places. Use the statement of income and statement of financial position."
    )

    assert plan.task_type is TaskType.CALCULATION
    assert [entity.canonical_name for entity in plan.entities] == ["ACTIVISIONBLIZZARD"]
    assert plan.output.answer_kind is AnswerKind.RATIO
    assert plan.output.decimal_places == 2
    assert plan.formula_hint == "revenue / average(property_plant_equipment)"
    assert _need_keys(plan) == {
        ("revenue", "FY2019"),
        ("property_plant_equipment", "FY2018"),
        ("property_plant_equipment", "FY2019"),
    }
    assert plan.source_hints == ("balance_sheet", "income_statement")


def test_compiles_amount_currency_scale_and_statement_constraint():
    plan = compile_finance_query(
        "What is the FY2018 capital expenditure amount (in USD millions) for 3M? "
        "Rely on the cash flow statement."
    )

    assert plan.task_type is TaskType.EXTRACTION
    assert plan.output.answer_kind is AnswerKind.AMOUNT
    assert plan.output.currency == "USD"
    assert plan.output.scale is MagnitudeScale.MILLION
    assert {(need.metric, need.period) for need in plan.evidence_needs} == {
        ("capital_expenditure", "FY2018")
    }
    assert plan.evidence_needs[0].source_types == ("sec_filing",)


def test_attributable_does_not_accidentally_request_table_output():
    plan = compile_finance_query(
        "What is Amazon's FY2019 net income attributable to shareholders "
        "(in USD millions)?"
    )

    assert plan.output.answer_kind is AnswerKind.AMOUNT
    assert plan.output.presentation == "numeric"
    assert plan.task_type is TaskType.EXTRACTION


def test_generic_yoy_formula_binds_to_the_metric_named_by_the_question():
    plan = compile_finance_query(
        "What is Adobe's year-over-year change in unadjusted operating income "
        "from FY2015 to FY2016 (in units of percents)?"
    )

    assert plan.formula_hint == "(current - prior) / prior"
    assert _need_keys(plan) == {
        ("operating_income", "FY2015"),
        ("operating_income", "FY2016"),
    }


def test_expands_fiscal_range_for_multi_year_formula():
    plan = compile_finance_query(
        "What is the FY2017 - FY2019 3 year average of capex as a % of revenue "
        "for Activision Blizzard? Answer in units of percents and round to one decimal place."
    )

    assert [period.year for period in plan.periods] == [2017, 2018, 2019]
    assert plan.output.answer_kind is AnswerKind.PERCENTAGE
    assert plan.output.decimal_places == 1
    assert _need_keys(plan) == {
        (metric, f"FY{year}")
        for metric in ("capital_expenditure", "revenue")
        for year in (2017, 2018, 2019)
    }


def test_longest_company_alias_wins_and_canonical_entity_is_deduplicated():
    plan = compile_finance_query("What was CVS Health revenue in FY2022?")

    assert len(plan.entities) == 1
    assert plan.entities[0].surface_form == "CVS Health"
    assert plan.entities[0].canonical_name == "CVSHEALTH"


def test_conceptual_query_abstains_from_inventing_entity_and_time_constraints():
    plan = compile_finance_query("What is the difference between NOI and FFO for REITs?")

    assert plan.entities == ()
    assert plan.periods == ()
    assert set(plan.constraint_abstentions) == {"entity", "period"}
    assert plan.task_type is TaskType.COMPARISON
    assert len(plan.evidence_needs) == 1
    assert plan.evidence_needs[0].entity is None
    assert plan.evidence_needs[0].period is None


def test_finance_acronyms_are_not_mistaken_for_tickers():
    plan = compile_finance_query(
        "What drove the reduction in SG&A expense and adjusted EPS in FY2023?"
    )

    assert plan.entities == ()
    assert "entity" in plan.constraint_abstentions


def test_explicit_ticker_syntax_is_supported_without_loose_all_caps_guessing():
    plan = compile_finance_query("What was ticker AAPL revenue in FY2023?")

    assert [entity.canonical_name for entity in plan.entities] == ["AAPL"]


def test_relative_reporting_period_remains_unresolved_even_with_wall_clock_anchor():
    plan = compile_finance_query(
        "Summarize Costco's latest quarter results.",
        as_of="2026-08-29",
    )

    assert [entity.canonical_name for entity in plan.entities] == ["COSTCO"]
    assert len(plan.periods) == 1
    assert plan.periods[0].kind is PeriodKind.RELATIVE
    assert plan.periods[0].resolved is False
    assert plan.periods[0].anchor == "2026-08-29"
    assert plan.unresolved_constraints == ("period:latest reported quarter",)


def test_plan_to_dict_is_json_serializable_and_stable():
    plan = compile_finance_query(
        "What is Amazon's FY2017 days payable outstanding (DPO)? Round to two decimal places."
    )
    payload = plan.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["task_type"] == "calculation"
    assert payload["output"]["decimal_places"] == 2
    assert all(need["need_id"].startswith("need:") for need in payload["evidence_needs"])


def test_current_ratio_and_yoy_growth_compile_to_known_formulas():
    current_ratio = compile_finance_query(
        "What was Apple's current ratio in FY2022?"
    )
    yoy_growth = compile_finance_query(
        "What was Apple's revenue growth year-over-year from FY2021 to FY2022?"
    )

    assert current_ratio.requires_calculation
    assert current_ratio.formula_id == "working_capital_ratio"
    assert {need.metric for need in current_ratio.evidence_needs} == {
        "current_assets",
        "current_liabilities",
    }
    assert yoy_growth.requires_calculation
    assert yoy_growth.formula_id == "year_over_year_change"
    assert [need.period for need in yoy_growth.evidence_needs] == [
        "FY2021",
        "FY2022",
    ]


def test_conflicting_explicit_formula_fails_closed_instead_of_being_rewritten():
    plan = compile_finance_query(
        "What was Apple's current ratio in FY2023? Current ratio is defined as: "
        "(current assets - inventory) / current liabilities."
    )

    assert plan.task_type is TaskType.CALCULATION
    assert plan.formula_id is None
    assert plan.formula_hint is None
    assert plan.unresolved_constraints == (
        "formula:explicit_definition_untrusted",
    )
    assert {need.metric for need in plan.evidence_needs} == {
        "current_assets",
        "inventory",
        "current_liabilities",
    }


def test_allowlisted_definition_cannot_be_a_prefix_for_an_extra_operation():
    plan = compile_finance_query(
        "What was Apple's current ratio in FY2023? Current ratio is defined as: "
        "current assets / current liabilities, then multiplied by 2."
    )

    assert plan.requires_calculation
    assert plan.formula_id is None
    assert "formula:explicit_definition_untrusted" in plan.unresolved_constraints

    second_clause_plan = compile_finance_query(
        "What was Apple's current ratio in FY2023? Current ratio is defined as: "
        "current assets / current liabilities. Instead, use current liabilities "
        "/ current assets for this answer."
    )
    assert second_clause_plan.formula_id is None
    assert "formula:explicit_definition_untrusted" in (
        second_clause_plan.unresolved_constraints
    )


def test_equation_and_computed_by_variants_cannot_bypass_formula_allowlist():
    questions = (
        "Apple FY2023 current ratio = current liabilities / current assets.",
        "Apple FY2023 ROA = total assets / net income.",
        "Apple FY2023 current ratio formula: current liabilities / current assets.",
        "Apple FY2023 current ratio is computed by current liabilities divided "
        "by current assets.",
    )

    for question in questions:
        plan = compile_finance_query(question)
        assert plan.task_type is TaskType.CALCULATION
        assert plan.formula_id is None
        assert "formula:explicit_definition_untrusted" in (
            plan.unresolved_constraints
        )


def test_natural_language_arithmetic_cannot_bypass_formula_allowlist():
    questions = (
        "Apple FY2023 current ratio equals current liabilities over current assets.",
        "Compute Apple's FY2023 current ratio as the quotient of current "
        "liabilities and current assets.",
        "For Apple's FY2023 current ratio, take current liabilities as numerator "
        "and current assets as denominator.",
        "Apple FY2023 current ratio means current liabilities over current assets.",
        "For Apple's FY2023 current ratio, current liabilities are divided into "
        "current assets.",
    )

    for question in questions:
        plan = compile_finance_query(question)
        assert plan.task_type is TaskType.CALCULATION
        assert plan.formula_id is None
        assert "formula:explicit_definition_untrusted" in (
            plan.unresolved_constraints
        )


def test_formula_constants_basis_and_modifiers_require_complete_definition():
    questions = (
        "What is Apple's FY2023 DPO on a 360-day convention?",
        "What is Apple's FY2023 DPO using 366 instead of the conventional year?",
        "What is Apple's FY2023 ROA based on ending total assets?",
        "What is Apple's FY2023 asset turnover based on ending total assets "
        "rather than average total assets?",
        "What is Apple's FY2023 free cash flow before capital expenditures?",
    )

    for question in questions:
        plan = compile_finance_query(question)
        assert plan.task_type is TaskType.CALCULATION
        assert plan.formula_id is None
        assert "formula:explicit_definition_untrusted" in (
            plan.unresolved_constraints
        )


def test_single_metric_arithmetic_is_not_downgraded_to_extraction():
    plan = compile_finance_query("Calculate 10% of Apple's FY2023 revenue.")

    assert plan.task_type is TaskType.CALCULATION
    assert plan.formula_id is None
    assert "formula:unresolved" in plan.unresolved_constraints


def test_unknown_bundled_derived_questions_fail_closed_as_calculations():
    questions = (
        "What was MGM's interest coverage ratio using FY2022 Adjusted EBIT "
        "as the numerator and annual Interest Expense as the denominator?",
        "What percent of Ulta Beauty's total spend on stock repurchases for "
        "FY 2023 occurred in Q4 of FY2023?",
        "Based on the information provided primarily in the statement of "
        "income, what is the FY2018 - FY2019 change in unadjusted operating "
        "income % margin for Walmart? Answer in units of percents and round "
        "to one decimal place.",
        "According to the details clearly outlined within the P&L statement "
        "and the statement of cash flows, what is the FY2015 depreciation and "
        "amortization (D&A from cash flow statement) % margin for AMD?",
        "Did JnJ's net earnings as a percent of sales increase in Q2 of FY2023 "
        "compared to Q2 of FY2022?",
        "What is the FY2019 - FY2020 total revenue growth rate for Block?",
        "What is Lockheed Martin's 2 year total revenue CAGR from FY2020 to "
        "FY2022?",
        "How much has the effective tax rate of American Express changed "
        "between FY2021 and FY2022?",
        "Does American Water Works have positive working capital based on "
        "FY2022 data?",
        "Has AMCOR's quick ratio improved or declined between FY2023 and FY2022?",
        "Among operations, investing, and financing activities, which brought "
        "in the most (or lost the least) cash flow for AMD in FY2022?",
        "Has Microsoft increased its debt on balance sheet between FY2023 and "
        "the FY2022 period?",
    )

    for question in questions:
        plan = compile_finance_query(question)
        assert plan.task_type is TaskType.CALCULATION
        assert plan.formula_id is None
        assert "formula:unresolved" in plan.unresolved_constraints


def test_formula_intent_cannot_hide_behind_screening_or_monitoring_words():
    questions = (
        "Monitor current ratio, calculated as current assets divided by "
        "current liabilities, each quarter.",
        "Keep track of debt-to-equity ratio computed as total debt / equity.",
        "Screen companies whose operating margin equals operating income "
        "divided by revenue.",
        "Find all companies with EBITDA margin computed as EBITDA divided by "
        "revenue above 20%.",
    )

    for question in questions:
        plan = compile_finance_query(question)
        assert plan.task_type is TaskType.CALCULATION
        assert plan.unresolved_constraints


def test_discussing_or_extracting_a_named_derived_metric_is_not_arithmetic():
    expected = {
        "What did management attribute the revenue growth rate to?": (
            TaskType.QUALITATIVE
        ),
        "What was the reported revenue growth rate for FY2022?": (
            TaskType.EXTRACTION
        ),
        "Describe the company's working capital policy.": TaskType.QUALITATIVE,
        "Did management discuss positive working capital?": TaskType.QUALITATIVE,
        "What is the definition of CAGR?": TaskType.QUALITATIVE,
    }

    for question, task_type in expected.items():
        plan = compile_finance_query(question)
        assert plan.task_type is task_type
        assert "formula:unresolved" not in plan.unresolved_constraints


def test_bundled_extraction_boilerplate_is_an_exact_narrow_exception():
    exact_question = (
        "How much (in USD billions) did American Water Works pay out in cash "
        "dividends for FY2020? Compute or extract the answer by primarily using "
        "the details outlined in the statement of cash flows."
    )
    exact_plan = compile_finance_query(exact_question)
    near_match_plan = compile_finance_query(
        exact_question.replace("primarily using", "mostly using")
    )

    assert exact_plan.task_type is TaskType.EXTRACTION
    assert near_match_plan.task_type is TaskType.CALCULATION
    assert "formula:unresolved" in near_match_plan.unresolved_constraints


def test_generic_yoy_operation_requires_exactly_one_named_metric():
    plan = compile_finance_query(
        "Calculate Apple's year-over-year change in revenue and net income "
        "from FY2022 to FY2023."
    )

    assert plan.requires_calculation
    assert plan.formula_id is None
    assert "formula:ambiguous_metric" in plan.unresolved_constraints


def test_calendar_year_calculation_never_becomes_a_fiscal_year_contract():
    plan = compile_finance_query(
        "Calculate Apple's current ratio for calendar year 2023."
    )

    assert [(period.kind, period.label) for period in plan.periods] == [
        (PeriodKind.CALENDAR_YEAR, "CY2023")
    ]
    assert {need.period for need in plan.evidence_needs} == {"CY2023"}
    assert "period_semantics:calendar_year:CY2023" in plan.unresolved_constraints

    bare_year_plan = compile_finance_query(
        "Calculate Apple's current ratio during 2023."
    )
    assert [(period.kind, period.label) for period in bare_year_plan.periods] == [
        (PeriodKind.CALENDAR_YEAR, "2023")
    ]
    assert {need.period for need in bare_year_plan.evidence_needs} == {"2023"}
    assert "period_semantics:calendar_year:2023" in (
        bare_year_plan.unresolved_constraints
    )


def test_point_in_time_date_is_preserved_and_marked_for_resolution():
    plan = compile_finance_query(
        "Calculate Apple's current ratio as of December 31, 2023."
    )

    assert [(period.kind, period.label) for period in plan.periods] == [
        (PeriodKind.DATE, "2023-12-31")
    ]
    assert {need.period for need in plan.evidence_needs} == {"2023-12-31"}
    assert "period_semantics:date:2023-12-31" in plan.unresolved_constraints
