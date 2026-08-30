"""Adversarial tests for evidence-bound typed financial execution."""

import json
from decimal import Decimal

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from src.finance_program import (
    FinanceOperandSpec,
    FinanceQuestionSpec,
    FinanceProgram,
    IssueCode,
    ProgramExecutionError,
    execute_program,
    parse_finance_response,
    repair_program_result,
    verify_program,
)


QUICK_QUOTE = (
    "3M Company Consolidated Balance Sheet (Dollars in millions) "
    "June 30, 2023 Total current assets 15,754 Total inventories 5,280 "
    "Total current liabilities 10,936"
)


def _doc(
    content: str = QUICK_QUOTE,
    *,
    company: str = "3M",
    source: str = "3M_2023Q2_10Q.pdf",
    currency: str = "USD",
) -> Document:
    return Document(
        page_content=content,
        metadata={"company": company, "source": source, "currency": currency},
    )


def _money_operand(
    operand_id: str,
    value: str,
    value_text: str,
    metric: str,
    *,
    quote: str = QUICK_QUOTE,
    doc_id: str = "Doc1",
    entity: str = "3M",
    period: str = "FY2023Q2",
    period_label: str = "June 30, 2023",
    currency: str = "USD",
    scale: str = "million",
    metric_label: str | None = None,
    row_label: str | None = None,
    column_label: str | None = None,
) -> dict:
    evidence = {
        "doc_id": doc_id,
        "quote": quote,
        "value_text": value_text,
        "metric_label": metric_label or metric,
        "period_label": period_label,
    }
    if row_label is not None:
        evidence["row_label"] = row_label
    if column_label is not None:
        evidence["column_label"] = column_label
    return {
        "id": operand_id,
        "value": value,
        "currency": currency,
        "scale": scale,
        "unit": "money",
        "entity": entity,
        "period": period,
        "metric": metric,
        "evidence": evidence,
    }


def _quick_program_dict(answer_value: str = "0.96") -> dict:
    return {
        "schema_version": "1.0",
        "answer": {
            "value": answer_value,
            "unit": "ratio",
            "entity": "3M",
            "period": "FY2023Q2",
            "metric": "quick_ratio",
            "rounding": {"places": 2, "mode": "half_up"},
        },
        "operands": [
            _money_operand(
                "current_assets",
                "15754",
                "15,754",
                "Total current assets",
            ),
            _money_operand(
                "inventory",
                "5280",
                "5,280",
                "Total inventories",
            ),
            _money_operand(
                "current_liabilities",
                "10936",
                "10,936",
                "Total current liabilities",
            ),
        ],
        "expression": {
            "op": "div",
            "args": [
                {
                    "op": "sub",
                    "args": [
                        {"op": "ref", "operand_id": "current_assets"},
                        {"op": "ref", "operand_id": "inventory"},
                    ],
                },
                {"op": "ref", "operand_id": "current_liabilities"},
            ],
        },
    }


def _codes(result) -> set[str]:
    return {issue.code.value for issue in result.issues}


def test_quick_ratio_derived_result_passes_without_verbatim_final_value():
    assert "0.96" not in QUICK_QUOTE
    program = FinanceProgram.model_validate(_quick_program_dict())

    result = verify_program(
        program,
        [_doc()],
        "What is 3M's quick ratio for Q2 of FY2023?",
    )

    assert result.passed is True
    assert result.rendered_answer == "0.96x"
    assert result.evidence_coverage == 1.0
    assert result.execution is not None
    assert abs(
        result.execution.value - (Decimal("10474") / Decimal("10936"))
    ) < Decimal("1e-27")
    assert result.execution.trace[-1].op == "div"


def test_parser_strips_one_program_block_and_validates_json():
    program_json = json.dumps(_quick_program_dict())
    response = f"The verified answer is 0.96x.\n<finance_program>{program_json}</finance_program>"

    parsed = parse_finance_response(response, require_program=True)

    assert parsed.passed is True
    assert parsed.answer_text == "The verified answer is 0.96x."
    assert isinstance(parsed.program, FinanceProgram)
    assert "finance_program" not in parsed.answer_text


def test_parser_reports_missing_malformed_and_multiple_programs():
    optional = parse_finance_response("plain answer")
    required = parse_finance_response("plain answer", require_program=True)
    malformed = parse_finance_response("answer <finance_program>{}")
    multiple = parse_finance_response(
        "<finance_program>{}</finance_program>"
        "<finance_program>{}</finance_program>"
    )

    assert optional.passed and optional.program is None
    assert _codes(required) == {"missing_program"}
    assert _codes(malformed) == {"schema_invalid"}
    assert _codes(multiple) == {"schema_invalid"}


def test_parser_forbids_extra_fields_and_unknown_operator():
    extra = _quick_program_dict()
    extra["backdoor"] = "ignored?"
    parsed_extra = parse_finance_response(
        f"<finance_program>{json.dumps(extra)}</finance_program>"
    )

    unsupported = _quick_program_dict()
    unsupported["expression"]["op"] = "python_eval"
    parsed_unsupported = parse_finance_response(
        f"<finance_program>{json.dumps(unsupported)}</finance_program>"
    )

    assert "schema_invalid" in _codes(parsed_extra)
    assert "unsupported_operator" in _codes(parsed_unsupported)


def test_decimal_values_must_be_json_strings_not_floats():
    raw = _quick_program_dict()
    raw["operands"][0]["value"] = 15754.0

    with pytest.raises(ValidationError):
        FinanceProgram.model_validate(raw)

    exponent_bomb = _quick_program_dict()
    exponent_bomb["operands"][0]["value"] = "0e999999"
    with pytest.raises(ValidationError, match="exponent"):
        FinanceProgram.model_validate(exponent_bomb)

    scaled_overflow = _quick_program_dict()
    scaled_overflow["operands"][0]["value"] = "1e30"
    scaled_overflow["operands"][0]["scale"] = "trillion"
    with pytest.raises(ValidationError, match="scaled absolute value"):
        FinanceProgram.model_validate(scaled_overflow)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), b"1.0"),
        (("answer", "unit"), b"ratio"),
        (("operands", 0, "scale"), b"million"),
        (("expression", "op"), b"div"),
    ],
)
def test_mapping_validation_does_not_coerce_bytes(path, value):
    raw = _quick_program_dict()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        FinanceProgram.model_validate(raw)


def test_quote_must_exist_in_the_specific_cited_document():
    raw = _quick_program_dict()
    for operand in raw["operands"]:
        operand["evidence"]["doc_id"] = "Doc2"
    program = FinanceProgram.model_validate(raw)
    docs = [
        _doc(),
        _doc("A different 3M passage without the quoted balance sheet."),
    ]

    result = verify_program(program, docs, "3M FY2023 quick ratio")

    assert result.passed is False
    assert _codes(result) == {"invalid_citation"}


def test_wrong_operand_value_fails_even_when_all_anchors_are_valid():
    raw = _quick_program_dict()
    raw["operands"][0]["value"] = "15755"
    program = FinanceProgram.model_validate(raw)

    result = verify_program(program, [_doc()], "3M FY2023 quick ratio")

    assert "operand_value_mismatch" in _codes(result)


def test_value_text_cannot_select_a_substring_of_a_larger_number():
    raw = _quick_program_dict()
    raw["operands"][0]["value"] = "15"
    raw["operands"][0]["evidence"]["value_text"] = "15"
    program = FinanceProgram.model_validate(raw)

    result = verify_program(program, [_doc()], "3M FY2023 quick ratio")

    assert "missing_evidence" in _codes(result)


def test_period_year_cannot_masquerade_as_the_metric_value():
    quote = "3M Company Revenue FY2023"
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "2023",
                "unit": "number",
                "entity": "3M",
                "period": "FY2023",
                "metric": "revenue",
                "rounding": {"places": 0},
            },
            "operands": [
                {
                    "id": "revenue",
                    "value": "2023",
                    "unit": "number",
                    "entity": "3M",
                    "period": "FY2023",
                    "metric": "Revenue",
                    "evidence": {
                        "doc_id": "Doc1",
                        "quote": quote,
                        "value_text": "2023",
                        "metric_label": "Revenue",
                        "period_label": "FY2023",
                    },
                }
            ],
            "expression": {"op": "ref", "operand_id": "revenue"},
        }
    )

    result = verify_program(program, [_doc(quote)], "3M FY2023 revenue")

    assert "operand_value_mismatch" in _codes(result)


def test_scale_mismatch_is_not_silently_tolerated():
    raw = _quick_program_dict()
    raw["operands"][0]["scale"] = "billion"
    program = FinanceProgram.model_validate(raw)

    result = verify_program(program, [_doc()], "3M FY2023 quick ratio")

    assert "operand_unit_mismatch" in _codes(result)


def test_ambiguous_table_scale_fails_closed_even_for_scale_one():
    quote = (
        "3M FY2023 Revenue 1 (Dollars in millions) supplemental amounts in thousands"
    )
    operand = _money_operand(
        "revenue",
        "1",
        "1",
        "Revenue",
        quote=quote,
        period="FY2023",
        period_label="FY2023",
        scale="one",
    )
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "1",
                "currency": "USD",
                "unit": "money",
                "entity": "3M",
                "period": "FY2023",
                "metric": "Revenue",
                "rounding": {"places": 0},
            },
            "operands": [operand],
            "expression": {"op": "ref", "operand_id": "revenue"},
        }
    )

    result = verify_program(program, [_doc(quote)], "3M FY2023 revenue")

    assert "operand_unit_mismatch" in _codes(result)


def test_currency_mismatch_is_rejected():
    raw = _quick_program_dict()
    raw["operands"][0]["currency"] = "EUR"
    program = FinanceProgram.model_validate(raw)

    result = verify_program(program, [_doc()], "3M FY2023 quick ratio")

    assert "operand_currency_mismatch" in _codes(result)


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("metric_label", "Invented metric", "operand_metric_mismatch"),
        ("period_label", "December 31, 1999", "operand_period_mismatch"),
        ("row_label", "Invented row", "operand_metric_mismatch"),
        ("column_label", "FY2099", "operand_period_mismatch"),
    ],
)
def test_metric_period_row_and_column_anchors_are_required_in_quote(
    field,
    value,
    expected_code,
):
    raw = _quick_program_dict()
    raw["operands"][0]["evidence"][field] = value
    program = FinanceProgram.model_validate(raw)

    result = verify_program(program, [_doc()], "3M FY2023 quick ratio")

    assert expected_code in _codes(result)


def test_period_semantics_reject_wrong_year_even_if_anchor_text_exists():
    raw = _quick_program_dict()
    raw["operands"][0]["period"] = "FY2022"
    program = FinanceProgram.model_validate(raw)

    result = verify_program(program, [_doc()], "3M FY2023 quick ratio")

    assert "operand_period_mismatch" in _codes(result)


def test_entity_must_match_document_metadata_or_source():
    raw = _quick_program_dict()
    raw["operands"][0]["entity"] = "Adobe"
    program = FinanceProgram.model_validate(raw)

    result = verify_program(program, [_doc()], "3M FY2023 quick ratio")

    assert "operand_entity_mismatch" in _codes(result)


def test_answer_entity_and_period_must_be_supported_by_operand_ledger():
    wrong_entity = _quick_program_dict()
    wrong_entity["answer"]["entity"] = "Adobe"
    entity_result = verify_program(
        FinanceProgram.model_validate(wrong_entity),
        [_doc()],
        "3M FY2023 quick ratio",
    )
    assert "operand_entity_mismatch" in _codes(entity_result)

    wrong_period = _quick_program_dict()
    wrong_period["answer"]["period"] = "FY2024Q2"
    period_result = verify_program(
        FinanceProgram.model_validate(wrong_period),
        [_doc()],
        "3M FY2023 quick ratio",
    )
    assert "operand_period_mismatch" in _codes(period_result)


def test_accounting_negative_can_be_made_positive_only_by_explicit_abs():
    quote = (
        "3M Company Cash Flows (Dollars in millions) FY2022 "
        "Capital expenditures (1,055)"
    )
    operand = _money_operand(
        "capex",
        "-1055",
        "(1,055)",
        "Capital expenditures",
        quote=quote,
        period="FY2022",
        period_label="FY2022",
    )
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "1055",
                "currency": "USD",
                "scale": "million",
                "unit": "money",
                "entity": "3M",
                "period": "FY2022",
                "metric": "capital_expenditures_absolute",
                "rounding": {"places": 0},
            },
            "operands": [operand],
            "expression": {
                "op": "abs",
                "args": [{"op": "ref", "operand_id": "capex"}],
            },
        }
    )

    result = verify_program(
        program,
        [_doc(quote, source="3M_2022_10K.pdf")],
        "What was 3M capital expenditure in FY2022?",
    )

    assert result.passed
    assert result.rendered_answer == "USD 1055 million"


def test_percent_change_and_basis_points_have_explicit_conversions():
    quote = (
        "3M Company (Dollars in millions) FY2021 FY2022 Revenue FY2021 100 "
        "Revenue FY2022 110"
    )
    old = _money_operand(
        "revenue_2021",
        "100",
        "100",
        "Revenue FY2021",
        quote=quote,
        period="FY2021",
        period_label="FY2021",
    )
    new = _money_operand(
        "revenue_2022",
        "110",
        "110",
        "Revenue FY2022",
        quote=quote,
        period="FY2022",
        period_label="FY2022",
    )
    percent_program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "10.0",
                "unit": "percent",
                "entity": "3M",
                "period": "FY2021-FY2022",
                "metric": "revenue_growth",
                "rounding": {"places": 1},
            },
            "operands": [old, new],
            "expression": {
                "op": "percent_change",
                "args": [
                    {"op": "ref", "operand_id": "revenue_2022"},
                    {"op": "ref", "operand_id": "revenue_2021"},
                ],
            },
        }
    )

    percent_result = verify_program(
        percent_program,
        [_doc(quote, source="3M_2022_10K.pdf")],
        "What was 3M revenue growth from FY2021 to FY2022?",
    )
    assert percent_result.passed
    assert percent_result.rendered_answer == "10.0%"

    bps_quote = "3M Company FY2022 Credit spread 150 bps"
    bps_program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "150.0",
                "unit": "basis_points",
                "entity": "3M",
                "period": "FY2022",
                "metric": "credit_spread",
                "rounding": {"places": 1},
            },
            "operands": [
                {
                    "id": "spread",
                    "value": "150",
                    "unit": "basis_points",
                    "entity": "3M",
                    "period": "FY2022",
                    "metric": "Credit spread",
                    "evidence": {
                        "doc_id": "Doc1",
                        "quote": bps_quote,
                        "value_text": "150 bps",
                        "metric_label": "Credit spread",
                        "period_label": "FY2022",
                    },
                }
            ],
            "expression": {"op": "ref", "operand_id": "spread"},
        }
    )
    bps_result = verify_program(
        bps_program,
        [_doc(bps_quote, source="3M_2022_10K.pdf")],
        "What was 3M's FY2022 credit spread?",
    )
    assert bps_result.passed
    assert bps_result.rendered_answer == "150.0 bps"


def test_bare_dimensionless_value_requires_an_explicit_unit_anchor():
    quote = "3M Company FY2022 Gross margin 43.3"
    raw = {
        "answer": {
            "value": "43.3",
            "unit": "percent",
            "entity": "3M",
            "period": "FY2022",
            "metric": "gross_margin",
            "rounding": {"places": 1},
        },
        "operands": [
            {
                "id": "margin",
                "value": "43.3",
                "unit": "percent",
                "entity": "3M",
                "period": "FY2022",
                "metric": "Gross margin",
                "evidence": {
                    "doc_id": "Doc1",
                    "quote": quote,
                    "value_text": "43.3",
                    "metric_label": "Gross margin",
                    "period_label": "FY2022",
                },
            }
        ],
        "expression": {"op": "ref", "operand_id": "margin"},
    }

    unanchored = verify_program(
        FinanceProgram.model_validate(raw),
        [_doc(quote, source="3M_2022_10K.pdf")],
        "What was 3M's gross margin percentage in FY2022?",
    )
    assert "operand_unit_mismatch" in _codes(unanchored)

    raw["operands"][0]["evidence"]["row_label"] = "Gross margin %"
    quote_with_anchor = "3M Company FY2022 Gross margin % 43.3"
    raw["operands"][0]["evidence"]["quote"] = quote_with_anchor
    anchored = verify_program(
        FinanceProgram.model_validate(raw),
        [_doc(quote_with_anchor, source="3M_2022_10K.pdf")],
        "What was 3M's gross margin percentage in FY2022?",
    )
    assert anchored.passed
    assert anchored.rendered_answer == "43.3%"


def test_nested_dpo_program_supports_question_bound_days_constant():
    quote = (
        "Amazon (Dollars in millions) FY2016 FY2017 Accounts payable FY2016 10 "
        "Accounts payable FY2017 14 Cost of sales FY2017 40 "
        "Inventory FY2016 5 Inventory FY2017 7"
    )
    operands = [
        _money_operand(
            "ap_2016", "10", "10", "Accounts payable FY2016",
            quote=quote, entity="Amazon", period="FY2016", period_label="FY2016",
        ),
        _money_operand(
            "ap_2017", "14", "14", "Accounts payable FY2017",
            quote=quote, entity="Amazon", period="FY2017", period_label="FY2017",
        ),
        _money_operand(
            "cogs_2017", "40", "40", "Cost of sales FY2017",
            quote=quote, entity="Amazon", period="FY2017", period_label="FY2017",
        ),
        _money_operand(
            "inventory_2016", "5", "5", "Inventory FY2016",
            quote=quote, entity="Amazon", period="FY2016", period_label="FY2016",
        ),
        _money_operand(
            "inventory_2017", "7", "7", "Inventory FY2017",
            quote=quote, entity="Amazon", period="FY2017", period_label="FY2017",
        ),
    ]
    expression = {
        "op": "div",
        "args": [
            {
                "op": "mul",
                "args": [
                    {
                        "op": "const",
                        "value": "365",
                        "unit": "days",
                        "source_text": "365",
                    },
                    {
                        "op": "avg",
                        "args": [
                            {"op": "ref", "operand_id": "ap_2016"},
                            {"op": "ref", "operand_id": "ap_2017"},
                        ],
                    },
                ],
            },
            {
                "op": "add",
                "args": [
                    {"op": "ref", "operand_id": "cogs_2017"},
                    {
                        "op": "sub",
                        "args": [
                            {"op": "ref", "operand_id": "inventory_2017"},
                            {"op": "ref", "operand_id": "inventory_2016"},
                        ],
                    },
                ],
            },
        ],
    }
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "104.29",
                "unit": "days",
                "entity": "Amazon",
                "period": "FY2017",
                "metric": "days_payable_outstanding",
                "rounding": {"places": 2},
            },
            "operands": operands,
            "expression": expression,
        }
    )
    question = "DPO is defined as 365 * average accounts payable / adjusted COGS."

    result = verify_program(
        program,
        [_doc(quote, company="Amazon", source="AMAZON_2017_10K.pdf")],
        question,
    )

    assert result.passed
    assert result.rendered_answer == "104.29 days"


def test_invented_constant_is_rejected_before_execution():
    raw = _quick_program_dict()
    original = raw["expression"]
    raw["expression"] = {
        "op": "mul",
        "args": [
            original,
            {
                "op": "const",
                "value": "999",
                "unit": "number",
                "source_text": "999",
            },
        ],
    }
    raw["answer"]["value"] = "957.80"
    program = FinanceProgram.model_validate(raw)

    result = verify_program(
        program,
        [_doc()],
        "What is 3M's FY2023 quick ratio?",
    )

    assert _codes(result) == {"constant_not_question_bound"}

    substring = _quick_program_dict()
    substring["expression"] = {
        "op": "mul",
        "args": [
            substring["expression"],
            {
                "op": "const",
                "value": "36",
                "unit": "number",
                "source_text": "36",
            },
        ],
    }
    substring["answer"]["value"] = "34.48"
    substring_result = verify_program(
        FinanceProgram.model_validate(substring),
        [_doc()],
        "The formula uses 365 days for 3M.",
    )
    assert _codes(substring_result) == {"constant_not_question_bound"}


def test_division_by_zero_is_a_machine_readable_arithmetic_error():
    quote = "3M FY2023 Numerator 1 Denominator 0"
    operands = []
    for operand_id, value, metric in (
        ("numerator", "1", "Numerator"),
        ("denominator", "0", "Denominator"),
    ):
        operands.append(
            {
                "id": operand_id,
                "value": value,
                "unit": "number",
                "entity": "3M",
                "period": "FY2023",
                "metric": metric,
                "evidence": {
                    "doc_id": "Doc1",
                    "quote": quote,
                    "value_text": value,
                    "metric_label": metric,
                    "period_label": "FY2023",
                },
            }
        )
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "0",
                "unit": "ratio",
                "entity": "3M",
                "period": "FY2023",
                "metric": "test_ratio",
            },
            "operands": operands,
            "expression": {
                "op": "div",
                "args": [
                    {"op": "ref", "operand_id": "numerator"},
                    {"op": "ref", "operand_id": "denominator"},
                ],
            },
        }
    )

    result = verify_program(
        program,
        [_doc(quote, source="3M_2023_10K.pdf")],
        "3M FY2023 test ratio",
    )

    assert _codes(result) == {"arithmetic_error"}
    assert result.execution is None


def test_mixed_currency_addition_fails_safe_execution():
    quote_usd = "Acme FY2023 (Dollars in millions) Revenue 10"
    quote_eur = "Acme FY2023 (EUR in millions) Expense €2 million"
    usd = _money_operand(
        "usd_revenue", "10", "10", "Revenue", quote=quote_usd,
        entity="Acme", period="FY2023", period_label="FY2023",
    )
    eur = _money_operand(
        "eur_expense", "2", "€2 million", "Expense", quote=quote_eur,
        doc_id="Doc2", entity="Acme", period="FY2023", period_label="FY2023",
        currency="EUR", scale="million",
    )
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "12",
                "currency": "USD",
                "scale": "million",
                "unit": "money",
                "entity": "Acme",
                "period": "FY2023",
                "metric": "invalid_sum",
            },
            "operands": [usd, eur],
            "expression": {
                "op": "add",
                "args": [
                    {"op": "ref", "operand_id": "usd_revenue"},
                    {"op": "ref", "operand_id": "eur_expense"},
                ],
            },
        }
    )

    with pytest.raises(ProgramExecutionError) as error:
        execute_program(program)

    assert error.value.code == IssueCode.OPERAND_UNIT_MISMATCH

    verified = verify_program(
        program,
        [
            _doc(quote_usd, company="Acme", source="ACME_2023_10K.pdf"),
            _doc(
                quote_eur,
                company="Acme",
                source="ACME_2023_10K.pdf",
                currency="EUR",
            ),
        ],
        "What is Acme's FY2023 invalid sum?",
    )
    assert "operand_currency_mismatch" in _codes(verified)


def test_expression_result_unit_and_declared_value_are_independently_checked():
    wrong_unit = _quick_program_dict()
    wrong_unit["answer"].update(
        {
            "value": "10474",
            "currency": "USD",
            "scale": "million",
            "unit": "money",
        }
    )
    wrong_unit["expression"] = wrong_unit["expression"]["args"][0]
    # current_liabilities is now unused; remove it to keep the schema honest.
    wrong_unit["operands"] = wrong_unit["operands"][:2]
    wrong_unit_program = FinanceProgram.model_validate(wrong_unit)
    wrong_unit_result = verify_program(
        wrong_unit_program,
        [_doc()],
        "3M FY2023 quick ratio",
    )
    # The expression is money and the answer is also money, so this is valid;
    # now deliberately declare a ratio for the same money expression.
    mismatched = wrong_unit_program.model_dump(mode="json")
    mismatched["answer"].pop("currency")
    mismatched["answer"]["scale"] = "one"
    mismatched["answer"]["unit"] = "ratio"
    mismatched_program = FinanceProgram.model_validate(mismatched)
    mismatched_result = verify_program(
        mismatched_program,
        [_doc()],
        "3M FY2023 quick ratio",
    )

    assert wrong_unit_result.passed
    assert _codes(mismatched_result) == {"result_unit_mismatch"}

    wrong_value = FinanceProgram.model_validate(_quick_program_dict("0.95"))
    wrong_value_result = verify_program(
        wrong_value,
        [_doc()],
        "3M FY2023 quick ratio",
    )
    assert _codes(wrong_value_result) == {"result_value_mismatch"}


def test_local_result_repair_returns_a_new_fully_reverified_program():
    original = FinanceProgram.model_validate(_quick_program_dict("0.95"))
    expected = FinanceQuestionSpec(
        entity="3M",
        period="FY2023Q2",
        metric="quick ratio",
        unit="ratio",
        expression=original.expression,
        operands=tuple(
            FinanceOperandSpec(
                id=operand.id,
                entity=operand.entity,
                period=operand.period,
                metric=operand.metric,
                unit=operand.unit,
                currency=operand.currency,
            )
            for operand in original.operands
        ),
    )

    repaired, verification = repair_program_result(
        original,
        [_doc()],
        "What is the quick ratio?",
        question_spec=expected,
    )

    assert repaired is not None
    assert repaired.answer.value == "0.96"
    assert original.answer.value == "0.95"
    assert verification.passed is True
    assert verification.fully_verified is True
    assert verification.rendered_answer == "0.96x"


def test_rounding_is_decimal_half_up_not_binary_float_or_bankers_rounding():
    quote = "3M FY2023 Numerator 1 Denominator 8"
    operands = [
        {
            "id": "one",
            "value": "1",
            "unit": "number",
            "entity": "3M",
            "period": "FY2023",
            "metric": "Numerator",
            "evidence": {
                "doc_id": "Doc1",
                "quote": quote,
                "value_text": "1",
                "metric_label": "Numerator",
                "period_label": "FY2023",
            },
        },
        {
            "id": "eight",
            "value": "8",
            "unit": "number",
            "entity": "3M",
            "period": "FY2023",
            "metric": "Denominator",
            "evidence": {
                "doc_id": "Doc1",
                "quote": quote,
                "value_text": "8",
                "metric_label": "Denominator",
                "period_label": "FY2023",
            },
        },
    ]
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "0.13",
                "unit": "ratio",
                "entity": "3M",
                "period": "FY2023",
                "metric": "test_ratio",
                "rounding": {"places": 2, "mode": "half_up"},
            },
            "operands": operands,
            "expression": {
                "op": "div",
                "args": [
                    {"op": "ref", "operand_id": "one"},
                    {"op": "ref", "operand_id": "eight"},
                ],
            },
        }
    )

    result = verify_program(
        program,
        [_doc(quote, source="3M_2023_10K.pdf")],
        "3M FY2023 test ratio",
    )

    assert result.passed
    assert result.rendered_answer == "0.13x"


def test_conflicting_values_for_same_typed_fact_are_rejected():
    quote = (
        "3M Company (Dollars in millions) FY2023 Reported revenue 100 "
        "Restated revenue 101"
    )
    first = _money_operand(
        "revenue_a", "100", "100", "Revenue", quote=quote,
        period="FY2023", period_label="FY2023", metric_label="Reported revenue",
    )
    second = _money_operand(
        "revenue_b", "101", "101", "Revenue", quote=quote,
        period="FY2023", period_label="FY2023", metric_label="Restated revenue",
    )
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "100.5",
                "currency": "USD",
                "scale": "million",
                "unit": "money",
                "entity": "3M",
                "period": "FY2023",
                "metric": "Revenue",
                "rounding": {"places": 1},
            },
            "operands": [first, second],
            "expression": {
                "op": "avg",
                "args": [
                    {"op": "ref", "operand_id": "revenue_a"},
                    {"op": "ref", "operand_id": "revenue_b"},
                ],
            },
        }
    )

    result = verify_program(program, [_doc(quote)], "3M FY2023 revenue")

    assert "conflicting_evidence" in _codes(result)


def test_schema_rejects_unknown_unused_duplicate_and_excessively_deep_graphs():
    unknown = _quick_program_dict()
    unknown["expression"]["args"][1]["operand_id"] = "missing"
    with pytest.raises(ValidationError, match="unknown operands"):
        FinanceProgram.model_validate(unknown)

    unused = _quick_program_dict()
    unused["expression"] = {"op": "ref", "operand_id": "current_assets"}
    with pytest.raises(ValidationError, match="unused operands"):
        FinanceProgram.model_validate(unused)

    duplicate = _quick_program_dict()
    duplicate["operands"][1]["id"] = "current_assets"
    duplicate["expression"]["args"][0]["args"][1]["operand_id"] = "current_assets"
    with pytest.raises(ValidationError, match="unique"):
        FinanceProgram.model_validate(duplicate)

    deep = _quick_program_dict()
    expression = {"op": "ref", "operand_id": "current_assets"}
    for _ in range(13):
        expression = {"op": "abs", "args": [expression]}
    deep["expression"] = expression
    deep["operands"] = deep["operands"][:1]
    with pytest.raises(ValidationError, match="depth"):
        FinanceProgram.model_validate(deep)


def test_verification_result_is_json_serializable_for_audit_logs():
    result = verify_program(
        FinanceProgram.model_validate(_quick_program_dict()),
        [_doc()],
        "3M FY2023 quick ratio",
    )

    encoded = json.dumps(result.to_dict(), sort_keys=True)

    assert '"passed": true' in encoded
    assert '"semantic_unit": "ratio"' in encoded
    assert '"evidence_coverage": 1.0' in encoded


def test_verify_program_accepts_raw_mapping_and_returns_schema_issues():
    raw = _quick_program_dict()
    del raw["operands"][0]["evidence"]

    result = verify_program(raw, [_doc()], "3M FY2023 quick ratio")

    assert result.passed is False
    assert "missing_evidence" in _codes(result)


def test_evidence_occurrence_is_enforced_for_repeated_value_text():
    raw = _quick_program_dict()
    raw["operands"][0]["evidence"]["occurrence"] = 2
    program = FinanceProgram.model_validate(raw)

    result = verify_program(program, [_doc()], "3M FY2023 quick ratio")

    assert "missing_evidence" in _codes(result)


def test_model_dump_contract_contains_only_json_safe_schema_fields():
    program = FinanceProgram.model_validate(_quick_program_dict())

    payload = program.model_dump(mode="json")
    reparsed = FinanceProgram.model_validate_json(json.dumps(payload))

    assert reparsed == program
    assert payload["schema_version"] == "1.0"
    assert payload["expression"]["op"] == "div"


def test_parser_rejects_duplicate_json_keys_and_unmatched_extra_tags():
    raw_json = json.dumps(_quick_program_dict()).replace(
        '"schema_version": "1.0"',
        '"schema_version": "1.0", "schema_version": "1.0"',
        1,
    )
    duplicate = parse_finance_response(
        f"<finance_program>{raw_json}</finance_program>"
    )
    stray = parse_finance_response(
        "answer "
        f"<finance_program>{json.dumps(_quick_program_dict())}</finance_program>"
        "</finance_program>"
    )

    assert _codes(duplicate) == {"schema_invalid"}
    assert _codes(stray) == {"schema_invalid"}


def test_declared_metric_must_match_the_evidence_metric_anchor():
    raw = _quick_program_dict()
    raw["operands"][0]["metric"] = "Cash and equivalents"

    result = verify_program(
        FinanceProgram.model_validate(raw),
        [_doc()],
        "3M FY2023 quick ratio",
    )

    assert "operand_metric_mismatch" in _codes(result)


def test_percent_change_rejects_mixed_ratio_and_percent_semantics():
    quote = "Acme FY2023 Trading multiple 2x Gross margin 50%"
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "300",
                "unit": "percent",
                "entity": "Acme",
                "period": "FY2023",
                "metric": "invalid_change",
                "rounding": {"places": 0},
            },
            "operands": [
                {
                    "id": "multiple",
                    "value": "2",
                    "unit": "ratio",
                    "entity": "Acme",
                    "period": "FY2023",
                    "metric": "Trading multiple",
                    "evidence": {
                        "doc_id": "Doc1",
                        "quote": quote,
                        "value_text": "2x",
                        "metric_label": "Trading multiple",
                        "period_label": "FY2023",
                    },
                },
                {
                    "id": "margin",
                    "value": "50",
                    "unit": "percent",
                    "entity": "Acme",
                    "period": "FY2023",
                    "metric": "Gross margin",
                    "evidence": {
                        "doc_id": "Doc1",
                        "quote": quote,
                        "value_text": "50%",
                        "metric_label": "Gross margin",
                        "period_label": "FY2023",
                    },
                },
            ],
            "expression": {
                "op": "percent_change",
                "args": [
                    {"op": "ref", "operand_id": "multiple"},
                    {"op": "ref", "operand_id": "margin"},
                ],
            },
        }
    )

    result = verify_program(
        program,
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 percent change",
    )

    assert _codes(result) == {"operand_unit_mismatch"}


def test_percentage_in_a_dollar_scaled_quote_is_not_mistyped_as_money():
    quote = "3M (Dollars in millions) FY2023 Gross margin 20%"
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "20.0",
                "unit": "percent",
                "entity": "3M",
                "period": "FY2023",
                "metric": "gross_margin",
                "rounding": {"places": 1},
            },
            "operands": [
                {
                    "id": "margin",
                    "value": "20",
                    "unit": "percent",
                    "entity": "3M",
                    "period": "FY2023",
                    "metric": "Gross margin",
                    "evidence": {
                        "doc_id": "Doc1",
                        "quote": quote,
                        "value_text": "20%",
                        "metric_label": "Gross margin",
                        "period_label": "FY2023",
                    },
                }
            ],
            "expression": {"op": "ref", "operand_id": "margin"},
        }
    )

    result = verify_program(
        program,
        [_doc(quote, source="3M_2023_10K.pdf")],
        "3M FY2023 gross margin",
    )

    assert result.passed
    assert result.rendered_answer == "20.0%"


@pytest.mark.parametrize(
    ("quote", "value_text", "metric"),
    [
        ("Acme FY2023 Loss (1,055)", "1,055", "Loss"),
        ("Acme FY2023 Change -5", "5", "Change"),
        ("Acme FY2023 Revenue $2 million", "2", "Revenue"),
        ("Acme FY2023 Margin 5%", "5", "Margin"),
        ("Acme FY2023 Spread 150 bps", "150", "Spread"),
    ],
)
def test_value_text_must_cover_the_complete_signed_scaled_unit_token(
    quote,
    value_text,
    metric,
):
    plain_value = value_text.replace(",", "")
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": plain_value,
                "unit": "number",
                "entity": "Acme",
                "period": "FY2023",
                "metric": metric,
                "rounding": {"places": 0},
            },
            "operands": [
                {
                    "id": "value",
                    "value": plain_value,
                    "unit": "number",
                    "entity": "Acme",
                    "period": "FY2023",
                    "metric": metric,
                    "evidence": {
                        "doc_id": "Doc1",
                        "quote": quote,
                        "value_text": value_text,
                        "metric_label": metric,
                        "period_label": "FY2023",
                    },
                }
            ],
            "expression": {"op": "ref", "operand_id": "value"},
        }
    )

    result = verify_program(
        program,
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        f"Acme FY2023 {metric}",
    )

    assert "missing_evidence" in _codes(result)


def test_numeric_text_inside_a_metric_anchor_cannot_be_used_as_the_value():
    quote = "Acme FY2023 ASC 606 Revenue 100"
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "606",
                "unit": "number",
                "entity": "Acme",
                "period": "FY2023",
                "metric": "Revenue",
                "rounding": {"places": 0},
            },
            "operands": [
                {
                    "id": "revenue",
                    "value": "606",
                    "unit": "number",
                    "entity": "Acme",
                    "period": "FY2023",
                    "metric": "Revenue",
                    "evidence": {
                        "doc_id": "Doc1",
                        "quote": quote,
                        "value_text": "606",
                        "metric_label": "ASC 606 Revenue",
                        "period_label": "FY2023",
                    },
                }
            ],
            "expression": {"op": "ref", "operand_id": "revenue"},
        }
    )

    result = verify_program(
        program,
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 revenue",
    )

    assert "operand_value_mismatch" in _codes(result)


@pytest.mark.parametrize(
    "revenue_label",
    ["Revenue", "Revenue 100 Expenses"],
)
def test_values_cannot_be_swapped_or_anchor_widened_between_metrics(
    revenue_label,
):
    quote = "Acme FY2023 Revenue 100 Expenses 200"
    operands = []
    for operand_id, metric, value, label in (
        ("revenue", "Revenue", "200", revenue_label),
        ("expenses", "Expenses", "100", "Expenses"),
    ):
        operands.append(
            {
                "id": operand_id,
                "value": value,
                "unit": "number",
                "entity": "Acme",
                "period": "FY2023",
                "metric": metric,
                "evidence": {
                    "doc_id": "Doc1",
                    "quote": quote,
                    "value_text": value,
                    "metric_label": label,
                    "period_label": "FY2023",
                },
            }
        )
    program = FinanceProgram.model_validate(
        {
            "answer": {
                "value": "100",
                "unit": "number",
                "entity": "Acme",
                "period": "FY2023",
                "metric": "profit",
                "rounding": {"places": 0},
            },
            "operands": operands,
            "expression": {
                "op": "sub",
                "args": [
                    {"op": "ref", "operand_id": "revenue"},
                    {"op": "ref", "operand_id": "expenses"},
                ],
            },
        }
    )

    result = verify_program(
        program,
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 profit",
    )

    assert "operand_metric_mismatch" in _codes(result)


def test_value_association_cannot_cross_a_missing_row_or_sentence_boundary():
    quote = "Acme FY2023 Revenue N/A. Expenses 200"
    raw = {
        "answer": {
            "value": "200",
            "unit": "number",
            "entity": "Acme",
            "period": "FY2023",
            "metric": "Revenue",
            "rounding": {"places": 0},
        },
        "operands": [
            {
                "id": "revenue",
                "value": "200",
                "unit": "number",
                "entity": "Acme",
                "period": "FY2023",
                "metric": "Revenue",
                "evidence": {
                    "doc_id": "Doc1",
                    "quote": quote,
                    "value_text": "200",
                    "metric_label": "Revenue",
                    "period_label": "FY2023",
                },
            }
        ],
        "expression": {"op": "ref", "operand_id": "revenue"},
    }

    result = verify_program(
        FinanceProgram.model_validate(raw),
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 revenue",
        question_spec={
            "entity": "Acme",
            "period": "FY2023",
            "metric": "Revenue",
            "unit": "number",
            "rounding": {"places": 0},
            "expression": raw["expression"],
        },
    )

    assert "operand_metric_mismatch" in _codes(result)

    for field in ("column_label", "period_label"):
        widened = FinanceProgram.model_validate(raw).model_dump(mode="json")
        widened["operands"][0]["evidence"][field] = "N/A. Expenses"
        widened_result = verify_program(
            FinanceProgram.model_validate(widened),
            [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
            "Acme FY2023 revenue",
            question_spec={
                "entity": "Acme",
                "period": "FY2023",
                "metric": "Revenue",
                "unit": "number",
                "rounding": {"places": 0},
                "expression": raw["expression"],
            },
        )
        assert not widened_result.passed
        assert _codes(widened_result) & {
            "operand_metric_mismatch",
            "operand_period_mismatch",
        }


def test_trusted_question_spec_binds_entity_period_metric_and_formula():
    program = FinanceProgram.model_validate(_quick_program_dict())
    expected = FinanceQuestionSpec(
        entity="3M",
        period="FY2023Q2",
        metric="quick ratio",
        unit="ratio",
        expression=program.expression,
        operands=tuple(
            FinanceOperandSpec(
                id=operand.id,
                entity=operand.entity,
                period=operand.period,
                metric=operand.metric,
                unit=operand.unit,
                currency=operand.currency,
            )
            for operand in program.operands
        ),
    )
    valid = verify_program(
        program,
        [_doc()],
        "What is the quick ratio?",
        question_spec=expected,
    )
    assert valid.passed
    assert valid.assurance_level.value == "evidence_arithmetic"
    assert not valid.fully_verified

    full = verify_program(
        program,
        [_doc()],
        "What is the quick ratio?",
        question_spec=expected,
        answer_text="0.96x",
        require_full_contract=True,
    )
    assert full.passed and full.fully_verified
    assert full.assurance_level.value == "full_contract"

    missing_contract = verify_program(
        program,
        [_doc()],
        "What is the quick ratio?",
        require_full_contract=True,
    )
    assert _codes(missing_contract) == {"unsupported_claim"}

    mismatches = {
        "operand_entity_mismatch": expected.model_copy(update={"entity": "Adobe"}),
        "operand_period_mismatch": expected.model_copy(
            update={"period": "FY2022Q2"}
        ),
        "operand_metric_mismatch": expected.model_copy(
            update={"metric": "debt to equity ratio"}
        ),
        "formula_mismatch": expected.model_copy(
            update={
                "expression": {
                    "op": "div",
                    "args": [
                        {"op": "ref", "operand_id": "current_assets"},
                        {"op": "ref", "operand_id": "current_liabilities"},
                    ],
                }
            }
        ),
    }
    for issue_code, wrong_spec in mismatches.items():
        result = verify_program(
            program,
            [_doc()],
            "Model-controlled prose cannot override the trusted spec",
            question_spec=wrong_spec,
        )
        assert issue_code in _codes(result)

    wrong_output = program.model_dump(mode="json")
    wrong_output["answer"].update({"value": "95.78", "unit": "percent"})
    output_result = verify_program(
        FinanceProgram.model_validate(wrong_output),
        [_doc()],
        "What is the quick ratio?",
        question_spec=expected,
        answer_text="95.78%",
    )
    assert "result_unit_mismatch" in _codes(output_result)


def test_trusted_operand_ids_prevent_semantic_value_swaps():
    original = FinanceProgram.model_validate(_quick_program_dict())
    expected = FinanceQuestionSpec(
        entity="3M",
        period="FY2023Q2",
        metric="quick ratio",
        unit="ratio",
        expression=original.expression,
        operands=tuple(
            FinanceOperandSpec(
                id=operand.id,
                entity=operand.entity,
                period=operand.period,
                metric=operand.metric,
                unit=operand.unit,
                currency=operand.currency,
            )
            for operand in original.operands
        ),
    )
    swapped = original.model_dump(mode="json")
    swapped["operands"][0]["id"] = "current_liabilities"
    swapped["operands"][2]["id"] = "current_assets"
    swapped["answer"]["value"] = "0.36"

    result = verify_program(
        FinanceProgram.model_validate(swapped),
        [_doc()],
        "What is the quick ratio?",
        question_spec=expected,
        answer_text="0.36x",
        require_full_contract=True,
    )

    assert not result.passed
    assert "operand_metric_mismatch" in _codes(result)


def test_trusted_operand_unit_prevents_currency_erasure():
    original = FinanceProgram.model_validate(_quick_program_dict())
    expected = FinanceQuestionSpec(
        entity="3M",
        period="FY2023Q2",
        metric="quick ratio",
        unit="ratio",
        expression=original.expression,
        operands=tuple(
            FinanceOperandSpec(
                id=operand.id,
                entity=operand.entity,
                period=operand.period,
                metric=operand.metric,
                unit="money",
                currency="USD",
            )
            for operand in original.operands
        ),
    )
    erased = original.model_dump(mode="json")
    for operand in erased["operands"]:
        operand["unit"] = "number"
        operand.pop("currency", None)

    result = verify_program(
        FinanceProgram.model_validate(erased),
        [_doc()],
        "What is the quick ratio?",
        question_spec=expected,
        answer_text="0.96x",
        require_full_contract=True,
    )

    assert not result.passed
    assert "operand_unit_mismatch" in _codes(result)


def test_model_cannot_crop_headers_out_of_a_flattened_source_line():
    content = "FY2022 FY2023 Revenue 100 200"
    raw = {
        "answer": {
            "value": "100",
            "unit": "money",
            "currency": "USD",
            "scale": "one",
            "entity": "Acme",
            "period": "FY2023",
            "metric": "Revenue",
            "rounding": {"places": 0},
        },
        "operands": [
            {
                "id": "revenue",
                "value": "100",
                "unit": "money",
                "currency": "USD",
                "scale": "one",
                "entity": "Acme",
                "period": "FY2023",
                "metric": "Revenue",
                "evidence": {
                    "doc_id": "Doc1",
                    "quote": "FY2023 Revenue 100",
                    "value_text": "100",
                    "metric_label": "Revenue",
                    "period_label": "FY2023",
                },
            }
        ],
        "expression": {"op": "ref", "operand_id": "revenue"},
    }

    result = verify_program(
        FinanceProgram.model_validate(raw),
        [_doc(content, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 revenue",
    )

    assert not result.passed
    assert "invalid_citation" in _codes(result)


def test_qualified_metric_suffix_cannot_masquerade_as_base_metric():
    content = "FY2023 Deferred revenue 100"
    raw = {
        "answer": {
            "value": "100",
            "unit": "money",
            "currency": "USD",
            "scale": "one",
            "entity": "Acme",
            "period": "FY2023",
            "metric": "Revenue",
            "rounding": {"places": 0},
        },
        "operands": [
            {
                "id": "revenue",
                "value": "100",
                "unit": "money",
                "currency": "USD",
                "scale": "one",
                "entity": "Acme",
                "period": "FY2023",
                "metric": "Revenue",
                "evidence": {
                    "doc_id": "Doc1",
                    "quote": content,
                    "value_text": "100",
                    "metric_label": "revenue",
                    "period_label": "FY2023",
                },
            }
        ],
        "expression": {"op": "ref", "operand_id": "revenue"},
    }

    result = verify_program(
        FinanceProgram.model_validate(raw),
        [_doc(content, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 revenue",
    )

    assert "operand_metric_mismatch" in _codes(result)


@pytest.mark.parametrize(
    ("operand_period", "evidence_period"),
    [
        ("December 31, 2023", "June 30, 2023"),
        ("TTM", "FY2022"),
    ],
)
def test_incompatible_date_and_relative_period_semantics_fail_closed(
    operand_period,
    evidence_period,
):
    raw = _quick_program_dict()
    raw["operands"][0]["period"] = operand_period
    raw["operands"][0]["evidence"]["period_label"] = evidence_period
    raw["operands"][0]["evidence"]["quote"] = (
        f"{raw['operands'][0]['evidence']['quote']} {evidence_period}"
    )
    doc_content = raw["operands"][0]["evidence"]["quote"]

    result = verify_program(
        FinanceProgram.model_validate(raw),
        [_doc(doc_content)],
        "3M period test",
    )

    assert "operand_period_mismatch" in _codes(result)


def test_trusted_period_preserves_exact_date_semantics():
    raw = _quick_program_dict()
    raw["answer"]["period"] = "June 30, 2023"
    for operand in raw["operands"]:
        operand["period"] = "June 30, 2023"
    program = FinanceProgram.model_validate(raw)

    result = verify_program(
        program,
        [_doc()],
        "3M balance sheet date",
        question_spec={
            "entity": "3M",
            "period": "December 31, 2023",
            "metric": "quick ratio",
            "unit": "ratio",
            "expression": program.expression,
        },
    )

    assert "operand_period_mismatch" in _codes(result)


def test_displayed_answer_requires_a_dedicated_deterministic_value_field():
    program = FinanceProgram.model_validate(_quick_program_dict())

    correct = verify_program(
        program,
        [_doc()],
        "3M FY2023 quick ratio",
        answer_text="0.96x",
    )
    wrong = verify_program(
        program,
        [_doc()],
        "3M FY2023 quick ratio",
        answer_text="1.44x",
    )
    ambiguous = verify_program(
        program,
        [_doc()],
        "3M FY2023 quick ratio",
        answer_text="The quick ratio was not 0.96x.",
    )

    assert correct.passed
    assert _codes(wrong) == {"answer_result_mismatch"}
    assert _codes(ambiguous) == {"answer_result_mismatch"}
    assert wrong.rendered_answer == "0.96x"


def test_pre_2000_periods_and_ambiguous_dollar_currencies_fail_closed():
    quote = "Acme FY1999 (Canadian dollars in millions) Revenue 100"
    operand = _money_operand(
        "revenue",
        "100",
        "100",
        "Revenue",
        quote=quote,
        entity="Acme",
        period="FY1999",
        period_label="FY1999",
    )
    raw = {
        "answer": {
            "value": "100",
            "currency": "USD",
            "scale": "million",
            "unit": "money",
            "entity": "Acme",
            "period": "FY1998",
            "metric": "Revenue",
            "rounding": {"places": 0},
        },
        "operands": [operand],
        "expression": {"op": "ref", "operand_id": "revenue"},
    }

    result = verify_program(
        FinanceProgram.model_validate(raw),
        [_doc(quote, company="Acme", source="ACME_1999_10K.pdf", currency="CAD")],
        "Acme FY1999 revenue",
    )
    assert "operand_period_mismatch" in _codes(result)
    assert "operand_currency_mismatch" in _codes(result)

    generic_quote = "Acme FY1999 (Dollars in millions) Revenue 100"
    generic = _money_operand(
        "revenue",
        "100",
        "100",
        "Revenue",
        quote=generic_quote,
        entity="Acme",
        period="FY1999",
        period_label="FY1999",
    )
    generic_raw = dict(raw)
    generic_raw["answer"] = dict(raw["answer"], period="FY1999")
    generic_raw["operands"] = [generic]
    no_currency_metadata = Document(
        page_content=generic_quote,
        metadata={"company": "Acme", "source": "ACME_1999_10K.pdf"},
    )
    generic_result = verify_program(
        FinanceProgram.model_validate(generic_raw),
        [no_currency_metadata],
        "Acme FY1999 revenue",
    )
    assert "operand_currency_mismatch" in _codes(generic_result)


@pytest.mark.parametrize(
    ("quote", "period"),
    [
        ("Acme FY2022 FY2023 Revenue 100 200", "FY2023"),
        (
            "Acme June 30, 2023 December 31, 2023 Revenue 100 200",
            "December 31, 2023",
        ),
        (
            "Acme Three months ended June 30, 2023 Year ended "
            "December 31, 2023 Revenue 100 200",
            "December 31, 2023",
        ),
    ],
)
def test_flattened_multi_column_rows_require_structured_cell_binding(
    quote,
    period,
):
    raw = {
        "answer": {
            "value": "100",
            "unit": "number",
            "entity": "Acme",
            "period": period,
            "metric": "Revenue",
            "rounding": {"places": 0},
        },
        "operands": [
            {
                "id": "revenue",
                "value": "100",
                "unit": "number",
                "entity": "Acme",
                "period": period,
                "metric": "Revenue",
                "evidence": {
                    "doc_id": "Doc1",
                    "quote": quote,
                    "value_text": "100",
                    "metric_label": "Revenue",
                    "period_label": period,
                },
            }
        ],
        "expression": {"op": "ref", "operand_id": "revenue"},
    }
    result = verify_program(
        FinanceProgram.model_validate(raw),
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 revenue",
        question_spec={
            "entity": "Acme",
            "period": period,
            "metric": "Revenue",
            "unit": "number",
            "rounding": {"places": 0},
            "expression": raw["expression"],
        },
        answer_text="100",
        require_full_contract=True,
    )

    assert not result.passed
    assert "operand_metric_mismatch" in _codes(result)


def test_document_entity_identity_must_be_authoritative_and_consistent():
    program = FinanceProgram.model_validate(_quick_program_dict())
    conflicting = Document(
        page_content=QUICK_QUOTE,
        metadata={
            "company": "Adobe",
            "source": "3M_2023Q2_10Q.pdf",
            "currency": "USD",
        },
    )
    missing = Document(
        page_content=QUICK_QUOTE,
        metadata={
            "currency": "USD",
            "fiscal_period": "FY2023Q2",
            "doc_type": "10Q",
            "quarter": "Q2",
        },
    )

    conflicting_result = verify_program(
        program,
        [conflicting],
        "3M FY2023 quick ratio",
    )
    missing_result = verify_program(
        program,
        [missing],
        "3M FY2023 quick ratio",
    )

    assert _codes(conflicting_result) == {"operand_entity_mismatch"}
    assert _codes(missing_result) == {"operand_entity_mismatch"}


def _single_money_program(
    *,
    entity="Acme",
    period="FY2023",
    metric="Revenue",
    quote="Acme FY2023 Revenue 100",
    value="100",
    metric_label=None,
    period_label=None,
    row_label=None,
    column_label=None,
):
    evidence = {
        "doc_id": "Doc1",
        "quote": quote,
        "value_text": value,
        "metric_label": metric_label or metric,
        "period_label": period_label or period,
    }
    if row_label is not None:
        evidence["row_label"] = row_label
    if column_label is not None:
        evidence["column_label"] = column_label
    return FinanceProgram.model_validate(
        {
            "answer": {
                "value": value.replace(",", ""),
                "unit": "money",
                "currency": "USD",
                "scale": "one",
                "entity": entity,
                "period": period,
                "metric": metric,
                "rounding": {"places": 0},
            },
            "operands": [
                {
                    "id": "value",
                    "value": value.replace(",", ""),
                    "unit": "money",
                    "currency": "USD",
                    "scale": "one",
                    "entity": entity,
                    "period": period,
                    "metric": metric,
                    "evidence": evidence,
                }
            ],
            "expression": {"op": "ref", "operand_id": "value"},
        }
    )


@pytest.mark.parametrize(
    ("period_label", "source", "metadata"),
    [
        ("Q2 FY2023", "ACME_2023_10K.pdf", {}),
        ("FY2023", "ACME_2023Q2_10Q.pdf", {}),
        (
            "FY2023",
            "ACME_2023_10K.pdf",
            {"fiscal_period": "FY2023Q2", "doc_type": "10Q", "quarter": "Q2"},
        ),
        ("FY2023", "notes.txt", {"fiscal_period": "Q2 FY2023"}),
    ],
)
def test_annual_fiscal_period_rejects_quarterly_evidence_context(
    period_label,
    source,
    metadata,
):
    quote = f"Acme {period_label} Revenue 100"
    program = _single_money_program(
        quote=quote,
        period="FY2023",
        period_label=period_label,
    )
    document = Document(
        page_content=quote,
        metadata={
            "company": "Acme",
            "currency": "USD",
            "source": source,
            **metadata,
        },
    )

    result = verify_program(program, [document], "Acme FY2023 revenue")

    assert "operand_period_mismatch" in _codes(result)


@pytest.mark.parametrize(
    ("period", "period_label", "source", "quote"),
    [
        (
            "FY2023",
            "September 30, 2023",
            "APPLE_2023_10K.pdf",
            "Apple Year ended September 30, 2023 Net sales 100",
        ),
        (
            "FY2024Q1",
            "December 30, 2023",
            "APPLE_2024Q1_10Q.pdf",
            "Apple Three months ended December 30, 2023 Net sales 100",
        ),
    ],
)
def test_authoritative_metadata_supports_off_calendar_fiscal_periods(
    period,
    period_label,
    source,
    quote,
):
    program = _single_money_program(
        entity="Apple",
        period=period,
        metric="Revenue",
        quote=quote,
        metric_label="Net sales",
        period_label=period_label,
    )

    result = verify_program(
        program,
        [_doc(quote, company="Apple", source=source)],
        f"Apple {period} revenue",
    )

    assert result.passed


@pytest.mark.parametrize(
    ("declared_metric", "row_metric", "metric_label"),
    [
        ("Operating income", "Adjusted operating income", "Operating income"),
        ("Adjusted operating income", "Unadjusted operating income", "Operating income"),
        ("Revenue", "Consolidated net sales", "Net sales"),
        ("Consolidated revenue", "Net sales", "Net sales"),
    ],
)
def test_full_row_qualifiers_cannot_change_metric_semantics(
    declared_metric,
    row_metric,
    metric_label,
):
    quote = f"Acme FY2023 {row_metric} 100"
    program = _single_money_program(
        metric=declared_metric,
        quote=quote,
        metric_label=metric_label,
        row_label=row_metric,
    )

    result = verify_program(
        program,
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 metric",
    )

    assert "operand_metric_mismatch" in _codes(result)


def test_model_cannot_crop_qualifier_out_of_declared_row_label():
    quote = "Acme FY2023 Consolidated revenue 100"
    program = _single_money_program(
        metric="Revenue",
        quote=quote,
        metric_label="Revenue",
        row_label="Revenue",
    )

    result = verify_program(
        program,
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 revenue",
    )

    assert "operand_metric_mismatch" in _codes(result)


@pytest.mark.parametrize(
    ("declared_metric", "row_metric"),
    [
        ("Adjusted operating income", "Adjusted operating income"),
        ("Consolidated revenue", "Consolidated net sales"),
    ],
)
def test_trusted_metric_may_explicitly_include_row_qualifiers(
    declared_metric,
    row_metric,
):
    quote = f"Acme FY2023 {row_metric} 100"
    program = _single_money_program(
        metric=declared_metric,
        quote=quote,
        metric_label=row_metric,
        row_label=row_metric,
    )

    result = verify_program(
        program,
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 qualified metric",
    )

    assert result.passed


@pytest.mark.parametrize(
    ("declared_metric", "source_metric"),
    [
        ("Revenue", "Net sales"),
        ("Operating cash flow", "Net cash provided by operating activities"),
        ("Capex", "Purchases of property, plant and equipment"),
        ("Net income incl NCI", "Net income including noncontrolling interests"),
        ("PP&E net", "Property, plant and equipment, net"),
    ],
)
def test_financebench_metric_aliases_are_canonical_and_safe(
    declared_metric,
    source_metric,
):
    quote = f"Acme FY2023 {source_metric} 100"
    program = _single_money_program(
        metric=declared_metric,
        quote=quote,
        metric_label=source_metric,
        row_label=source_metric,
    )

    result = verify_program(
        program,
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        "Acme FY2023 aliased metric",
    )

    assert result.passed


@pytest.mark.parametrize(
    ("trusted_metric", "source_metric"),
    [
        ("net_income", "Net income including noncontrolling interests"),
        ("property_plant_equipment", "Property, plant and equipment, net"),
    ],
)
def test_trusted_operand_contract_accepts_exact_financebench_metric_vocabulary(
    trusted_metric,
    source_metric,
):
    quote = f"Acme FY2023 {source_metric} 100"
    program = _single_money_program(
        metric=trusted_metric,
        quote=quote,
        metric_label=source_metric,
        row_label=source_metric,
    )
    expected = FinanceQuestionSpec(
        entity="Acme",
        period="FY2023",
        metric=trusted_metric,
        unit="money",
        currency="USD",
        scale="one",
        rounding={"places": 0},
        expression=program.expression,
        operands=(
            FinanceOperandSpec(
                id="value",
                entity="Acme",
                period="FY2023",
                metric=trusted_metric,
                unit="money",
                currency="USD",
                scale="one",
            ),
        ),
    )

    result = verify_program(
        program,
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        f"Acme FY2023 {trusted_metric}",
        question_spec=expected,
        answer_text="USD 100",
        require_full_contract=True,
    )

    assert result.passed
    assert result.fully_verified


@pytest.mark.parametrize(
    ("trusted_metric", "source_metric"),
    [
        ("net_income", "Net income attributable to Acme shareholders"),
        ("property_plant_equipment", "Gross property, plant and equipment"),
    ],
)
def test_contract_aliases_do_not_erase_attribution_or_gross_qualifiers(
    trusted_metric,
    source_metric,
):
    quote = f"Acme FY2023 {source_metric} 100"
    program = _single_money_program(
        metric=trusted_metric,
        quote=quote,
        metric_label=source_metric,
        row_label=source_metric,
    )

    result = verify_program(
        program,
        [_doc(quote, company="Acme", source="ACME_2023_10K.pdf")],
        f"Acme FY2023 {trusted_metric}",
    )

    assert "operand_metric_mismatch" in _codes(result)


def test_markdown_table_header_binds_value_to_the_requested_column():
    content = (
        "| Metric | FY2022 | FY2023 |\n"
        "|:---|---:|---:|\n"
        "| Gross profit | 50 | 60 |\n"
        "| Net sales | 100 | 200 |"
    )
    row = "| Net sales | 100 | 200 |"
    valid = _single_money_program(
        metric="Revenue",
        quote=row,
        value="200",
        metric_label="Net sales",
        row_label="Net sales",
        column_label="FY2023",
    )
    swapped = _single_money_program(
        metric="Revenue",
        quote=row,
        value="100",
        metric_label="Net sales",
        row_label="Net sales",
        column_label="FY2023",
    )
    document = _doc(content, company="Acme", source="ACME_2023_10K.pdf")

    valid_result = verify_program(valid, [document], "Acme FY2023 revenue")
    swapped_result = verify_program(swapped, [document], "Acme FY2023 revenue")

    assert valid_result.passed
    assert "operand_metric_mismatch" in _codes(swapped_result)


def test_rounding_contract_mismatch_routes_to_replan_without_new_issue_code():
    program = FinanceProgram.model_validate(_quick_program_dict())
    expected = FinanceQuestionSpec(
        entity="3M",
        period="FY2023Q2",
        metric="quick ratio",
        unit="ratio",
        rounding={"places": 3},
        expression=program.expression,
    )

    result = verify_program(
        program,
        [_doc()],
        "3M FY2023 quick ratio",
        question_spec=expected,
        answer_text="0.96x",
    )

    assert _codes(result) == {"formula_mismatch"}
    assert result.execution is None
