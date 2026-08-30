"""Regression tests for unit-aware financial evidence verification."""

from langchain_core.documents import Document

from evaluation.deterministic_verify import deterministic_verify
from evaluation.numeric_check import (
    augmented_judge,
    extract_numbers,
    numeric_match,
    numbers_match,
)
from src.postprocessing.numeric_verify import verify_numeric_answer
from src.providers.base import LLMResponse


def test_scale_aliases_normalize_to_same_value():
    billion = extract_numbers("$1.5 billion")[0]
    millions = extract_numbers("$1,500mn")[0]

    assert billion.value == 1_500_000_000
    assert millions.value == 1_500_000_000
    assert numbers_match(billion, millions)


def test_basis_points_normalize_to_percentage_points():
    basis_points = extract_numbers("150 bps")[0]
    percent = extract_numbers("1.5%")[0]

    assert basis_points.value == 1.5
    assert numbers_match(basis_points, percent)


def test_accounting_parentheses_preserve_negative_sign():
    value = extract_numbers("(1,234) million")[0]

    assert value.value == -1_234_000_000


def test_numeric_match_checks_all_reference_values():
    matched, _ = numeric_match(
        "Revenue was $10 million and margin was 12%.",
        "Margin was 12%; revenue was $10M.",
    )
    missing, explanation = numeric_match(
        "Revenue was $10 million and margin was 12%.",
        "Revenue was $10M.",
    )

    assert matched is True
    assert missing is False
    assert "12%" in explanation


def test_calendar_years_require_exact_equality():
    matched, explanation = numeric_match("2023", "2022")

    assert matched is False
    assert "2023" in explanation


def test_citation_values_do_not_leak_into_answer_metric():
    matched, explanation = numeric_match(
        "$2 billion",
        "Revenue was not stated [Doc1: 'Revenue was $2 billion']",
    )

    assert matched is False
    assert "contains no numeric value" in explanation


def test_numeric_reference_without_prediction_is_a_failure():
    matched, _ = numeric_match("$2 billion", "Revenue was not stated")

    assert matched is False


def test_magnitude_error_is_reported():
    matched, explanation = numeric_match("$1.5 billion", "$1.5 million")

    assert matched is False
    assert "1,000x" in explanation


def test_citation_must_point_to_the_named_document():
    docs = [
        Document(page_content="Revenue was $2 billion."),
        Document(page_content="Revenue was $1 million."),
    ]
    result = deterministic_verify(
        "Revenue was $2 billion [Doc2: 'Revenue was $2 billion']",
        docs,
    )

    assert result.passed is False
    assert "invalid_citation" in result.reason_codes


def test_quote_must_support_claim_value_and_scale():
    docs = [Document(page_content="Revenue was $1.5 million.")]
    result = deterministic_verify(
        "Revenue was $1.5 billion [Doc1: 'Revenue was $1.5 million']",
        docs,
    )

    assert result.passed is False
    assert result.mismatched_claims == ["$1.5 billion"]
    assert "numeric_evidence_mismatch" in result.reason_codes


def test_numeric_citation_rejects_wrong_metric_with_same_value():
    docs = [Document(page_content="Operating expenses were $100 million.")]
    result = deterministic_verify(
        "Revenue was $100 million "
        "[Doc1: 'Operating expenses were $100 million']",
        docs,
        question="What was revenue?",
    )

    assert result.passed is False
    assert "citation_claim_mismatch" in result.reason_codes


def test_numeric_citation_rejects_wrong_entity_with_same_metric_and_value():
    docs = [Document(page_content="Microsoft revenue was $100 million.")]
    result = deterministic_verify(
        "Apple revenue was $100 million "
        "[Doc1: 'Microsoft revenue was $100 million']",
        docs,
        question="What was Apple's revenue?",
    )

    assert result.passed is False
    assert "citation_claim_mismatch" in result.reason_codes


def test_numeric_citation_rejects_negation_of_affirmative_quote():
    docs = [Document(page_content="Revenue was $100 million.")]
    result = deterministic_verify(
        "Revenue was not $100 million [Doc1: 'Revenue was $100 million']",
        docs,
        question="Was revenue $100 million?",
    )

    assert result.passed is False
    assert "citation_claim_mismatch" in result.reason_codes


def test_citation_immediately_after_period_supports_claim():
    docs = [Document(page_content="Revenue was $1.5 billion.")]
    result = deterministic_verify(
        "Revenue was $1.5 billion. [Doc1: 'Revenue was $1.5 billion']",
        docs,
    )

    assert result.passed is True


def test_table_header_scale_is_applied_to_bare_cells():
    docs = [
        Document(
            page_content=(
                "Consolidated cash flows (Dollars in millions)\n"
                "Purchases of property, plant and equipment (1,577)"
            )
        )
    ]
    result = deterministic_verify(
        "Capital expenditures were $1,577 million "
        "[Doc1: 'Consolidated cash flows (Dollars in millions) "
        "Purchases of property, plant and equipment (1,577)']",
        docs,
        question="What was the capital expenditure amount?",
    )

    assert result.passed is True


def test_table_scale_is_not_applied_to_years():
    docs = [Document(page_content="Income statement (Dollars in millions) FY 2018")]
    result = deterministic_verify(
        "$2.018 billion [Doc1: 'FY 2018']",
        docs,
    )

    assert result.passed is False
    assert result.mismatched_claims == ["$2.018 billion"]


def test_derived_percentage_format_matches_basis_point_quote():
    docs = [Document(page_content="Gross margin improved by 150 bps.")]
    result = deterministic_verify(
        "Gross margin improved 1.5% [Doc1: 'Gross margin improved by 150 bps']",
        docs,
    )

    assert result.passed is True


def test_numbers_inside_quotes_are_not_treated_as_answer_claims():
    docs = [Document(page_content="Revenue grew to $2 billion.")]
    result = deterministic_verify(
        "Revenue grew [Doc1: 'Revenue grew to $2 billion']",
        docs,
    )

    assert result.passed is True
    assert result.coverage_ratio == 1.0


def test_nonnumeric_factual_answer_requires_a_citation():
    docs = [Document(page_content="Products are shipped after payment.")]

    result = deterministic_verify(
        "Apple recognizes revenue when products are shipped.",
        docs,
        question="When does Apple recognize revenue?",
    )

    assert result.passed is False
    assert result.coverage_ratio == 0.0
    assert "missing_citation" in result.reason_codes


def test_nonnumeric_citation_must_be_lexically_attributed_to_claim():
    docs = [Document(page_content="The annual report was published by Apple.")]

    result = deterministic_verify(
        "Apple recognizes revenue when products are shipped "
        "[Doc1: 'The annual report was published by Apple']",
        docs,
        question="When does Apple recognize revenue?",
    )

    assert result.passed is False
    assert "citation_claim_mismatch" in result.reason_codes


def test_nonnumeric_exact_quote_with_claim_anchors_passes_lexical_gate():
    docs = [
        Document(
            page_content=(
                "Apple recognizes revenue when control of shipped products "
                "transfers to the customer."
            )
        )
    ]

    result = deterministic_verify(
        "Apple recognizes revenue when products are shipped "
        "[Doc1: 'Apple recognizes revenue when control of shipped products "
        "transfers to the customer']",
        docs,
        question="When does Apple recognize revenue?",
    )

    assert result.passed is True
    assert "semantic attribution" in result.message


def test_nonnumeric_citation_rejects_negation_of_affirmative_quote():
    docs = [
        Document(
            page_content="The Company recognizes revenue when control transfers."
        )
    ]

    result = deterministic_verify(
        "The Company does not recognize revenue when control transfers "
        "[Doc1: 'The Company recognizes revenue when control transfers']",
        docs,
        question="When does the Company recognize revenue?",
    )

    assert result.passed is False
    assert "citation_claim_mismatch" in result.reason_codes


def test_nonnumeric_citation_rejects_opposite_direction():
    docs = [Document(page_content="Gross margin declined during the year.")]

    result = deterministic_verify(
        "Gross margin increased during the year "
        "[Doc1: 'Gross margin declined during the year']",
        docs,
        question="How did gross margin change?",
    )

    assert result.passed is False
    assert "citation_claim_mismatch" in result.reason_codes


def test_standard_numeric_source_gate_respects_cli_config_flag():
    from src.bulk_testing import BulkTestConfig, BulkTestRunner

    class Pipeline:
        def retrieve(self, question):
            return [Document(page_content="Revenue was $10 million")]

    class Provider:
        def generate(self, **kwargs):
            return LLMResponse(
                content="$10 million",
                model="fake",
                provider="fake",
            )

    disabled = BulkTestRunner(
        BulkTestConfig(dataset_name="financebench", use_numeric_verify=False)
    )
    disabled.pipeline = Pipeline()
    disabled.llm_provider = Provider()
    disabled_result = disabled.process_single_question("Revenue?", "q0")

    enabled = BulkTestRunner(
        BulkTestConfig(dataset_name="financebench", use_numeric_verify=True)
    )
    enabled.pipeline = Pipeline()
    enabled.llm_provider = Provider()
    enabled_result = enabled.process_single_question("Revenue?", "q1")

    assert "numeric_score" not in disabled_result
    assert enabled_result["numeric_score"] == 1.0


def test_postprocessing_uses_unit_compatibility():
    docs = [Document(page_content="Revenue was $1.5 million and margin was 1.5%.")]

    result = verify_numeric_answer("Revenue was $1.5 billion.", docs)
    percentage = verify_numeric_answer("Margin was 150 bps.", docs)

    assert result.score == 0.0
    assert result.flagged_numbers == ["$1.5 billion"]
    assert percentage.score == 1.0


def test_fuzzy_quote_cannot_fabricate_a_numeric_value():
    docs = [
        Document(
            page_content="one two three four revenue was $1 million"
        )
    ]
    result = deterministic_verify(
        "Revenue was $9 million "
        "[Doc1: 'one two three four revenue was $9 million']",
        docs,
    )

    assert result.passed is False
    assert "invalid_citation" in result.reason_codes


def test_smart_quote_citation_delimiters_are_supported():
    docs = [Document(page_content="Revenue was $2 million.")]

    single = deterministic_verify(
        "Revenue was $2 million [Doc1: ‘Revenue was $2 million’]",
        docs,
    )
    double = deterministic_verify(
        "Revenue was $2 million [Doc1: “Revenue was $2 million”]",
        docs,
    )

    assert single.passed is True
    assert double.passed is True


def test_smart_quote_citation_values_do_not_leak_into_answer_metric():
    matched, explanation = numeric_match(
        "$2 million",
        "Revenue was not stated [Doc1: ‘Revenue was $2 million’]",
    )

    assert matched is False
    assert "contains no numeric value" in explanation


def test_unicode_minus_is_preserved_as_a_negative_sign():
    parsed = extract_numbers("Net income was −$2 million.")
    docs = [Document(page_content="Net income was $2 million.")]
    result = deterministic_verify(
        "Net income was −$2 million [Doc1: 'Net income was $2 million']",
        docs,
    )

    assert parsed[0].value == -2_000_000
    assert result.passed is False
    assert result.mismatched_claims == ["−$2 million"]


def test_scientific_notation_is_a_numeric_claim():
    parsed = extract_numbers("Revenue was 1.5e6.")
    result = deterministic_verify(
        "Revenue was 1.5e6.",
        [Document(page_content="Revenue was 9.")],
    )

    assert parsed[0].value == 1_500_000
    assert result.passed is False
    assert result.ungrounded_claims == ["1.5e6"]


def test_scale_does_not_bleed_from_uncited_chunk_context():
    docs = [
        Document(
            page_content=(
                "Table A (Dollars in millions)\n"
                "Narrative: The filing fee was $2."
            )
        )
    ]
    result = deterministic_verify(
        "The filing fee was $2 million [Doc1: 'The filing fee was $2']",
        docs,
    )

    assert result.passed is False
    assert result.mismatched_claims == ["$2 million"]


def test_cited_scale_has_only_the_scaled_interpretation():
    docs = [
        Document(
            page_content=(
                "Consolidated cash flows (Dollars in millions) "
                "Capital expenditures (1,577)"
            )
        )
    ]
    result = deterministic_verify(
        "Capital expenditures were $1,577 "
        "[Doc1: 'Consolidated cash flows (Dollars in millions) "
        "Capital expenditures (1,577)']",
        docs,
        question="What was capital expenditure?",
    )

    assert result.passed is False
    assert result.mismatched_claims == ["$1,577"]


def test_sign_insensitivity_is_not_enabled_by_unrelated_loss_text():
    docs = [
        Document(
            page_content="Revenue was $10 million and net loss was ($2 million)."
        )
    ]
    result = deterministic_verify(
        "Revenue was -$10 million and net loss was $2 million "
        "[Doc1: 'Revenue was $10 million and net loss was ($2 million)']",
        docs,
        question="What were revenue and net loss?",
    )

    assert result.passed is False
    assert result.mismatched_claims == ["-$10 million", "$2 million"]


def test_single_capex_claim_can_match_accounting_outflow_sign():
    docs = [Document(page_content="Capital expenditures were ($2 million).")]
    result = deterministic_verify(
        "$2 million [Doc1: 'Capital expenditures were ($2 million)']",
        docs,
        question="What was capital expenditure?",
    )

    assert result.passed is True


def test_capex_sign_rule_does_not_leak_to_neighboring_revenue_claim():
    docs = [
        Document(
            page_content="Revenue was $10 million and capex was ($2 million)."
        )
    ]
    result = deterministic_verify(
        "Revenue was -$10 million and capex was $2 million "
        "[Doc1: 'Revenue was $10 million and capex was ($2 million)']",
        docs,
    )

    assert result.passed is False
    assert result.supported_claims == ["$2 million"]
    assert result.mismatched_claims == ["-$10 million"]


def test_numbered_list_marker_is_not_a_numeric_claim():
    docs = [Document(page_content="Revenue was $2 million.")]
    result = deterministic_verify(
        "1. Revenue was $2 million [Doc1: 'Revenue was $2 million']",
        docs,
    )

    assert result.passed is True
    assert result.supported_claims == ["$2 million"]


def test_parenthesized_numbered_list_marker_is_not_a_numeric_claim():
    docs = [Document(page_content="Revenue was $2 million.")]
    result = deterministic_verify(
        "1) Revenue was $2 million [Doc1: 'Revenue was $2 million']",
        docs,
    )

    assert result.passed is True
    assert result.supported_claims == ["$2 million"]


def test_augmented_judge_does_not_boost_conflicting_extra_quantity():
    predicted = (
        "The source mentions $10 million, but the final answer is $1 million."
    )
    matched, _ = numeric_match("$10 million", predicted)
    score, explanation = augmented_judge(
        "What was the amount?",
        "$10 million",
        predicted,
        0.1,
        "The final answer is incorrect",
    )

    assert matched is True
    assert score == 0.1
    assert "CONFLICTING" in explanation


def test_strict_numeric_metric_rejects_conflicting_extra_quantity():
    matched, explanation = numeric_match(
        "$10 million",
        "The source mentions $10 million, but the final answer is $1 million.",
        reject_conflicting_extras=True,
    )

    assert matched is False
    assert "Conflicting extra" in explanation
