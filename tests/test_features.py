"""Unit tests for feature extraction (meta-learning router).

These tests verify the rule-based feature extraction that powers
the meta-learning router. All tests are fast (no API calls) and
can run in parallel with pytest-xdist.

Usage:
    pytest tests/test_features.py -n 4    # Parallel execution
    pytest tests/test_features.py -v      # Verbose output
"""

import pytest


class TestQuestionFeatures:
    """Tests for the QuestionFeatures dataclass."""

    def test_features_to_dict(self):
        """Test features convert to dictionary with float values."""
        from src.routing.features import QuestionFeatures

        features = QuestionFeatures(
            has_year=True,
            word_count=10,
            finance_density=0.15,
        )
        d = features.to_dict()

        assert isinstance(d, dict)
        assert d["has_year"] == 1.0  # bool converted to float
        assert d["word_count"] == 10.0  # int converted to float
        assert d["finance_density"] == 0.15  # float unchanged

    def test_features_to_vector(self):
        """Test features convert to list for ML models."""
        from src.routing.features import QuestionFeatures

        features = QuestionFeatures()
        vector = features.to_vector()

        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)

    def test_feature_names(self):
        """Test feature names match vector order."""
        from src.routing.features import QuestionFeatures

        names = QuestionFeatures.feature_names()
        features = QuestionFeatures()
        d = features.to_dict()

        assert list(d.keys()) == names


class TestTemporalFeatures:
    """Tests for temporal feature extraction (years, quarters, fiscal)."""

    def test_year_detection(self, feature_extractor):
        """Test year detection in questions."""
        features = feature_extractor.extract("What was Apple's revenue in 2022?")

        assert features.has_year is True
        assert features.year_count == 1

    def test_multiple_years(self, feature_extractor):
        """Test detection of multiple years."""
        features = feature_extractor.extract(
            "Compare revenue between 2020 and 2022."
        )

        assert features.has_year is True
        assert features.year_count == 2

    def test_no_year(self, feature_extractor):
        """Test questions without years."""
        features = feature_extractor.extract("What is the company's main product?")

        assert features.has_year is False
        assert features.year_count == 0

    def test_quarter_detection(self, feature_extractor):
        """Test quarter detection (Q1, Q2, etc.)."""
        q1 = feature_extractor.extract("Revenue in Q1 2023?")
        q_word = feature_extractor.extract("First quarter results?")

        assert q1.has_quarter is True
        assert q_word.has_quarter is True

    def test_fiscal_indicator(self, feature_extractor):
        """Test fiscal year indicator detection."""
        fy = feature_extractor.extract("FY2018 capital expenditure?")
        fiscal = feature_extractor.extract("Fiscal 2022 earnings?")

        assert fy.has_fiscal_indicator is True
        assert fiscal.has_fiscal_indicator is True


class TestEntityFeatures:
    """Tests for entity feature extraction (companies, metrics)."""

    def test_company_indicator(self, feature_extractor):
        """Test company name detection."""
        features = feature_extractor.extract("What was Apple's revenue?")

        assert features.has_company_indicator is True
        assert features.capitalized_word_count >= 1

    def test_metric_detection(self, feature_extractor):
        """Test financial metric detection."""
        revenue = feature_extractor.extract("What is the revenue?")
        ebitda = feature_extractor.extract("Calculate EBITDA margin.")

        assert revenue.has_metric_name is True
        assert ebitda.has_metric_name is True

    def test_no_metric(self, feature_extractor):
        """Test questions without financial metrics."""
        features = feature_extractor.extract("Who is the CEO?")

        assert features.has_metric_name is False


class TestQuestionStructure:
    """Tests for question structure detection."""

    def test_what_question(self, feature_extractor):
        """Test 'what' question detection."""
        features = feature_extractor.extract("What is the total revenue?")

        assert features.is_what is True
        assert features.is_how is False
        assert features.is_why is False

    def test_how_question(self, feature_extractor):
        """Test 'how' question detection."""
        features = feature_extractor.extract("How much did profit increase?")

        assert features.is_how is True
        assert features.expects_explanation is True

    def test_why_question(self, feature_extractor):
        """Test 'why' question detection."""
        features = feature_extractor.extract("Why did margins decline?")

        assert features.is_why is True
        assert features.expects_explanation is True
        assert features.needs_reasoning is True

    def test_yes_no_question(self, feature_extractor):
        """Test yes/no question detection."""
        is_q = feature_extractor.extract("Is Apple a profitable company?")
        does_q = feature_extractor.extract("Does revenue exceed $100M?")

        assert is_q.is_yes_no is True
        assert does_q.is_yes_no is True

    def test_numeric_expectation(self, feature_extractor):
        """Test numeric answer expectation."""
        how_much = feature_extractor.extract("How much revenue was generated?")
        total = feature_extractor.extract("What is the total amount?")

        assert how_much.expects_number is True
        assert total.expects_number is True


class TestDomainSignals:
    """Tests for domain keyword density calculations."""

    def test_finance_density(self, feature_extractor):
        """Test finance keyword density."""
        finance_q = feature_extractor.extract(
            "What was the revenue, profit, and EBITDA margin?"
        )
        non_finance = feature_extractor.extract("What is the weather today?")

        assert finance_q.finance_density > 0
        assert non_finance.finance_density == 0

    def test_medical_density(self, feature_extractor):
        """Test medical keyword density."""
        medical_q = feature_extractor.extract(
            "What treatment is available for the patient's diagnosis?"
        )

        assert medical_q.medical_density > 0

    def test_legal_density(self, feature_extractor):
        """Test legal keyword density."""
        legal_q = feature_extractor.extract(
            "Explain the termination clause and liability provisions."
        )

        assert legal_q.legal_density > 0


class TestComplexitySignals:
    """Tests for complexity signal detection."""

    def test_needs_reasoning(self, feature_extractor):
        """Test reasoning requirement detection."""
        why_q = feature_extractor.extract("Why did margins decline?")
        compare_q = feature_extractor.extract("Compare the two companies.")
        simple_q = feature_extractor.extract("What is the revenue?")

        assert why_q.needs_reasoning is True
        assert compare_q.needs_reasoning is True
        assert simple_q.needs_reasoning is False

    def test_multi_part_question(self, feature_extractor):
        """Test multi-part question detection."""
        multi = feature_extractor.extract(
            "What is revenue? And what is profit?"
        )
        single = feature_extractor.extract("What is the revenue?")

        assert multi.multi_part_question is True
        assert single.multi_part_question is False


class TestTextStatistics:
    """Tests for text statistics features."""

    def test_word_count(self, feature_extractor):
        """Test word count calculation."""
        short = feature_extractor.extract("Revenue?")
        long = feature_extractor.extract(
            "What was the total revenue for Apple Inc in fiscal year 2022?"
        )

        assert short.word_count == 1
        assert long.word_count == 12

    def test_char_count(self, feature_extractor):
        """Test character count calculation."""
        features = feature_extractor.extract("Hello world")

        assert features.char_count == 11

    def test_avg_word_length(self, feature_extractor):
        """Test average word length calculation."""
        features = feature_extractor.extract("What is the revenue?")

        # "What" (4) + "is" (2) + "the" (3) + "revenue?" (8) = 17 / 4 = 4.25
        assert 4.0 <= features.avg_word_length <= 4.5


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_extract_features(self):
        """Test extract_features returns dict."""
        from src.routing.features import extract_features

        result = extract_features("What is revenue?")

        assert isinstance(result, dict)
        assert "has_year" in result
        assert "finance_density" in result

    def test_extract_features_batch(self, sample_questions):
        """Test batch feature extraction."""
        from src.routing.features import extract_features_batch

        results = extract_features_batch(sample_questions)

        assert len(results) == len(sample_questions)
        assert all(isinstance(r, dict) for r in results)

    def test_global_extractor_singleton(self):
        """Test global extractor is reused (singleton pattern)."""
        from src.routing.features import get_extractor

        ext1 = get_extractor()
        ext2 = get_extractor()

        assert ext1 is ext2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_question(self, feature_extractor):
        """Test handling of empty question."""
        features = feature_extractor.extract("")

        assert features.word_count == 0
        assert features.char_count == 0
        # avg_word_length should handle division by zero
        assert features.avg_word_length == 0.0

    def test_special_characters(self, feature_extractor):
        """Test handling of special characters."""
        features = feature_extractor.extract("Revenue: $100M (est.)?")

        assert features.word_count > 0
        assert features.expects_number is True  # Contains $

    def test_unicode_characters(self, feature_extractor):
        """Test handling of unicode characters."""
        features = feature_extractor.extract("What is Apple's revenue?")

        assert features.has_company_indicator is True
