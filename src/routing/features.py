"""Feature extraction for meta-router question classification.

This module extracts features from questions that help predict which
retrieval pipeline will perform best. All extraction is rule-based
(no API calls) for speed and reproducibility.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set


# =============================================================================
# Domain Keyword Sets
# =============================================================================

FINANCE_KEYWORDS = {
    'revenue', 'profit', 'income', 'earnings', 'ebitda', 'margin',
    'capex', 'capital expenditure', 'assets', 'liabilities', 'debt',
    'equity', 'cash flow', 'balance sheet', 'income statement',
    '10-k', '10k', '10-q', 'sec filing', 'annual report',
    'fiscal', 'quarter', 'dividend', 'eps', 'pe ratio', 'market cap',
    'depreciation', 'amortization', 'goodwill', 'inventory', 'receivables'
}

MEDICAL_KEYWORDS = {
    'patient', 'treatment', 'diagnosis', 'symptom', 'disease',
    'clinical', 'trial', 'drug', 'therapy', 'efficacy', 'safety',
    'adverse', 'dosage', 'medication', 'prognosis', 'mortality',
    'morbidity', 'biomarker', 'pathology', 'oncology', 'cardiology'
}

LEGAL_KEYWORDS = {
    'contract', 'agreement', 'clause', 'party', 'parties',
    'termination', 'liability', 'indemnification', 'warranty',
    'confidential', 'governing law', 'jurisdiction', 'arbitration',
    'breach', 'damages', 'remedies', 'assignment', 'license'
}

REASONING_KEYWORDS = {
    'why', 'explain', 'analyze', 'compare', 'evaluate', 'assess',
    'impact', 'effect', 'cause', 'reason', 'implication', 'significance',
    'based on', 'according to', 'in light of', 'considering'
}

NUMERIC_INDICATORS = {
    'how much', 'how many', 'what is the', 'what was the',
    'total', 'amount', 'number', 'percentage', 'ratio', 'rate',
    '$', 'million', 'billion', 'percent', '%'
}


# =============================================================================
# Feature Extraction
# =============================================================================

@dataclass
class QuestionFeatures:
    """Features extracted from a question for routing prediction."""

    # Temporal features
    has_year: bool = False
    has_quarter: bool = False
    has_fiscal_indicator: bool = False
    year_count: int = 0

    # Entity features
    has_company_indicator: bool = False
    capitalized_word_count: int = 0
    has_metric_name: bool = False

    # Question structure
    is_what: bool = False
    is_how: bool = False
    is_why: bool = False
    is_yes_no: bool = False
    expects_number: bool = False
    expects_explanation: bool = False

    # Text statistics
    word_count: int = 0
    char_count: int = 0
    avg_word_length: float = 0.0

    # Domain signals (keyword density)
    finance_density: float = 0.0
    medical_density: float = 0.0
    legal_density: float = 0.0

    # Complexity signals
    needs_reasoning: bool = False
    multi_part_question: bool = False

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with all values as floats for ML."""
        d = asdict(self)
        return {k: float(v) if isinstance(v, (bool, int)) else v for k, v in d.items()}

    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML models."""
        return list(self.to_dict().values())

    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names."""
        return list(asdict(QuestionFeatures()).keys())


class FeatureExtractor:
    """Extracts routing features from questions.

    All extraction is rule-based (no API calls) for:
    - Speed: <1ms per question
    - Reproducibility: Same question always gives same features
    - Cost: Free to run
    """

    def __init__(self):
        self.finance_keywords = FINANCE_KEYWORDS
        self.medical_keywords = MEDICAL_KEYWORDS
        self.legal_keywords = LEGAL_KEYWORDS
        self.reasoning_keywords = REASONING_KEYWORDS
        self.numeric_indicators = NUMERIC_INDICATORS

    def extract(self, question: str) -> QuestionFeatures:
        """Extract all features from a question.

        Args:
            question: The question text

        Returns:
            QuestionFeatures dataclass with all extracted features
        """
        features = QuestionFeatures()
        q_lower = question.lower()
        words = question.split()

        # --- Temporal Features ---
        features.has_year = bool(re.search(r'\b(20\d{2}|19\d{2})\b', question))
        features.year_count = len(re.findall(r'\b(20\d{2}|19\d{2})\b', question))
        features.has_quarter = bool(re.search(r'\b[Qq][1-4]\b|first quarter|second quarter|third quarter|fourth quarter', question, re.I))
        features.has_fiscal_indicator = 'fy' in q_lower or 'fiscal' in q_lower

        # --- Entity Features ---
        # Count capitalized words (potential company names), excluding sentence starts
        cap_words = re.findall(r'(?<!^)(?<!\. )\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        features.capitalized_word_count = len(cap_words)
        features.has_company_indicator = features.capitalized_word_count > 0 or any(
            ind in q_lower for ind in ['company', 'corporation', 'inc', 'corp', 'llc', "'s"]
        )

        # Check for financial metric names
        features.has_metric_name = any(kw in q_lower for kw in [
            'revenue', 'profit', 'income', 'earnings', 'ebitda', 'capex',
            'margin', 'assets', 'liabilities', 'cash flow', 'eps'
        ])

        # --- Question Structure ---
        features.is_what = q_lower.startswith('what')
        features.is_how = q_lower.startswith('how')
        features.is_why = q_lower.startswith('why')
        features.is_yes_no = any(q_lower.startswith(w) for w in ['is ', 'are ', 'does ', 'do ', 'was ', 'were ', 'can ', 'could '])

        # Numeric expectation
        features.expects_number = any(ind in q_lower for ind in self.numeric_indicators)

        # Explanation expectation
        features.expects_explanation = features.is_why or features.is_how or any(
            kw in q_lower for kw in ['explain', 'describe', 'analyze', 'compare']
        )

        # --- Text Statistics ---
        features.word_count = len(words)
        features.char_count = len(question)
        features.avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

        # --- Domain Signals ---
        features.finance_density = self._keyword_density(q_lower, self.finance_keywords)
        features.medical_density = self._keyword_density(q_lower, self.medical_keywords)
        features.legal_density = self._keyword_density(q_lower, self.legal_keywords)

        # --- Complexity Signals ---
        features.needs_reasoning = any(kw in q_lower for kw in self.reasoning_keywords)
        features.multi_part_question = question.count('?') > 1 or ' and ' in q_lower

        return features

    def _keyword_density(self, text: str, keywords: Set[str]) -> float:
        """Calculate keyword density (0.0 to 1.0)."""
        if not keywords:
            return 0.0
        matches = sum(1 for kw in keywords if kw in text)
        return matches / len(keywords)


# =============================================================================
# Convenience Functions
# =============================================================================

# Global extractor instance
_extractor: Optional[FeatureExtractor] = None


def get_extractor() -> FeatureExtractor:
    """Get or create the global feature extractor."""
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


def extract_features(question: str) -> Dict[str, float]:
    """Extract features from a question as a dictionary.

    This is the main entry point for feature extraction.

    Args:
        question: The question text

    Returns:
        Dictionary mapping feature names to values (all floats)
    """
    extractor = get_extractor()
    features = extractor.extract(question)
    return features.to_dict()


def extract_features_batch(
    questions: List[str],
    show_progress: bool = False,
) -> List[Dict[str, float]]:
    """Extract features from multiple questions.

    Args:
        questions: List of question texts
        show_progress: Whether to show tqdm progress bar

    Returns:
        List of feature dictionaries
    """
    extractor = get_extractor()

    if show_progress:
        from tqdm import tqdm
        return [extractor.extract(q).to_dict() for q in tqdm(questions, desc="Extracting features")]
    else:
        return [extractor.extract(q).to_dict() for q in questions]


# =============================================================================
# Analysis Utilities
# =============================================================================

def analyze_feature_distribution(questions: List[str]) -> Dict[str, Dict[str, float]]:
    """Analyze feature distribution across a set of questions.

    Args:
        questions: List of question texts

    Returns:
        Dictionary with mean, std, min, max for each feature
    """
    import numpy as np

    features_list = extract_features_batch(questions)
    if not features_list:
        return {}

    # Convert to arrays
    feature_names = list(features_list[0].keys())
    arrays = {name: [] for name in feature_names}

    for feat_dict in features_list:
        for name, value in feat_dict.items():
            arrays[name].append(value)

    # Calculate statistics
    stats = {}
    for name, values in arrays.items():
        arr = np.array(values)
        stats[name] = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max()),
        }

    return stats


# =============================================================================
# CLI / Testing
# =============================================================================

if __name__ == "__main__":
    import json

    test_questions = [
        "What is the FY2018 capital expenditure amount for 3M?",
        "What was Apple's revenue in 2022?",
        "Is 3M a capital-intensive business based on FY2022 data?",
        "Why did Microsoft's profit margin decline in Q4 2021?",
        "Explain the termination clause in the contract.",
        "What treatment options are available for type 2 diabetes?",
    ]

    print("=" * 60)
    print("FEATURE EXTRACTION TEST")
    print("=" * 60)

    extractor = FeatureExtractor()

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("-" * 60)

        features = extractor.extract(q)
        feat_dict = features.to_dict()

        # Show non-zero features
        active = {k: v for k, v in feat_dict.items() if v > 0}
        print(f"Active features ({len(active)}):")
        for k, v in sorted(active.items()):
            print(f"  {k}: {v}")

        # Predict likely pipeline
        if features.has_year and features.has_company_indicator:
            predicted = "hybrid_filter"
        elif features.needs_reasoning or features.expects_explanation:
            predicted = "hybrid_filter_rerank"
        elif features.has_metric_name:
            predicted = "hybrid"
        else:
            predicted = "semantic"

        print(f"\nPredicted pipeline: {predicted}")
