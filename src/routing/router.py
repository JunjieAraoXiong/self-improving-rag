"""Meta-router for adaptive pipeline selection.

This module implements the trained router that predicts the best
retrieval pipeline for each question based on extracted features.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import extract_features, FeatureExtractor


# Pipeline class mapping
PIPELINE_CLASSES = ['semantic', 'hybrid', 'hybrid_filter', 'hybrid_filter_rerank']
PIPELINE_TO_IDX = {p: i for i, p in enumerate(PIPELINE_CLASSES)}
IDX_TO_PIPELINE = {i: p for i, p in enumerate(PIPELINE_CLASSES)}


@dataclass
class RouterPrediction:
    """Result of router prediction."""
    pipeline: str
    confidence: float
    probabilities: Dict[str, float]


class Router:
    """Meta-router that predicts the best retrieval pipeline.

    Uses a trained classifier to analyze question features and
    predict which pipeline will perform best.
    """

    def __init__(
        self,
        model=None,
        scaler=None,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        """Initialize router.

        Args:
            model: Trained sklearn classifier
            scaler: Fitted StandardScaler for features
            feature_extractor: Feature extractor instance
        """
        self.model = model
        self.scaler = scaler
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.feature_names: Optional[List[str]] = None

    def predict(self, question: str) -> str:
        """Predict the best pipeline for a question.

        Args:
            question: The question text

        Returns:
            Pipeline name ('semantic', 'hybrid', etc.)
        """
        prediction = self.predict_with_confidence(question)
        return prediction.pipeline

    def predict_with_confidence(self, question: str) -> RouterPrediction:
        """Predict pipeline with confidence scores.

        Args:
            question: The question text

        Returns:
            RouterPrediction with pipeline, confidence, and probabilities
        """
        # Extract features
        features = self.feature_extractor.extract(question)
        feature_vector = np.array([features.to_vector()])

        # Scale features
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)

        # Predict
        pred_idx = self.model.predict(feature_vector)[0]
        pipeline = IDX_TO_PIPELINE[pred_idx]

        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(feature_vector)[0]
            probabilities = {IDX_TO_PIPELINE[i]: float(p) for i, p in enumerate(probs)}
            confidence = float(probs[pred_idx])
        else:
            probabilities = {pipeline: 1.0}
            confidence = 1.0

        return RouterPrediction(
            pipeline=pipeline,
            confidence=confidence,
            probabilities=probabilities
        )

    def predict_batch(self, questions: List[str]) -> List[str]:
        """Predict pipelines for multiple questions.

        Args:
            questions: List of question texts

        Returns:
            List of pipeline names
        """
        # Extract features for all questions
        feature_vectors = []
        for q in questions:
            features = self.feature_extractor.extract(q)
            feature_vectors.append(features.to_vector())

        X = np.array(feature_vectors)

        # Scale
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Predict
        pred_indices = self.model.predict(X)
        return [IDX_TO_PIPELINE[idx] for idx in pred_indices]

    @classmethod
    def load(cls, model_dir: str) -> 'Router':
        """Load a trained router from disk.

        Args:
            model_dir: Directory containing model files

        Returns:
            Loaded Router instance
        """
        import joblib

        model_path = Path(model_dir)

        # Load model
        model_file = model_path / 'router_lr.joblib'
        if not model_file.exists():
            model_file = model_path / 'router_rf.joblib'
        model = joblib.load(model_file)

        # Load scaler
        scaler_file = model_path / 'scaler.joblib'
        scaler = joblib.load(scaler_file) if scaler_file.exists() else None

        # Load metadata
        metadata_file = model_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        router = cls(model=model, scaler=scaler)
        router.feature_names = metadata.get('feature_names')

        return router

    def save(self, model_dir: str, model_type: str = 'logistic_regression'):
        """Save router to disk.

        Args:
            model_dir: Directory to save model files
            model_type: Type of model ('logistic_regression' or 'random_forest')
        """
        import joblib

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = 'router_lr.joblib' if 'logistic' in model_type else 'router_rf.joblib'
        joblib.dump(self.model, model_path / model_file)

        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, model_path / 'scaler.joblib')

        # Save metadata
        metadata = {
            'model_type': model_type,
            'feature_names': self.feature_names or list(extract_features("test").keys()),
            'pipeline_classes': PIPELINE_CLASSES,
        }
        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


class RuleBasedRouter:
    """Simple rule-based router (no training required).

    Use as a baseline or fallback when no trained model is available.
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def predict(self, question: str) -> str:
        """Predict pipeline using simple rules.

        Rules:
        1. If needs_reasoning or expects_explanation → hybrid_filter_rerank
        2. If has_year AND has_company_indicator → hybrid_filter
        3. If has_metric_name → hybrid
        4. Otherwise → semantic
        """
        features = self.feature_extractor.extract(question)

        # Rule 1: Complex reasoning questions need full pipeline
        if features.needs_reasoning or features.expects_explanation:
            return 'hybrid_filter_rerank'

        # Rule 2: Specific company + time queries need filtering
        if features.has_year and features.has_company_indicator:
            return 'hybrid_filter'

        # Rule 3: Financial metrics benefit from hybrid search
        if features.has_metric_name:
            return 'hybrid'

        # Rule 4: Default to semantic for simple queries
        return 'semantic'

    def predict_with_confidence(self, question: str) -> RouterPrediction:
        """Predict with fixed confidence (rule-based has no learned confidence)."""
        pipeline = self.predict(question)
        return RouterPrediction(
            pipeline=pipeline,
            confidence=0.8,  # Fixed confidence for rule-based
            probabilities={pipeline: 0.8}
        )


def get_router(model_dir: Optional[str] = None) -> Router:
    """Get a router instance.

    Args:
        model_dir: Path to trained model. If None, uses rule-based router.

    Returns:
        Router instance (trained or rule-based)
    """
    if model_dir and Path(model_dir).exists():
        return Router.load(model_dir)
    else:
        # Fall back to rule-based
        rule_router = RuleBasedRouter()
        # Wrap in Router interface
        router = Router()
        router.predict = rule_router.predict
        router.predict_with_confidence = rule_router.predict_with_confidence
        return router


# =============================================================================
# CLI / Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ROUTER TEST (Rule-Based)")
    print("=" * 60)

    router = RuleBasedRouter()

    test_questions = [
        "What is the FY2018 capital expenditure amount for 3M?",
        "What was Apple's revenue in 2022?",
        "Is 3M a capital-intensive business based on FY2022 data?",
        "Why did Microsoft's profit margin decline in Q4 2021?",
        "Explain the termination clause in the contract.",
        "What is revenue?",  # Simple conceptual question
    ]

    for q in test_questions:
        prediction = router.predict_with_confidence(q)
        print(f"\nQ: {q}")
        print(f"   → {prediction.pipeline} (confidence: {prediction.confidence:.2f})")
