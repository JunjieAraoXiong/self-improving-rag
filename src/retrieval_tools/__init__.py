# Retrieval tools package

from .base import RetrievalPipeline, RetrieverProtocol
from .tool_registry import (
    SimplePipeline,
    build_pipeline,
    build_retriever_for_pipeline,
    build_routed_pipeline,
    list_pipelines,
)
from .router import (
    RoutedPipeline,
    LLMQuestionClassifier,
    HyDEGenerator,
    HyDERetriever,
    TablePreferenceFilter,
    ClassificationResult,
    QuestionType,
)

__all__ = [
    # Protocols
    "RetrievalPipeline",
    "RetrieverProtocol",
    # Pipelines
    "SimplePipeline",
    "RoutedPipeline",
    # Factory functions
    "build_pipeline",
    "build_retriever_for_pipeline",
    "build_routed_pipeline",
    "list_pipelines",
    # Router components
    "LLMQuestionClassifier",
    "HyDEGenerator",
    "HyDERetriever",
    "TablePreferenceFilter",
    "ClassificationResult",
    "QuestionType",
]
