"""Shared fixtures for parallel test execution with pytest-xdist.

This module provides:
- Parallel-safe fixtures that don't share mutable state
- Cached resources (models, embeddings) loaded once per worker
- Mock data generators for unit testing
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
RAG_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(RAG_ROOT))
sys.path.insert(0, str(RAG_ROOT / "src"))


# =============================================================================
# Session-scoped Fixtures (loaded once per worker)
# =============================================================================

@pytest.fixture(scope="session")
def rag_root() -> Path:
    """Return the RAG project root directory."""
    return RAG_ROOT


@pytest.fixture(scope="session")
def chroma_path(rag_root) -> Path:
    """Return path to ChromaDB directory.

    Prefers chroma_docling (FinanceBench) if available, falls back to chroma.
    """
    docling_path = rag_root / "chroma_docling"
    if docling_path.exists():
        return docling_path
    return rag_root / "chroma"


@pytest.fixture(scope="session")
def feature_extractor():
    """Get or create the feature extractor (session-scoped for efficiency).

    This is loaded once per pytest-xdist worker to avoid redundant initialization.
    """
    from src.routing.features import FeatureExtractor
    return FeatureExtractor()


@pytest.fixture(scope="session")
def sample_questions() -> list:
    """Sample questions for testing feature extraction and routing."""
    return [
        "What is the FY2018 capital expenditure amount for 3M?",
        "What was Apple's revenue in 2022?",
        "Is 3M a capital-intensive business based on FY2022 data?",
        "Why did Microsoft's profit margin decline in Q4 2021?",
        "Explain the termination clause in the contract.",
        "What treatment options are available for type 2 diabetes?",
        "How much debt did Tesla have in 2023?",
        "Compare the P/E ratios of Google and Amazon.",
    ]


# =============================================================================
# Function-scoped Fixtures (fresh for each test)
# =============================================================================

@pytest.fixture
def mock_documents():
    """Create mock LangChain documents for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Apple reported $394 billion in revenue for fiscal year 2022.",
            metadata={"source": "apple_10k_2022.pdf", "page": 1}
        ),
        Document(
            page_content="3M's capital expenditure was $1.5 billion in FY2018.",
            metadata={"source": "3m_10k_2018.pdf", "page": 5}
        ),
        Document(
            page_content="The termination clause allows either party to exit with 30 days notice.",
            metadata={"source": "contract_a.pdf", "page": 12}
        ),
    ]


@pytest.fixture
def mock_agent_decision():
    """Create a mock agent decision for testing."""
    from src.agents.base import AgentDecision

    return AgentDecision(
        agent_name="TestAgent",
        decision_type="test_decision",
        decision_value={"result": "pass"},
        confidence=0.85,
        reasoning="Test decision for unit testing",
        metadata={"attempt": 0}
    )


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.fixture(scope="session")
def has_together_key() -> bool:
    """Check if Together API key is available."""
    return bool(os.environ.get("TOGETHER_API_KEY"))


@pytest.fixture(scope="session")
def has_chromadb(chroma_path) -> bool:
    """Check if ChromaDB exists and has data."""
    return chroma_path.exists() and any(chroma_path.iterdir())


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (API calls, model loading)"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API keys"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


# =============================================================================
# Parallel Safety Hooks
# =============================================================================

@pytest.fixture(autouse=True)
def isolate_test_state():
    """Ensure each test starts with clean state.

    This fixture runs before each test to prevent state leakage
    between parallel workers.
    """
    # Reset any global state that might leak between tests
    yield
    # Cleanup after test if needed


def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test names."""
    for item in items:
        # Mark API tests
        if "api" in item.nodeid.lower() or "openai" in item.nodeid.lower():
            item.add_marker(pytest.mark.api)

        # Mark slow tests
        if any(kw in item.nodeid.lower() for kw in ["reranker", "chromadb", "embedding"]):
            item.add_marker(pytest.mark.slow)
