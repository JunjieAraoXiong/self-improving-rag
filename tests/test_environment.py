"""Parallel environment smoke tests using pytest-xdist.

These tests verify all pipeline components work before running full evaluations.
With pytest-xdist, all 5 tests run simultaneously (~10-15s vs ~60s sequential).

Usage:
    pytest -n 4 tests/test_environment.py       # 4 parallel workers
    pytest -n auto tests/test_environment.py    # Auto-detect CPU cores
    pytest tests/test_environment.py -m "not slow"  # Skip API tests
"""

import os
import pytest


class TestEnvironment:
    """Environment and dependency smoke tests.

    Each test is independent and can run in parallel with pytest-xdist.
    Tests are ordered from fastest to slowest for optimal scheduling.
    """

    def test_pyarrow_fix(self):
        """Test PyArrow/HuggingFace Datasets compatibility.

        This verifies the critical fix for the PyArrow segfault issue
        that can crash the entire pipeline when using HF Datasets.
        """
        import pandas as pd
        from datasets import Dataset

        df = pd.DataFrame({"test": [1, 2, 3], "name": ["a", "b", "c"]})
        ds = Dataset.from_pandas(df)

        assert len(ds) == 3
        assert ds[0]["test"] == 1
        assert ds[0]["name"] == "a"

    def test_reranker_loads(self):
        """Test reranker model loading and inference.

        Verifies the CrossEncoder reranker can be loaded and produces
        valid similarity scores. This model is used for hybrid_filter_rerank.
        """
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

        # Test inference
        score = model.predict([("query about revenue", "Revenue was $10 billion")])

        assert score is not None
        # Score can be a float, list, or numpy array
        import numpy as np
        if isinstance(score, np.ndarray):
            assert score.shape[0] > 0
        else:
            assert isinstance(score, (float, list))

    @pytest.mark.slow
    @pytest.mark.api
    def test_chromadb_connection(self, chroma_path, has_chromadb, has_openai_key):
        """Test ChromaDB has data.

        Verifies the vector database is accessible and contains chunks.
        Uses OpenAI embeddings to match the indexed embeddings.
        """
        if not has_chromadb:
            pytest.skip("ChromaDB not available at expected path")
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set (needed for ChromaDB embeddings)")

        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        # Use OpenAI embeddings to match indexed data
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        db = Chroma(persist_directory=str(chroma_path), embedding_function=embeddings)

        count = db._collection.count()
        assert count > 0, f"ChromaDB empty: {count} chunks"

    @pytest.mark.slow
    @pytest.mark.api
    def test_openai_embedding_api(self, has_openai_key):
        """Test OpenAI embedding API connectivity.

        Verifies the text-embedding-3-large model works and returns
        the expected 3072-dimensional vectors.
        """
        if not has_openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        result = embeddings.embed_query("test query for embedding")

        assert len(result) == 3072
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.slow
    @pytest.mark.api
    def test_together_api(self, has_together_key):
        """Test Together API (LLM) connectivity.

        Verifies the Llama 70B model can be called through Together's
        OpenAI-compatible API endpoint.
        """
        if not has_together_key:
            pytest.skip("TOGETHER_API_KEY not set")

        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
        )

        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 0


class TestImports:
    """Verify all critical imports work without errors.

    These tests catch import-time issues that might not surface
    until runtime in the main pipeline.
    """

    def test_agents_import(self):
        """Test agent module imports."""
        from src.agents.base import BaseAgent, AgentDecision
        from src.agents.judge_agent import JudgeAgent

        assert BaseAgent is not None
        assert AgentDecision is not None
        assert JudgeAgent is not None

    def test_routing_import(self):
        """Test routing module imports."""
        from src.routing.features import FeatureExtractor, QuestionFeatures

        assert FeatureExtractor is not None
        assert QuestionFeatures is not None

    def test_retrieval_tools_import(self):
        """Test retrieval tools imports."""
        from src.retrieval_tools.tool_registry import list_pipelines, build_pipeline

        pipelines = list_pipelines()
        assert pipelines is not None
        assert len(pipelines) > 0
        assert "semantic" in pipelines

    def test_config_import(self):
        """Test configuration imports."""
        from src.config import DEFAULTS

        assert DEFAULTS is not None
        # Config uses llm_model, not default_model
        assert hasattr(DEFAULTS, "llm_model")
        assert hasattr(DEFAULTS, "embedding_model")
