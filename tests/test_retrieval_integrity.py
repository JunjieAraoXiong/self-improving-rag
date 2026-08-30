"""Regression tests for retrieval filtering, ordering, and provenance."""

import json
import importlib
from pathlib import Path

from langchain_core.documents import Document

from src.config import DEFAULTS
from src.ingest_docling import add_chunk_identity
from src.metadata_utils import (
    extract_metadata_from_question,
    filter_chunks_by_metadata,
    normalize_company_name,
    parse_filename,
)
from src.retrieval_tools.rerank import Reranker
from src.retrieval_tools.rse import get_chunk_position
from src.retrieval_tools.tool_registry import SimplePipeline, _build_chroma_filter


def test_default_retrieval_depth_matches_paper_schedule():
    assert DEFAULTS.top_k == 10


def test_legacy_retrieval_module_has_no_classic_or_api_key_import_side_effect():
    module = importlib.import_module("src.retrieval")

    assert callable(module.create_retriever)


def test_multiword_company_uses_ingested_metadata_key():
    metadata_filter = _build_chroma_filter(
        {"companies": ["Best Buy"], "years": [2022]}
    )

    assert normalize_company_name("Best Buy") == "BESTBUY"
    assert metadata_filter == {
        "$and": [{"company": "BESTBUY"}, {"year": 2022}]
    }


def test_metadata_filter_fails_closed_instead_of_returning_wrong_company():
    docs = [
        Document(
            page_content="Apple revenue",
            metadata={"company": "APPLE", "year": 2022, "source": "APPLE_2022_10K"},
        )
    ]

    filtered = filter_chunks_by_metadata(
        docs, {"companies": ["Best Buy"], "years": [2022], "doc_types": []}
    )

    assert filtered == []


def test_legacy_source_filename_alias_is_canonicalized_during_postfilter():
    docs = [
        Document(
            page_content="Activision evidence",
            metadata={"source": "/legacy/ACTIVSIONBLIZZARD_2023Q2_10Q.pdf"},
        )
    ]

    filtered = filter_chunks_by_metadata(
        docs,
        {
            "companies": ["Activision Blizzard"],
            "years": [2023],
            "doc_types": ["10q"],
        },
    )

    assert filtered == docs


def test_all_financebench_document_names_parse_to_idempotent_company_keys():
    dataset = (
        Path(__file__).parents[1]
        / "data"
        / "question_sets"
        / "financebench_document_information.jsonl"
    )

    for line in dataset.read_text().splitlines():
        row = json.loads(line)
        metadata = parse_filename(row["doc_name"])
        assert metadata is not None, row["doc_name"]
        assert metadata.company == normalize_company_name(metadata.company)
        assert metadata.company == normalize_company_name(row["company"])


def test_financebench_question_aliases_resolve_to_the_expected_company():
    dataset = (
        Path(__file__).parents[1]
        / "data"
        / "question_sets"
        / "financebench_open_source.jsonl"
    )
    questions_without_entity = []

    for index, line in enumerate(dataset.read_text().splitlines()):
        row = json.loads(line)
        extracted = extract_metadata_from_question(row["question"])
        targets = {
            normalize_company_name(company)
            for company in extracted["companies"]
        }
        if not targets:
            questions_without_entity.append(index)
            continue
        assert normalize_company_name(row["company"]) in targets, (
            index,
            row["company"],
            row["question"],
            targets,
        )

    assert len(questions_without_entity) == 3


def test_prefilter_miss_falls_back_to_strict_source_postfilter():
    class BaseRetriever:
        def invoke(self, question):
            return [
                Document(
                    page_content="Apple evidence",
                    metadata={"company": "APPLE", "year": 2022},
                ),
                Document(
                    page_content="Best Buy evidence",
                    metadata={"company": "BESTBUY", "year": 2022},
                ),
            ]

    class LegacyDB:
        def get(self, where=None):
            return {"documents": [], "metadatas": []}

    pipeline = SimplePipeline(
        retriever=BaseRetriever(),
        top_k=10,
        use_metadata_filter=True,
        use_rerank=False,
        initial_k_factor=1.0,
        set_k_fn=lambda retriever, k: None,
        take_top_k_fn=lambda docs, k: docs[:k],
        db=LegacyDB(),
        use_hybrid=True,
    )

    docs = pipeline.retrieve("What was Best Buy revenue in 2022?")

    assert [doc.page_content for doc in docs] == ["Best Buy evidence"]


def test_empty_prefilter_result_also_uses_strict_postfilter_fallback():
    class EmptyRetriever:
        def invoke(self, question):
            return []

    class BroadRetriever:
        def invoke(self, question):
            return [
                Document(
                    page_content="Best Buy evidence",
                    metadata={"company": "BESTBUY", "year": 2022},
                )
            ]

    pipeline = SimplePipeline(
        retriever=BroadRetriever(),
        top_k=10,
        use_metadata_filter=True,
        use_rerank=False,
        initial_k_factor=1.0,
        set_k_fn=lambda retriever, k: None,
        take_top_k_fn=lambda docs, k: docs[:k],
        db=object(),
        use_hybrid=True,
    )
    pipeline._get_filtered_retriever = lambda metadata_filter, k: EmptyRetriever()

    docs = pipeline.retrieve("What was Best Buy revenue in 2022?")

    assert [doc.page_content for doc in docs] == ["Best Buy evidence"]


def test_rse_order_prefers_chunk_index_over_shared_page():
    doc = Document(page_content="chunk", metadata={"page": 7, "chunk_index": 12})

    assert get_chunk_position(doc) == 12


def test_ingestion_assigns_stable_source_local_chunk_identity():
    docs = add_chunk_identity(
        [Document(page_content="first"), Document(page_content="second")],
        "BESTBUY_2022_10K.pdf",
    )

    assert [doc.metadata["chunk_index"] for doc in docs] == [0, 1]
    assert [doc.metadata["chunk_id"] for doc in docs] == [
        "BESTBUY_2022_10K:chunk:00000",
        "BESTBUY_2022_10K:chunk:00001",
    ]


def test_local_reranker_preserves_scores_for_rse():
    class FakeModel:
        def predict(self, pairs):
            return [0.2, 0.9]

    reranker = Reranker.__new__(Reranker)
    reranker.model_name = "fake"
    reranker.model = FakeModel()
    docs = [Document(page_content="a"), Document(page_content="b")]

    ranked = reranker.rerank("query", docs, top_k=2)

    assert [doc.page_content for doc in ranked] == ["b", "a"]
    assert [doc.metadata["rerank_score"] for doc in ranked] == [0.9, 0.2]


def test_rse_segment_documents_preserve_source_and_child_ids():
    pipeline = SimplePipeline.__new__(SimplePipeline)
    pipeline.rse_preset = "balanced"
    pipeline.retrieve = lambda question: [
        Document(
            page_content="first",
            metadata={
                "source": "BESTBUY_2022_10K.pdf",
                "page": 3,
                "chunk_index": 10,
                "chunk_id": "c10",
                "rerank_score": 1.0,
            },
        ),
        Document(
            page_content="second",
            metadata={
                "source": "BESTBUY_2022_10K.pdf",
                "page": 3,
                "chunk_index": 11,
                "chunk_id": "c11",
                "rerank_score": 0.9,
            },
        ),
    ]

    segments = pipeline.retrieve_segment_documents("query")

    assert len(segments) == 1
    assert segments[0].metadata["source"] == "BESTBUY_2022_10K.pdf"
    assert segments[0].metadata["positions"] == [10, 11]
    assert segments[0].metadata["child_chunk_ids"] == ["c10", "c11"]
