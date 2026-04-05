"""Registry for retrieval pipelines/policies."""

from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.documents import Document

from .base import RetrievalPipeline
from .semantic import build_semantic_retriever, set_retriever_k as set_semantic_k, take_top_k as take_semantic_top_k
from .hybrid import build_hybrid_retriever, set_retriever_k as set_hybrid_k, take_top_k as take_hybrid_top_k
from .metadata_filter import filter_with_question_metadata
from .rerank import get_reranker
from .rse import extract_relevant_segments
from src.config import DEFAULTS
from src.metadata_utils import extract_metadata_from_question


def _build_chroma_filter(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a ChromaDB where filter from extracted question metadata.

    Args:
        metadata: Dict with 'companies', 'years', 'doc_types' lists

    Returns:
        ChromaDB where filter dict, or None if no useful filters
    """
    companies = metadata.get('companies', [])
    years = metadata.get('years', [])

    # Only build filter if we have BOTH company AND year (high confidence)
    # Single-signal filtering (just company or just year) is too aggressive
    if not companies or not years:
        return None

    # Build filter conditions
    conditions = []

    # Company filter (case-insensitive matching via uppercase)
    if len(companies) == 1:
        conditions.append({"company": companies[0].upper()})
    else:
        conditions.append({"company": {"$in": [c.upper() for c in companies]}})

    # Year filter
    if len(years) == 1:
        conditions.append({"year": years[0]})
    else:
        conditions.append({"year": {"$in": years}})

    # Combine with $and
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class SimplePipeline(RetrievalPipeline):
    """Composable retrieval pipeline with pre-filtering support.

    Supports:
    - PRE-FILTERING: When question has company+year, builds filtered retriever
      to search only relevant documents (fixes BM25 keyword dilution issue)
    - POST-FILTERING: Additional metadata filtering after retrieval
    - RSE (Relevant Segment Extraction): Merges adjacent chunks into segments
    """

    def __init__(
        self,
        retriever,
        top_k: int,
        use_metadata_filter: bool,
        use_rerank: bool,
        initial_k_factor: float,
        set_k_fn,
        take_top_k_fn,
        reranker_model: str = None,
        use_rse: bool = False,
        rse_preset: str = "balanced",
        db=None,  # ChromaDB instance for pre-filtering
        use_hybrid: bool = False,  # Whether this is a hybrid pipeline
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.use_metadata_filter = use_metadata_filter
        self.use_rerank = use_rerank
        self.initial_k_factor = max(1.0, float(initial_k_factor))
        self._set_k = set_k_fn
        self._take_top_k = take_top_k_fn
        self._reranker = None
        self._reranker_model = reranker_model
        self.use_rse = use_rse
        self.rse_preset = rse_preset
        self._db = db
        self._use_hybrid = use_hybrid
        # Cache for filtered retrievers (by filter hash)
        self._filtered_retriever_cache: Dict[str, Any] = {}

    def _get_filtered_retriever(self, metadata_filter: Dict[str, Any], k: int) -> Any:
        """Get or build a filtered retriever for the given metadata filter."""
        # Create cache key from filter
        cache_key = str(sorted(metadata_filter.items()) if isinstance(metadata_filter, dict) else metadata_filter)

        if cache_key not in self._filtered_retriever_cache:
            if self._use_hybrid:
                retriever = build_hybrid_retriever(
                    self._db,
                    top_k=k,
                    weights=DEFAULTS.ensemble_weights,
                    metadata_filter=metadata_filter,
                )
            else:
                # For semantic-only, use filter in search_kwargs
                retriever = self._db.as_retriever(
                    search_kwargs={"k": k, "filter": metadata_filter}
                )
            self._filtered_retriever_cache[cache_key] = retriever

        return self._filtered_retriever_cache[cache_key]

    def retrieve(self, question: str) -> List[Document]:
        """Retrieve documents for a question.

        Uses PRE-FILTERING when question contains company+year metadata,
        then applies post-filtering and reranking as configured.
        """
        multiplier = self.initial_k_factor if (self.use_metadata_filter or self.use_rerank) else 1.0
        initial_k = max(self.top_k, int(self.top_k * multiplier))

        # Try PRE-FILTERING: Extract metadata and build filtered retriever
        use_prefilter = False
        if self._db is not None and self.use_metadata_filter:
            metadata = extract_metadata_from_question(question)
            chroma_filter = _build_chroma_filter(metadata)
            if chroma_filter:
                try:
                    retriever = self._get_filtered_retriever(chroma_filter, initial_k)
                    docs = retriever.invoke(question)
                    use_prefilter = True
                    if docs:
                        print(f"✓ Pre-filter: {len(docs)} docs for {metadata.get('companies')} {metadata.get('years')}")
                except Exception as e:
                    print(f"⚠️ Pre-filter failed: {e}, falling back to standard retrieval")
                    use_prefilter = False

        # Fall back to standard retrieval if pre-filtering wasn't used
        if not use_prefilter:
            self._set_k(self.retriever, initial_k)
            docs = self.retriever.invoke(question)

        # POST-FILTERING: Apply additional metadata filtering (still useful for edge cases)
        if self.use_metadata_filter and not use_prefilter:
            filtered_docs, used_metadata = filter_with_question_metadata(question, docs)
            if used_metadata:
                docs = filtered_docs
            elif filtered_docs:
                docs = filtered_docs

        if self.use_rerank:
            self._reranker = get_reranker(self._reranker, model_name=self._reranker_model) if self._reranker_model else get_reranker(self._reranker)
            docs = self._reranker.rerank(question, docs, self.top_k)
        else:
            docs = self._take_top_k(docs, self.top_k)

        return docs

    def retrieve_segments(self, question: str) -> List[str]:
        """Retrieve and merge into segments using RSE.

        Always applies RSE regardless of use_rse flag.
        Use this when you want segment-level context.
        """
        docs = self.retrieve(question)
        if not docs:
            return []

        # Get relevance scores from metadata if available (set by Cohere reranker)
        relevance_scores = [
            doc.metadata.get("rerank_score", 1.0 / (i + 1))
            for i, doc in enumerate(docs)
        ]

        return extract_relevant_segments(
            docs,
            relevance_scores=relevance_scores,
            preset=self.rse_preset,
        )


def _pipeline_flags(pipeline_id: str) -> Tuple[bool, bool, bool]:
    """Return (use_hybrid, use_filter, use_rerank) for a pipeline id."""
    mapping = {
        "semantic": (False, False, False),
        "hybrid": (True, False, False),
        "hybrid_filter": (True, True, False),
        "hybrid_filter_rerank": (True, True, True),
    }
    if pipeline_id not in mapping:
        raise ValueError(f"Unknown pipeline_id '{pipeline_id}'")
    return mapping[pipeline_id]


def build_retriever_for_pipeline(pipeline_id: str, db, top_k: int):
    """Return a retriever, helpers, and metadata for the given pipeline.

    Returns:
        tuple: (retriever, set_k_fn, take_top_k_fn, use_hybrid)
    """
    use_hybrid, _, _ = _pipeline_flags(pipeline_id)
    if use_hybrid:
        retriever = build_hybrid_retriever(
            db,
            top_k=top_k,
            weights=DEFAULTS.ensemble_weights,  # Use config weights (BM25, semantic)
        )
        return retriever, set_hybrid_k, take_hybrid_top_k, use_hybrid
    retriever = build_semantic_retriever(db, top_k=top_k)
    return retriever, set_semantic_k, take_semantic_top_k, use_hybrid


def build_pipeline(
    pipeline_id: str,
    retriever,
    top_k: int,
    initial_k_factor: float,
    set_k_fn,
    take_top_k_fn,
    reranker_model: str = None,
    use_rse: bool = False,
    rse_preset: str = "balanced",
    db=None,  # ChromaDB instance for pre-filtering
    use_hybrid: bool = False,  # Whether this is a hybrid pipeline
) -> SimplePipeline:
    """Construct a SimplePipeline for the pipeline id.

    Args:
        pipeline_id: One of "semantic", "hybrid", "hybrid_filter", "hybrid_filter_rerank"
        retriever: The base retriever to use
        top_k: Number of documents to return
        initial_k_factor: Multiplier for initial retrieval (before filtering/reranking)
        set_k_fn: Function to set k on the retriever
        take_top_k_fn: Function to take top k from results
        reranker_model: Reranker model name (if using reranking)
        use_rse: Enable RSE (Relevant Segment Extraction) for retrieve_segments()
        rse_preset: RSE preset ("balanced", "precision", "find_all")
        db: ChromaDB instance for pre-filtering (enables dynamic filtered retriever)
        use_hybrid: Whether this is a hybrid (BM25+semantic) pipeline
    """
    _, use_filter, use_rerank = _pipeline_flags(pipeline_id)
    return SimplePipeline(
        retriever=retriever,
        top_k=top_k,
        use_metadata_filter=use_filter,
        use_rerank=use_rerank,
        initial_k_factor=initial_k_factor,
        set_k_fn=set_k_fn,
        take_top_k_fn=take_top_k_fn,
        reranker_model=reranker_model,
        use_rse=use_rse,
        rse_preset=rse_preset,
        db=db,
        use_hybrid=use_hybrid,
    )


def list_pipelines() -> List[str]:
    """Return supported pipeline ids."""
    return ["semantic", "hybrid", "hybrid_filter", "hybrid_filter_rerank", "routed"]


# Re-export routed pipeline builder for convenience
from .router import build_routed_pipeline
