"""Retrieval Agent: Decides retrieval strategy for each question."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from .base import AgentDecision, BaseAgent


@dataclass
class RetrievalStrategy:
    """Encapsulates a retrieval strategy configuration."""
    pipeline_id: str
    top_k: int
    initial_k_factor: float
    use_hyde: bool = False
    use_rse: bool = False
    use_rerank: bool = True


# Escalation strategies: each retry uses a more aggressive configuration
ESCALATION_STRATEGIES = [
    # Attempt 0: Standard balanced retrieval
    RetrievalStrategy(
        pipeline_id="hybrid_filter_rerank",
        top_k=10,
        initial_k_factor=3.0,
        use_hyde=False,
        use_rerank=True,
    ),
    # Attempt 1: More documents, higher initial retrieval
    RetrievalStrategy(
        pipeline_id="hybrid_filter_rerank",
        top_k=20,
        initial_k_factor=4.0,
        use_hyde=False,
        use_rerank=True,
    ),
    # Attempt 2: Use RSE for richer context windows
    RetrievalStrategy(
        pipeline_id="hybrid_filter_rerank",
        top_k=25,
        initial_k_factor=5.0,
        use_hyde=False,
        use_rse=True,
        use_rerank=True,
    ),
    # Attempt 3: Maximum recall configuration with RSE
    RetrievalStrategy(
        pipeline_id="hybrid_filter_rerank",
        top_k=30,
        initial_k_factor=6.0,
        use_hyde=False,
        use_rse=True,
        use_rerank=True,
    ),
]


class RetrievalAgent(BaseAgent):
    """Agent A: Decides retrieval strategy based on question characteristics.

    This agent analyzes the question and decides:
    1. Which pipeline to use (semantic, hybrid, hybrid_filter_rerank)
    2. How many documents to retrieve (top_k)
    3. Whether to use HyDE for query expansion
    4. How aggressive the initial retrieval should be (initial_k_factor)

    The agent can escalate its strategy on retry, retrieving more
    documents or using different techniques if the initial attempt fails.
    """

    def __init__(
        self,
        db=None,
        embedding_fn=None,
        reranker_model: str = None,
        use_rule_router: bool = True,
        use_rse: bool = False,
        disable_escalation: bool = False,
        disable_hyde: bool = False,
    ):
        """Initialize the retrieval agent.

        Args:
            db: ChromaDB instance
            embedding_fn: Embedding function for HyDE
            reranker_model: Model name for reranking
            use_rule_router: Whether to use rule-based classification
            use_rse: Whether to use Relevant Segment Extraction
            disable_escalation: Ablation flag - always use attempt 0 strategy
            disable_hyde: Ablation flag - never enable HyDE
        """
        super().__init__("RetrievalAgent")
        self.db = db
        self.embedding_fn = embedding_fn
        self.reranker_model = reranker_model
        self.use_rule_router = use_rule_router
        self.use_rse = use_rse
        self.disable_escalation = disable_escalation
        self.disable_hyde = disable_hyde

        # Lazy-loaded components
        self._classifier = None
        self._hyde_generator = None
        self._pipeline = None  # Pre-built pipeline (set via set_pipeline)
        self._pipeline_cache: Dict[tuple, Any] = {}

        # Current strategy (can be escalated on retry)
        self._current_strategy: Optional[RetrievalStrategy] = None

    @property
    def classifier(self):
        """Lazy-load the question classifier."""
        if self._classifier is None:
            if self.use_rule_router:
                from src.retrieval_tools.router import RuleBasedClassifier
                self._classifier = RuleBasedClassifier()
            else:
                from src.retrieval_tools.router import get_classifier
                self._classifier = get_classifier()
        return self._classifier

    @property
    def hyde_generator(self):
        """Lazy-load the HyDE generator."""
        if self._hyde_generator is None:
            from src.retrieval_tools.router import get_hyde_generator
            self._hyde_generator = get_hyde_generator()
        return self._hyde_generator

    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question to extract features for strategy selection.

        Args:
            question: The question text

        Returns:
            Dictionary of extracted features
        """
        classification = self.classifier.classify(question)

        # Extract additional features
        features = {
            "question_type": classification.question_type,
            "classification_confidence": classification.confidence,
            "classification_reasoning": classification.reasoning,
            "has_company_name": self._detect_company_name(question),
            "has_fiscal_year": self._detect_fiscal_year(question),
            "has_numeric_ask": self._detect_numeric_ask(question),
            "question_length": len(question.split()),
        }

        return features

    def _detect_company_name(self, question: str) -> bool:
        """Check if question mentions a company name."""
        # Simple heuristic: check for capitalized words that could be company names
        import re
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, question)
        # Filter out common non-company words
        common_words = {'What', 'How', 'Why', 'When', 'Where', 'Which', 'The', 'In', 'For'}
        company_candidates = [m for m in matches if m not in common_words]
        return len(company_candidates) > 0

    def _detect_fiscal_year(self, question: str) -> bool:
        """Check if question mentions a fiscal year."""
        import re
        patterns = [
            r'\b20\d{2}\b',  # Years like 2020, 2021, etc.
            r'\bFY\s*\d{2,4}\b',  # FY22, FY2022
            r'\bfiscal\s+year\b',
            r'\bQ[1-4]\s*\d{2,4}\b',  # Q1 2022, Q4 22
        ]
        for pattern in patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return True
        return False

    def _detect_numeric_ask(self, question: str) -> bool:
        """Check if question asks for a numeric answer."""
        numeric_patterns = [
            r'\bhow much\b',
            r'\bhow many\b',
            r'\bwhat is the\s+\w*\s*(ratio|percentage|amount|value|number|revenue|income|cost)\b',
            r'\bcalculate\b',
            r'\b\d+\s*(million|billion|thousand|percent|%)\b',
        ]
        question_lower = question.lower()
        for pattern in numeric_patterns:
            if __import__('re').search(pattern, question_lower):
                return True
        return False

    def get_strategy(self, attempt: int) -> RetrievalStrategy:
        """Get the retrieval strategy for the given attempt.

        Args:
            attempt: The attempt number (0-indexed)

        Returns:
            The retrieval strategy to use
        """
        # Ablation: disable escalation - always use attempt 0 strategy
        if self.disable_escalation:
            strategy_idx = 0
        else:
            strategy_idx = min(attempt, len(ESCALATION_STRATEGIES) - 1)

        strategy = ESCALATION_STRATEGIES[strategy_idx]

        # Ablation: disable HyDE - override use_hyde to False
        if self.disable_hyde and strategy.use_hyde:
            # Create a copy with HyDE disabled
            strategy = RetrievalStrategy(
                pipeline_id=strategy.pipeline_id,
                top_k=strategy.top_k,
                initial_k_factor=strategy.initial_k_factor,
                use_hyde=False,  # Override
                use_rerank=strategy.use_rerank,
            )

        return strategy

    def decide(self, context: Dict[str, Any]) -> AgentDecision:
        """Decide on retrieval strategy based on question and attempt number.

        Args:
            context: Must contain 'question' key, optionally 'attempt'

        Returns:
            AgentDecision with pipeline configuration
        """
        question = context["question"]
        attempt = context.get("attempt", self._attempt)

        # Analyze the question
        features = self.analyze_question(question)

        # Get strategy based on attempt (escalation on retry)
        strategy = self.get_strategy(attempt)
        self._current_strategy = strategy

        # Build reasoning
        if attempt == 0:
            reasoning = (
                f"Initial retrieval using {strategy.pipeline_id} pipeline. "
                f"Question type: {features['question_type']} "
                f"(confidence: {features['classification_confidence']:.2f}). "
                f"Retrieving top-{strategy.top_k} with {strategy.initial_k_factor}x initial factor."
            )
        else:
            reasoning = (
                f"Retry #{attempt}: Escalating retrieval strategy. "
                f"Increased to top-{strategy.top_k} documents. "
                f"{'Using HyDE for query expansion.' if strategy.use_hyde else ''}"
            )

        # Determine confidence based on features
        confidence = features["classification_confidence"]
        if features["has_company_name"] and features["has_fiscal_year"]:
            confidence = min(1.0, confidence + 0.1)  # Boost for well-specified queries

        decision = AgentDecision(
            agent_name=self.name,
            decision_type="pipeline_selection",
            decision_value={
                "pipeline_id": strategy.pipeline_id,
                "top_k": strategy.top_k,
                "initial_k_factor": strategy.initial_k_factor,
                "use_hyde": strategy.use_hyde,
                "use_rerank": strategy.use_rerank,
            },
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "attempt": attempt,
                "question_features": features,
            }
        )

        self.log_decision(decision)
        return decision

    def retrieve(self, question: str, decision: AgentDecision = None) -> List[Document]:
        """Execute retrieval based on the current strategy.

        Args:
            question: The question to retrieve documents for
            decision: Optional decision to use (uses current strategy if not provided)

        Returns:
            List of retrieved documents (or synthetic Documents from RSE segments)
        """
        if decision is None:
            decision = self.get_last_decision()

        if decision is None:
            # No decision made yet, use default
            decision = self.decide({"question": question})

        strategy_values = decision.decision_value
        top_k = strategy_values["top_k"]

        # Handle HyDE if enabled
        if strategy_values.get("use_hyde"):
            hypothetical_doc = self.hyde_generator.generate_hypothetical_document(question)
            query = f"{question}\n\n{hypothetical_doc}"
        else:
            query = question

        # Use pre-built pipeline if available
        if self._pipeline is not None:
            # Update top_k dynamically
            if hasattr(self._pipeline, 'top_k'):
                self._pipeline.top_k = top_k

            # Use RSE if enabled and pipeline supports it
            if self.use_rse and hasattr(self._pipeline, 'retrieve_segments'):
                segments = self._pipeline.retrieve_segments(query)
                # Convert segments to Documents for downstream compatibility
                return [
                    Document(page_content=seg, metadata={"source": "RSE segment", "segment_idx": i})
                    for i, seg in enumerate(segments)
                ] if segments else []

            return self._pipeline.retrieve(query)

        # Extract metadata from question for filtering
        # This ensures company-specific queries (e.g., "AMD revenue FY22") retrieve
        # the correct company's documents rather than semantically similar but wrong docs
        from src.metadata_utils import extract_metadata_from_question, filter_chunks_by_metadata
        question_metadata = extract_metadata_from_question(question)
        companies = question_metadata.get('companies', [])
        years = question_metadata.get('years', [])

        # Build ChromaDB filter if BOTH company AND year are specified (high confidence)
        chroma_filter = None
        if companies and years:
            filter_conditions = []
            # Normalize company names to match ChromaDB format (uppercase, no spaces)
            # ChromaDB stores: "BESTBUY", "AMERICANWATERWORKS", "COCACOLA" etc.
            normalized_companies = [
                c.upper().replace(" ", "").replace("-", "").replace("&", "AND")
                for c in companies
            ]
            if len(normalized_companies) == 1:
                filter_conditions.append({"company": normalized_companies[0]})
            else:
                filter_conditions.append({"company": {"$in": normalized_companies}})

            # Years stored as integers in metadata
            if len(years) == 1:
                filter_conditions.append({"year": years[0]})
            else:
                filter_conditions.append({"year": {"$in": years}})

            if len(filter_conditions) == 1:
                chroma_filter = filter_conditions[0]
            else:
                chroma_filter = {"$and": filter_conditions}

        # Try PRE-FILTERING at retrieval time (most efficient)
        # ChromaDB 1.3.5 has a bug with similarity_search(filter=...), so we use
        # a two-step approach: get filtered docs first, then do BM25 search within them
        docs = None
        if chroma_filter:
            try:
                # Step 1: Get filtered documents using db.get() which works reliably
                filtered_result = self.db.get(where=chroma_filter, limit=top_k * 3)
                if filtered_result and filtered_result.get("documents"):
                    # Step 2: Do BM25 search within filtered docs for relevance ranking
                    from langchain_community.retrievers import BM25Retriever
                    filtered_docs = [
                        Document(page_content=text, metadata=meta)
                        for text, meta in zip(
                            filtered_result["documents"],
                            filtered_result["metadatas"]
                        )
                    ]
                    if len(filtered_docs) > 0:
                        bm25 = BM25Retriever.from_documents(filtered_docs)
                        bm25.k = min(top_k, len(filtered_docs))
                        docs = bm25.invoke(query)
                        print(f"✓ Pre-filter + BM25: {len(docs)} docs for {companies} {years}")
            except Exception as e:
                print(f"⚠️ Pre-filter failed: {e}, using unfiltered search")
                docs = None

        # Fallback: unfiltered semantic search + post-retrieval filtering
        if docs is None:
            docs = self.db.similarity_search(query, k=top_k)
            # Apply POST-FILTERING if we have metadata (may return partial results)
            if companies or years:
                docs = filter_chunks_by_metadata(docs, question_metadata)

        # Apply reranking if configured
        if strategy_values.get("use_rerank") and self.reranker_model:
            from src.retrieval_tools.rerank import get_reranker
            reranker = get_reranker(model_name=self.reranker_model)
            docs = reranker.rerank(question, docs, top_k)

        return docs

    def set_pipeline(self, pipeline) -> None:
        """Set a pre-built pipeline to use for retrieval.

        This allows reusing an existing pipeline rather than rebuilding.

        Args:
            pipeline: A pre-built RetrievalPipeline instance
        """
        self._pipeline = pipeline

    def escalate_strategy(self) -> None:
        """Escalate to a more aggressive retrieval strategy."""
        self._attempt += 1
