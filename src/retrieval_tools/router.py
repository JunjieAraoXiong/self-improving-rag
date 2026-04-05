"""Question-type router for optimized retrieval strategies."""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Any

from langchain_core.documents import Document

from .base import RetrievalPipeline
from .tool_registry import build_retriever_for_pipeline, build_pipeline
from .rerank import get_reranker


QuestionType = Literal["metrics-generated", "domain-relevant", "novel-generated"]


@dataclass
class ClassificationResult:
    """Result of question classification."""
    question_type: QuestionType
    confidence: float
    reasoning: str


# =============================================================================
# Rule-Based Classifier (FREE - no API calls)
# =============================================================================

class RuleBasedClassifier:
    """Free rule-based classifier using routing.RuleBasedRouter.

    Maps pipeline predictions to question types:
    - hybrid_filter_rerank → novel-generated (complex reasoning)
    - hybrid_filter → metrics-generated (specific company + time)
    - hybrid → domain-relevant (has financial metrics)
    - semantic → domain-relevant (simple queries)
    """

    def __init__(self):
        from src.routing.router import RuleBasedRouter
        self._router = RuleBasedRouter()
        self._pipeline_to_type = {
            'hybrid_filter_rerank': 'novel-generated',
            'hybrid_filter': 'metrics-generated',
            'hybrid': 'domain-relevant',
            'semantic': 'domain-relevant',
        }

    def classify(self, question: str) -> ClassificationResult:
        """Classify question using rule-based heuristics (instant, free)."""
        pipeline = self._router.predict(question)
        question_type = self._pipeline_to_type.get(pipeline, 'domain-relevant')

        return ClassificationResult(
            question_type=question_type,
            confidence=0.8,  # Fixed confidence for rule-based
            reasoning=f"Rule-based: {pipeline}"
        )


# =============================================================================
# LLM-Based Question Classifier
# =============================================================================

_classifier_cache: Dict[str, "LLMQuestionClassifier"] = {}


class LLMQuestionClassifier:
    """LLM-based question classifier with caching."""

    SYSTEM_PROMPT = """You are a question classifier for financial document QA.
Classify questions into exactly one of these types:

1. metrics-generated: Questions asking for specific numerical values, ratios,
   percentages, or financial metrics from tables. Examples:
   - "What was the revenue in 2022?"
   - "What is the debt-to-equity ratio?"
   - "How much did operating income increase?"

2. domain-relevant: Questions requiring understanding of financial concepts,
   definitions, or domain knowledge. Examples:
   - "What accounting method does the company use?"
   - "What are the main risk factors mentioned?"
   - "How does the company recognize revenue?"

3. novel-generated: Questions requiring inference, synthesis across multiple
   facts, or multi-step reasoning. Examples:
   - "Why did profitability decline despite revenue growth?"
   - "What strategic initiatives drove the margin improvement?"
   - "How might the acquisition affect future earnings?"

Respond with JSON only: {"type": "<type>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}"""

    def __init__(
        self,
        model_name: str = None,
        cache_enabled: bool = True
    ):
        from src.config import DEFAULTS
        self._model_name = model_name or DEFAULTS.router_classifier_model
        self._provider = None
        self._result_cache: Dict[int, ClassificationResult] = {}
        self._cache_enabled = cache_enabled

    @property
    def provider(self):
        """Lazy-load the LLM provider."""
        if self._provider is None:
            from src.providers.factory import get_provider
            self._provider = get_provider(self._model_name)
        return self._provider

    def classify(self, question: str) -> ClassificationResult:
        """Classify question into a type."""
        cache_key = question  # Use question directly as key (stable, no collision risk)
        if self._cache_enabled and cache_key in self._result_cache:
            return self._result_cache[cache_key]

        try:
            user_prompt = f"Classify this question: {question}"

            response = self.provider.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=150,
                temperature=0.0
            )

            result = self._parse_response(response.content)
        except Exception as e:
            # Fall back to default classification on any error
            result = ClassificationResult(
                question_type="domain-relevant",
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}"
            )

        if self._cache_enabled:
            self._result_cache[cache_key] = result

        return result

    def _parse_response(self, response: str) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        try:
            # Handle potential markdown code blocks more robustly
            content = response.strip()
            if content.startswith("```"):
                parts = content.split("```")
                if len(parts) >= 3:  # Proper code block: ```...```
                    content = parts[1]
                    if content.startswith("json"):
                        content = content[4:]
            content = content.strip()

            data = json.loads(content)
            question_type = data.get("type", "domain-relevant")

            # Validate question type
            if question_type not in ("metrics-generated", "domain-relevant", "novel-generated"):
                question_type = "domain-relevant"

            return ClassificationResult(
                question_type=question_type,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "")
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return ClassificationResult(
                question_type="domain-relevant",
                confidence=0.3,
                reasoning="Failed to parse classification response"
            )


def get_classifier(
    existing: Optional[LLMQuestionClassifier] = None,
    model_name: str = None
) -> LLMQuestionClassifier:
    """Get or create a cached classifier instance."""
    from src.config import DEFAULTS
    model = model_name or DEFAULTS.router_classifier_model

    if existing and existing._model_name == model:
        return existing

    if model in _classifier_cache:
        return _classifier_cache[model]

    classifier = LLMQuestionClassifier(model_name=model)
    _classifier_cache[model] = classifier
    return classifier


# =============================================================================
# HyDE (Hypothetical Document Embeddings)
# =============================================================================

_hyde_cache: Dict[str, "HyDEGenerator"] = {}


class HyDEGenerator:
    """Generates hypothetical document embeddings for improved retrieval."""

    SYSTEM_PROMPT = """You are a financial document passage generator.
Given a question about a financial document (10-K, 10-Q filing), write a
hypothetical passage that would contain the answer to this question.

Write as if you are quoting directly from a financial report. Include:
- Realistic financial terminology
- Placeholder numbers that fit the context
- Company-neutral language

The passage should be 2-3 sentences and sound like an actual excerpt."""

    def __init__(self, model_name: str = None):
        from src.config import DEFAULTS
        self._model_name = model_name or DEFAULTS.router_hyde_model
        self._provider = None

    @property
    def provider(self):
        """Lazy-load the LLM provider."""
        if self._provider is None:
            from src.providers.factory import get_provider
            self._provider = get_provider(self._model_name)
        return self._provider

    def generate_hypothetical_document(self, question: str) -> str:
        """Generate a hypothetical answer passage for the question.

        Returns the question itself as fallback if generation fails.
        """
        try:
            user_prompt = f"Generate a hypothetical passage answering: {question}"

            response = self.provider.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=200,
                temperature=0.3
            )

            return response.content
        except Exception:
            # Fall back to using the question itself for embedding
            return question


def get_hyde_generator(
    existing: Optional[HyDEGenerator] = None,
    model_name: str = None
) -> HyDEGenerator:
    """Get or create a cached HyDE generator instance."""
    from src.config import DEFAULTS
    model = model_name or DEFAULTS.router_hyde_model

    if existing and existing._model_name == model:
        return existing

    if model in _hyde_cache:
        return _hyde_cache[model]

    generator = HyDEGenerator(model_name=model)
    _hyde_cache[model] = generator
    return generator


class HyDERetriever:
    """Retriever that uses HyDE for semantic search."""

    def __init__(
        self,
        db,
        embedding_fn,
        hyde_generator: HyDEGenerator,
        top_k: int
    ):
        self.db = db
        self.embedding_fn = embedding_fn
        self.hyde_generator = hyde_generator
        self.top_k = top_k

    def invoke(self, query: str) -> List[Document]:
        """Retrieve using hypothetical document embedding."""
        hypothetical_doc = self.hyde_generator.generate_hypothetical_document(query)
        embedding = self.embedding_fn.embed_query(hypothetical_doc)

        results = self.db.similarity_search_by_vector(
            embedding=embedding,
            k=self.top_k
        )
        return results


# =============================================================================
# Table Preference Filter
# =============================================================================

class TablePreferenceFilter:
    """Boosts table chunks for metrics-generated questions."""

    def __init__(self, table_quota_ratio: float = 0.6):
        self.table_quota_ratio = table_quota_ratio

    def filter_and_boost(
        self,
        docs: List[Document],
        question_type: QuestionType,
        top_k: int
    ) -> List[Document]:
        """
        For metrics-generated questions, prioritize table chunks.
        Uses element_type metadata from ingest.py.
        """
        if question_type != "metrics-generated":
            return docs[:top_k]

        # Separate table and non-table docs
        table_docs = []
        other_docs = []

        for doc in docs:
            element_type = doc.metadata.get("element_type", "other")
            if element_type == "table":
                table_docs.append(doc)
            else:
                other_docs.append(doc)

        # Prioritize tables: fill quota with tables first, then others
        result = []
        table_quota = min(len(table_docs), int(top_k * self.table_quota_ratio))

        result.extend(table_docs[:table_quota])
        remaining = top_k - len(result)
        result.extend(other_docs[:remaining])

        # If we still need more, add remaining tables
        if len(result) < top_k:
            result.extend(table_docs[table_quota:top_k - len(result)])

        return result[:top_k]


# =============================================================================
# Routed Pipeline
# =============================================================================

class RoutedPipeline(RetrievalPipeline):
    """
    Question-type router that delegates to optimized SimplePipeline instances.
    Implements wrapper pattern around existing SimplePipeline.
    """

    def __init__(
        self,
        db,
        embedding_fn,
        classifier: LLMQuestionClassifier,
        routes: Dict[str, Any],
        hyde_generator: HyDEGenerator = None,
        table_filter: TablePreferenceFilter = None,
        reranker_model: str = None,
    ):
        self.db = db
        self.embedding_fn = embedding_fn
        self.classifier = classifier
        self.routes = routes
        self.hyde_generator = hyde_generator or HyDEGenerator()
        self.table_filter = table_filter or TablePreferenceFilter()
        self._reranker_model = reranker_model

        # Cache for pipelines (lazy initialization)
        self._pipeline_cache: Dict[tuple, Any] = {}
        self._hyde_retriever: Optional[HyDERetriever] = None

    def _get_effective_pipeline_id(self, route) -> str:
        """Get the effective pipeline_id, respecting skip_rerank flag.

        If skip_rerank is True and the pipeline uses reranking,
        returns the non-rerank variant.
        """
        pipeline_id = route.pipeline_id
        skip_rerank = getattr(route, 'skip_rerank', False)

        if skip_rerank and pipeline_id.endswith('_rerank'):
            # Strip _rerank suffix to get base pipeline
            return pipeline_id[:-7]  # Remove "_rerank"
        return pipeline_id

    def _get_pipeline(self, route) -> Any:
        """Get or create a SimplePipeline for the route."""
        # Use effective pipeline_id that respects skip_rerank
        effective_pipeline_id = self._get_effective_pipeline_id(route)
        cache_key = (effective_pipeline_id, route.top_k, route.initial_k_factor)

        if cache_key not in self._pipeline_cache:
            retriever, set_k_fn, take_top_k_fn, use_hybrid = build_retriever_for_pipeline(
                effective_pipeline_id, self.db, top_k=route.top_k
            )
            pipeline = build_pipeline(
                pipeline_id=effective_pipeline_id,
                retriever=retriever,
                top_k=route.top_k,
                initial_k_factor=route.initial_k_factor,
                set_k_fn=set_k_fn,
                take_top_k_fn=take_top_k_fn,
                reranker_model=self._reranker_model,
                db=self.db,
                use_hybrid=use_hybrid,
            )
            self._pipeline_cache[cache_key] = pipeline

        return self._pipeline_cache[cache_key]

    def _get_hyde_retriever(self, top_k: int) -> HyDERetriever:
        """Get or create HyDE retriever."""
        if self._hyde_retriever is None or self._hyde_retriever.top_k != top_k:
            self._hyde_retriever = HyDERetriever(
                db=self.db,
                embedding_fn=self.embedding_fn,
                hyde_generator=self.hyde_generator,
                top_k=top_k
            )
        return self._hyde_retriever

    def retrieve(self, question: str) -> List[Document]:
        """
        Classify question, route to appropriate strategy, and retrieve.

        Respects skip_rerank flag from route config:
        - For HyDE path: skips reranker if skip_rerank=True
        - For standard path: uses non-rerank pipeline variant if skip_rerank=True
        """
        # Step 1: Classify the question
        classification = self.classifier.classify(question)
        question_type = classification.question_type

        # Step 2: Get route configuration
        route = self.routes.get(question_type, self.routes["domain-relevant"])
        skip_rerank = getattr(route, 'skip_rerank', False)

        # Step 3: Retrieve based on route configuration
        if route.use_hyde:
            # Use HyDE for novel-generated questions
            initial_k = int(route.top_k * route.initial_k_factor)
            hyde_retriever = self._get_hyde_retriever(initial_k)
            docs = hyde_retriever.invoke(question)

            # Apply reranking unless skip_rerank is set
            if not skip_rerank:
                reranker = get_reranker(model_name=self._reranker_model)
                docs = reranker.rerank(question, docs, route.top_k)
            else:
                # Just truncate to top_k without reranking
                docs = docs[:route.top_k]
        else:
            # Use standard SimplePipeline (respects skip_rerank via _get_pipeline)
            pipeline = self._get_pipeline(route)
            docs = pipeline.retrieve(question)

        # Step 4: Apply table preference filter if needed
        if route.use_table_preference:
            docs = self.table_filter.filter_and_boost(
                docs, question_type, route.top_k
            )

        return docs

    def retrieve_with_metadata(
        self, question: str
    ) -> tuple[List[Document], Dict[str, Any]]:
        """
        Retrieve documents and return routing metadata for analysis.
        Useful for debugging and evaluation.
        """
        classification = self.classifier.classify(question)
        docs = self.retrieve(question)

        route = self.routes.get(classification.question_type, self.routes["domain-relevant"])

        metadata = {
            "question_type": classification.question_type,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning,
            "route_config": asdict(route),
            "num_docs": len(docs),
        }

        return docs, metadata


# =============================================================================
# Factory Function
# =============================================================================

def build_routed_pipeline(
    db,
    embedding_fn,
    classifier_model: str = None,
    hyde_model: str = None,
    routes: dict = None,
    reranker_model: str = None,
    use_rule_router: bool = False,
    domain: str = None,
) -> RoutedPipeline:
    """
    Build a RoutedPipeline that classifies questions and routes
    to optimal retrieval strategies.

    Args:
        db: ChromaDB instance
        embedding_fn: Embedding function for HyDE
        classifier_model: Model for question classification
        hyde_model: Model for HyDE generation
        routes: Custom route configurations (optional, overrides domain)
        reranker_model: Model for reranking
        use_rule_router: If True, use free RuleBasedClassifier instead of LLM
        domain: Domain for route selection ("finance", "legal", "medical").
                Uses domain-specific routes that may skip reranking.
                Ignored if routes is explicitly provided.

    Returns:
        RoutedPipeline instance
    """
    from src.config import ROUTES, DEFAULTS, get_routes_for_domain

    # Choose classifier: rule-based (free) or LLM-based (costs API calls)
    if use_rule_router:
        classifier = RuleBasedClassifier()
    else:
        classifier = get_classifier(model_name=classifier_model)

    hyde_gen = get_hyde_generator(model_name=hyde_model)
    table_filter = TablePreferenceFilter()

    # Determine routes: explicit > domain-specific > default
    if routes is not None:
        selected_routes = routes
    elif domain is not None:
        selected_routes = get_routes_for_domain(domain)
    else:
        selected_routes = ROUTES

    return RoutedPipeline(
        db=db,
        embedding_fn=embedding_fn,
        classifier=classifier,
        routes=selected_routes,
        hyde_generator=hyde_gen,
        table_filter=table_filter,
        reranker_model=reranker_model or DEFAULTS.reranker_model,
    )
