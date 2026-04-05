"""Central configuration for RAG system."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Embedding Configuration
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    name: str
    model_id: str
    provider: str  # "local", "openai", "azure", "cohere"
    dimension: int
    description: str


class CohereAsymmetricEmbeddings:
    """Cohere embeddings wrapper that uses different input_type for queries vs documents.

    This implements the dsRAG insight: Cohere's embed-v3 performs significantly better
    when using:
    - input_type="search_document" for embedding documents (indexing)
    - input_type="search_query" for embedding queries (retrieval)

    This asymmetric embedding approach improves retrieval because:
    1. Documents are statements of fact
    2. Queries are questions or information needs
    3. The model learns different representations for each
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy-load the Cohere client."""
        if self._client is None:
            import cohere
            self._client = cohere.ClientV2(api_key=self.api_key)
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using input_type='search_document'.

        Cohere has a limit of 96 texts per request, so we batch accordingly.
        """
        if not texts:
            return []

        client = self._get_client()
        all_embeddings = []
        batch_size = 96  # Cohere's maximum

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document",
                embedding_types=["float"],
            )
            all_embeddings.extend([list(emb) for emb in response.embeddings.float_])

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using input_type='search_query'."""
        client = self._get_client()
        response = client.embed(
            texts=[text],
            model=self.model,
            input_type="search_query",
            embedding_types=["float"],
        )
        return list(response.embeddings.float_[0])


class VoyageAsymmetricEmbeddings:
    """Voyage AI embeddings wrapper with asymmetric input types.

    Voyage AI (like Cohere) supports distinct input_type for queries vs documents:
    - input_type="document" for embedding documents (indexing)
    - input_type="query" for embedding queries (retrieval)

    Voyage AI advantages over Cohere:
    - +20% average improvement across 100 retrieval datasets
    - 32K context window (vs Cohere's 512 tokens)
    - Domain-specific models (voyage-finance-2, voyage-law-2)
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy-load the Voyage client."""
        if self._client is None:
            import voyageai
            self._client = voyageai.Client(api_key=self.api_key)
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using input_type='document'."""
        if not texts:
            return []
        client = self._get_client()
        # Voyage recommends batching - max 128 texts per call
        all_embeddings = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = client.embed(
                batch,
                model=self.model,
                input_type="document",
            )
            all_embeddings.extend(result.embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using input_type='query'."""
        client = self._get_client()
        result = client.embed(
            [text],
            model=self.model,
            input_type="query",
        )
        return result.embeddings[0]


# Embedding registry - local models are FREE
EMBEDDINGS: Dict[str, EmbeddingConfig] = {
    # FREE local models (recommended)
    "bge-large": EmbeddingConfig(
        name="bge-large",
        model_id="BAAI/bge-large-en-v1.5",
        provider="local",
        dimension=1024,
        description="Best free option - comparable to OpenAI",
    ),
    "bge-base": EmbeddingConfig(
        name="bge-base",
        model_id="BAAI/bge-base-en-v1.5",
        provider="local",
        dimension=768,
        description="Good quality, smaller/faster",
    ),
    "gte-large": EmbeddingConfig(
        name="gte-large",
        model_id="Alibaba-NLP/gte-large-en-v1.5",
        provider="local",
        dimension=1024,
        description="Excellent quality, slightly slower",
    ),
    "nomic": EmbeddingConfig(
        name="nomic",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        provider="local",
        dimension=768,
        description="Good quality, fast",
    ),
    # Paid OpenAI models (expensive - avoid)
    "openai-large": EmbeddingConfig(
        name="openai-large",
        model_id="text-embedding-3-large",
        provider="openai",
        dimension=3072,
        description="OpenAI - $0.13/1M tokens (PAID)",
    ),
    "openai-small": EmbeddingConfig(
        name="openai-small",
        model_id="text-embedding-3-small",
        provider="openai",
        dimension=1536,
        description="OpenAI - $0.02/1M tokens (PAID)",
    ),
    # Azure OpenAI models (use with Azure credits)
    "azure-small": EmbeddingConfig(
        name="azure-small",
        model_id="text-embedding-3-small",
        provider="azure",
        dimension=1536,
        description="Azure OpenAI - use with Azure credits",
    ),
    "azure-large": EmbeddingConfig(
        name="azure-large",
        model_id="text-embedding-3-large",
        provider="azure",
        dimension=3072,
        description="Azure OpenAI - use with Azure credits",
    ),
    # Cohere embeddings - dsRAG benchmark SOTA (PAID)
    # Key feature: uses distinct input_type for queries vs documents
    "cohere-v3": EmbeddingConfig(
        name="cohere-v3",
        model_id="embed-english-v3.0",
        provider="cohere",
        dimension=1024,
        description="Cohere embed-v3 - SOTA retrieval (~$0.10/1M tokens)",
    ),
    "cohere-v3-multilingual": EmbeddingConfig(
        name="cohere-v3-multilingual",
        model_id="embed-multilingual-v3.0",
        provider="cohere",
        dimension=1024,
        description="Cohere embed-v3 multilingual - for non-English docs",
    ),
    # Voyage AI embeddings - SOTA retrieval, +20% over Cohere (PAID)
    # 32K context window, supports input_type for query/document
    "voyage-3-large": EmbeddingConfig(
        name="voyage-3-large",
        model_id="voyage-3-large",
        provider="voyage",
        dimension=1024,
        description="Voyage AI - SOTA retrieval, +20% over Cohere (~$0.06/1M tokens)",
    ),
    "voyage-finance-2": EmbeddingConfig(
        name="voyage-finance-2",
        model_id="voyage-finance-2",
        provider="voyage",
        dimension=1024,
        description="Voyage AI - Finance domain optimized (~$0.12/1M tokens)",
    ),
    "voyage-law-2": EmbeddingConfig(
        name="voyage-law-2",
        model_id="voyage-law-2",
        provider="voyage",
        dimension=1024,
        description="Voyage AI - Legal domain optimized (~$0.12/1M tokens)",
    ),
}


def get_embedding_model(
    embedding_name: str = "bge-large",
    input_type: Optional[str] = None
):
    """Get an embedding model instance (lazy loaded).

    Args:
        embedding_name: Name of embedding model from EMBEDDINGS registry
        input_type: For Cohere embeddings, specifies "search_document" or "search_query".
                   - "search_document": Use when embedding documents for storage
                   - "search_query": Use when embedding queries for retrieval
                   If None, defaults to "search_document" for Cohere.

    Returns:
        LangChain embedding model instance
    """
    if embedding_name not in EMBEDDINGS:
        raise ValueError(f"Unknown embedding: {embedding_name}. Available: {list(EMBEDDINGS.keys())}")

    config = EMBEDDINGS[embedding_name]

    if config.provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        import torch
        # Device selection: CUDA > MPS (Apple Metal) > CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Using device: {device} for embeddings")
        return HuggingFaceEmbeddings(
            model_name=config.model_id,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif config.provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.model_id)
    elif config.provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        # Requires: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_deployment = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", config.model_id)
        if not azure_endpoint or not azure_api_key:
            raise ValueError(
                "Azure embeddings require AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars"
            )
        return AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            azure_deployment=azure_deployment,
            model=config.model_id,
        )
    elif config.provider == "cohere":
        # Cohere uses distinct input_type for queries vs documents
        # This is the key dsRAG insight for SOTA retrieval
        api_key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        if not api_key:
            raise ValueError("Cohere embeddings require COHERE_API_KEY or CO_API_KEY env var")

        if input_type is not None:
            # Explicit input_type requested - create fixed-type embeddings
            # Use this for ingestion (search_document) or testing
            from langchain_cohere import CohereEmbeddings
            if input_type not in ("search_document", "search_query"):
                raise ValueError(
                    f"Cohere input_type must be 'search_document' or 'search_query', got: {input_type}"
                )
            print(f"Using Cohere embeddings with fixed input_type={input_type}")
            return CohereEmbeddings(
                model=config.model_id,
                cohere_api_key=api_key,
                user_agent="dsrag-style-rag",
            )
        else:
            # No explicit input_type - return wrapper that uses correct type automatically
            # embed_documents() uses "search_document", embed_query() uses "search_query"
            print("Using Cohere embeddings with automatic input_type switching")
            return CohereAsymmetricEmbeddings(
                model=config.model_id,
                api_key=api_key,
            )
    elif config.provider == "voyage":
        # Voyage AI - SOTA retrieval embeddings (+20% over Cohere)
        # Supports asymmetric input_type (document vs query)
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("Voyage embeddings require VOYAGE_API_KEY env var")

        if input_type is not None:
            # Explicit input_type requested - use LangChain wrapper
            from langchain_voyageai import VoyageAIEmbeddings
            if input_type not in ("document", "query"):
                raise ValueError(
                    f"Voyage input_type must be 'document' or 'query', got: {input_type}"
                )
            print(f"Using Voyage embeddings with fixed input_type={input_type}")
            return VoyageAIEmbeddings(
                model=config.model_id,
                voyage_api_key=api_key,
            )
        else:
            # No explicit input_type - return wrapper that uses correct type automatically
            print("Using Voyage embeddings with automatic input_type switching")
            return VoyageAsymmetricEmbeddings(
                model=config.model_id,
                api_key=api_key,
            )
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def is_cohere_embedding(embedding_name: str) -> bool:
    """Check if the embedding model is a Cohere model."""
    if embedding_name not in EMBEDDINGS:
        return False
    return EMBEDDINGS[embedding_name].provider == "cohere"


def is_voyage_embedding(embedding_name: str) -> bool:
    """Check if the embedding model is a Voyage AI model."""
    if embedding_name not in EMBEDDINGS:
        return False
    return EMBEDDINGS[embedding_name].provider == "voyage"


def is_asymmetric_embedding(embedding_name: str) -> bool:
    """Check if the embedding model uses asymmetric input types (query vs document).

    Both Cohere and Voyage AI use different representations for queries vs documents,
    which is the key insight from dsRAG for achieving SOTA retrieval.
    """
    if embedding_name not in EMBEDDINGS:
        return False
    return EMBEDDINGS[embedding_name].provider in ("cohere", "voyage")


# =============================================================================
# LLM Provider Configuration
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    base_url: Optional[str]
    api_key_env: str
    models: List[str]

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


# Provider registry - add new providers here
PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        models=["gpt-5.2", "gpt-5.2-mini", "gpt-4o", "gpt-4o-mini"],
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        base_url=None,
        api_key_env="ANTHROPIC_API_KEY",
        models=[
            "claude-sonnet-4-5-20250514",
            "claude-opus-4-5-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-haiku-20240307",  # Cheapest - $0.25/$1.25 per 1M tokens
        ],
    ),
    "google": ProviderConfig(
        name="google",
        base_url=None,
        api_key_env="GOOGLE_API_KEY",
        models=["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.0-flash"],
    ),
    "together": ProviderConfig(
        name="together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        models=[
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",  # Serverless (Qwen3-235B needs dedicated)
            "moonshotai/Kimi-K2-Instruct",
        ],
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        models=["deepseek-chat", "deepseek-reasoner"],
    ),
    "local-vllm": ProviderConfig(
        name="local-vllm",
        base_url="http://localhost:8000/v1",
        api_key_env="EMPTY_KEY",  # vLLM doesn't need a real key, but provider might check existence
        models=["meta-llama/Meta-Llama-3.1-70B-Instruct"],
    ),
    "xai": ProviderConfig(
        name="xai",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        models=["grok-3", "grok-3-mini", "grok-4-1-fast-non-reasoning"],
    ),
}


def get_provider_for_model(model_name: str) -> str:
    """Determine which provider to use based on model name."""
    model_lower = model_name.lower()
    if model_lower.startswith("gpt-"):
        return "openai"
    elif model_lower.startswith("claude-"):
        return "anthropic"
    elif model_lower.startswith("gemini-"):
        return "google"
    elif model_lower.startswith("grok-"):
        return "xai"
    elif "deepseek" in model_lower and not model_lower.startswith("deepseek-ai/"):
        return "deepseek"
    elif "meta-llama" in model_lower:
        if "turbo" in model_lower:
            return "together"  # Turbo models use Together API
        return "local-vllm"
    elif "qwen" in model_lower or "kimi" in model_lower or "moonshotai" in model_lower:
        return "together"
    return "together"


def get_provider_config(model_name: str) -> ProviderConfig:
    """Get provider configuration for a model."""
    provider_name = get_provider_for_model(model_name)
    return PROVIDERS[provider_name]


# =============================================================================
# Reranker Configuration
# =============================================================================

@dataclass
class RerankerConfig:
    """Configuration for reranker models."""
    name: str
    description: str


RERANKERS: Dict[str, RerankerConfig] = {
    # Local models (free)
    "BAAI/bge-reranker-large": RerankerConfig(
        name="BAAI/bge-reranker-large",
        description="High quality, slower (FREE)",
    ),
    "BAAI/bge-reranker-base": RerankerConfig(
        name="BAAI/bge-reranker-base",
        description="Good quality, medium speed (FREE)",
    ),
    "cross-encoder/ms-marco-MiniLM-L-6-v2": RerankerConfig(
        name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Fast, lower quality (FREE)",
    ),
    # API models (paid but SOTA quality)
    "cohere": RerankerConfig(
        name="cohere",
        description="Cohere rerank-v3 - SOTA quality, +28% NDCG ($1/1K queries)",
    ),
}


# =============================================================================
# Router Configuration
# =============================================================================

@dataclass
class RouteConfig:
    """Configuration for a specific question type route."""
    pipeline_id: str
    top_k: int
    initial_k_factor: float
    use_hyde: bool = False
    use_table_preference: bool = False
    table_quota_ratio: float = 0.6
    skip_rerank: bool = False  # Override pipeline reranking (e.g., for legal domain)


ROUTES: Dict[str, RouteConfig] = {
    "metrics-generated": RouteConfig(
        pipeline_id="hybrid_filter_rerank",
        top_k=10,  # Was 5 - increased to retrieve more table chunks
        initial_k_factor=6.0,  # Was 4.0 - retrieve 60 docs initially
        use_hyde=False,
        use_table_preference=True,
        table_quota_ratio=0.9,  # Was 0.6 - prioritize 90% table chunks
    ),
    "domain-relevant": RouteConfig(
        pipeline_id="hybrid_filter_rerank",
        top_k=5,
        initial_k_factor=3.0,
        use_hyde=False,
        use_table_preference=False,
    ),
    "novel-generated": RouteConfig(
        pipeline_id="hybrid_filter_rerank",
        top_k=8,
        initial_k_factor=3.0,
        use_hyde=True,
        use_table_preference=False,
    ),
}


# Domain-specific route configurations
# LegalBench-RAG finding: general-purpose rerankers hurt legal text retrieval
LEGAL_ROUTES: Dict[str, RouteConfig] = {
    "metrics-generated": RouteConfig(
        pipeline_id="hybrid_filter",  # No rerank in pipeline_id
        top_k=10,
        initial_k_factor=4.0,
        skip_rerank=True,  # Per LegalBench-RAG findings
        use_table_preference=True,
        table_quota_ratio=0.8,  # Legal contracts have many structured clauses
    ),
    "domain-relevant": RouteConfig(
        pipeline_id="hybrid_filter",
        top_k=5,
        initial_k_factor=3.0,
        skip_rerank=True,
    ),
    "novel-generated": RouteConfig(
        pipeline_id="hybrid_filter",
        top_k=8,
        initial_k_factor=3.0,
        use_hyde=True,
        skip_rerank=True,  # Skip rerank even with HyDE for legal
    ),
}

# Medical routes - reranking may help with biomedical terminology
MEDICAL_ROUTES: Dict[str, RouteConfig] = {
    "metrics-generated": RouteConfig(
        pipeline_id="hybrid_filter_rerank",
        top_k=10,
        initial_k_factor=4.0,
        use_table_preference=True,
        table_quota_ratio=0.7,
    ),
    "domain-relevant": RouteConfig(
        pipeline_id="hybrid_filter_rerank",
        top_k=5,
        initial_k_factor=3.0,
    ),
    "novel-generated": RouteConfig(
        pipeline_id="hybrid_filter_rerank",
        top_k=8,
        initial_k_factor=3.0,
        use_hyde=True,
    ),
}


# Registry mapping domain names to their route configurations
DOMAIN_ROUTES: Dict[str, Dict[str, RouteConfig]] = {
    "finance": ROUTES,  # Default with reranking
    "legal": LEGAL_ROUTES,  # Skip reranking per LegalBench-RAG
    "medical": MEDICAL_ROUTES,  # Keep reranking for biomedical text
}


def get_routes_for_domain(domain: str) -> Dict[str, RouteConfig]:
    """Get route configuration for a specific domain.

    Args:
        domain: One of "finance", "legal", "medical"

    Returns:
        Route configuration dictionary for the domain
    """
    if domain not in DOMAIN_ROUTES:
        raise ValueError(f"Unknown domain '{domain}'. Available: {list(DOMAIN_ROUTES.keys())}")
    return DOMAIN_ROUTES[domain]


# =============================================================================
# Default Settings
# =============================================================================

@dataclass
class Defaults:
    """Default values for the RAG system."""
    # Model defaults
    llm_model: str = "gpt-4o-mini"  # Cost-effective, works reliably
    embedding_model: str = "bge-large"  # FREE local embedding (was text-embedding-3-large)
    reranker_model: str = "BAAI/bge-reranker-large"  # Also FREE local
    judge_model: str = "gpt-4o-mini"  # Was claude-sonnet-4-5-20250514 (404 error)

    # Retrieval defaults
    top_k: int = 5
    initial_k_factor: float = 3.0
    pipeline_id: str = "hybrid_filter_rerank"
    ensemble_weights: tuple = (0.5, 0.5)  # (BM25, semantic) - balanced for entity + semantic matching
    rerank_threshold: float = 0.0  # Minimum reranker score (0.0 = no filtering, try 0.1-0.3)

    # Generation defaults
    temperature: float = 0.0
    max_tokens: int = 512

    # Router defaults
    router_classifier_model: str = "gpt-4o-mini"
    router_hyde_model: str = "gpt-4o-mini"

    # Paths (relative to project root)
    chroma_path: str = "chroma_docling"
    output_dir: str = "bulk_runs"


DEFAULTS = Defaults()


# =============================================================================
# Model Name Abbreviations (for filenames)
# =============================================================================

MODEL_ABBREVS: Dict[str, str] = {
    # Claude
    "claude-sonnet-4-5": "claude45-sonnet",
    "claude-opus-4-5": "claude45-opus",
    "claude-sonnet-4": "claude4-sonnet",
    "claude-3-haiku": "claude3-haiku",
    # GPT
    "gpt-5.2-mini": "gpt52-mini",
    "gpt-5.2": "gpt52",
    "gpt-4o-mini": "gpt4o-mini",
    "gpt-4o": "gpt4o",
    # Gemini
    "gemini-3-flash": "gemini3-flash",
    "gemini-3-pro": "gemini3-pro",
    "gemini-2": "gemini2-flash",
    # Llama
    "llama-4": "llama4",
    "llama-3.1-70b": "llama31-70b",
    "llama-3.1-8b": "llama31-8b",
    # DeepSeek
    "deepseek-v3": "deepseek-v3",
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-r1",
    # Qwen
    "qwen3-235b": "qwen3-235b",
    "qwen2.5-72b": "qwen25-72b",
    "qwen3-coder": "qwen3-coder",
    # Kimi
    "kimi-k2": "kimi-k2",
    # Grok (xAI)
    "grok-3-mini": "grok3-mini",
    "grok-3": "grok3",
    "grok-4-1-fast": "grok4-fast",
}


def get_model_abbrev(model_name: str) -> str:
    """Get abbreviated model name for filenames."""
    model_lower = model_name.lower()
    for pattern, abbrev in MODEL_ABBREVS.items():
        if pattern in model_lower:
            return abbrev
    # Fallback: use last part of model name
    return model_name.split("/")[-1][:20]


# =============================================================================
# Cost Tracking (USD per 1K tokens)
# =============================================================================

COST_PER_1K_TOKENS: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-5.2": {"input": 0.003, "output": 0.012},
    "gpt-5.2-mini": {"input": 0.0004, "output": 0.0016},
    # Anthropic
    "claude-sonnet-4-5-20250514": {"input": 0.003, "output": 0.015},
    "claude-opus-4-5-20250514": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    # Google
    "gemini-3-pro-preview": {"input": 0.00125, "output": 0.005},
    "gemini-3-flash-preview": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    # Together / Open-source
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {"input": 0.00027, "output": 0.00085},
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {"input": 0.00088, "output": 0.00088},
    "deepseek-ai/DeepSeek-V3": {"input": 0.0005, "output": 0.0005},
    "Qwen/Qwen2.5-72B-Instruct-Turbo": {"input": 0.0012, "output": 0.0012},
    "moonshotai/Kimi-K2-Instruct": {"input": 0.0006, "output": 0.0006},
    # DeepSeek direct
    "deepseek-chat": {"input": 0.00027, "output": 0.0011},
    "deepseek-reasoner": {"input": 0.00055, "output": 0.0022},
    # xAI
    "grok-3": {"input": 0.003, "output": 0.015},
    "grok-3-mini": {"input": 0.0003, "output": 0.0005},
    "grok-4-1-fast-non-reasoning": {"input": 0.003, "output": 0.015},
}


def calculate_cost(model_name: str, usage: dict) -> float:
    """Calculate USD cost from token usage.

    Args:
        model_name: Model name (matched against COST_PER_1K_TOKENS keys)
        usage: Dict with 'prompt_tokens' and 'completion_tokens'

    Returns:
        Cost in USD
    """
    # Find matching cost entry (partial match for model name prefixes)
    rates = COST_PER_1K_TOKENS.get(model_name)
    if rates is None:
        # Try partial match
        model_lower = model_name.lower()
        for key, val in COST_PER_1K_TOKENS.items():
            if key.lower() in model_lower or model_lower in key.lower():
                rates = val
                break
    if rates is None:
        return 0.0

    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    input_cost = (prompt_tokens / 1000) * rates["input"]
    output_cost = (completion_tokens / 1000) * rates["output"]
    return input_cost + output_cost


# =============================================================================
# Pipeline Configuration
# =============================================================================

PIPELINES = ["semantic", "hybrid", "hybrid_filter", "hybrid_filter_rerank", "routed"]


def get_pipeline_flags(pipeline_id: str) -> tuple:
    """Return (use_hybrid, use_filter, use_rerank) for a pipeline id."""
    mapping = {
        "semantic": (False, False, False),
        "hybrid": (True, False, False),
        "hybrid_filter": (True, True, False),
        "hybrid_filter_rerank": (True, True, True),
    }
    if pipeline_id not in mapping:
        raise ValueError(f"Unknown pipeline_id '{pipeline_id}'. Available: {PIPELINES}")
    return mapping[pipeline_id]
