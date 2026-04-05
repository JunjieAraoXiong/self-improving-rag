"""Factory for creating LLM providers."""

from typing import Dict, Optional
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider

# Import config - uses relative import to avoid circular deps
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_provider_config, get_provider_for_model


# Cache providers to avoid recreating clients
_provider_cache: Dict[str, LLMProvider] = {}


def get_provider(model_name: str, use_cache: bool = True) -> LLMProvider:
    """
    Get an LLM provider for the given model.

    Args:
        model_name: The model name (e.g., "claude-sonnet-4-5-20250514", "gpt-5.2")
        use_cache: Whether to cache and reuse providers

    Returns:
        LLMProvider instance configured for the model
    """
    cache_key = model_name

    if use_cache and cache_key in _provider_cache:
        return _provider_cache[cache_key]

    provider_config = get_provider_config(model_name)
    provider_name = provider_config.name
    api_key = provider_config.api_key

    # local-vllm doesn't need an API key
    if not api_key and provider_name != "local-vllm":
        raise ValueError(
            f"API key not found for provider '{provider_name}'. "
            f"Set {provider_config.api_key_env} environment variable."
        )

    # Create appropriate provider
    if provider_name == "anthropic":
        provider = AnthropicProvider(model_name=model_name, api_key=api_key)
    elif provider_name == "google":
        provider = GoogleProvider(model_name=model_name, api_key=api_key)
    elif provider_name in ("openai", "together", "deepseek", "local-vllm", "xai"):
        # local-vllm uses OpenAI-compatible API
        provider = OpenAIProvider(
            model_name=model_name,
            api_key=api_key or "not-needed",  # vLLM doesn't require real key
            base_url=provider_config.base_url,
            provider_name_override=provider_name,
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    if use_cache:
        _provider_cache[cache_key] = provider

    return provider


def clear_provider_cache():
    """Clear the provider cache."""
    _provider_cache.clear()
