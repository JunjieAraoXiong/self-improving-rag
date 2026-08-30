"""Offline regression tests for provider request reproducibility metadata."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from src.agents.orchestrator import _summarize_usage_records
from src.agents.reasoning_agent import ReasoningAgent
from src.bulk_testing import BulkTestConfig
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import get_usage_tracker
from src.providers.google_provider import GoogleProvider
from src.providers.openai_provider import OpenAIProvider


@pytest.fixture(autouse=True)
def _reset_usage_tracker():
    tracker = get_usage_tracker()
    tracker.reset()
    yield
    tracker.reset()


def _openai_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=3,
            total_tokens=14,
        ),
        model="gpt-4o-mini-2024-07-18",
        id="chatcmpl-test",
        _request_id="req-test",
        system_fingerprint="fp-test",
    )


def test_openai_forwards_supported_seed_and_records_response_identity():
    create = Mock(return_value=_openai_response())
    provider = OpenAIProvider("gpt-4o-mini", "unused")
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    response = provider.generate(
        "system",
        "user",
        max_tokens=123,
        temperature=0.25,
        seed=17,
    )

    request = create.call_args.kwargs
    assert request["temperature"] == 0.25
    assert request["max_tokens"] == 123
    assert request["seed"] == 17
    assert response.metadata == {
        "requested_model": "gpt-4o-mini",
        "response_model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.25,
        "max_tokens": 123,
        "seed_requested": 17,
        "seed_applied": True,
        "provider_supports_seed": True,
        "request_id": "req-test",
        "system_fingerprint": "fp-test",
        "sdk": "openai",
        "sdk_version": response.metadata["sdk_version"],
    }

    records = get_usage_tracker().records_since(0)
    summary = _summarize_usage_records(records, fallback_model="unused")
    assert summary["by_model"]["gpt-4o-mini"]["requests"] == [response.metadata]


def test_openai_compatible_provider_does_not_guess_seed_support():
    create = Mock(return_value=_openai_response())
    provider = OpenAIProvider(
        "deepseek-chat",
        "unused",
        provider_name_override="deepseek",
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    response = provider.generate("system", "user", seed=23)

    assert "seed" not in create.call_args.kwargs
    assert response.metadata["seed_requested"] == 23
    assert response.metadata["seed_applied"] is False
    assert response.metadata["provider_supports_seed"] is False


def test_anthropic_forwards_temperature_and_records_unsupported_seed():
    create = Mock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(text="answer")],
            usage=SimpleNamespace(input_tokens=7, output_tokens=2),
            model="claude-sonnet-4-5-20250514",
            id="msg-test",
            _request_id="req-anthropic",
        )
    )
    provider = AnthropicProvider("claude-sonnet-4-5-20250514", "unused")
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))

    response = provider.generate(
        "system",
        "user",
        max_tokens=321,
        temperature=0.4,
        seed=29,
    )

    request = create.call_args.kwargs
    assert request["temperature"] == 0.4
    assert request["max_tokens"] == 321
    assert "seed" not in request
    assert response.metadata["response_model"] == "claude-sonnet-4-5-20250514"
    assert response.metadata["request_id"] == "req-anthropic"
    assert response.metadata["seed_requested"] == 29
    assert response.metadata["seed_applied"] is False
    assert response.metadata["provider_supports_seed"] is False


def test_google_genai_forwards_seed_and_records_resolved_model():
    generate_content = Mock(
        return_value=SimpleNamespace(
            text="answer",
            usage_metadata=SimpleNamespace(
                prompt_token_count=5,
                candidates_token_count=2,
                total_token_count=7,
            ),
            model_version="gemini-2.0-flash-001",
            response_id="gemini-response-test",
        )
    )
    provider = GoogleProvider("gemini-2.0-flash", "unused")
    provider._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )

    response = provider.generate(
        "system",
        "user",
        max_tokens=222,
        temperature=0.3,
        seed=43,
    )

    request = generate_content.call_args.kwargs
    assert request["model"] == "gemini-2.0-flash"
    assert request["contents"] == "user"
    assert request["config"] == {
        "system_instruction": "system",
        "max_output_tokens": 222,
        "temperature": 0.3,
        "seed": 43,
    }
    assert response.metadata["response_model"] == "gemini-2.0-flash-001"
    assert response.metadata["request_id"] == "gemini-response-test"
    assert response.metadata["seed_requested"] == 43
    assert response.metadata["seed_applied"] is True
    assert response.metadata["provider_supports_seed"] is True
    assert response.metadata["sdk"] == "google-genai"


def test_bulk_config_distinguishes_local_and_generation_seed_scope():
    config = BulkTestConfig(
        dataset_name="financebench",
        model_name="gpt-4o-mini",
        seed=31,
        generation_seed=37,
        temperature=0.1,
        max_tokens=456,
        timestamp="fixed",
    )

    metadata = config.reproducibility_metadata()

    assert metadata == {
        "local_seed": 31,
        "local_seed_scope": "python_random_and_numpy_only",
        "generation_seed_requested": 37,
        "generation_seed_scope": "provider_best_effort",
        "hosted_generation_deterministic": False,
        "requested_model": "gpt-4o-mini",
        "provider": "openai",
        "temperature": 0.1,
        "max_tokens": 456,
    }


def test_agentic_reasoning_forwards_only_explicit_generation_seed():
    class RecordingProvider:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(content="grounded answer")

    provider = RecordingProvider()
    agent = ReasoningAgent(generation_seed=41)
    agent._provider = provider

    agent.decide(
        {
            "question": "What was revenue?",
            "documents": [Document(page_content="Revenue was $10 million")],
        }
    )

    assert provider.kwargs["seed"] == 41
