"""Anthropic Claude provider."""

from typing import Optional

from .base import LLMProvider, LLMResponse, package_version


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models."""

    def _create_client(self):
        import anthropic
        return anthropic.Anthropic(api_key=self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        content = ""
        if response and response.content:
            content = response.content[0].text or ""

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        metadata = self.request_metadata(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            seed_applied=False,
            response_model=getattr(response, "model", None),
            request_id=(
                getattr(response, "_request_id", None)
                or getattr(response, "id", None)
            ),
            sdk="anthropic",
            sdk_version=package_version("anthropic"),
        )
        response_obj = LLMResponse(
            content=content,
            model=self.model_name,
            provider="anthropic",
            usage=usage,
            metadata=metadata,
        )

        # Record usage in global tracker
        from .base import get_usage_tracker
        get_usage_tracker().record(
            usage,
            model=self.model_name,
            provider="anthropic",
            request_metadata=metadata,
        )

        return response_obj

    @property
    def provider_name(self) -> str:
        return "anthropic"
