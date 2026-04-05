"""Anthropic Claude provider."""

from .base import LLMProvider, LLMResponse


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
    ) -> LLMResponse:
        # Check cache first
        from .cache import get_cache
        cache = get_cache()
        cached = cache.get(self.model_name, system_prompt, user_prompt, temperature, max_tokens)
        if cached is not None:
            return cached

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
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

        response_obj = LLMResponse(
            content=content,
            model=self.model_name,
            provider="anthropic",
            usage=usage,
        )

        # Record usage in global tracker
        from .base import get_usage_tracker
        get_usage_tracker().record(usage)

        # Store in cache
        cache.put(self.model_name, system_prompt, user_prompt, temperature, max_tokens, response_obj)

        return response_obj

    @property
    def provider_name(self) -> str:
        return "anthropic"
