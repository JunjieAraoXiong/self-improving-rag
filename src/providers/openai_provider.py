"""OpenAI and OpenAI-compatible provider (Together, DeepSeek)."""

from typing import Optional
import openai

from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI and OpenAI-compatible APIs."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        provider_name_override: str = "openai",
    ):
        super().__init__(model_name, api_key)
        self.base_url = base_url
        self._provider_name = provider_name_override

    def _create_client(self):
        if self.base_url:
            return openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return openai.OpenAI(api_key=self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # GPT-5.x models require max_completion_tokens instead of max_tokens
        use_new_param = self.model_name.startswith("gpt-5") or "o1" in self.model_name or "o3" in self.model_name

        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        if use_new_param:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)

        content = ""
        if response and response.choices:
            content = response.choices[0].message.content or ""

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        response_obj = LLMResponse(
            content=content,
            model=self.model_name,
            provider=self._provider_name,
            usage=usage,
        )

        # Record usage in global tracker
        from .base import get_usage_tracker
        get_usage_tracker().record(usage)

        return response_obj

    @property
    def provider_name(self) -> str:
        return self._provider_name
