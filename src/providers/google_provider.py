"""Google Gemini provider."""

from typing import Optional

from .base import LLMProvider, LLMResponse, package_version


class GoogleProvider(LLMProvider):
    """Provider for Google Gemini models."""

    def _create_client(self):
        from google import genai

        return genai.Client(api_key=self.api_key)

    @property
    def supports_seed(self) -> bool:
        """The current Google Gen AI SDK accepts best-effort request seeds."""

        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        generation_config = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            generation_config["seed"] = seed

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=generation_config,
        )

        content = ""
        if response and response.text:
            content = response.text

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        metadata = self.request_metadata(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            seed_applied=seed is not None,
            response_model=getattr(response, "model_version", None),
            request_id=getattr(response, "response_id", None),
            sdk="google-genai",
            sdk_version=package_version("google-genai"),
        )
        response_obj = LLMResponse(
            content=content,
            model=self.model_name,
            provider="google",
            usage=usage,
            metadata=metadata,
        )

        # Record usage in global tracker
        from .base import get_usage_tracker
        get_usage_tracker().record(
            usage,
            model=self.model_name,
            provider="google",
            request_metadata=metadata,
        )

        return response_obj

    @property
    def provider_name(self) -> str:
        return "google"
