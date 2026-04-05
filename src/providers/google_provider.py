"""Google Gemini provider."""

from .base import LLMProvider, LLMResponse


class GoogleProvider(LLMProvider):
    """Provider for Google Gemini models."""

    def _create_client(self):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model_name)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        # Gemini combines system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Lower safety settings for medical/scientific content
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = self.client.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
            safety_settings=safety_settings,
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

        response_obj = LLMResponse(
            content=content,
            model=self.model_name,
            provider="google",
            usage=usage,
        )

        # Record usage in global tracker
        from .base import get_usage_tracker
        get_usage_tracker().record(usage)

        return response_obj

    @property
    def provider_name(self) -> str:
        return "google"
