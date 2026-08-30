"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from threading import Lock
from typing import Any, Optional


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    usage: Optional[dict] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def package_version(distribution: str) -> Optional[str]:
    """Return an installed SDK version without making provider calls fail."""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


class UsageTracker:
    """Global token/cost accumulator across all provider calls."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self._records: list[dict] = []
        self._lock = Lock()

    def record(
        self,
        usage: Optional[dict],
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        request_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record one provider call with model-level attribution."""

        usage = usage or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        record = {
            "model": model,
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if request_metadata:
            record["request_metadata"] = request_metadata.copy()
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_calls += 1
            self._records.append(record)

    def cursor(self) -> int:
        """Return a stable cursor for later per-question usage deltas."""

        with self._lock:
            return len(self._records)

    def records_since(self, cursor: int) -> list[dict]:
        """Return copies of records added after ``cursor``."""

        with self._lock:
            if cursor < 0 or cursor > len(self._records):
                raise ValueError("usage cursor is out of range")
            return [record.copy() for record in self._records[cursor:]]

    def reset(self):
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_calls = 0
            self._records = []

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_calls": self.total_calls,
                "records": [record.copy() for record in self._records],
            }


# Global usage tracker -- all providers record here
_usage_tracker = UsageTracker()


def get_usage_tracker() -> UsageTracker:
    """Get the global usage tracker."""
    return _usage_tracker


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Lazy-load the client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @abstractmethod
    def _create_client(self):
        """Create the provider-specific client."""
        pass

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a response, applying ``seed`` only when supported."""
        pass

    @property
    def supports_seed(self) -> bool:
        """Whether this provider can forward a best-effort request seed."""

        return False

    def request_metadata(
        self,
        *,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        seed_applied: bool,
        response_model: Optional[str] = None,
        request_id: Optional[str] = None,
        system_fingerprint: Optional[str] = None,
        sdk: Optional[str] = None,
        sdk_version: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build JSON-safe, non-prompt metadata for one remote request."""

        metadata: dict[str, Any] = {
            "requested_model": self.model_name,
            "response_model": response_model or self.model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed_requested": seed,
            "seed_applied": seed_applied,
            "provider_supports_seed": self.supports_seed,
        }
        optional_values = {
            "request_id": request_id,
            "system_fingerprint": system_fingerprint,
            "sdk": sdk,
            "sdk_version": sdk_version,
        }
        metadata.update(
            {key: value for key, value in optional_values.items() if value is not None}
        )
        return metadata

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
