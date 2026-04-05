"""LLM response cache using SQLite.

Caches deterministic LLM calls (temperature=0) to avoid redundant API
calls during iterative development. Saves money and ensures exact
reproducibility of any previously computed result.

Usage:
    from src.providers.cache import LLMCache
    cache = LLMCache()  # creates .cache/llm_cache.db

    # Check cache before calling API
    cached = cache.get(model, system_prompt, user_prompt, temperature, max_tokens)
    if cached:
        return cached  # LLMResponse

    # After API call, store result
    cache.put(model, system_prompt, user_prompt, temperature, max_tokens, response)
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

from .base import LLMResponse


class LLMCache:
    """SQLite-backed cache for LLM responses."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / ".cache" / "llm_cache.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                content TEXT,
                model TEXT,
                provider TEXT,
                usage_json TEXT
            )
        """)
        self._conn.commit()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(model: str, system_prompt: str, user_prompt: str,
                  temperature: float, max_tokens: int) -> str:
        raw = f"{model}|{system_prompt}|{user_prompt}|{temperature}|{max_tokens}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model: str, system_prompt: str, user_prompt: str,
            temperature: float, max_tokens: int) -> Optional[LLMResponse]:
        # Only cache deterministic calls
        if temperature > 0:
            return None

        key = self._make_key(model, system_prompt, user_prompt, temperature, max_tokens)
        row = self._conn.execute(
            "SELECT content, model, provider, usage_json FROM cache WHERE key = ?", (key,)
        ).fetchone()

        if row:
            self._hits += 1
            usage = json.loads(row[3]) if row[3] else None
            return LLMResponse(content=row[0], model=row[1], provider=row[2], usage=usage)

        self._misses += 1
        return None

    def put(self, model: str, system_prompt: str, user_prompt: str,
            temperature: float, max_tokens: int, response: LLMResponse):
        if temperature > 0:
            return

        key = self._make_key(model, system_prompt, user_prompt, temperature, max_tokens)
        usage_json = json.dumps(response.usage) if response.usage else None
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, content, model, provider, usage_json) VALUES (?, ?, ?, ?, ?)",
            (key, response.content, response.model, response.provider, usage_json)
        )
        self._conn.commit()

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


# Global cache instance
_cache: Optional[LLMCache] = None


def get_cache() -> LLMCache:
    """Get or create global cache."""
    global _cache
    if _cache is None:
        _cache = LLMCache()
    return _cache
