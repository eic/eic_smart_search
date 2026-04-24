from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from typing import Any, Protocol

from app.core.config import Settings

logger = logging.getLogger(__name__)


REWRITE_SYSTEM_PROMPT = (
    "You rewrite a search query for a hybrid (lexical + semantic) retriever. "
    "Produce {max_variants} alternative phrasings that are semantically equivalent "
    "but use different vocabulary (synonyms, different grammatical form, "
    "expanded abbreviations, translated-to-English if applicable). "
    "Rules: one variant per line. No numbering. No preamble. No explanations. "
    "Never invent topics not implied by the query. If the query is already highly "
    "specific, output fewer variants. Each variant must be under 15 words."
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class QueryRewriter(Protocol):
    name: str

    def rewrite(self, query: str) -> tuple[list[str], dict]: ...


class NullQueryRewriter:
    name = "none"

    def rewrite(self, query: str) -> tuple[list[str], dict]:
        return [], {"provider": self.name, "variants": 0, "skipped": True}


class OpenAIQueryRewriter:
    """Paraphrase a search query using an OpenAI chat model.

    Output is a list of variant queries (excluding the original). Empty list
    means the rewriter chose not to expand (or the query was too long).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        timeout: float = 15.0,
        max_variants: int = 2,
        trigger_max_words: int = 8,
        max_output_tokens: int = 200,
        temperature: float | None = None,
        cache_size: int = 512,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            timeout=timeout,
        )
        self.name = f"openai:{model}"
        self.model = model
        self.max_variants = max(1, max_variants)
        self.trigger_max_words = max(1, trigger_max_words)
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._cache: OrderedDict[str, list[str]] = OrderedDict()
        self._cache_size = max(1, cache_size)

    def rewrite(self, query: str) -> tuple[list[str], dict]:
        clean = query.strip()
        if not clean:
            return [], {"provider": self.name, "variants": 0, "skipped": True, "reason": "empty"}

        word_count = len(_WORD_RE.findall(clean))
        if word_count > self.trigger_max_words:
            return [], {"provider": self.name, "variants": 0, "skipped": True, "reason": "too_long", "word_count": word_count}

        cached = self._cache.get(clean)
        if cached is not None:
            self._cache.move_to_end(clean)
            return list(cached), {"provider": self.name, "variants": len(cached), "cache": "hit"}

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT.format(max_variants=self.max_variants)},
                {"role": "user", "content": clean},
            ],
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature

        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.warning("query_rewrite_failed", extra={"error": str(exc), "provider": self.name})
            return [], {"provider": self.name, "variants": 0, "skipped": True, "reason": "error", "error": str(exc)[:200]}

        latency_ms = int((time.perf_counter() - started) * 1000)
        raw = (completion.choices[0].message.content or "").strip()
        variants = self._parse(raw, original=clean)
        self._cache_put(clean, variants)

        usage = completion.usage
        return variants, {
            "provider": self.name,
            "variants": len(variants),
            "latency_ms": latency_ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "cache": "miss",
        }

    def _parse(self, raw: str, original: str) -> list[str]:
        lowered_original = original.lower()
        seen = {lowered_original}
        out: list[str] = []
        for line in raw.splitlines():
            candidate = line.strip().lstrip("-*0123456789.) ").strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            if len(_WORD_RE.findall(candidate)) > 20:
                continue
            seen.add(lowered)
            out.append(candidate)
            if len(out) >= self.max_variants:
                break
        return out

    def _cache_put(self, key: str, value: list[str]) -> None:
        self._cache[key] = list(value)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def build_query_rewriter(settings: Settings) -> QueryRewriter:
    provider = getattr(settings, "QUERY_REWRITE_PROVIDER", "none")
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when QUERY_REWRITE_PROVIDER=openai")
        return OpenAIQueryRewriter(
            api_key=settings.OPENAI_API_KEY,
            model=settings.QUERY_REWRITE_MODEL or settings.OPENAI_GENERATION_MODEL,
            base_url=settings.OPENAI_BASE_URL,
            timeout=settings.QUERY_REWRITE_TIMEOUT_S,
            max_variants=settings.QUERY_REWRITE_MAX_VARIANTS,
            trigger_max_words=settings.QUERY_REWRITE_TRIGGER_MAX_WORDS,
            max_output_tokens=settings.QUERY_REWRITE_MAX_OUTPUT_TOKENS,
            temperature=settings.OPENAI_TEMPERATURE,
            cache_size=settings.QUERY_REWRITE_CACHE_SIZE,
        )
    return NullQueryRewriter()


__all__ = [
    "QueryRewriter",
    "NullQueryRewriter",
    "OpenAIQueryRewriter",
    "build_query_rewriter",
]
