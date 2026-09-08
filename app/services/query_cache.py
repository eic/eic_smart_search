"""In-memory LRU+TTL cache for /query responses.

Keyed on (query, scope, generate_answer). Two users asking the same
popular question within the TTL get the second one for free. The cache
is cleared on ingest so stale content never lingers.

The cached payload is the rendered answer + citations (post parent-expansion,
post rerank). We recompute fresh ``query_log_id`` and ``latency_ms`` per
request so the query log stays an accurate per-call record.

``top_k`` is intentionally not part of the key: an entry with N cached
citations satisfies any request with top_k <= N (citations are sliced down
at read time). QueryService also falls back from a preview
(generate_answer=false) miss to the matching submit (generate_answer=true)
entry, since a submit entry is a strict superset of what a preview needs.
See app/services/query.py.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    payload: dict[str, Any]
    created_at: float


class QueryCache:
    def __init__(self, max_size: int = 500, ttl_s: float = 3600.0) -> None:
        self.max_size = max(1, int(max_size))
        self.ttl_s = max(0.001, float(ttl_s))
        self._store: dict[tuple[str, str, bool], _Entry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(query: str, scope: str, generate_answer: bool) -> tuple[str, str, bool]:
        return (query, scope, bool(generate_answer))

    # ---- public api ----------------------------------------------------------

    def get(self, query: str, scope: str, generate_answer: bool) -> dict[str, Any] | None:
        key = self._key(query, scope, generate_answer)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if now - entry.created_at > self.ttl_s:
                self._store.pop(key, None)
                return None
            # LRU touch: move this key to end-of-insertion (most-recently-used).
            self._store.pop(key)
            self._store[key] = entry
            return {"payload": entry.payload, "age_s": now - entry.created_at}

    def set(self, query: str, scope: str, generate_answer: bool, payload: dict[str, Any]) -> None:
        key = self._key(query, scope, generate_answer)
        with self._lock:
            self._store[key] = _Entry(payload=payload, created_at=time.time())
            while len(self._store) > self.max_size:
                oldest_key = next(iter(self._store))
                self._store.pop(oldest_key, None)

    def clear(self) -> int:
        with self._lock:
            n = len(self._store)
            self._store.clear()
            return n


# Module-level singleton. Factory resets it when settings change.
_INSTANCE: QueryCache | None = None
_INSTANCE_LOCK = threading.Lock()


def get_query_cache(max_size: int | None = None, ttl_s: float | None = None) -> QueryCache:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            # Fall back to the configured values, not hardcoded copies — an
            # argless first call used to pin a 1h TTL for process lifetime
            # while the Settings default was 5 days.
            from app.core.config import get_settings

            settings = get_settings()
            _INSTANCE = QueryCache(
                max_size=max_size if max_size is not None else settings.QUERY_CACHE_MAX_SIZE,
                ttl_s=ttl_s if ttl_s is not None else settings.QUERY_CACHE_TTL_S,
            )
        return _INSTANCE


def reset_query_cache() -> None:
    """Clear the singleton. Called from ingest/reindex paths."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            cleared = _INSTANCE.clear()
            logger.info("query_cache_cleared", extra={"entries_removed": cleared})
