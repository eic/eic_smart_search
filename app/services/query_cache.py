"""In-memory LRU+TTL cache for /query responses.

Keyed on (query, scope, top_k, generate_answer, filters). Two users asking
the same popular question within the TTL get the second one for free. The
cache is cleared on ingest so stale content never lingers.

The cached payload is the rendered answer + citations (post parent-expansion,
post rerank). We recompute fresh ``query_log_id`` and ``latency_ms`` per
request so the query log stays an accurate per-call record.
"""
from __future__ import annotations

import hashlib
import json
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
        self._store: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ---- key -----------------------------------------------------------------

    @staticmethod
    def _key(
        query: str,
        scope: str,
        top_k: int,
        generate_answer: bool,
        filters: dict[str, Any],
    ) -> str:
        blob = json.dumps(
            {"q": query, "s": scope, "k": int(top_k), "g": bool(generate_answer), "f": filters or {}},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ---- public api ----------------------------------------------------------

    def get(
        self,
        query: str,
        scope: str,
        top_k: int,
        generate_answer: bool,
        filters: dict[str, Any],
    ) -> dict[str, Any] | None:
        key = self._key(query, scope, top_k, generate_answer, filters)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if now - entry.created_at > self.ttl_s:
                self._store.pop(key, None)
                self._misses += 1
                return None
            # LRU touch: move this key to end-of-insertion (most-recently-used).
            self._store.pop(key)
            self._store[key] = entry
            self._hits += 1
            return {"payload": entry.payload, "age_s": now - entry.created_at}

    def set(
        self,
        query: str,
        scope: str,
        top_k: int,
        generate_answer: bool,
        filters: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        key = self._key(query, scope, top_k, generate_answer, filters)
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

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "ttl_s": self.ttl_s,
                "hits": self._hits,
                "misses": self._misses,
            }


# Module-level singleton. Factory resets it when settings change.
_INSTANCE: QueryCache | None = None
_INSTANCE_LOCK = threading.Lock()


def get_query_cache(max_size: int | None = None, ttl_s: float | None = None) -> QueryCache:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = QueryCache(
                max_size=max_size if max_size is not None else 500,
                ttl_s=ttl_s if ttl_s is not None else 3600.0,
            )
        return _INSTANCE


def reset_query_cache() -> None:
    """Clear the singleton. Called from ingest/reindex paths."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            cleared = _INSTANCE.clear()
            logger.info("query_cache_cleared", extra={"entries_removed": cleared})
