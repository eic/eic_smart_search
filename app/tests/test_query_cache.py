from __future__ import annotations

import time

import pytest

from app.services.query_cache import QueryCache, get_query_cache, reset_query_cache


def test_get_miss_returns_none() -> None:
    cache = QueryCache(max_size=10, ttl_s=60)
    assert cache.get("q", "public", 5, True, {}) is None
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1


def test_set_then_get_hit() -> None:
    cache = QueryCache(max_size=10, ttl_s=60)
    cache.set("q", "public", 5, True, {}, {"answer": "hi", "citations": []})
    hit = cache.get("q", "public", 5, True, {})
    assert hit is not None
    assert hit["payload"]["answer"] == "hi"
    assert hit["age_s"] >= 0
    assert cache.stats()["hits"] == 1


def test_keys_differ_by_top_k_and_generate_answer() -> None:
    cache = QueryCache(max_size=10, ttl_s=60)
    cache.set("q", "public", 5, True, {}, {"answer": "a", "citations": []})
    cache.set("q", "public", 5, False, {}, {"answer": "b", "citations": []})
    cache.set("q", "public", 8, True, {}, {"answer": "c", "citations": []})
    assert cache.get("q", "public", 5, True, {})["payload"]["answer"] == "a"
    assert cache.get("q", "public", 5, False, {})["payload"]["answer"] == "b"
    assert cache.get("q", "public", 8, True, {})["payload"]["answer"] == "c"


def test_filters_affect_key() -> None:
    cache = QueryCache(max_size=10, ttl_s=60)
    cache.set("q", "public", 5, True, {"source_names": ["a"]}, {"answer": "a-only", "citations": []})
    cache.set("q", "public", 5, True, {"source_names": ["b"]}, {"answer": "b-only", "citations": []})
    assert cache.get("q", "public", 5, True, {"source_names": ["a"]})["payload"]["answer"] == "a-only"
    assert cache.get("q", "public", 5, True, {"source_names": ["b"]})["payload"]["answer"] == "b-only"


def test_ttl_expiry_evicts_entry() -> None:
    cache = QueryCache(max_size=10, ttl_s=0.05)
    cache.set("q", "public", 5, True, {}, {"answer": "hi", "citations": []})
    assert cache.get("q", "public", 5, True, {}) is not None
    time.sleep(0.1)
    assert cache.get("q", "public", 5, True, {}) is None


def test_lru_eviction_on_overflow() -> None:
    cache = QueryCache(max_size=2, ttl_s=60)
    cache.set("a", "public", 5, True, {}, {"answer": "A", "citations": []})
    cache.set("b", "public", 5, True, {}, {"answer": "B", "citations": []})
    cache.set("c", "public", 5, True, {}, {"answer": "C", "citations": []})
    assert cache.get("a", "public", 5, True, {}) is None
    assert cache.get("b", "public", 5, True, {}) is not None
    assert cache.get("c", "public", 5, True, {}) is not None


def test_lru_touch_protects_recently_used() -> None:
    cache = QueryCache(max_size=2, ttl_s=60)
    cache.set("a", "public", 5, True, {}, {"answer": "A", "citations": []})
    cache.set("b", "public", 5, True, {}, {"answer": "B", "citations": []})
    cache.get("a", "public", 5, True, {})
    cache.set("c", "public", 5, True, {}, {"answer": "C", "citations": []})
    # 'a' was touched, so 'b' should have been evicted, not 'a'.
    assert cache.get("a", "public", 5, True, {}) is not None
    assert cache.get("b", "public", 5, True, {}) is None
    assert cache.get("c", "public", 5, True, {}) is not None


def test_clear_empties_store() -> None:
    cache = QueryCache(max_size=10, ttl_s=60)
    cache.set("a", "public", 5, True, {}, {"answer": "A", "citations": []})
    assert cache.clear() == 1
    assert cache.get("a", "public", 5, True, {}) is None


def test_singleton_and_reset() -> None:
    c1 = get_query_cache(max_size=50, ttl_s=30)
    c2 = get_query_cache()
    assert c1 is c2
    c1.set("q", "public", 5, True, {}, {"answer": "x", "citations": []})
    assert c2.get("q", "public", 5, True, {}) is not None
    reset_query_cache()
    assert c2.get("q", "public", 5, True, {}) is None
