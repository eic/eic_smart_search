from __future__ import annotations

import time


from app.services.query_cache import QueryCache, get_query_cache, reset_query_cache


def test_get_miss_returns_none() -> None:
    cache = QueryCache(max_size=10, ttl_s=60)
    assert cache.get("q", "public", True) is None


def test_set_then_get_hit() -> None:
    cache = QueryCache(max_size=10, ttl_s=60)
    cache.set("q", "public", True, {"answer": "hi", "citations": []})
    hit = cache.get("q", "public", True)
    assert hit is not None
    assert hit["payload"]["answer"] == "hi"
    assert hit["age_s"] >= 0


def test_keys_differ_by_generate_answer() -> None:
    cache = QueryCache(max_size=10, ttl_s=60)
    cache.set("q", "public", True, {"answer": "submit", "citations": []})
    cache.set("q", "public", False, {"answer": "preview", "citations": []})
    assert cache.get("q", "public", True)["payload"]["answer"] == "submit"
    assert cache.get("q", "public", False)["payload"]["answer"] == "preview"


def test_keys_differ_by_scope() -> None:
    cache = QueryCache(max_size=10, ttl_s=60)
    cache.set("q", "public", True, {"answer": "pub", "citations": []})
    cache.set("q", "internal", True, {"answer": "int", "citations": []})
    assert cache.get("q", "public", True)["payload"]["answer"] == "pub"
    assert cache.get("q", "internal", True)["payload"]["answer"] == "int"


def test_ttl_expiry_evicts_entry() -> None:
    cache = QueryCache(max_size=10, ttl_s=0.05)
    cache.set("q", "public", True, {"answer": "hi", "citations": []})
    assert cache.get("q", "public", True) is not None
    time.sleep(0.1)
    assert cache.get("q", "public", True) is None


def test_lru_eviction_on_overflow() -> None:
    cache = QueryCache(max_size=2, ttl_s=60)
    cache.set("a", "public", True, {"answer": "A", "citations": []})
    cache.set("b", "public", True, {"answer": "B", "citations": []})
    cache.set("c", "public", True, {"answer": "C", "citations": []})
    assert cache.get("a", "public", True) is None
    assert cache.get("b", "public", True) is not None
    assert cache.get("c", "public", True) is not None


def test_lru_touch_protects_recently_used() -> None:
    cache = QueryCache(max_size=2, ttl_s=60)
    cache.set("a", "public", True, {"answer": "A", "citations": []})
    cache.set("b", "public", True, {"answer": "B", "citations": []})
    cache.get("a", "public", True)
    cache.set("c", "public", True, {"answer": "C", "citations": []})
    # 'a' was touched, so 'b' should have been evicted, not 'a'.
    assert cache.get("a", "public", True) is not None
    assert cache.get("b", "public", True) is None
    assert cache.get("c", "public", True) is not None


def test_clear_empties_store() -> None:
    cache = QueryCache(max_size=10, ttl_s=60)
    cache.set("a", "public", True, {"answer": "A", "citations": []})
    assert cache.clear() == 1
    assert cache.get("a", "public", True) is None


def test_singleton_and_reset() -> None:
    c1 = get_query_cache(max_size=50, ttl_s=30)
    c2 = get_query_cache()
    assert c1 is c2
    c1.set("q", "public", True, {"answer": "x", "citations": []})
    assert c2.get("q", "public", True) is not None
    reset_query_cache()
    assert c2.get("q", "public", True) is None
