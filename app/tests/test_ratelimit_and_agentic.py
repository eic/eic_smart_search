"""Unit tests: public-endpoint rate limiter + search_index agentic tool."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

import app.api.ratelimit as rl


class FakeRequest:
    def __init__(self, ip: str = "1.2.3.4", forwarded: str | None = None) -> None:
        self.headers = {"x-forwarded-for": forwarded} if forwarded else {}

        class _Client:
            host = ip

        self.client = _Client()


def test_rate_limit_allows_then_blocks_then_isolates_clients() -> None:
    rl._hits.clear()
    req_a = FakeRequest(ip="10.0.0.1")
    for _ in range(3):
        rl.check_rate_limit(req_a, limit=3)
    with pytest.raises(HTTPException) as exc:
        rl.check_rate_limit(req_a, limit=3)
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers
    # Different client keeps its own window.
    rl.check_rate_limit(FakeRequest(ip="10.0.0.2"), limit=3)


def test_rate_limit_prefers_forwarded_header_and_zero_disables() -> None:
    rl._hits.clear()
    req = FakeRequest(ip="127.0.0.1", forwarded="203.0.113.9, 10.0.0.1")
    rl.check_rate_limit(req, limit=1)
    with pytest.raises(HTTPException):
        rl.check_rate_limit(req, limit=1)
    # limit=0 → disabled, never raises
    for _ in range(10):
        rl.check_rate_limit(req, limit=0)


def test_index_search_client_shapes_results(monkeypatch) -> None:
    from app.integrations.index_search import IndexSearchClient
    from app.retrieval.types import RetrievedChunk

    class FakeRetriever:
        def search(self, db, query, filters, top_k, rerank=True):
            assert rerank is False
            assert filters.scope == "public"
            chunks = [
                RetrievedChunk(
                    chunk_id=f"c{i}", document_id=f"d{i}", source_name="eic_website",
                    source_type="website", title=f"T{i}", url=f"https://x/{i}",
                    content="body " * 400, score=1.0 - i * 0.1,
                )
                for i in range(top_k)
            ]
            return chunks, {}

    import app.integrations.index_search as mod

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod, "SessionLocal", FakeSession)
    client = IndexSearchClient(FakeRetriever(), top_k=4)
    hits = client.search("pfRICH radiator design")
    assert len(hits) == 4
    assert set(hits[0]) == {"title", "url", "source", "content"}
    assert len(hits[0]["content"]) <= 1200
    assert client.search("") == []
    # top_k clamped to 12
    assert len(client.search("q", top_k=99)) == 12


def test_search_index_tool_wired_into_generator() -> None:
    from app.integrations.index_search import SEARCH_INDEX_TOOL_SCHEMA
    from app.llm.generation import OpenAIAnswerGenerator

    generator = OpenAIAnswerGenerator.__new__(OpenAIAnswerGenerator)
    generator._indico = None
    generator._people = None
    generator._zenodo_search = None

    class FakeIndexSearch:
        def search(self, query, top_k=None):
            return [{"title": "T", "url": "u", "source": "s", "content": "c"}]

    generator._index_search = FakeIndexSearch()
    result = generator._run_tool("search_index", {"query": "dRICH aerogel", "top_k": "3"})
    assert result["count"] == 1
    assert SEARCH_INDEX_TOOL_SCHEMA["function"]["name"] == "search_index"


def test_safe_fetch_blocks_ssrf() -> None:
    from app.ingestion.safe_fetch import BlockedURLError, check_url

    check_url("https://zenodo.org/records/1/files/x.pdf")  # allowed
    for bad in [
        "http://qdrant:6333/collections",           # internal service
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "https://evil.example.com/x",                # not allowlisted
        "file:///etc/passwd",                        # scheme
    ]:
        try:
            check_url(bad)
        except BlockedURLError:
            continue
        raise AssertionError(f"should have blocked: {bad}")


def test_answer_sanitizer_strips_images_and_foreign_links() -> None:
    from app.retrieval.types import RetrievedChunk
    from app.services.query import QueryService

    chunks = [
        RetrievedChunk(chunk_id="c1", document_id="d1", source_name="s", source_type="t",
                       title="T", url="https://zenodo.org/records/1", content="x", score=1.0),
    ]
    answer = (
        "See [the record](https://zenodo.org/records/1) and "
        "[evil](https://evil.com/leak) ![beacon](https://evil.com/?q=secret)."
    )
    out = QueryService._sanitize_answer(answer, chunks)
    assert "https://zenodo.org/records/1" in out       # allowed link kept
    assert "evil.com" not in out                        # foreign link + image gone
    assert "the record" in out and "evil" in out        # link text preserved


def test_client_ip_prefers_cf_and_last_hop() -> None:
    import app.api.ratelimit as rl

    class R:
        def __init__(self, headers, host="127.0.0.1"):
            self.headers = headers

            class C:
                pass
            self.client = C()
            self.client.host = host

    assert rl._client_ip(R({"cf-connecting-ip": "9.9.9.9", "x-forwarded-for": "1.1.1.1"})) == "9.9.9.9"
    # No CF header: take the LAST (trusted-proxy-added) hop, never client hop 0.
    assert rl._client_ip(R({"x-forwarded-for": "1.1.1.1, 2.2.2.2, 3.3.3.3"})) == "3.3.3.3"
