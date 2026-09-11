from app.retrieval.types import RetrievedChunk
from app.schemas.api import QueryRequest
from app.services.query import QueryService


class FakeRetriever:
    def __init__(self) -> None:
        self.scope = None
        self.allowed_visibilities = None
        self.extra_queries: list[str] | None = None

    def search(self, db, query, filters, top_k, extra_queries=None, rerank=None):
        self.scope = filters.scope
        self.allowed_visibilities = filters.allowed_visibilities
        self.extra_queries = extra_queries
        return (
            [
                RetrievedChunk(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    source_name="eic_website",
                    source_type="website",
                    title="Tutorials",
                    url="https://eic.github.io/documentation/tutorials.html",
                    content="Simulation tutorials explain how to run examples.",
                    score=0.9,
                )
            ],
            {"test": True},
        )


class FakeGenerator:
    def generate(self, query, chunks, min_support_score, *, history=None):
        self.last_history = history
        return "answer", {"mode": "fake"}


class FakeSettings:
    MIN_SUPPORT_SCORE = 0.16
    CONFIDENCE_VEC_LOW = 0.30
    CONFIDENCE_VEC_HIGH = 0.55


class FakeDb:
    def __init__(self) -> None:
        self.added = []

    def add(self, record) -> None:
        self.added.append(record)

    def commit(self) -> None:
        return None


def test_query_service_public_scope_never_includes_internal_visibility() -> None:
    retriever = FakeRetriever()
    service = QueryService(FakeSettings(), retriever, FakeGenerator())

    response = service.query(FakeDb(), QueryRequest(query="How do I run tutorials?", scope="public"))

    assert response.citations
    assert retriever.scope == "public"
    assert retriever.allowed_visibilities == ["public"]


def test_align_citations_reorders_and_remaps() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            document_id=f"d{i}",
            source_name="eic_website",
            source_type="website",
            title=f"T{i}",
            url=f"https://example/{i}",
            content="...",
            score=1.0 - i * 0.01,
        )
        for i in range(1, 11)
    ]
    # LLM cited sources 7 and 3 in that order out of 10 candidates.
    answer = "Foo [7] bar [3]. More [7]."
    remapped, new_chunks = QueryService._align_citations(answer, chunks, display_k=5)
    assert [c.chunk_id for c in new_chunks[:2]] == ["c7", "c3"]
    # Remaining slots filled with highest-scored non-cited chunks (c1, c2, c4).
    assert [c.chunk_id for c in new_chunks[2:5]] == ["c1", "c2", "c4"]
    assert remapped == "Foo [1] bar [2]. More [1]."


def test_align_citations_drops_out_of_range_references() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            document_id=f"d{i}",
            source_name="eic_website",
            source_type="website",
            title=f"T{i}",
            url=f"https://example/{i}",
            content="...",
            score=1.0 - i * 0.01,
        )
        for i in range(1, 4)
    ]
    # LLM cited source 99 — doesn't exist. Must be stripped.
    answer = "Claim [99]."
    remapped, new_chunks = QueryService._align_citations(answer, chunks, display_k=1)
    assert [c.chunk_id for c in new_chunks] == ["c1"]
    assert remapped == "Claim ."


def test_query_service_cache_hit_skips_retriever_and_generator() -> None:
    """On a second call with the same query/scope/top_k/generate_answer/filters,
    the cached payload is served without re-hitting the retriever, rewriter,
    or generator. The log still records the hit with cache_hit=true.
    """
    from app.services.query_cache import QueryCache

    retriever = FakeRetriever()
    generator = FakeGenerator()
    cache = QueryCache(max_size=10, ttl_s=60)
    service = QueryService(FakeSettings(), retriever, generator, query_cache=cache)

    db1 = FakeDb()
    req = QueryRequest(query="what are rucio tags?", scope="public", top_k=5, generate_answer=True)

    first = service.query(db1, req)
    assert first.retrieval_debug.get("cache_hit") is False
    assert first.citations
    # Reset FakeRetriever so we can detect whether it's called a second time.
    retriever.scope = None
    retriever.extra_queries = None

    db2 = FakeDb()
    second = service.query(db2, req)
    assert second.retrieval_debug.get("cache_hit") is True
    assert second.retrieval_debug.get("cache_age_s") is not None
    assert second.answer == first.answer
    assert [c.chunk_id for c in second.citations] == [c.chunk_id for c in first.citations]
    # Retriever must not have been called on the second query.
    assert retriever.scope is None
    assert retriever.extra_queries is None
    # But a new query log entry was still recorded for analytics.
    assert len(db2.added) == 1


class _MultiChunkRetriever:
    """Stub retriever that returns N stub chunks, tracking call count."""

    def __init__(self, n: int = 10) -> None:
        self.n = n
        self.calls = 0

    def search(self, db, query, filters, top_k, extra_queries=None, rerank=None):
        self.calls += 1
        chunks = [
            RetrievedChunk(
                chunk_id=f"c{i}",
                document_id=f"d{i}",
                source_name="eic_website",
                source_type="website",
                title=f"T{i}",
                url=f"https://example/{i}",
                content="content",
                score=1.0 - i * 0.01,
            )
            for i in range(self.n)
        ]
        return chunks[:top_k], {}


def test_cache_smaller_top_k_reuses_larger_entry() -> None:
    """A submit cached at top_k=10 must satisfy a later submit for top_k=3,
    sliced down. Retriever fires only once."""
    from app.services.query_cache import QueryCache

    retriever = _MultiChunkRetriever(n=10)
    cache = QueryCache(max_size=10, ttl_s=60)
    service = QueryService(FakeSettings(), retriever, FakeGenerator(), query_cache=cache)

    first = service.query(FakeDb(), QueryRequest(query="q", scope="public", top_k=10, generate_answer=True))
    assert len(first.citations) == 10
    assert retriever.calls == 1

    second = service.query(FakeDb(), QueryRequest(query="q", scope="public", top_k=3, generate_answer=True))
    assert second.retrieval_debug.get("cache_hit") is True
    assert len(second.citations) == 3
    # Sliced citations come from the head of the cached list.
    assert [c.chunk_id for c in second.citations] == [c.chunk_id for c in first.citations[:3]]
    assert retriever.calls == 1  # not re-queried


def test_cache_preview_reuses_submit_entry() -> None:
    """After a submit caches an entry, a same-query preview hits the
    submit entry (preview discards the bonus answer)."""
    from app.services.query_cache import QueryCache

    retriever = _MultiChunkRetriever(n=10)
    cache = QueryCache(max_size=10, ttl_s=60)
    service = QueryService(FakeSettings(), retriever, FakeGenerator(), query_cache=cache)

    service.query(FakeDb(), QueryRequest(query="q", scope="public", top_k=5, generate_answer=True))
    assert retriever.calls == 1

    preview = service.query(FakeDb(), QueryRequest(query="q", scope="public", top_k=5, generate_answer=False))
    assert preview.retrieval_debug.get("cache_hit") is True
    assert len(preview.citations) == 5
    assert retriever.calls == 1  # not re-queried


def test_cache_larger_top_k_request_misses_smaller_entry() -> None:
    """A submit cached at top_k=3 must NOT satisfy a later submit for
    top_k=8 — the cached entry doesn't have enough rows to honor the
    bigger request, so retrieval re-runs."""
    from app.services.query_cache import QueryCache

    retriever = _MultiChunkRetriever(n=10)
    cache = QueryCache(max_size=10, ttl_s=60)
    service = QueryService(FakeSettings(), retriever, FakeGenerator(), query_cache=cache)

    service.query(FakeDb(), QueryRequest(query="q", scope="public", top_k=3, generate_answer=True))
    assert retriever.calls == 1

    second = service.query(FakeDb(), QueryRequest(query="q", scope="public", top_k=8, generate_answer=True))
    assert second.retrieval_debug.get("cache_hit") is False
    assert len(second.citations) == 8
    assert retriever.calls == 2



def test_reduce_duplicates_collapses_same_paper_across_sources() -> None:
    from app.retrieval.hybrid import HybridRetriever

    def paper(doc_id, source_type, title, score, **meta):
        return RetrievedChunk(
            chunk_id=f"c-{doc_id}",
            document_id=doc_id,
            source_name=source_type,
            source_type=source_type,
            title=title,
            url=f"https://x/{doc_id}",
            content="body " + doc_id,
            score=score,
            metadata=meta,
        )

    retriever = HybridRetriever.__new__(HybridRetriever)
    chunks = [
        # Same talk uploaded to Zenodo three times (fresh DOI each version)
        paper("z1", "zenodo_community", "The ePIC SVT: design and studies", 0.9, doi="10.5281/zenodo.1"),
        paper("z2", "zenodo_community", "The ePIC SVT: Design and Studies", 0.8, doi="10.5281/zenodo.2"),
        # Same work via INSPIRE (journal DOI differs, title matches)
        paper("i1", "inspire_literature", "The ePIC SVT: design and studies", 0.7, doi="10.1000/j.1"),
        # Different paper, must survive
        paper("i2", "inspire_literature", "EIC vertex and tracking detector", 0.6, doi="10.1000/j.2"),
        # Meeting agendas share titles legitimately — must NOT collapse
        paper("e1", "indico", "ePIC dRICH WG Meeting", 0.5),
        paper("e2", "indico", "ePIC dRICH WG Meeting", 0.4),
    ]
    out = retriever._reduce_duplicates(chunks)
    doc_ids = [c.document_id for c in out]
    assert "z1" in doc_ids
    assert "z2" not in doc_ids  # zenodo re-upload collapsed by title
    assert "i1" not in doc_ids  # cross-source duplicate collapsed
    assert "i2" in doc_ids
    assert "e1" in doc_ids and "e2" in doc_ids


def test_reduce_duplicates_keeps_multichunk_document() -> None:
    from app.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever.__new__(HybridRetriever)
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            document_id="d1",
            source_name="zenodo_epic",
            source_type="zenodo_community",
            title="Long PDF paper",
            url="https://x/d1",
            content=f"section {i}",
            score=0.9 - i * 0.1,
            heading_path=[f"h{i}"],
            metadata={"doi": "10.5281/zenodo.7"},
        )
        for i in range(2)
    ]
    out = retriever._reduce_duplicates(chunks)
    # Both chunks of the SAME document survive (in-doc context preserved).
    assert len(out) == 2


def test_query_stream_emits_citations_delta_done() -> None:
    import json as _json

    from app.services.query_cache import QueryCache

    class StreamingGenerator:
        def generate(self, query, chunks, min_support_score, *, history=None):
            return "full", {"mode": "fake"}

        def generate_stream(self, query, chunks, min_support_score, *, history=None):
            yield ("delta", "Hello ")
            yield ("delta", "world [1].")
            yield ("final", ("Hello world [1].", {"mode": "fake", "contexts_sent": len(chunks)}))

    retriever = _MultiChunkRetriever(n=5)
    cache = QueryCache(max_size=10, ttl_s=60)
    service = QueryService(FakeSettings(), retriever, StreamingGenerator(), query_cache=cache)

    frames = list(
        service.query_stream(FakeDb(), QueryRequest(query="q", scope="public", top_k=3, generate_answer=True))
    )
    events = []
    for frame in frames:
        lines = frame.strip().split("\n")
        event = lines[0].removeprefix("event: ")
        data = _json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))

    kinds = [e for e, _ in events]
    assert kinds[0] == "citations"
    assert kinds.count("delta") == 2
    assert kinds[-1] == "done"
    done = events[-1][1]
    # Final answer has [1] remapped/aligned and citations capped at top_k.
    assert done["answer"].startswith("Hello world")
    assert len(done["citations"]) == 3
    # Provisional citations arrived before any delta.
    assert events[0][1]["provisional"] is True

    # Second identical query must be served from cache — still full SSE shape.
    frames2 = list(
        service.query_stream(FakeDb(), QueryRequest(query="q", scope="public", top_k=3, generate_answer=True))
    )
    assert retriever.calls == 1
    assert "event: done" in frames2[-1]


def test_expert_hint_on_low_confidence(monkeypatch) -> None:
    """Low-confidence answers append a who-to-ask referral from the people graph."""

    class LowConfRetriever(_MultiChunkRetriever):
        def search(self, db, query, filters, top_k, extra_queries=None, rerank=None):
            chunks, dbg = super().search(db, query, filters, top_k, extra_queries, rerank)
            for c in chunks:
                c.vector_score = 0.1  # below CONFIDENCE_VEC_LOW -> "low"
                c.metadata = {"authors": ["Smith, Jane (BNL)"]}
            return chunks, dbg

    class FakePerson:
        canonical_name = "Jane Smith"
        affiliations = ["BNL"]
        activity = [{"title": "dRICH status", "url": "https://indico/x", "when": "2026-07-14"}]

    import app.people.service as people_service

    monkeypatch.setattr(people_service, "lookup", lambda db, q, limit=1: [FakePerson()])

    service = QueryService(FakeSettings(), LowConfRetriever(n=3), FakeGenerator())
    resp = service.query(FakeDb(), QueryRequest(query="obscure topic", scope="public", top_k=3, generate_answer=True))
    assert resp.expert_hint is not None
    assert "Jane Smith" in resp.expert_hint
    assert "Who to ask" in resp.expert_hint  # structured field, not in answer text
    assert "Who to ask" not in resp.answer


def test_no_expert_hint_on_high_confidence() -> None:
    class HighConfRetriever(_MultiChunkRetriever):
        def search(self, db, query, filters, top_k, extra_queries=None, rerank=None):
            chunks, dbg = super().search(db, query, filters, top_k, extra_queries, rerank)
            for c in chunks:
                c.vector_score = 0.9
            return chunks, dbg

    service = QueryService(FakeSettings(), HighConfRetriever(n=3), FakeGenerator())
    resp = service.query(FakeDb(), QueryRequest(query="q", scope="public", top_k=3, generate_answer=True))
    assert resp.expert_hint is None
    assert "Who to ask" not in resp.answer


def test_reduce_duplicates_prefers_newest_version() -> None:
    """Zenodo re-uploads / arXiv v2: newest version owns the identity even
    when an older version ranks higher."""
    from datetime import datetime

    from app.retrieval.hybrid import HybridRetriever

    def paper(doc_id, title, score, when):
        return RetrievedChunk(
            chunk_id=f"c-{doc_id}", document_id=doc_id, source_name="zenodo_epic",
            source_type="zenodo_community", title=title, url=f"https://x/{doc_id}",
            content="body", score=score, last_updated=when,
        )

    retriever = HybridRetriever.__new__(HybridRetriever)
    chunks = [
        paper("v1", "The ePIC TDR", 0.9, datetime(2025, 1, 1)),  # older, ranks higher
        paper("v2", "The ePIC TDR", 0.8, datetime(2026, 6, 1)),  # newest wins
    ]
    out = retriever._reduce_duplicates(chunks)
    assert [c.document_id for c in out] == ["v2"]


def test_history_bypasses_cache_and_reaches_generator() -> None:
    """A history-bearing request must not read or write the shared response
    cache (anaphoric queries would collide across conversations), and the
    prior turns must reach the generator as chat messages."""
    from app.schemas.api import HistoryTurn
    from app.services.query_cache import QueryCache

    cache = QueryCache(max_size=10, ttl_s=60)
    generator = FakeGenerator()
    service = QueryService(FakeSettings(), FakeRetriever(), generator, query_cache=cache)
    history = [
        HistoryTurn(role="user", content="what is a picoDst?"),
        HistoryTurn(role="assistant", content="A compact analysis format."),
    ]
    request = QueryRequest(
        query="how do I read one?", scope="public", top_k=3,
        generate_answer=True, history=history,
    )
    resp = service.query(FakeDb(), request)
    assert resp.answer == "answer"
    assert generator.last_history == [
        {"role": "user", "content": "what is a picoDst?"},
        {"role": "assistant", "content": "A compact analysis format."},
    ]
    # Nothing cached: an identical follow-up in another conversation must miss.
    assert cache.get("how do I read one?", "public", True) is None
    # And a history-bearing probe never serves a cached history-free answer.
    cache.set("how do I read one?", "public", True, {"answer": "stale", "citations": [], "top_k": 3})
    resp2 = service.query(FakeDb(), request)
    assert resp2.answer == "answer"


def test_client_tag_is_logged_in_debug() -> None:
    from app.services.query_cache import QueryCache

    service = QueryService(
        FakeSettings(), FakeRetriever(), FakeGenerator(), query_cache=QueryCache(max_size=10, ttl_s=60)
    )
    request = QueryRequest(query="what is EICrecon", top_k=3, client="cli")
    resp = service.query(FakeDb(), request)
    assert resp.retrieval_debug["client"] == "cli"
    resp2 = service.query(FakeDb(), request)  # cache hit path keeps the tag
    assert resp2.retrieval_debug["cache_hit"] is True
    assert resp2.retrieval_debug["client"] == "cli"


def test_history_adds_previous_user_turn_as_extra_query() -> None:
    """Retrieval for an anaphoric follow-up fuses in the previous user
    question so the topic's chunks stay in the pool."""
    from app.schemas.api import HistoryTurn

    class RecordingRetriever(FakeRetriever):
        def search(self, db, query, filters, top_k, extra_queries=None, rerank=None):
            self.extra_queries = extra_queries
            return super().search(db, query, filters, top_k, extra_queries, rerank)

    retriever = RecordingRetriever()
    service = QueryService(FakeSettings(), retriever, FakeGenerator())
    service.query(
        FakeDb(),
        QueryRequest(
            query="how do I read one?", scope="public", top_k=3, generate_answer=True,
            history=[
                HistoryTurn(role="user", content="what is a picoDst?"),
                HistoryTurn(role="assistant", content="A compact analysis format."),
            ],
        ),
    )
    assert retriever.extra_queries == ["what is a picoDst?"]


def test_probe_requests_skip_generation_and_cache() -> None:
    from app.services.query_cache import QueryCache

    class CountingGenerator(FakeGenerator):
        calls = 0

        def generate(self, *args, **kwargs):
            CountingGenerator.calls += 1
            return super().generate(*args, **kwargs)

    service = QueryService(
        FakeSettings(), FakeRetriever(), CountingGenerator(), query_cache=QueryCache(max_size=10, ttl_s=60)
    )
    resp = service.query(FakeDb(), QueryRequest(query="init", top_k=3, client="cli"))
    assert CountingGenerator.calls == 0
    assert resp.answer == ""
    assert resp.retrieval_debug["client"] == "probe"
    assert resp.retrieval_debug["generation"]["skipped"] == "short-cli"
    # A person typing the same word in the widget must not get the empty
    # probe answer replayed from the cache.
    resp2 = service.query(FakeDb(), QueryRequest(query="init", top_k=3))
    assert resp2.retrieval_debug["cache_hit"] is False
    assert CountingGenerator.calls == 1
