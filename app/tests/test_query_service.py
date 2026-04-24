from app.retrieval.types import RetrievedChunk
from app.schemas.api import QueryRequest
from app.services.query import QueryService


class FakeRetriever:
    def __init__(self) -> None:
        self.scope = None
        self.allowed_visibilities = None
        self.extra_queries: list[str] | None = None

    def search(self, db, query, filters, top_k, extra_queries=None):
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
    def generate(self, query, chunks, min_support_score):
        return "answer", {"mode": "fake"}


class FakeSettings:
    MIN_SUPPORT_SCORE = 0.16


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

