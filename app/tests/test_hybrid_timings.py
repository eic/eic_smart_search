from app.retrieval.hybrid import HybridRetriever
from app.retrieval.types import RetrievedChunk, RetrievalFilters


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, document_id=f"d-{chunk_id}", source_name="eic_website",
        source_type="website", title=f"T {chunk_id}", url=f"https://x/{chunk_id}",
        content=f"content {chunk_id}", score=0.0, content_hash=chunk_id,
    )


class FakeVectorStore:
    def search(self, query, filters, limit):
        return [{"chunk_id": "a", "score": 0.7}, {"chunk_id": "b", "score": 0.5}]


class FakeLexical:
    def search(self, db, query, filters, limit):
        c = _chunk("c")
        c.lexical_score = 0.4
        return [c]

    def fetch_by_chunk_ids(self, db, chunk_scores):
        return [_chunk(cid) for cid in chunk_scores]


def test_search_reports_stage_timings() -> None:
    retriever = HybridRetriever(FakeVectorStore(), lexical=FakeLexical())
    chunks, debug = retriever.search(
        None, "how to run TDR tutorial", RetrievalFilters(scope="public"), top_k=3,
        extra_queries=["another phrasing"],
    )
    assert chunks
    timings = debug["timings_ms"]
    assert set(timings) == {"lexical", "vector", "fuse", "rerank", "dedupe"}
    for value in timings.values():
        assert isinstance(value, int) and value >= 0
    # Per-variant timings are folded into the total, not duplicated.
    assert all("timings_ms" not in q for q in debug["per_query"])
