from __future__ import annotations

from app.retrieval.rerank import LocalCrossEncoderReranker, NullReranker
from app.retrieval.types import RetrievedChunk


def _chunk(chunk_id: str, score: float, content: str = "body", title: str = "t") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"d-{chunk_id}",
        source_name="s",
        source_type="website",
        title=title,
        url=f"https://example.com/{chunk_id}",
        content=content,
        score=score,
    )


class _StubModel:
    def __init__(self, score_by_content: dict[str, float]) -> None:
        self.score_by_content = score_by_content
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs, convert_to_numpy=True):  # noqa: ARG002 — matches CrossEncoder signature
        self.calls.append(pairs)
        return [self.score_by_content.get(pair[1].split(". ", 1)[-1], 0.0) for pair in pairs]


def _build(model: _StubModel, **overrides) -> LocalCrossEncoderReranker:
    reranker = LocalCrossEncoderReranker.__new__(LocalCrossEncoderReranker)
    reranker._model = model  # type: ignore[attr-defined]
    reranker.name = "cross_encoder:stub"
    reranker.max_candidates = overrides.get("max_candidates", 40)
    reranker.score_weight = overrides.get("score_weight", 0.7)
    reranker.max_passage_chars = overrides.get("max_passage_chars", 1500)
    reranker.timeout_s = overrides.get("timeout_s", None)
    reranker._executor = overrides.get("_executor", None)  # type: ignore[attr-defined]
    return reranker


def test_null_reranker_passes_through() -> None:
    reranker = NullReranker()
    chunks = [_chunk("a", 0.9), _chunk("b", 0.5)]

    out, debug = reranker.rerank("q", chunks)

    assert out == chunks
    assert debug["provider"] == "none"
    assert debug["reranked"] == 0


def test_cross_encoder_promotes_better_match() -> None:
    chunks = [
        _chunk("a", 0.9, content="unrelated text"),
        _chunk("b", 0.4, content="the exact answer"),
        _chunk("c", 0.3, content="partially relevant"),
    ]
    model = _StubModel({"unrelated text": 0.1, "the exact answer": 0.95, "partially relevant": 0.5})
    reranker = _build(model)

    out, debug = reranker.rerank("what is the answer?", chunks)

    assert [c.chunk_id for c in out] == ["b", "c", "a"]
    assert debug["reranked"] == 3
    assert debug["max_promotion"] >= 1
    assert out[0].metadata["rerank_score"] == 0.95


def test_cross_encoder_respects_max_candidates() -> None:
    chunks = [_chunk(str(i), 1.0 - i * 0.01, content=f"c{i}") for i in range(10)]
    model = _StubModel({f"c{i}": 0.1 * i for i in range(10)})
    reranker = _build(model, max_candidates=3)

    out, debug = reranker.rerank("q", chunks)

    assert debug["reranked"] == 3
    assert len(out) == len(chunks)
    # Items past the cutoff should retain their original relative order.
    assert [c.chunk_id for c in out[3:]] == [str(i) for i in range(3, 10)]


def test_cross_encoder_falls_back_on_error() -> None:
    class _BrokenModel:
        def predict(self, pairs, convert_to_numpy=True):  # noqa: ARG002
            raise RuntimeError("oom")

    chunks = [_chunk("a", 0.9), _chunk("b", 0.5)]
    reranker = _build(_BrokenModel())  # type: ignore[arg-type]

    out, debug = reranker.rerank("q", chunks)

    assert out == chunks
    assert "error" in debug
    assert debug["reranked"] == 0


def test_cross_encoder_falls_back_on_timeout() -> None:
    import concurrent.futures
    import time as _time

    class _SlowModel:
        def predict(self, pairs, convert_to_numpy=True):  # noqa: ARG002
            _time.sleep(0.5)
            return [0.5] * len(pairs)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        reranker = _build(_SlowModel(), timeout_s=0.05, _executor=executor)  # type: ignore[arg-type]
        chunks = [_chunk("a", 0.9), _chunk("b", 0.5)]
        out, debug = reranker.rerank("q", chunks)

        assert out == chunks
        assert debug["error"] == "timeout"
        assert debug["timeout_s"] == 0.05
        assert debug["reranked"] == 0
    finally:
        executor.shutdown(wait=False)
