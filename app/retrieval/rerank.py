from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Protocol

from app.core.config import Settings
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], dict]: ...


class NullReranker:
    name = "none"

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], dict]:
        return chunks, {"provider": self.name, "reranked": 0}


class LocalCrossEncoderReranker:
    """Cross-encoder reranker that scores (query, passage) pairs.

    Uses sentence-transformers' CrossEncoder (already a project dependency).
    Models are downloaded on first use and cached. Default is a multilingual
    reranker suitable for mixed-source corpora.
    """

    def __init__(
        self,
        model: str,
        *,
        max_candidates: int = 40,
        score_weight: float = 0.7,
        max_passage_chars: int = 1500,
        timeout_s: float | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model)
        self.name = f"cross_encoder:{model}"
        self.max_candidates = max_candidates
        self.score_weight = score_weight
        self.max_passage_chars = max_passage_chars
        self.timeout_s = timeout_s
        self._executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rerank")
            if timeout_s
            else None
        )

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], dict]:
        if not chunks:
            return chunks, {"provider": self.name, "reranked": 0}

        head = chunks[: self.max_candidates]
        tail = chunks[self.max_candidates :]
        pairs = [[query, self._prepare_passage(chunk)] for chunk in head]
        started = time.perf_counter()
        try:
            if self._executor is not None and self.timeout_s:
                future = self._executor.submit(self._model.predict, pairs, convert_to_numpy=True)
                raw_scores = future.result(timeout=self.timeout_s)
            else:
                raw_scores = self._model.predict(pairs, convert_to_numpy=True)
        except concurrent.futures.TimeoutError:  # noqa: PERF203
            logger.warning(
                "rerank_timeout",
                extra={"provider": self.name, "timeout_s": self.timeout_s, "candidates": len(head)},
            )
            return chunks, {
                "provider": self.name,
                "reranked": 0,
                "error": "timeout",
                "timeout_s": self.timeout_s,
                "candidates": len(head),
            }
        except BaseException as exc:  # NotImplementedError, MemoryError, torch errors all caught
            logger.warning("rerank_failed", extra={"error": str(exc), "provider": self.name})
            return chunks, {"provider": self.name, "reranked": 0, "error": str(exc)[:200]}
        latency_ms = int((time.perf_counter() - started) * 1000)

        scores = [float(value) for value in raw_scores]
        normalized = self._min_max_normalize(scores)
        original_ranks = {chunk.chunk_id: index for index, chunk in enumerate(head)}
        for chunk, norm_score, raw_score in zip(head, normalized, scores):
            chunk.metadata = {**chunk.metadata, "rerank_score": raw_score}
            chunk.score = self.score_weight * norm_score + (1.0 - self.score_weight) * chunk.score

        head.sort(key=lambda item: item.score, reverse=True)
        new_ranks = {chunk.chunk_id: index for index, chunk in enumerate(head)}
        rank_deltas = [
            original_ranks[chunk_id] - new_ranks[chunk_id]
            for chunk_id in new_ranks
        ]
        max_promotion = max(rank_deltas, default=0)
        max_demotion = -min(rank_deltas, default=0)

        return head + tail, {
            "provider": self.name,
            "reranked": len(head),
            "latency_ms": latency_ms,
            "max_promotion": max_promotion,
            "max_demotion": max_demotion,
            "score_weight": self.score_weight,
        }

    def _prepare_passage(self, chunk: RetrievedChunk) -> str:
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""
        prefix = f"{chunk.title}. {heading}. " if heading else f"{chunk.title}. "
        body = chunk.content
        if len(body) > self.max_passage_chars:
            body = body[: self.max_passage_chars - 3].rstrip() + "..."
        return prefix + body

    @staticmethod
    def _min_max_normalize(values: list[float]) -> list[float]:
        if not values:
            return values
        lo = min(values)
        hi = max(values)
        span = hi - lo
        if span <= 1e-9:
            return [0.5 for _ in values]
        return [(value - lo) / span for value in values]


def build_reranker(settings: Settings) -> Reranker:
    provider = getattr(settings, "RERANK_PROVIDER", "none")
    if provider == "cross_encoder":
        timeout = getattr(settings, "RERANK_TIMEOUT_S", None)
        return LocalCrossEncoderReranker(
            model=settings.RERANK_MODEL,
            max_candidates=settings.RERANK_CANDIDATES,
            score_weight=settings.RERANK_SCORE_WEIGHT,
            max_passage_chars=settings.RERANK_MAX_PASSAGE_CHARS,
            timeout_s=timeout if timeout and timeout > 0 else None,
        )
    return NullReranker()
