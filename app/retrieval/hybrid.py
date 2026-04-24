import concurrent.futures
import logging
import re
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.retrieval.lexical import PostgresLexicalRetriever
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.rerank import NullReranker, Reranker
from app.retrieval.types import RetrievedChunk, RetrievalFilters

logger = logging.getLogger(__name__)

# One shared worker for the vector-search side of every query. httpx (used
# by qdrant-client) and the embedding model release the GIL during IO /
# tensor ops, so a single background thread is enough to overlap with the
# Postgres call that runs on the request thread.
_VECTOR_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="vec-search"
)


HOW_TO_RE = re.compile(r"\b(how do i|how to|tutorial|guide|run|setup|install|configure)\b", re.I)


class HybridRetriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        lexical: PostgresLexicalRetriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.lexical = lexical or PostgresLexicalRetriever()
        self.reranker: Reranker = reranker or NullReranker()

    def search(
        self,
        db: Session,
        query: str,
        filters: RetrievalFilters,
        top_k: int,
        extra_queries: list[str] | None = None,
    ) -> tuple[list[RetrievedChunk], dict]:
        limit = max(top_k * 4, 20)
        all_queries = [query] + list(extra_queries or [])
        merged: dict[str, RetrievedChunk] = {}
        per_query_debug: list[dict] = []
        total_lexical = 0
        total_vector = 0
        vector_errors: list[str] = []

        for variant in all_queries:
            fused, variant_debug = self._fuse_single_query(db, variant, filters, limit)
            per_query_debug.append({"query": variant, **variant_debug})
            total_lexical += variant_debug["lexical_count"]
            total_vector += variant_debug["vector_count"]
            if variant_debug.get("vector_error"):
                vector_errors.append(variant_debug["vector_error"])
            for chunk in fused:
                existing = merged.get(chunk.chunk_id)
                if existing is None or chunk.score > existing.score:
                    merged[chunk.chunk_id] = chunk

        union = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        reranked, rerank_debug = self.reranker.rerank(query, union)
        deduped = self._reduce_duplicates(reranked)
        debug = {
            "queries": len(all_queries),
            "extra_queries": list(extra_queries or []),
            "lexical_count": total_lexical,
            "vector_count": total_vector,
            "fused_count": len(union),
            "returned_count": min(len(deduped), top_k),
            "vector_errors": vector_errors,
            "rerank": rerank_debug,
            "per_query": per_query_debug if extra_queries else None,
            "filters": {
                "scope": filters.scope,
                "allowed_visibilities": filters.allowed_visibilities,
                "source_names": filters.source_names,
                "source_types": filters.source_types,
                "section": filters.section,
                "repo_path_prefix": filters.repo_path_prefix,
                "filetypes": filters.filetypes,
            },
        }
        return deduped[:top_k], debug

    def _fuse_single_query(
        self,
        db: Session,
        query: str,
        filters: RetrievalFilters,
        limit: int,
    ) -> tuple[list[RetrievedChunk], dict]:
        # Run vector search in the background (Qdrant HTTP round-trip +
        # embedding inference) so it overlaps with the Postgres full-text
        # query on the request thread. The DB Session is not thread-safe,
        # so lexical stays on the caller's thread.
        vector_future = _VECTOR_POOL.submit(self.vector_store.search, query, filters, limit)
        lexical_results = self.lexical.search(db, query, filters, limit)
        vector_results: list[dict] = []
        vector_error: str | None = None
        try:
            vector_results = vector_future.result()
        except Exception as exc:  # pragma: no cover - exercised only when Qdrant is unavailable
            vector_error = str(exc)
            logger.warning("vector_search_failed", extra={"error": vector_error})

        lexical_scores = {chunk.chunk_id: chunk.lexical_score for chunk in lexical_results}
        vector_scores = {
            result["chunk_id"]: float(result["score"])
            for result in vector_results
            if result.get("chunk_id")
        }
        combined_ids = set(lexical_scores) | set(vector_scores)
        fetched = {
            chunk.chunk_id: chunk
            for chunk in self.lexical.fetch_by_chunk_ids(
                db,
                {
                    chunk_id: (vector_scores.get(chunk_id, 0.0), lexical_scores.get(chunk_id, 0.0))
                    for chunk_id in combined_ids
                },
            )
        }

        max_vector = max(vector_scores.values(), default=1.0) or 1.0
        max_lexical = max(lexical_scores.values(), default=1.0) or 1.0
        fused: list[RetrievedChunk] = []
        for chunk_id in combined_ids:
            chunk = fetched.get(chunk_id)
            if not chunk:
                continue
            normalized_vector = vector_scores.get(chunk_id, 0.0) / max_vector
            normalized_lexical = lexical_scores.get(chunk_id, 0.0) / max_lexical
            chunk.vector_score = vector_scores.get(chunk_id, 0.0)
            chunk.lexical_score = lexical_scores.get(chunk_id, 0.0)
            chunk.score = 0.62 * normalized_vector + 0.38 * normalized_lexical
            chunk.score += self._instructional_boost(query, chunk)
            chunk.score += self._freshness_boost(chunk)
            fused.append(chunk)

        return fused, {
            "lexical_count": len(lexical_results),
            "vector_count": len(vector_results),
            "vector_error": vector_error,
        }

    def _instructional_boost(self, query: str, chunk: RetrievedChunk) -> float:
        if not HOW_TO_RE.search(query):
            return 0.0
        haystack = " ".join(
            [
                chunk.title,
                chunk.repo_path or "",
                " ".join(chunk.section_path),
                " ".join(chunk.heading_path),
            ]
        ).lower()
        if any(term in haystack for term in ["tutorial", "guide", "how-to", "how to", "readme", "documentation"]):
            return 0.08
        return 0.0

    def _freshness_boost(self, chunk: RetrievedChunk) -> float:
        if not chunk.last_updated:
            return 0.0
        now = datetime.now(timezone.utc)
        last_updated = chunk.last_updated
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)
        age_days = max((now - last_updated).days, 0)
        if age_days > 730:
            return 0.0
        return max(0.0, 0.05 * (1.0 - age_days / 730.0))

    def _reduce_duplicates(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen_hashes: set[str] = set()
        seen_titles: set[str] = set()
        # Diversify: a single repo or document shouldn't flood the top
        # results. After the Nth chunk sharing a cluster key, decay the
        # score so chunks from other sources get a fair shot.
        cluster_counts: dict[str, int] = {}
        DIVERSITY_CAP = 2
        DIVERSITY_DECAY = 0.55
        output: list[RetrievedChunk] = []
        for chunk in chunks:
            key_hash = chunk.content_hash or ""
            title_key = f"{chunk.title.lower()}::{chunk.heading_path[-1].lower() if chunk.heading_path else ''}"
            if key_hash and key_hash in seen_hashes:
                continue
            if title_key in seen_titles and chunk.source_type == "github_repo":
                chunk.score *= 0.82
            cluster_key = self._cluster_key(chunk)
            count = cluster_counts.get(cluster_key, 0)
            if count >= DIVERSITY_CAP:
                chunk.score *= DIVERSITY_DECAY ** (count - DIVERSITY_CAP + 1)
            cluster_counts[cluster_key] = count + 1
            if key_hash:
                seen_hashes.add(key_hash)
            seen_titles.add(title_key)
            output.append(chunk)
        output.sort(key=lambda item: item.score, reverse=True)
        return output

    @staticmethod
    def _cluster_key(chunk: RetrievedChunk) -> str:
        # GitHub repos share a metadata.repo (e.g. "eic/zenodo-mcp-server");
        # use that so 5 README-like chunks from one repo don't crowd out
        # everything else. Fall back to document_id for other sources.
        repo = (chunk.metadata or {}).get("repo") if chunk.metadata else None
        if repo:
            return f"repo::{repo}"
        return f"doc::{chunk.document_id}"

