import concurrent.futures
import logging
import re
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.profile import load_synonyms
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


RECENCY_RE = re.compile(r"\b(most\s+recent|latest|newest|up[- ]to[- ]date|current\s+status|recently)\b", re.I)
# Domain acronym -> expansion. Applied as an EXTRA query variant (multi-
# query RRF fusion), so "give me TDR" also searches "technical design
# report" — the authoritative records spell the phrase out and never the
# acronym (real failure: the preTDR Zenodo record invisible to "TDR").
# Collaboration-specific: loaded from profiles/<PROFILE>/synonyms.yaml.
QUERY_SYNONYMS: dict[str, str] = load_synonyms()
_SYNONYM_RE = re.compile(r"[a-z0-9]+")


def expand_acronyms(query: str) -> str | None:
    """Return an expanded variant of the query, or None if nothing matched."""
    tokens = _SYNONYM_RE.findall(query.lower())
    replaced = False
    out: list[str] = []
    for token in tokens:
        expansion = QUERY_SYNONYMS.get(token)
        if expansion:
            out.append(expansion)
            replaced = True
        else:
            out.append(token)
    return " ".join(out) if replaced else None


HOW_TO_RE = re.compile(r"\b(how do i|how to|how can i|tutorial|guide|setup|set up|install|configure)\b", re.I)


class HybridRetriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        lexical: PostgresLexicalRetriever | None = None,
        reranker: Reranker | None = None,
        *,
        fusion: str = "rrf",
        expand_queries: bool = True,
        dedupe: bool = True,
        boosts: bool = True,
    ) -> None:
        self.vector_store = vector_store
        self.lexical = lexical or PostgresLexicalRetriever()
        self.reranker: Reranker = reranker or NullReranker()
        # Ablation switches. Defaults are the production configuration; the
        # evaluation sweep (app/eval/sweep.py) flips them to reconstruct the
        # original baseline pipeline and measure what each stage is worth.
        self.fusion = fusion
        self.expand_queries = expand_queries
        self.dedupe = dedupe
        self.boosts = boosts

    def search(
        self,
        db: Session,
        query: str,
        filters: RetrievalFilters,
        top_k: int,
        extra_queries: list[str] | None = None,
        rerank: bool = True,
    ) -> tuple[list[RetrievedChunk], dict]:
        limit = max(top_k * 4, 20)
        all_queries = [query] + list(extra_queries or [])
        expanded = expand_acronyms(query) if self.expand_queries else None
        if expanded and expanded not in all_queries:
            all_queries.append(expanded)
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
        # Rerank against the acronym-expanded phrasing as well — the CE
        # otherwise buries documents that spell the phrase out ("Preliminary
        # Technical Design Report") under material that echoes the acronym.
        rerank_query = f"{query} ({expanded})" if expanded else query
        if rerank:
            reranked, rerank_debug = self.reranker.rerank(rerank_query, union)
        else:
            reranked, rerank_debug = union, {"provider": "skipped", "reranked": 0, "reason": "preview"}
        deduped = self._reduce_duplicates(reranked, query) if self.dedupe else reranked
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

        # Reciprocal Rank Fusion. The previous weighted-score fusion
        # (0.62·vec + 0.38·lex on per-batch-normalized scores) structurally
        # buried single-engine hits: a chunk found only by lexical search
        # carries vector_score 0, so even lexical rank ~20 landed below the
        # LLM-context cutoff (real failure: the LFHCal 60-layer design chunk).
        # RRF is rank-based — an engine that ranks a chunk well keeps it
        # visible regardless of the other engine's score scale — and needs
        # no hand-tuned weights. K=60 is the standard damping constant.
        RRF_K = 60

        def _ranks(scores: dict[str, float]) -> dict[str, int]:
            return {
                chunk_id: rank
                for rank, (chunk_id, _) in enumerate(
                    sorted(scores.items(), key=lambda kv: kv[1], reverse=True), start=1
                )
            }

        vector_ranks = _ranks(vector_scores)
        lexical_ranks = _ranks(lexical_scores)
        # Normalize so the top fused score sits near 1.0 — keeps the boost
        # magnitudes and MIN_SUPPORT_SCORE semantics from the old scheme.
        max_rrf = 2.0 / (RRF_K + 1)
        # Batch maxima for the legacy weighted scheme (ablation arm only).
        max_vec = max(vector_scores.values(), default=0.0) or 1.0
        max_lex = max(lexical_scores.values(), default=0.0) or 1.0
        fused: list[RetrievedChunk] = []
        for chunk_id in combined_ids:
            chunk = fetched.get(chunk_id)
            if not chunk:
                continue
            chunk.vector_score = vector_scores.get(chunk_id, 0.0)
            chunk.lexical_score = lexical_scores.get(chunk_id, 0.0)
            if self.fusion == "weighted":
                chunk.score = (
                    0.62 * (chunk.vector_score / max_vec) + 0.38 * (chunk.lexical_score / max_lex)
                )
            else:
                rrf = 0.0
                if chunk_id in vector_ranks:
                    rrf += 1.0 / (RRF_K + vector_ranks[chunk_id])
                if chunk_id in lexical_ranks:
                    rrf += 1.0 / (RRF_K + lexical_ranks[chunk_id])
                chunk.score = rrf / max_rrf
            if self.boosts:
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

    # Source types whose documents represent *papers* — the same work
    # routinely enters via several of them (Zenodo upload, INSPIRE record,
    # arXiv preprint) and as multiple Zenodo versions. For these, a shared
    # normalized title is treated as the same work.
    _PAPER_SOURCE_TYPES = frozenset({"zenodo_community", "inspire_literature", "arxiv"})

    # Queries that actually ask about beam tests / prototypes.
    _TESTBEAM_QUERY_RE = re.compile(r"\b(test\s*beams?|beam\s*tests?|prototypes?|campaigns?|testbeams?)\b", re.I)
    _TESTBEAM_TITLE_RE = re.compile(r"\b(test\s*beams?|beam\s*tests?|testbeams?)\b", re.I)

    # "give me / where is / link to <document>" — the user wants the record
    # itself. Boost document repositories (Zenodo/INSPIRE/arXiv) and demote
    # meeting material: hundreds of slide decks MENTION the TDR; only the
    # repository holds it.
    _DOC_REQUEST_RE = re.compile(
        r"\b(give\s+me|where\s+is|link\s+to|find|download|get)\b.*"
        r"\b(tdr|report|proposal|paper|document|note|draft|technical\s+design)\b",
        re.I,
    )

    def _doc_request_adjustment(self, query: str, chunk: RetrievedChunk) -> float:
        if not self._DOC_REQUEST_RE.search(query):
            return 1.0
        if chunk.source_type in self._PAPER_SOURCE_TYPES:
            return 1.4
        if chunk.source_type == "indico":
            return 0.7
        return 1.0

    # Queries that actually want meeting/slide content rank slides normally.
    _MEETING_QUERY_RE = re.compile(
        r"\b(meeting|meetings|slides?|talks?|present(ed|ation)?s?|discuss(ed|ion)?s?|agenda|minutes)\b", re.I
    )

    @staticmethod
    def _is_slide_deck(chunk: RetrievedChunk) -> bool:
        """Slide-deck detection by CONTENT SHAPE, not source_type — so a deck
        uploaded to Zenodo is caught the same as one from Indico (real
        failure: a Zenodo 'Slides for Workshop' deck beat the Forward_Hcal
        wiki on 'how many layers in lfhcal?'). Slides are line-fragmented and
        sentence-poor: many short lines, few full stops.
        """
        if chunk.filetype not in ("pdf", "pptx"):
            return False
        title = (chunk.title or "").lower()
        if "slide" in title or "presentation" in title or chunk.source_type == "indico":
            return True
        text = chunk.content or ""
        if len(text) < 200:
            return False
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return False
        short_ratio = sum(1 for ln in lines if len(ln) < 45) / len(lines)
        sentence_density = text.count(". ") / max(len(text) / 500, 1)
        return short_ratio > 0.6 and sentence_density < 2.0

    def _slides_demotion(self, query: str, chunk: RetrievedChunk) -> float:
        """Demote slide decks (any source) for non-meeting factual queries.

        Decks are numerous and keyword-dense, so they crowd out curated
        design/docs pages. They remain first-class for
        "what was presented/discussed" asks.
        """
        if not self._is_slide_deck(chunk):
            return 1.0
        if self._MEETING_QUERY_RE.search(query):
            return 1.0
        return 0.85

    def _testbeam_demotion(self, query: str, chunk: RetrievedChunk) -> float:
        """Demote test-beam/prototype reports for generic detector questions.

        Test-beam wiki pages describe partial prototype configurations in
        design-like language ("the final 1.32 m design ... 32 layers") and
        routinely outrank the actual detector-design pages, misleading the
        generator (real failure: LFHCal layer count). When the query itself
        is about a test beam they rank normally.
        """
        if self._TESTBEAM_QUERY_RE.search(query):
            return 1.0
        haystack = " ".join([chunk.title] + list(chunk.section_path) + list(chunk.heading_path))
        if self._TESTBEAM_TITLE_RE.search(haystack):
            return 0.8
        return 1.0

    def _reduce_duplicates(self, chunks: list[RetrievedChunk], query: str = "") -> list[RetrievedChunk]:
        seen_hashes: set[str] = set()
        seen_titles: set[str] = set()
        # Cross-source identity: one document per *work*. Ownership prefers
        # the NEWEST version when candidates carry dates (Zenodo re-uploads,
        # arXiv v2+, revised TDRs have near-identical content and therefore
        # near-identical ranks — recency is the correct tiebreak, per the
        # supersedes-chain roadmap item); otherwise the best-scored document
        # claims the key. Chunks of the owning document pass through so
        # multi-chunk PDFs keep their in-document context.
        identity_owner: dict[str, str] = {}
        # Pre-pass: for every identity key, pick the newest dated document.
        newest_by_key: dict[str, tuple] = {}
        for chunk in chunks:
            for key in self._identity_keys(chunk):
                if chunk.last_updated is None:
                    continue
                current = newest_by_key.get(key)
                if current is None or chunk.last_updated > current[0]:
                    newest_by_key[key] = (chunk.last_updated, chunk.document_id)
        for key, (_, doc_id) in newest_by_key.items():
            identity_owner[key] = doc_id
        # Diversify: a single repo or document shouldn't flood the top
        # results. After the Nth chunk sharing a cluster key, decay the
        # score so chunks from other sources get a fair shot.
        # CAP 4 / DECAY 0.7 (was 2 / 0.55): the old compounding decay
        # crushed the 3rd+ chunk of a document to ~0.3× even when the
        # cross-encoder ranked it top-15 — for section-rich wiki pages the
        # extra chunks are usually complementary sections, not spam
        # (real failure: the LFHCal 60-layer design section).
        cluster_counts: dict[str, int] = {}
        DIVERSITY_CAP = 4
        DIVERSITY_DECAY = 0.7
        recency_intent = bool(query and RECENCY_RE.search(query))
        now = datetime.now(timezone.utc)
        output: list[RetrievedChunk] = []
        for chunk in chunks:
            chunk.score *= self._testbeam_demotion(query, chunk)
            if recency_intent and chunk.last_updated is not None:
                # "most recent / latest" queries: recency IS the ranking
                # criterion, and it must be applied AFTER the cross-encoder
                # (whose relevance score would otherwise drown it). Steep
                # 180-day horizon: this month's status deck outranks
                # keyword-denser material from last year.
                lu = chunk.last_updated if chunk.last_updated.tzinfo else chunk.last_updated.replace(tzinfo=timezone.utc)
                age_days = max((now - lu).days, 0)
                chunk.score += 0.35 * max(0.0, 1.0 - age_days / 180.0)
            chunk.score *= self._slides_demotion(query, chunk)
            chunk.score *= self._doc_request_adjustment(query, chunk)
            key_hash = chunk.content_hash or ""
            title_key = f"{chunk.title.lower()}::{chunk.heading_path[-1].lower() if chunk.heading_path else ''}"
            if key_hash and key_hash in seen_hashes:
                continue
            identity_keys = self._identity_keys(chunk)
            if identity_keys and any(
                identity_owner.get(key) not in (None, chunk.document_id)
                for key in identity_keys
            ):
                continue
            for key in identity_keys:
                identity_owner.setdefault(key, chunk.document_id)
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

    @classmethod
    def _identity_keys(cls, chunk: RetrievedChunk) -> list[str]:
        """Cross-source identity keys of the underlying *work* (may be []).

        A document claims every key it carries; a later document matching
        ANY already-claimed key is a duplicate. DOI and arXiv id are
        authoritative when present, but Zenodo re-uploads of the same talk
        mint fresh DOIs — so for paper-type sources the normalized title is
        a key too (meeting agendas and wiki pages legitimately share titles
        across documents, hence the source-type restriction).
        """
        meta = chunk.metadata or {}
        keys: list[str] = []
        doi = str(meta.get("doi") or "").strip().lower()
        if doi:
            keys.append(f"doi::{doi}")
        arxiv_id = str(meta.get("arxiv_id") or "").strip().lower()
        if arxiv_id:
            keys.append(f"arxiv::{arxiv_id}")
        if chunk.source_type in cls._PAPER_SOURCE_TYPES:
            normalized_title = " ".join(chunk.title.lower().split())
            if normalized_title:
                keys.append(f"title::{normalized_title}")
        return keys

    @staticmethod
    def _cluster_key(chunk: RetrievedChunk) -> str:
        # GitHub repos share a metadata.repo (e.g. "eic/zenodo-mcp-server");
        # use that so 5 README-like chunks from one repo don't crowd out
        # everything else. Fall back to document_id for other sources.
        repo = (chunk.metadata or {}).get("repo") if chunk.metadata else None
        if repo:
            return f"repo::{repo}"
        return f"doc::{chunk.document_id}"

