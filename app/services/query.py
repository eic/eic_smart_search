import re
import time
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.llm.generation import AnswerGenerator
from app.models.entities import QueryLog
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.parent_expand import ParentExpansionConfig, expand_parents
from app.retrieval.query_rewrite import NullQueryRewriter, QueryRewriter
from app.retrieval.types import RetrievedChunk, RetrievalFilters
from app.schemas.api import Citation, Filters, QueryRequest, QueryResponse
from app.services.query_cache import QueryCache


class QueryService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        answer_generator: AnswerGenerator,
        query_rewriter: QueryRewriter | None = None,
        query_cache: QueryCache | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.parent_expansion = ParentExpansionConfig.from_settings(settings)
        self.query_rewriter: QueryRewriter = query_rewriter or NullQueryRewriter()
        self.query_cache: QueryCache | None = query_cache

    def query(self, db: Session, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        filters_payload = request.filters.model_dump()

        # Cache probe. On hit we skip retrieval / rewrite / generation entirely
        # and just record a query log for analytics. Meeting / schedule queries
        # bypass both the probe and the write below — Indico data moves in
        # real time, so an hour-old cached "next meeting" answer is worse than
        # no answer. The IndicoClient's own 5-minute cache still deduplicates
        # rapid repeat calls at the HTTP layer.
        looks_time_sensitive = self._is_time_sensitive_query(request.query)
        if self.query_cache is not None and not looks_time_sensitive:
            cached = self.query_cache.get(
                request.query, request.scope, request.top_k, request.generate_answer, filters_payload
            )
            if cached is not None:
                return self._hit(db, request, filters_payload, cached, started)

        filters = self._filters(request.scope, request.filters)
        # Only pay the rewriter hop on submit (generate_answer=true). Preview
        # fires on every keystroke, and burning one OpenAI call per character
        # turns typing latency into 1-2s per stroke with no win: the extra
        # paraphrases only really matter when the LLM is about to read the
        # sources anyway.
        if request.generate_answer:
            variants, rewrite_debug = self.query_rewriter.rewrite(request.query)
        else:
            variants, rewrite_debug = [], {"skipped": "preview"}
        # When generating an answer, fetch a broader pool (LLM_CONTEXT_TOP_K)
        # so the model sees more supporting context, then we truncate the
        # returned citations to request.top_k after generation.
        llm_context_k = int(getattr(self.settings, "LLM_CONTEXT_TOP_K", 0) or 0)
        retrieval_k = (
            max(request.top_k, llm_context_k)
            if request.generate_answer and llm_context_k > 0
            else request.top_k
        )
        chunks, retrieval_debug = self.retriever.search(
            db, request.query, filters, retrieval_k, extra_queries=variants
        )
        retrieval_debug["query_rewrite"] = rewrite_debug
        chunks, parent_debug = expand_parents(db, chunks, self.parent_expansion)
        retrieval_debug["parent_expansion"] = parent_debug
        answer = ""
        generation_debug: dict[str, Any] = {"enabled": request.generate_answer}
        if request.generate_answer:
            answer, generation_debug = self.answer_generator.generate(
                request.query,
                chunks,
                min_support_score=self.settings.MIN_SUPPORT_SCORE,
            )
            # If the LLM invoked the Indico tool and the answer actually
            # references those event URLs, show the events as citations
            # instead of the (irrelevant) static retrieval cards. Detect by
            # scanning the answer for event URLs that appear in the tool's
            # returned events.
            indico_chunks = self._indico_citations(answer, generation_debug.get("indico_events") or [])
            if indico_chunks:
                answer, chunks = self._align_citations(answer, indico_chunks, request.top_k)
                retrieval_debug["indico_citations"] = len(chunks)
            else:
                # Reorder chunks so LLM-cited ones come first (in citation
                # order), then pad with top-scored non-cited chunks to
                # request.top_k. The answer's [N] markers are remapped so
                # they still resolve against the trimmed citation list.
                answer, chunks = self._align_citations(answer, chunks, request.top_k)
            retrieval_debug["llm_context_sent"] = generation_debug.get("contexts_sent", len(chunks))
            retrieval_debug["citations_returned"] = len(chunks)
        else:
            chunks = chunks[: request.top_k]
        citations = [self._citation(request.query, chunk) for chunk in chunks]
        latency_ms = int((time.perf_counter() - started) * 1000)
        debug = {**retrieval_debug, "generation": generation_debug, "latency_ms": latency_ms, "cache_hit": False}
        top_score = chunks[0].score if chunks else None
        # Don't persist retrieval-only preview calls (generate_answer=false):
        # the typing-debounced prefires from the widget would otherwise flood
        # analytics with partial-word rows ("wha", "what ar", …) that carry
        # no user intent beyond telemetry.
        query_log = QueryLog(
            query=request.query,
            scope=request.scope,
            filters=filters_payload,
            top_k=request.top_k,
            answer_generated=request.generate_answer,
            result_count=len(citations),
            latency_ms=latency_ms,
            retrieval_debug=debug,
            embedding_provider=getattr(self.settings, "EMBEDDING_PROVIDER", None),
            embedding_model=getattr(self.settings, "EMBEDDING_MODEL", None),
            generation_provider=getattr(self.settings, "GENERATION_PROVIDER", None) if request.generate_answer else None,
            generation_model=generation_debug.get("model") if request.generate_answer else None,
            prompt_tokens=generation_debug.get("prompt_tokens"),
            completion_tokens=generation_debug.get("completion_tokens"),
            total_tokens=generation_debug.get("total_tokens"),
            cost_usd=generation_debug.get("cost_usd"),
            confidence=self._confidence(top_score),
            top_score=top_score,
            answer=(answer or None) if request.generate_answer else None,
        )
        if request.generate_answer:
            db.add(query_log)
            db.commit()
            log_id = query_log.id
        else:
            log_id = None

        # Only cache substantive, successful responses — don't cache insufficient-support
        # refusals or zero-result queries (those are cheap to re-run anyway).
        # Also bypass the cache whenever the Indico live tool was invoked:
        # meeting schedules change (new events, rescheduled times, cancelled
        # slots) and we do not want to replay a 1h-old answer as "next". The
        # IndicoClient has its own short in-process cache (~5 min) that still
        # de-duplicates the HTTP call within a rapid burst.
        tool_was_used = bool(generation_debug.get("tool_rounds"))
        if (
            self.query_cache is not None
            and citations
            and not tool_was_used
            and (not request.generate_answer or generation_debug.get("mode") not in ("openai", "http", "extractive") or generation_debug.get("support") != "insufficient")
        ):
            self.query_cache.set(
                request.query,
                request.scope,
                request.top_k,
                request.generate_answer,
                filters_payload,
                payload={
                    "answer": answer,
                    "citations": [c.model_dump() for c in citations],
                    "top_score": top_score,
                },
            )

        return QueryResponse(answer=answer, citations=citations, retrieval_debug=debug, query_log_id=log_id)

    def _hit(
        self,
        db: Session,
        request: QueryRequest,
        filters_payload: dict[str, Any],
        cached: dict[str, Any],
        started: float,
    ) -> QueryResponse:
        payload = cached["payload"]
        age_s = cached["age_s"]
        answer = payload.get("answer", "")
        citations = [Citation(**c) for c in payload.get("citations", [])]
        top_score = payload.get("top_score")
        latency_ms = int((time.perf_counter() - started) * 1000)
        debug = {
            "cache_hit": True,
            "cache_age_s": round(age_s, 3),
            "latency_ms": latency_ms,
            "generation": {"enabled": request.generate_answer, "mode": "cached"},
            "returned_count": len(citations),
        }
        query_log = QueryLog(
            query=request.query,
            scope=request.scope,
            filters=filters_payload,
            top_k=request.top_k,
            answer_generated=request.generate_answer,
            result_count=len(citations),
            latency_ms=latency_ms,
            retrieval_debug=debug,
            embedding_provider=getattr(self.settings, "EMBEDDING_PROVIDER", None),
            embedding_model=getattr(self.settings, "EMBEDDING_MODEL", None),
            generation_provider="cache" if request.generate_answer else None,
            generation_model=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            confidence=self._confidence(top_score),
            top_score=top_score,
            answer=(answer or None) if request.generate_answer else None,
        )
        log_id = None
        if request.generate_answer:
            db.add(query_log)
            db.commit()
            log_id = query_log.id
        return QueryResponse(answer=answer, citations=citations, retrieval_debug=debug, query_log_id=log_id)

    def _confidence(self, top_score: float | None) -> str:
        if top_score is None:
            return "none"
        min_support = self.settings.MIN_SUPPORT_SCORE
        if top_score < min_support:
            return "low"
        if top_score < min_support * 2.5:
            return "medium"
        return "high"

    def _filters(self, scope: str, filters: Filters) -> RetrievalFilters:
        return RetrievalFilters(
            scope=scope,
            source_names=filters.source_names,
            source_types=filters.source_types,
            section=filters.section,
            repo_path_prefix=filters.repo_path_prefix,
            filetypes=filters.filetypes,
        )

    def _citation(self, query: str, chunk: RetrievedChunk) -> Citation:
        return Citation(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            url=chunk.url,
            snippet=self._snippet(query, chunk.content),
            source_name=chunk.source_name,
            source_type=chunk.source_type,
            score=round(chunk.score, 6),
            metadata={
                **chunk.metadata,
                "visibility": chunk.visibility,
                "repo_path": chunk.repo_path,
                "filetype": chunk.filetype,
                "section_path": chunk.section_path,
                "heading_path": chunk.heading_path,
                "vector_score": chunk.vector_score,
                "lexical_score": chunk.lexical_score,
                "last_updated": chunk.last_updated.isoformat() if chunk.last_updated else None,
            },
        )

    # Keywords that should always hit a live lookup, never the response cache.
    # Kept in sync with OpenAIAnswerGenerator._FORCE_INDICO_KEYWORDS — this
    # is the query-side mirror so the cache doesn't short-circuit the call
    # before the generator ever runs.
    _TIME_SENSITIVE_KEYWORDS: tuple[str, ...] = (
        " meeting", "meetings", "agenda", "agendas", "schedule", "scheduled",
        "when is", "when's", "when are", "when will", "what's next", "what is next",
        "upcoming", "next week", "this week", "next month", "today's",
    )

    @classmethod
    def _is_time_sensitive_query(cls, query: str) -> bool:
        q = f" {query.lower()} "
        return any(k in q for k in cls._TIME_SENSITIVE_KEYWORDS)

    @staticmethod
    def _indico_citations(answer: str, events: list[dict[str, Any]]) -> list[RetrievedChunk]:
        """Build virtual RetrievedChunks for the Indico events the LLM cited.

        Returns [] when the answer doesn't reference any of the tool's events
        — in that case the caller keeps the normal static citation flow. When
        non-empty, the list preserves the order the events first appear in
        the answer text, so [1] in the answer maps to the first-mentioned
        event after `_align_citations` runs.
        """
        if not events:
            return []
        by_id: dict[str, dict[str, Any]] = {}
        for ev in events:
            ev_id = str(ev.get("id") or "")
            if ev_id and ev_id not in by_id:
                by_id[ev_id] = ev
        if not by_id:
            return []
        # Indico event URLs look like https://<host>/event/<id>/
        cited_ids: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"https?://[^\s)\"']*indico[^\s)\"']*/event/(\d+)", answer or ""):
            ev_id = match.group(1)
            if ev_id in by_id and ev_id not in seen:
                seen.add(ev_id)
                cited_ids.append(ev_id)
        if not cited_ids:
            return []
        chunks: list[RetrievedChunk] = []
        for ev_id in cited_ids:
            ev = by_id[ev_id]
            title = str(ev.get("title") or "")
            category = str(ev.get("category") or "")
            display_title = f"{title} — {category}" if category and category not in title else title
            content_parts = [
                ev.get("start") and f"Start: {ev['start']}",
                ev.get("end") and f"End: {ev['end']}",
                ev.get("location") and f"Location: {ev['location']}",
                ev.get("description"),
            ]
            content = "\n".join([p for p in content_parts if p])
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"indico-{ev_id}",
                    document_id=f"indico-{ev_id}",
                    source_name="indico_live",
                    source_type="indico_event",
                    title=display_title or f"Indico event {ev_id}",
                    url=str(ev.get("url") or f"https://indico.bnl.gov/event/{ev_id}/"),
                    content=content,
                    score=1.0,
                    metadata={
                        "category": category,
                        "start": ev.get("start"),
                        "end": ev.get("end"),
                        "location": ev.get("location"),
                        "live": True,
                    },
                )
            )
        return chunks

    @staticmethod
    def _align_citations(
        answer: str, chunks: list[RetrievedChunk], display_k: int
    ) -> tuple[str, list[RetrievedChunk]]:
        """Reorder `chunks` so cited ones lead, cap to `display_k`, remap [N] markers.

        The LLM saw a larger pool than the user will see. We take the chunks
        it actually cited (in citation order, deduped) first, then fill up to
        `display_k` with the highest-scored non-cited chunks so every widget
        slot still has content. The answer's bracketed references are
        rewritten to match the new positions; any reference pointing outside
        the trimmed list is dropped.
        """
        if not chunks:
            return answer, chunks
        cited_order: list[int] = []
        seen: set[int] = set()
        for match in re.finditer(r"\[(\d+)\]", answer or ""):
            idx = int(match.group(1))
            if 1 <= idx <= len(chunks) and idx not in seen:
                seen.add(idx)
                cited_order.append(idx)
        new_chunks: list[RetrievedChunk] = [chunks[i - 1] for i in cited_order[:display_k]]
        if len(new_chunks) < display_k:
            for i, chunk in enumerate(chunks, start=1):
                if i in seen:
                    continue
                new_chunks.append(chunk)
                if len(new_chunks) >= display_k:
                    break
        old_to_new: dict[int, int] = {old: new + 1 for new, old in enumerate(cited_order[:display_k])}

        def _remap(match: re.Match[str]) -> str:
            old = int(match.group(1))
            new = old_to_new.get(old)
            return f"[{new}]" if new else ""

        remapped = re.sub(r"\[(\d+)\]", _remap, answer or "")
        return remapped, new_chunks

    def _snippet(self, query: str, content: str, max_chars: int = 520) -> str:
        terms = [term.lower() for term in re.findall(r"[A-Za-z0-9_+-]+", query) if len(term) > 2]
        lowered = content.lower()
        positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
        if positions:
            center = min(positions)
            start = max(0, center - max_chars // 3)
        else:
            start = 0
        end = min(len(content), start + max_chars)
        snippet = content[start:end].strip()
        if start > 0:
            snippet = "... " + snippet
        if end < len(content):
            snippet = snippet.rstrip() + " ..."
        return snippet

