import json
import logging
import re
import time
from typing import Any, Protocol

import httpx

from app.core.config import Settings
from app.integrations.indico import INDICO_TOOL_SCHEMA, IndicoClient
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


class AnswerGenerator(Protocol):
    def generate(self, query: str, chunks: list[RetrievedChunk], min_support_score: float) -> tuple[str, dict]: ...


class ExtractiveAnswerGenerator:
    """Grounded fallback answer generator.

    It never invents facts outside retrieved snippets. It produces a short
    synthesis when support is strong enough and an explicit insufficient-support
    answer otherwise.
    """

    def generate(self, query: str, chunks: list[RetrievedChunk], min_support_score: float) -> tuple[str, dict]:
        if not chunks or chunks[0].score < min_support_score:
            return (
                "I couldn't find enough support in the indexed sources.",
                {"mode": "extractive", "support": "insufficient"},
            )

        snippets = [self._best_sentence(query, chunk.content) for chunk in chunks[:4]]
        snippets = [snippet for snippet in snippets if snippet]
        if not snippets:
            return (
                "I couldn't find enough support in the indexed sources.",
                {"mode": "extractive", "support": "insufficient"},
            )

        answer_lines = ["Based on the indexed sources:"]
        for index, snippet in enumerate(snippets[:3], start=1):
            title = chunks[index - 1].title
            answer_lines.append(f"{index}. {snippet} ({title})")
        return "\n".join(answer_lines), {"mode": "extractive", "support": "retrieved_snippets"}

    def _best_sentence(self, query: str, text: str) -> str:
        terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_+-]+", query) if len(term) > 2}
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences:
            return text[:360].strip()
        ranked = sorted(
            sentences,
            key=lambda sentence: sum(1 for term in terms if term in sentence.lower()),
            reverse=True,
        )
        candidate = ranked[0].strip()
        if len(candidate) > 420:
            candidate = candidate[:417].rstrip() + "..."
        return candidate


class HttpAnswerGenerator:
    """Generic HTTP generation adapter.

    Expected request:
        {"query": "...", "contexts": [{"title": "...", "url": "...", "content": "..."}], "model": "..."}

    Expected response:
        {"answer": "...", "metadata": {...}}
    """

    def __init__(self, url: str, model: str | None = None, timeout: float = 60.0) -> None:
        self.url = url
        self.model = model
        self.timeout = timeout

    def generate(self, query: str, chunks: list[RetrievedChunk], min_support_score: float) -> tuple[str, dict]:
        if not chunks or chunks[0].score < min_support_score:
            return (
                "I couldn't find enough support in the indexed sources.",
                {"mode": "http", "support": "insufficient"},
            )
        payload: dict[str, object] = {
            "query": query,
            "contexts": [
                {
                    "title": chunk.title,
                    "url": chunk.url,
                    "content": chunk.content,
                    "score": chunk.score,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
        }
        if self.model:
            payload["model"] = self.model
        response = httpx.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer")
        if not isinstance(answer, str):
            raise ValueError("Generation HTTP provider did not return an answer string")
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        metadata.setdefault("mode", "http")
        return answer, metadata


SYSTEM_PROMPT = (
    "You are a careful assistant for an enterprise knowledge base. "
    "Answer the user's question using ONLY the numbered sources provided. "
    "The user may use different vocabulary than the sources (e.g. 'flags' vs "
    "'tags', 'params' vs 'parameters', translations, abbreviations); treat "
    "these as synonyms and answer from the sources when the topic matches, "
    "even if the exact word is absent. "
    "Cite every factual claim with bracketed source numbers like [1] or [2][3]. "
    "When the question asks for a document, record, paper, report, or link "
    "(e.g. 'give me the TDR', 'where is the proposal', 'link to the draft'), "
    "and any of the sources is that primary record (its title matches the "
    "requested document — e.g. title contains 'Technical Design Report' for a "
    "TDR request, or the URL is a Zenodo/arXiv/DOI record for a paper request), "
    "lead the answer with that direct URL in a Markdown link and cite it. "
    "Do NOT summarize a secondary 'about' page when the primary record is "
    "present in the sources. "
    "Keep answers concise: 3-5 sentences, no preamble, no restating the question. "
    "Format shell commands, multi-line code, config snippets, and URLs the "
    "user should literally run/copy as Markdown fenced code blocks with a "
    "language identifier (```bash for shell, ```python, ```yaml, ```json, "
    "```xml, ```cpp, ```text for plain). Prefer fenced blocks even for a "
    "single shell command, since they render with a copy button. Use single "
    "backticks only for inline identifiers like tag names, flag names, "
    "variable names, or short file names within a sentence. "
    "If you can give a substantive answer from the sources, give it and stop — "
    "do not append a hedging disclaimer. "
    "Only when the sources contain nothing on the topic at all, reply exactly: "
    "\"I couldn't find enough support in the indexed sources.\" and say nothing else. "
    "Never invent URLs, numbers, names, or sources that are not listed below. "
    "TOOL RULES (very important): "
    "If the `search_indico_events` tool is available AND the question mentions "
    "a meeting, schedule, agenda, or any time reference — 'next', 'upcoming', "
    "'today', 'tomorrow', 'this week', 'recent', 'when is', 'what's on' — "
    "you MUST call search_indico_events BEFORE answering. The numbered sources "
    "are a static snapshot and NEVER contain meeting dates; trying to answer "
    "meeting/schedule questions from them will produce stale or wrong results. "
    "DISAMBIGUATION: 'ePIC/EICUG Collaboration Meeting' (quarterly/annual, "
    "category 455) is NOT the same as the 'ePIC Collaboration Council meeting'. "
    "When the user asks about a Collaboration Meeting or EICUG meeting, search "
    "Indico for keywords like 'collaboration' / 'EICUG' / 'joint' — do NOT cite "
    "the ePIC Council page from static sources as if it answered the question. "
    "Pick a sensible `query` keyword (e.g. 'DIRC' for 'DIRC meeting', "
    "'collaboration' or 'EICUG' for collaboration meetings), set `from_date` to "
    "'today' for upcoming-only and '-7d' for a retrospective, and `to_date` to "
    "'30d' for regular WG meetings, '365d' for collaboration meetings / annual "
    "reviews. When answering from the tool's events, quote the title, start "
    "time with its timezone exactly as returned, and include the event URL as "
    "a Markdown link. Do not fabricate dates."
)


class OpenAIAnswerGenerator:
    """OpenAI chat-completions answer generator.

    Uses citation-anchored prompting: each retrieved chunk is numbered and the
    model is instructed to cite with [N]. On any OpenAI error we fall back to
    the extractive generator so the /query endpoint never 5xxs on an LLM outage.
    """

    MAX_TOOL_ROUNDS = 2  # cap to keep pathological loops from running up tokens

    # Queries with these keywords almost always want a live schedule lookup;
    # force the tool on round 1 so the LLM can't wave it off and answer from
    # the (stale) static index. Intentionally narrow — we don't want to force
    # the tool on something like "meeting minutes from 2023".
    _FORCE_INDICO_KEYWORDS = (
        " meeting", "meetings", "agenda", "agendas", "schedule", "scheduled",
        "when is", "when's", "when are", "when will", "what's next", "what is next",
        "upcoming", "next week", "this week", "next month", "today's",
    )

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_output_tokens: int = 600,
        temperature: float | None = None,
        max_context_chars: int = 1800,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0,
        fallback: AnswerGenerator | None = None,
        indico_client: IndicoClient | None = None,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            timeout=timeout,
        )
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self._fallback = fallback or ExtractiveAnswerGenerator()
        self._indico = indico_client

    def generate(self, query: str, chunks: list[RetrievedChunk], min_support_score: float) -> tuple[str, dict]:
        # Indico tool-call flow: if the Indico integration is wired, expose
        # the tool so the model can pull live meeting data for time-relative
        # questions ("next DIRC meeting", "schedule this week"). For plain
        # content questions it simply won't call the tool and we behave
        # identically to the pre-tool path.
        indico_available = self._indico is not None
        if not indico_available and (not chunks or chunks[0].score < min_support_score):
            return (
                "I couldn't find enough support in the indexed sources.",
                {"mode": "openai", "model": self.model, "support": "insufficient"},
            )

        force_indico = False
        if indico_available:
            q_lower = f" {query.lower()} "
            if any(k in q_lower for k in self._FORCE_INDICO_KEYWORDS):
                force_indico = True

        # When we force the Indico tool, send the model ZERO static sources.
        # The static index has things like the "Collaboration Council" page
        # that look tantalizingly answer-shaped for meeting questions; the
        # model cannot resist them even with strong prompt directives. With
        # no static sources it has nothing to fall back on except the tool
        # output (or an explicit "no upcoming meetings" if the tool is empty).
        prompt_chunks = [] if force_indico else chunks
        user_prompt = self._build_user_prompt(query, prompt_chunks)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        base_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            base_kwargs["temperature"] = self.temperature
        if indico_available:
            base_kwargs["tools"] = [INDICO_TOOL_SCHEMA]
            base_kwargs["tool_choice"] = "auto"

        prompt_tokens_total = 0
        completion_tokens_total = 0
        total_tokens_total = 0
        tool_rounds: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        started = time.perf_counter()
        finish_reason: str | None = None
        answer = ""

        try:
            for round_idx in range(self.MAX_TOOL_ROUNDS + 1):
                round_kwargs = dict(base_kwargs)
                if round_idx == 0 and force_indico:
                    round_kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": "search_indico_events"},
                    }
                completion = self._client.chat.completions.create(messages=messages, **round_kwargs)
                usage = completion.usage
                prompt_tokens_total += getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens_total += getattr(usage, "completion_tokens", 0) or 0
                total_tokens_total += getattr(usage, "total_tokens", 0) or 0
                choice = completion.choices[0]
                finish_reason = choice.finish_reason
                tool_calls = getattr(choice.message, "tool_calls", None) or []

                if not tool_calls or round_idx == self.MAX_TOOL_ROUNDS:
                    answer = (choice.message.content or "").strip()
                    break

                # Append the model's tool-call request verbatim, then resolve each call.
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    tool_name = tc.function.name
                    try:
                        raw_args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        raw_args = {}
                    result = self._run_tool(tool_name, raw_args)
                    tool_rounds.append({"name": tool_name, "args": raw_args, "result_items": len(result.get("events", []))})
                    if tool_name == "search_indico_events":
                        for ev in result.get("events", []):
                            if isinstance(ev, dict):
                                tool_events.append(ev)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False)[:12000],
                        }
                    )
        except Exception as exc:
            logger.warning("OpenAI generation failed, falling back to extractive", extra={"error": str(exc)})
            fallback_answer, fallback_meta = self._fallback.generate(query, chunks, min_support_score)
            return fallback_answer, {
                "mode": "openai",
                "model": self.model,
                "fallback": fallback_meta.get("mode", "extractive"),
                "error": str(exc)[:200],
            }

        latency_ms = int((time.perf_counter() - started) * 1000)
        cost_usd = (
            prompt_tokens_total * self.input_cost_per_1m / 1_000_000
            + completion_tokens_total * self.output_cost_per_1m / 1_000_000
        )

        cited_indices = self._extract_citations(answer, len(chunks))
        metadata = {
            "mode": "openai",
            "model": self.model,
            "prompt_tokens": prompt_tokens_total,
            "completion_tokens": completion_tokens_total,
            "total_tokens": total_tokens_total,
            "cost_usd": round(cost_usd, 8) if cost_usd else 0.0,
            "latency_ms": latency_ms,
            "cited_source_indices": cited_indices,
            "contexts_sent": len(chunks),
            "finish_reason": finish_reason,
            "tool_rounds": tool_rounds,
            "indico_events": tool_events,
        }
        return answer, metadata

    def _run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "search_indico_events" and self._indico is not None:
            events = self._indico.search(
                query=args.get("query") or None,
                from_date=str(args.get("from_date") or "-7d"),
                to_date=str(args.get("to_date") or "30d"),
                limit=int(args.get("limit") or 10),
            )
            return {
                "category_url": self._indico.category_url,
                "now": self._indico.now_iso(),
                "count": len(events),
                "events": [e.to_prompt_dict() for e in events],
            }
        return {"error": f"unknown tool {name!r}"}

    def _build_user_prompt(self, query: str, chunks: list[RetrievedChunk]) -> str:
        from datetime import datetime, timezone as _tz
        now_iso = datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"CURRENT TIME: {now_iso}", f"QUESTION: {query}", ""]
        if chunks:
            lines.append("SOURCES:")
            for index, chunk in enumerate(chunks, start=1):
                content = chunk.content.strip()
                if len(content) > self.max_context_chars:
                    content = content[: self.max_context_chars - 3].rstrip() + "..."
                lines.append(f"[{index}] {chunk.title}")
                if chunk.url:
                    lines.append(f"URL: {chunk.url}")
                lines.append(content)
                lines.append("")
            lines.append("Answer in 3-5 sentences with [N] citations. If unsupported, say so exactly.")
        else:
            # Called for time-sensitive meeting queries — the generator
            # intentionally withholds static sources so the model relies
            # entirely on the search_indico_events tool.
            lines.append(
                "No pre-fetched sources are attached — this is a time-sensitive "
                "question. Call search_indico_events and answer using ONLY what "
                "the tool returns. If the tool returns zero events, reply exactly: "
                "\"No upcoming meetings found for that query.\""
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_citations(answer: str, max_index: int) -> list[int]:
        seen: list[int] = []
        for match in re.finditer(r"\[(\d+)\]", answer):
            index = int(match.group(1))
            if 1 <= index <= max_index and index not in seen:
                seen.append(index)
        return seen


def build_answer_generator(settings: Settings) -> AnswerGenerator:
    if settings.GENERATION_PROVIDER == "http":
        if not settings.GENERATION_HTTP_URL:
            raise ValueError("GENERATION_HTTP_URL is required when GENERATION_PROVIDER=http")
        return HttpAnswerGenerator(str(settings.GENERATION_HTTP_URL), model=settings.GENERATION_HTTP_MODEL)
    if settings.GENERATION_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when GENERATION_PROVIDER=openai")
        indico_client: IndicoClient | None = None
        if getattr(settings, "INDICO_ENABLED", False):
            indico_client = IndicoClient(
                category_url=settings.INDICO_CATEGORY_URL,
                cache_ttl_s=settings.INDICO_CACHE_TTL_S,
                timeout_s=settings.INDICO_TIMEOUT_S,
            )
        return OpenAIAnswerGenerator(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_GENERATION_MODEL,
            base_url=settings.OPENAI_BASE_URL,
            timeout=settings.OPENAI_REQUEST_TIMEOUT_S,
            max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
            temperature=settings.OPENAI_TEMPERATURE,
            max_context_chars=settings.OPENAI_MAX_CONTEXT_CHARS,
            input_cost_per_1m=settings.OPENAI_INPUT_COST_PER_1M,
            output_cost_per_1m=settings.OPENAI_OUTPUT_COST_PER_1M,
            indico_client=indico_client,
        )
    return ExtractiveAnswerGenerator()
