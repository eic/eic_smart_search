import logging
import re
import time
from typing import Any, Protocol

import httpx

from app.core.config import Settings
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
    "Never invent URLs, numbers, names, or sources that are not listed below."
)


class OpenAIAnswerGenerator:
    """OpenAI chat-completions answer generator.

    Uses citation-anchored prompting: each retrieved chunk is numbered and the
    model is instructed to cite with [N]. On any OpenAI error we fall back to
    the extractive generator so the /query endpoint never 5xxs on an LLM outage.
    """

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

    def generate(self, query: str, chunks: list[RetrievedChunk], min_support_score: float) -> tuple[str, dict]:
        if not chunks or chunks[0].score < min_support_score:
            return (
                "I couldn't find enough support in the indexed sources.",
                {"mode": "openai", "model": self.model, "support": "insufficient"},
            )

        user_prompt = self._build_user_prompt(query, chunks)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.warning("OpenAI generation failed, falling back to extractive", extra={"error": str(exc)})
            answer, fallback_meta = self._fallback.generate(query, chunks, min_support_score)
            return answer, {
                "mode": "openai",
                "model": self.model,
                "fallback": fallback_meta.get("mode", "extractive"),
                "error": str(exc)[:200],
            }

        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = (completion.choices[0].message.content or "").strip()
        usage = completion.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
        cost_usd = (
            prompt_tokens * self.input_cost_per_1m / 1_000_000
            + completion_tokens * self.output_cost_per_1m / 1_000_000
        )

        cited_indices = self._extract_citations(answer, len(chunks))
        metadata = {
            "mode": "openai",
            "model": self.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost_usd, 8) if cost_usd else 0.0,
            "latency_ms": latency_ms,
            "cited_source_indices": cited_indices,
            "contexts_sent": len(chunks),
            "finish_reason": completion.choices[0].finish_reason,
        }
        return answer, metadata

    def _build_user_prompt(self, query: str, chunks: list[RetrievedChunk]) -> str:
        lines = [f"QUESTION: {query}", "", "SOURCES:"]
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
        )
    return ExtractiveAnswerGenerator()
