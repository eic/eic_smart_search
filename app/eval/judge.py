"""LLM-as-a-judge scoring for generated answers.

One cheap-model call per case, JSON verdict with three 1-5 scores:

  relevance     — does the answer address the question?
  faithfulness  — is every claim supported by the provided sources?
  completeness  — does it cover the key points the sources allow?

Failures return None — the eval run records "judge unavailable" for the
case instead of dying.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import Settings

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You are a strict evaluator of a retrieval-augmented search assistant for "
    "the Electron-Ion Collider (EIC/ePIC) community. Score the ANSWER for the "
    "QUESTION against the SOURCES on three axes, integers 1-5:\n"
    "  relevance: does it address the question? (1=off-topic, 5=direct)\n"
    "  faithfulness: is every factual claim supported by the sources? "
    "(1=fabricated, 5=fully grounded)\n"
    "  completeness: does it cover what the sources allow? (1=misses the "
    "point, 5=complete)\n"
    "Judge only against the sources — do not reward outside knowledge.\n"
    'Reply with ONLY a JSON object: {"relevance": n, "faithfulness": n, '
    '"completeness": n, "comment": "<one short sentence>"}'
)


class AnswerJudge:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        # Judge must not be the same model that wrote the answer (it would
        # grade itself). Fall back to a stronger default, not the generator.
        self.model = (
            getattr(settings, "EVAL_JUDGE_MODEL", None)
            or "gpt-5.4-mini"
        )
        # Explicit base_url: an empty OPENAI_BASE_URL env var would otherwise
        # override the SDK default with "" and break every request.
        self._client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or "https://api.openai.com/v1",
            timeout=30,
        )

    def judge(self, query: str, answer: str, snippets: list[str]) -> dict[str, Any] | None:
        if not answer.strip():
            return None
        sources = "\n\n".join(
            f"SOURCE [{i + 1}]:\n{snippet[:1200]}" for i, snippet in enumerate(snippets[:8])
        )
        user = f"QUESTION: {query}\n\n{sources}\n\nANSWER:\n{answer[:6000]}"
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=2000,
            )
            raw = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("judge_call_failed", extra={"error": str(exc)[:200]})
            return None
        return _parse_verdict(raw)


def _parse_verdict(raw: str) -> dict[str, Any] | None:
    """Parse the judge JSON, tolerating markdown fences and stray text."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    verdict: dict[str, Any] = {}
    for key in ("relevance", "faithfulness", "completeness"):
        try:
            verdict[key] = max(1, min(5, int(data.get(key))))
        except (TypeError, ValueError):
            return None
    comment = data.get("comment")
    verdict["comment"] = str(comment)[:300] if comment else ""
    return verdict
