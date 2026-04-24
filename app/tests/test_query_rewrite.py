from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.retrieval.query_rewrite import NullQueryRewriter, OpenAIQueryRewriter


@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Usage:
    prompt_tokens: int = 40
    completion_tokens: int = 20
    total_tokens: int = 60


@dataclass
class _Completion:
    choices: list[_Choice]
    usage: _Usage


class _Completions:
    def __init__(self, completion: _Completion | Exception) -> None:
        self._completion = completion
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        if isinstance(self._completion, Exception):
            raise self._completion
        return self._completion


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, completion: _Completion | Exception) -> None:
        self.chat = _Chat(_Completions(completion))


def _build(client: _Client, **overrides: Any) -> OpenAIQueryRewriter:
    from collections import OrderedDict

    rewriter = OpenAIQueryRewriter.__new__(OpenAIQueryRewriter)
    rewriter._client = client  # type: ignore[attr-defined]
    rewriter.name = "openai:test"
    rewriter.model = overrides.get("model", "gpt-5.4-nano")
    rewriter.max_variants = overrides.get("max_variants", 2)
    rewriter.trigger_max_words = overrides.get("trigger_max_words", 8)
    rewriter.max_output_tokens = overrides.get("max_output_tokens", 200)
    rewriter.temperature = overrides.get("temperature", None)
    rewriter._cache = OrderedDict()  # type: ignore[attr-defined]
    rewriter._cache_size = overrides.get("cache_size", 16)  # type: ignore[attr-defined]
    return rewriter


def test_null_rewriter_returns_no_variants() -> None:
    variants, debug = NullQueryRewriter().rewrite("anything")
    assert variants == []
    assert debug["skipped"] is True


def test_long_query_is_skipped_without_api_call() -> None:
    client = _Client(_Completion([_Choice(_Msg("x\ny"))], _Usage()))
    rewriter = _build(client, trigger_max_words=5)

    variants, debug = rewriter.rewrite("this query has many words and is quite long indeed")

    assert variants == []
    assert debug["skipped"] is True
    assert debug["reason"] == "too_long"
    assert client.chat.completions.calls == []


def test_empty_query_is_skipped() -> None:
    client = _Client(_Completion([_Choice(_Msg(""))], _Usage()))
    rewriter = _build(client)

    variants, debug = rewriter.rewrite("   ")

    assert variants == []
    assert debug["reason"] == "empty"
    assert client.chat.completions.calls == []


def test_variants_parsed_and_deduped() -> None:
    raw = "installation instructions\nsetup guide\n- getting started\nhow to install\nhow to install"
    client = _Client(_Completion([_Choice(_Msg(raw))], _Usage()))
    rewriter = _build(client, max_variants=3)

    variants, debug = rewriter.rewrite("how to install")

    # "how to install" matches original (case-insensitive) and should be excluded.
    assert "installation instructions" in variants
    assert "setup guide" in variants
    assert "getting started" in variants
    assert all(v.lower() != "how to install" for v in variants)
    assert len(variants) == 3
    assert debug["variants"] == 3
    assert debug["cache"] == "miss"


def test_variants_cap_at_max_variants() -> None:
    raw = "\n".join(f"variant {i}" for i in range(10))
    client = _Client(_Completion([_Choice(_Msg(raw))], _Usage()))
    rewriter = _build(client, max_variants=2)

    variants, _ = rewriter.rewrite("q")

    assert len(variants) == 2


def test_cache_returns_hit_on_repeat_query_without_api_call() -> None:
    raw = "alpha\nbeta"
    client = _Client(_Completion([_Choice(_Msg(raw))], _Usage()))
    rewriter = _build(client, max_variants=2)

    first_variants, first_debug = rewriter.rewrite("repeated")
    second_variants, second_debug = rewriter.rewrite("repeated")

    assert first_variants == second_variants == ["alpha", "beta"]
    assert first_debug["cache"] == "miss"
    assert second_debug["cache"] == "hit"
    assert len(client.chat.completions.calls) == 1


def test_api_error_returns_empty_variants() -> None:
    client = _Client(RuntimeError("rate limit"))
    rewriter = _build(client)

    variants, debug = rewriter.rewrite("short")

    assert variants == []
    assert debug["skipped"] is True
    assert debug["reason"] == "error"
    assert "rate limit" in debug["error"]


def test_respects_temperature_flag() -> None:
    client = _Client(_Completion([_Choice(_Msg("a\nb"))], _Usage()))
    rewriter_no_temp = _build(client, temperature=None)
    rewriter_no_temp.rewrite("q1")
    assert "temperature" not in client.chat.completions.calls[0]

    client2 = _Client(_Completion([_Choice(_Msg("a\nb"))], _Usage()))
    rewriter_with_temp = _build(client2, temperature=0.3)
    rewriter_with_temp.rewrite("q2")
    assert client2.chat.completions.calls[0]["temperature"] == 0.3


def test_rejects_overly_long_variants() -> None:
    long_variant = " ".join(["word"] * 30)
    raw = f"good alternative\n{long_variant}\nshort"
    client = _Client(_Completion([_Choice(_Msg(raw))], _Usage()))
    rewriter = _build(client, max_variants=5)

    variants, _ = rewriter.rewrite("q")

    assert "good alternative" in variants
    assert "short" in variants
    assert long_variant not in variants
