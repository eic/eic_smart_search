from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.llm.generation import ExtractiveAnswerGenerator, OpenAIAnswerGenerator
from app.retrieval.types import RetrievedChunk


def _chunk(score: float, title: str = "Tutorials", content: str = "Run examples via make query.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        source_name="eic_website",
        source_type="website",
        title=title,
        url="https://example.com/t",
        content=content,
        score=score,
    )


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str = "stop"


@dataclass
class _FakeUsage:
    prompt_tokens: int = 120
    completion_tokens: int = 45
    total_tokens: int = 165


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]
    usage: _FakeUsage


class _FakeChatCompletions:
    def __init__(self, completion: _FakeCompletion | Exception) -> None:
        self._completion = completion
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):  # noqa: ANN401 — matches OpenAI signature
        self.calls.append(kwargs)
        if isinstance(self._completion, Exception):
            raise self._completion
        return self._completion


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completion: _FakeCompletion | Exception) -> None:
        self.chat = _FakeChat(_FakeChatCompletions(completion))


def _build_generator(client: _FakeClient, **overrides: Any) -> OpenAIAnswerGenerator:
    generator = OpenAIAnswerGenerator.__new__(OpenAIAnswerGenerator)
    generator._client = client  # type: ignore[attr-defined]
    generator.model = overrides.get("model", "gpt-5.4-nano")
    generator.max_output_tokens = overrides.get("max_output_tokens", 600)
    generator.temperature = overrides.get("temperature", None)
    generator.max_context_chars = overrides.get("max_context_chars", 1800)
    generator.input_cost_per_1m = overrides.get("input_cost_per_1m", 0.40)
    generator.output_cost_per_1m = overrides.get("output_cost_per_1m", 1.60)
    generator._fallback = overrides.get("fallback", ExtractiveAnswerGenerator())  # type: ignore[attr-defined]
    return generator


def test_openai_generator_returns_insufficient_below_min_support() -> None:
    client = _FakeClient(_FakeCompletion([_FakeChoice(_FakeMessage("ignored"))], _FakeUsage()))
    generator = _build_generator(client)

    answer, metadata = generator.generate("irrelevant", [_chunk(score=0.05)], min_support_score=0.16)

    assert "couldn't find enough support" in answer
    assert metadata["support"] == "insufficient"
    assert client.chat.completions.calls == []


def test_openai_generator_records_usage_and_citations() -> None:
    completion = _FakeCompletion(
        choices=[_FakeChoice(_FakeMessage("The setup is in [1] and the tutorial in [2]."))],
        usage=_FakeUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250),
    )
    client = _FakeClient(completion)
    generator = _build_generator(client, input_cost_per_1m=0.40, output_cost_per_1m=1.60)

    chunks = [_chunk(score=0.8, title="Setup"), _chunk(score=0.7, title="Tutorial")]
    answer, metadata = generator.generate("How do I run tutorials?", chunks, min_support_score=0.16)

    assert "[1]" in answer and "[2]" in answer
    assert metadata["mode"] == "openai"
    assert metadata["prompt_tokens"] == 200
    assert metadata["completion_tokens"] == 50
    assert metadata["total_tokens"] == 250
    assert metadata["cited_source_indices"] == [1, 2]
    assert metadata["contexts_sent"] == 2
    # 200/1M * 0.40 + 50/1M * 1.60 = 0.00008 + 0.00008 = 0.00016
    assert metadata["cost_usd"] == pytest.approx(0.00016, rel=1e-6)

    sent = client.chat.completions.calls[0]
    assert sent["model"] == "gpt-5.4-nano"
    assert "temperature" not in sent
    assert sent["max_completion_tokens"] == 600
    assert sent["messages"][0]["role"] == "system"
    assert "[1]" in sent["messages"][1]["content"]
    assert "[2]" in sent["messages"][1]["content"]


def test_openai_generator_sends_temperature_when_configured() -> None:
    completion = _FakeCompletion([_FakeChoice(_FakeMessage("ok [1]"))], _FakeUsage())
    client = _FakeClient(completion)
    generator = _build_generator(client, temperature=0.2)

    generator.generate("q", [_chunk(score=0.8)], min_support_score=0.16)

    sent = client.chat.completions.calls[0]
    assert sent["temperature"] == 0.2


def test_openai_generator_falls_back_to_extractive_on_error() -> None:
    client = _FakeClient(RuntimeError("openai unreachable"))
    generator = _build_generator(client)

    chunks = [_chunk(score=0.8, content="Run make dev to start the local stack.")]
    answer, metadata = generator.generate("how to start", chunks, min_support_score=0.16)

    assert metadata["mode"] == "openai"
    assert metadata["fallback"] == "extractive"
    assert "openai unreachable" in metadata["error"]
    # Extractive fallback produces a "Based on the indexed sources" prefix when support is sufficient.
    assert "indexed sources" in answer.lower() or "make dev" in answer.lower()


def test_openai_generator_truncates_long_context() -> None:
    client = _FakeClient(_FakeCompletion([_FakeChoice(_FakeMessage("ok [1]"))], _FakeUsage()))
    generator = _build_generator(client, max_context_chars=50)

    long_content = "x" * 500
    chunks = [_chunk(score=0.8, content=long_content)]

    generator.generate("q", chunks, min_support_score=0.16)

    user_prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    # The truncated content block should not contain 500 x's; verify it was shortened.
    assert "x" * 500 not in user_prompt
    assert "..." in user_prompt
