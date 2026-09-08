"""Unit tests for the evaluation harness (no network, no DB)."""
from __future__ import annotations

from app.eval.judge import _parse_verdict
from app.eval.runner import (
    GOLDEN_PATH,
    _aggregate,
    _citation_markers_valid,
    _citation_rank,
    load_golden,
)
from app.schemas.api import Citation, QueryResponse


def _resp(urls: list[str]) -> QueryResponse:
    return QueryResponse(
        answer="",
        citations=[
            Citation(
                chunk_id=f"c{i}", document_id=f"d{i}", title=f"T{i}", url=u,
                snippet="", source_name="s", source_type="t", score=1.0, metadata={},
            )
            for i, u in enumerate(urls)
        ],
        retrieval_debug={},
        query_log_id=None,
    )


def test_golden_set_loads_and_is_wellformed() -> None:
    cases = load_golden(GOLDEN_PATH)
    assert len(cases) >= 10
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for case in cases:
        assert case.query.strip()
        # Negative/abstention cases have no source (the correct answer is a
        # refusal) — they assert on answer text instead.
        if "negative" in case.tags or "abstention" in case.tags:
            assert case.expect_answer_contains, f"{case.id} negative case needs answer expectation"
        else:
            assert case.expect_url_contains, f"{case.id} has no retrieval expectation"


def test_citation_rank_first_match_wins() -> None:
    resp = _resp([
        "https://example.com/a",
        "https://zenodo.org/records/1",
        "https://inspirehep.net/literature/2",
    ])
    assert _citation_rank(resp, ["zenodo.org", "inspirehep.net"]) == 2
    assert _citation_rank(resp, ["arxiv.org"]) is None
    assert _citation_rank(resp, []) is None


def test_citation_markers_valid() -> None:
    assert _citation_markers_valid("Claim [1] and [3].", 3) is True
    assert _citation_markers_valid("Claim [4].", 3) is False
    assert _citation_markers_valid("No markers at all.", 0) is True


def test_aggregate_metrics() -> None:
    results = [
        {
            "id": "a",
            "retrieval": {"hit": True, "reciprocal_rank": 1.0, "latency_ms": 100, "top_vector_score": 0.8},
            "answer": {"latency_ms": 2000, "expected_ok": True, "citation_markers_valid": True, "cost_usd": 0.001},
            "judge": {"relevance": 5, "faithfulness": 4, "completeness": 4},
        },
        {
            "id": "b",
            "retrieval": {"hit": False, "reciprocal_rank": 0.0, "latency_ms": 300, "top_vector_score": 0.2},
            "answer": {"latency_ms": 4000, "expected_ok": None, "citation_markers_valid": False, "cost_usd": 0.002},
        },
        {"id": "c", "error": "boom"},
    ]
    m = _aggregate(results)
    assert m["cases"] == 3
    assert m["errors"] == 1
    assert m["hit_rate"] == 0.5
    assert m["mrr"] == 0.5
    assert m["answer_expected_rate"] == 1.0
    assert m["citation_marker_valid_rate"] == 0.5
    assert m["judge_faithfulness_avg"] == 4.0
    assert abs(m["total_cost_usd"] - 0.003) < 1e-9


def test_judge_verdict_parsing() -> None:
    ok = _parse_verdict('{"relevance": 5, "faithfulness": 4, "completeness": 3, "comment": "good"}')
    assert ok == {"relevance": 5, "faithfulness": 4, "completeness": 3, "comment": "good"}
    fenced = _parse_verdict('```json\n{"relevance": 2, "faithfulness": 9, "completeness": 0}\n```')
    # Out-of-range scores clamp to 1..5.
    assert fenced["faithfulness"] == 5 and fenced["completeness"] == 1
    assert _parse_verdict("not json at all") is None
    assert _parse_verdict('{"relevance": "high"}') is None
