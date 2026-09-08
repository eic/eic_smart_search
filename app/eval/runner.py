"""Evaluation runner: golden set → per-case metrics → aggregate row.

Run after any retrieval/prompt/model change (``make eval``). Results land
in the ``eval_runs`` table and render at ``/admin/eval``.

Metrics:
  retrieval  hit@k (expected URL found in citations), reciprocal rank,
             top vector cosine, retrieval latency
  answer     expected-substring check, citation-marker validity,
             latency, tokens, cost
  judge      relevance / faithfulness / completeness 1-5 (LLM-as-a-judge,
             optional — needs OPENAI_API_KEY)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.profile import golden_path
from app.eval.metrics import mean_ci, rank_metrics
from app.models.entities import EvalRun
from app.schemas.api import QueryRequest, QueryResponse
from app.services.query import QueryService

logger = logging.getLogger(__name__)

# Benchmark lives in the collaboration profile (profiles/<PROFILE>/golden.yaml).
GOLDEN_PATH = golden_path()


@dataclass(slots=True)
class GoldenCase:
    id: str
    query: str
    expect_url_contains: list[str] = field(default_factory=list)
    expect_answer_contains: list[str] = field(default_factory=list)
    scope: str = "public"
    tags: list[str] = field(default_factory=list)


def load_golden(path: Path | str = GOLDEN_PATH) -> list[GoldenCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    cases: list[GoldenCase] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("id") or not item.get("query"):
            continue
        cases.append(
            GoldenCase(
                id=str(item["id"]),
                query=str(item["query"]),
                expect_url_contains=[str(u) for u in item.get("expect_url_contains") or []],
                expect_answer_contains=[str(a) for a in item.get("expect_answer_contains") or []],
                scope=str(item.get("scope") or "public"),
                tags=[str(t) for t in item.get("tags") or []],
            )
        )
    return cases


def _citation_rank(response: QueryResponse, needles: list[str]) -> int | None:
    """1-based rank of the first citation whose URL matches any needle."""
    if not needles:
        return None
    lowered = [n.lower() for n in needles]
    for i, citation in enumerate(response.citations, start=1):
        url = (citation.url or "").lower()
        if any(n in url for n in lowered):
            return i
    return None


def _citation_markers_valid(answer: str, n_citations: int) -> bool:
    import re

    refs = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer or "")]
    return all(1 <= r <= n_citations for r in refs) if refs else True


def run_eval(
    db: Session,
    service: QueryService,
    settings: Settings,
    *,
    label: str = "",
    generate: bool = True,
    judge: bool = True,
    top_k: int = 8,
    case_ids: list[str] | None = None,
    golden_path: Path | str = GOLDEN_PATH,
) -> EvalRun:
    cases = load_golden(golden_path)
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.id in wanted]

    judge_client = None
    if judge and generate and settings.GENERATION_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        from app.eval.judge import AnswerJudge

        try:
            judge_client = AnswerJudge(settings)
        except Exception as exc:  # pragma: no cover - import/config guard
            logger.warning("judge_init_failed", extra={"error": str(exc)[:200]})

    run = EvalRun(
        label=label or None,
        config={
            "generate": generate,
            "judge": judge_client is not None,
            "top_k": top_k,
            "cases": [c.id for c in cases],
            "embedding_model": settings.EMBEDDING_MODEL,
            "generation_provider": settings.GENERATION_PROVIDER,
            "generation_model": getattr(settings, "OPENAI_GENERATION_MODEL", None),
            "rerank_provider": settings.RERANK_PROVIDER,
        },
        status="running",
        cases_total=len(cases),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    def _progress(results_so_far: list[dict[str, Any]]) -> None:
        # Progress is visible to a polling dashboard between commits.
        run.results = list(results_so_far)
        db.commit()

    results = execute_cases(
        db, service, settings, cases,
        generate=generate, judge_client=judge_client, top_k=top_k, progress=_progress,
    )

    run.metrics = _aggregate(results)
    run.status = "completed"
    run.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(run)
    return run


def execute_cases(
    db: Session,
    service: QueryService,
    settings: Settings,
    cases: list[GoldenCase],
    *,
    generate: bool = True,
    judge_client=None,
    top_k: int = 8,
    progress=None,
) -> list[dict[str, Any]]:
    """Run every case through one QueryService and return raw per-case results.

    Split out of `run_eval` so the configuration sweep can drive the same
    cases through differently-configured services without writing a row per
    arm.
    """
    results: list[dict[str, Any]] = []
    for case in cases:
        results.append(_run_case(db, service, settings, case, generate, judge_client, top_k))
        if progress is not None:
            progress(results)
    return results


def _run_case(
    db: Session,
    service: QueryService,
    settings: Settings,
    case: GoldenCase,
    generate: bool,
    judge_client,
    top_k: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"id": case.id, "query": case.query, "tags": case.tags}

    # Retrieval-only pass — isolates search quality from generation.
    t0 = time.perf_counter()
    try:
        retrieval_resp = service.query(
            db,
            QueryRequest(
                query=case.query, scope=case.scope, top_k=top_k,
                generate_answer=False, eval_mode=True, force_rerank=True,
            ),
        )
    except Exception as exc:
        logger.exception("eval_case_retrieval_failed", extra={"case": case.id})
        result["error"] = str(exc)[:300]
        return result
    retrieval_ms = int((time.perf_counter() - t0) * 1000)
    rank = _citation_rank(retrieval_resp, case.expect_url_contains)
    top_vec = None
    if retrieval_resp.citations:
        top_vec = (retrieval_resp.citations[0].metadata or {}).get("vector_score")
    ranked = rank_metrics(
        [c.url or "" for c in retrieval_resp.citations], case.expect_url_contains
    )
    result["retrieval"] = {
        "hit": rank is not None if case.expect_url_contains else None,
        "rank": rank,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
        "results": len(retrieval_resp.citations),
        "top_vector_score": top_vec,
        "latency_ms": retrieval_ms,
        "top_urls": [c.url for c in retrieval_resp.citations[:3]],
        # nDCG / recall / success@k — rank-aware and coverage-aware, so a
        # config that surfaces two of three expected pages scores above one
        # that surfaces one. {} for abstention cases (no expected URL).
        **{k: v for k, v in ranked.items() if k not in ("hit", "rank", "reciprocal_rank")},
    }

    if not generate:
        return result

    t1 = time.perf_counter()
    try:
        full_resp = service.query(
            db,
            QueryRequest(
                query=case.query, scope=case.scope, top_k=top_k,
                generate_answer=True, eval_mode=True,
            ),
        )
    except Exception as exc:
        logger.exception("eval_case_generation_failed", extra={"case": case.id})
        result["error"] = str(exc)[:300]
        return result
    answer_ms = int((time.perf_counter() - t1) * 1000)
    answer = full_resp.answer or ""
    generation_debug = (full_resp.retrieval_debug or {}).get("generation") or {}
    # Normalize typographic quotes before matching — the LLM freely mixes
    # "couldn't" and "couldn’t", which used to flake the abstention cases.
    normalized = answer.lower().replace("’", "'").replace("‘", "'")
    missing = [
        needle for needle in case.expect_answer_contains
        if needle.lower() not in normalized
    ]
    result["answer"] = {
        "latency_ms": answer_ms,
        "length": len(answer),
        "expected_ok": not missing if case.expect_answer_contains else None,
        "missing_terms": missing,
        "citation_markers_valid": _citation_markers_valid(answer, len(full_resp.citations)),
        "insufficient": generation_debug.get("support") == "insufficient",
        "total_tokens": generation_debug.get("total_tokens"),
        "cost_usd": generation_debug.get("cost_usd"),
        "tool_time_ms": generation_debug.get("tool_time_ms"),
        # Full answer (capped) — failing answers get re-judged / re-generated
        # later with better models, so keep enough text to work with.
        "text": answer[:4000],
    }

    if judge_client is not None:
        snippets = [c.snippet or "" for c in full_resp.citations]
        result["judge"] = judge_client.judge(case.query, answer, snippets)
    return result


def case_failure_reasons(result: dict[str, Any]) -> list[str]:
    """Why a case counts as failing — [] means it passed.

    Used by /eval/failures to build the standing worklist of bad answers
    (re-run them after a fix, or feed them to a stronger model).
    """
    reasons: list[str] = []
    if result.get("error"):
        reasons.append("error")
    retrieval = result.get("retrieval") or {}
    if retrieval.get("hit") is False:
        reasons.append("retrieval-miss")
    answer = result.get("answer") or {}
    if answer.get("insufficient"):
        reasons.append("insufficient-support")
    if answer.get("expected_ok") is False:
        reasons.append("wrong-answer")
    if answer.get("citation_markers_valid") is False:
        reasons.append("bad-citation-markers")
    judge = result.get("judge") or {}
    if isinstance(judge.get("faithfulness"), int) and judge["faithfulness"] <= 2:
        reasons.append("low-faithfulness")
    if isinstance(judge.get("relevance"), int) and judge["relevance"] <= 2:
        reasons.append("low-relevance")
    return reasons


def collect_failures(runs: list["EvalRun"], *, max_runs: int = 5) -> list[dict[str, Any]]:
    """Failing cases from the latest completed runs, newest first, deduped by
    case id (a case fixed in a newer run drops off the list)."""
    seen: set[str] = set()
    failures: list[dict[str, Any]] = []
    completed = [r for r in runs if r.status == "completed"][:max_runs]
    for run in completed:
        for result in run.results or []:
            case_id = result.get("id")
            if not case_id or case_id in seen:
                continue
            seen.add(case_id)
            reasons = case_failure_reasons(result)
            if not reasons:
                continue
            answer = result.get("answer") or {}
            judge = result.get("judge") or {}
            failures.append(
                {
                    "case_id": case_id,
                    "query": result.get("query"),
                    "tags": result.get("tags") or [],
                    "reasons": reasons,
                    "run_id": run.id,
                    "run_label": run.label,
                    "run_created_at": run.created_at.isoformat() if run.created_at else None,
                    "answer_text": answer.get("text") or "",
                    "missing_terms": answer.get("missing_terms") or [],
                    "judge": judge or None,
                    "top_urls": (result.get("retrieval") or {}).get("top_urls") or [],
                }
            )
    return failures


def _by_tag(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-category success rates (ROOTER-style per-task breakdown):
    outsiders see WHERE the assistant is strong or weak, not one blended
    number."""
    tags: dict[str, dict[str, int]] = {}
    for r in results:
        for tag in r.get("tags") or ["untagged"]:
            bucket = tags.setdefault(tag, {"cases": 0, "hits": 0, "scored": 0, "answer_ok": 0, "answer_checked": 0})
            bucket["cases"] += 1
            retrieval = r.get("retrieval") or {}
            if retrieval.get("hit") is not None:
                bucket["scored"] += 1
                if retrieval["hit"]:
                    bucket["hits"] += 1
            answer = r.get("answer") or {}
            if answer.get("expected_ok") is not None:
                bucket["answer_checked"] += 1
                if answer["expected_ok"]:
                    bucket["answer_ok"] += 1
    out: dict[str, dict[str, Any]] = {}
    for tag, b in sorted(tags.items()):
        out[tag] = {
            "cases": b["cases"],
            "hit_rate": round(b["hits"] / b["scored"], 4) if b["scored"] else None,
            "answer_rate": round(b["answer_ok"] / b["answer_checked"], 4) if b["answer_checked"] else None,
        }
    return out


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    def _vals(path: list[str]) -> list[float]:
        out = []
        for r in results:
            node: Any = r
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            if isinstance(node, (int, float)) and not isinstance(node, bool):
                out.append(float(node))
        return out

    def _mean(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    def _pct(vals: list[float], q: float) -> float | None:
        if not vals:
            return None
        ordered = sorted(vals)
        idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return round(ordered[idx], 1)

    scored = [r for r in results if (r.get("retrieval") or {}).get("hit") is not None]
    hits = [r for r in scored if r["retrieval"]["hit"]]
    answer_checked = [
        r for r in results if (r.get("answer") or {}).get("expected_ok") is not None
    ]
    markers = [r for r in results if "answer" in r]
    return {
        "cases": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "hit_rate": round(len(hits) / len(scored), 4) if scored else None,
        # Bootstrap 95% CI on the headline rate — with ~60 questions a
        # few-point move is inside the noise band, and the dashboard says so.
        "hit_rate_ci": mean_ci([1.0 if r["retrieval"]["hit"] else 0.0 for r in scored]),
        "ndcg@5": _mean(_vals(["retrieval", "ndcg@5"])),
        "ndcg@5_ci": mean_ci(_vals(["retrieval", "ndcg@5"])),
        "ndcg@10": _mean(_vals(["retrieval", "ndcg@10"])),
        "precision@5": _mean(_vals(["retrieval", "precision@5"])),
        "precision@10": _mean(_vals(["retrieval", "precision@10"])),
        "success@1": _mean(_vals(["retrieval", "success@1"])),
        "success@3": _mean(_vals(["retrieval", "success@3"])),
        "mrr": _mean(_vals(["retrieval", "reciprocal_rank"])),
        "avg_top_vector_score": _mean(_vals(["retrieval", "top_vector_score"])),
        "retrieval_latency_ms_avg": _mean(_vals(["retrieval", "latency_ms"])),
        "retrieval_latency_ms_p50": _pct(_vals(["retrieval", "latency_ms"]), 0.5),
        "retrieval_latency_ms_p95": _pct(_vals(["retrieval", "latency_ms"]), 0.95),
        "answer_latency_ms_avg": _mean(_vals(["answer", "latency_ms"])),
        "answer_latency_ms_p50": _pct(_vals(["answer", "latency_ms"]), 0.5),
        "answer_latency_ms_p95": _pct(_vals(["answer", "latency_ms"]), 0.95),
        "tool_time_ms_avg": _mean(_vals(["answer", "tool_time_ms"])),
        "answer_expected_rate": (
            round(sum(1 for r in answer_checked if r["answer"]["expected_ok"]) / len(answer_checked), 4)
            if answer_checked else None
        ),
        "citation_marker_valid_rate": (
            round(sum(1 for r in markers if r["answer"]["citation_markers_valid"]) / len(markers), 4)
            if markers else None
        ),
        "by_tag": _by_tag(results),
        "judge_relevance_avg": _mean(_vals(["judge", "relevance"])),
        "judge_faithfulness_avg": _mean(_vals(["judge", "faithfulness"])),
        "judge_completeness_avg": _mean(_vals(["judge", "completeness"])),
        "total_cost_usd": round(sum(_vals(["answer", "cost_usd"])), 6),
    }
