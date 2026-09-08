"""Precomputed playground grid: every discrete retrieval configuration,
evaluated once, served as instant "draft" results to the playground UI.

The playground's retrieval knobs form a small discrete space —
fusion (2) x expansion (2) x dedupe (2) x boosts (2) x rerank (5) = 80
configurations. Retrieval-only evaluation of one configuration over the
golden set costs ~30-60 s and $0, so the whole grid is an hour of
background compute. With it cached, the UI can show a draft radar the
moment a knob flips — TensorFlow-playground feel — and only spend an
actual run (or an LLM sample) to confirm.

Results land in one eval_runs row with config.kind = "pgrid" (newest row
wins; the runs list filters kind != null like sweeps).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.entities import EvalRun

logger = logging.getLogger(__name__)

RERANK_LEVELS = ("off", "10", "20", "40", "60")


def grid_key(fusion: str, expand: bool, dedupe: bool, boosts: bool, rerank: str) -> str:
    return f"{fusion}|e{int(expand)}|d{int(dedupe)}|b{int(boosts)}|r{rerank}"


def grid_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    for fusion in ("rrf", "weighted"):
        for expand in (True, False):
            for dedupe in (True, False):
                for boosts in (True, False):
                    for rerank in RERANK_LEVELS:
                        yield (
                            grid_key(fusion, expand, dedupe, boosts, rerank),
                            {
                                "fusion": fusion,
                                "expand_queries": expand,
                                "dedupe": dedupe,
                                "boosts": boosts,
                                "rerank": rerank,
                            },
                        )


# The subset of aggregate metrics the draft radar needs — keep rows small.
_KEEP_METRICS = (
    "hit_rate",
    "ndcg@5",
    "mrr",
    "success@1",
    "success@3",
    "retrieval_latency_ms_p50",
)


def run_playground_grid(db: Session, settings: Settings, run_id: str) -> None:
    """Evaluate all 80 retrieval configurations; progress-commit per config."""
    from app.eval.runner import _aggregate, execute_cases, load_golden
    from app.eval.sweep import Arm, build_arm_service

    run = db.get(EvalRun, run_id)
    if run is None:
        return
    cases = load_golden()
    grid: dict[str, dict[str, Any]] = {}
    configs = list(grid_configs())
    try:
        for index, (key, cfg) in enumerate(configs, start=1):
            overrides: dict[str, Any] = {}
            if cfg["rerank"] == "off":
                overrides["RERANK_PROVIDER"] = "none"
            else:
                overrides["RERANK_CANDIDATES"] = int(cfg["rerank"])
            arm = Arm(
                "grid", key, "grid",
                settings=overrides,
                retriever={
                    "fusion": cfg["fusion"],
                    "expand_queries": cfg["expand_queries"],
                    "dedupe": cfg["dedupe"],
                    "boosts": cfg["boosts"],
                },
                top_k=8,
                generate=False,
            )
            service, arm_settings = build_arm_service(settings, arm)
            results = execute_cases(
                db, service, arm_settings, cases,
                generate=False, judge_client=None, top_k=8,
            )
            aggregate = _aggregate(results)
            grid[key] = {k: aggregate.get(k) for k in _KEEP_METRICS}
            run.metrics = {
                "progress": {"done": index, "total": len(configs)},
                "grid": grid,
            }
            db.commit()
        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        run.metrics = {
            "progress": {"done": len(configs), "total": len(configs)},
            "grid": grid,
            "cases": len(cases),
            "produced_at": datetime.now(timezone.utc).isoformat(),
        }
        db.commit()
        logger.info("playground_grid_completed", extra={"configs": len(grid)})
    except Exception as exc:
        db.rollback()
        run = db.get(EvalRun, run_id)
        if run:
            run.status = "failed"
            run.metrics = {**(run.metrics or {}), "error": str(exc)[:300]}
            db.commit()
        logger.exception("playground_grid_failed")


def latest_grid(db: Session) -> dict[str, Any] | None:
    """Newest completed grid row's payload, or None."""
    rows = db.execute(
        select(EvalRun)
        .where(EvalRun.status == "completed")
        .order_by(EvalRun.created_at.desc())
        .limit(50)
    ).scalars()
    for run in rows:
        if (run.config or {}).get("kind") == "pgrid":
            metrics = run.metrics or {}
            return {
                "grid": metrics.get("grid") or {},
                "cases": metrics.get("cases"),
                "produced_at": metrics.get("produced_at") or (
                    run.finished_at.isoformat() if run.finished_at else None
                ),
            }
    return None
