"""Configuration sweep — measure what each design decision is actually worth.

The golden-set runner answers "how good is the assistant *now*". This module
answers "*why* is it that good, and which knob bought which point". It replays
the same benchmark questions through deliberately different pipelines
(embedding model, fusion strategy, reranker, top_k, generation model) and
reports each arm against a baseline, with a paired bootstrap p-value so a
2-point wobble is not sold as an improvement.

Every arm is a real end-to-end run against the live index — no simulation.
The "Baseline (v0)" arm reconstructs the pipeline this project started with:
MiniLM-384 embeddings, weighted score fusion, no reranker, no query expansion,
no dedup / intent shaping.

Results are written as a *single* `eval_runs` row with `config.kind="sweep"`,
which the normal runs list filters out, so the sweep never pollutes the
release-over-release trend.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.eval.metrics import paired_bootstrap_p
from app.eval.runner import GOLDEN_PATH, _aggregate, execute_cases, load_golden
from app.integrations.index_search import IndexSearchClient
from app.llm.embeddings import build_embedding_provider
from app.llm.generation import build_answer_generator
from app.models.entities import EvalRun
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import PostgresLexicalRetriever
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.rerank import build_reranker
from app.services.query import QueryService

logger = logging.getLogger(__name__)

# The collection built with the original 384-d MiniLM embeddings. Kept around
# precisely so the embedding upgrade stays measurable instead of anecdotal.
BASELINE_COLLECTION = "collaboration_knowledge_chunks"
BASELINE_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
BASELINE_VECTOR_SIZE = 384


@dataclass(slots=True)
class Arm:
    key: str
    label: str
    group: str
    note: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    retriever: dict[str, Any] = field(default_factory=dict)
    top_k: int = 8
    generate: bool = False
    judge: bool = False


def _legacy() -> dict[str, Any]:
    return {
        "QDRANT_COLLECTION": BASELINE_COLLECTION,
        "EMBEDDING_MODEL": BASELINE_EMBEDDING,
        "VECTOR_SIZE": BASELINE_VECTOR_SIZE,
        "RERANK_PROVIDER": "none",
    }


def _off() -> dict[str, Any]:
    return {"fusion": "weighted", "expand_queries": False, "dedupe": False, "boosts": False}


def pipeline_arms() -> list[Arm]:
    """Cumulative build-up: each arm adds exactly one change to the previous."""
    return [
        Arm("v0_baseline", "Baseline (v0)", "pipeline",
            "MiniLM-384 · weighted fusion · no reranker · no expansion",
            settings=_legacy(), retriever=_off()),
        Arm("v1_bge", "+ bge-m3 embeddings", "pipeline",
            "1024-d embeddings replace MiniLM-384",
            settings={"RERANK_PROVIDER": "none"}, retriever=_off()),
        Arm("v2_rrf", "+ RRF fusion", "pipeline",
            "rank-based fusion replaces 0.62·vec + 0.38·lex score blending",
            settings={"RERANK_PROVIDER": "none"},
            retriever={"expand_queries": False, "dedupe": False, "boosts": False}),
        Arm("v3_expand", "+ acronym expansion", "pipeline",
            "TDR / SVT / LFHCAL … expanded into a second query variant",
            settings={"RERANK_PROVIDER": "none"},
            retriever={"dedupe": False, "boosts": False}),
        Arm("v4_rerank", "+ cross-encoder rerank", "pipeline",
            "top candidates re-scored by a cross-encoder on GPU",
            retriever={"dedupe": False, "boosts": False}),
        Arm("v5_current", "+ dedup & intent shaping (current)", "pipeline",
            "cross-source dedup, freshness / how-to / slide-deck shaping"),
    ]


def top_k_arms(values: tuple[int, ...] = (3, 5, 8, 12, 16, 20)) -> list[Arm]:
    return [
        Arm(f"top_k_{k}", f"top_k = {k}", "top_k",
            "current pipeline, only the number of returned sources changes", top_k=k)
        for k in values
    ]


def embedding_arms() -> list[Arm]:
    return [
        Arm("emb_minilm", "MiniLM-L6 (384-d)", "embedding",
            "original embedding model, current pipeline otherwise",
            settings=_legacy() | {"RERANK_PROVIDER": "cross_encoder"}),
        Arm("emb_bge_m3", "bge-m3 (1024-d)", "embedding", "current embedding model"),
    ]


def rerank_arms() -> list[Arm]:
    return [
        Arm("rr_off", "No reranker", "rerank", "hybrid fusion order used as-is",
            settings={"RERANK_PROVIDER": "none"}),
        Arm("rr_10", "Cross-encoder, 10 candidates", "rerank", "minimal reranking window",
            settings={"RERANK_CANDIDATES": 10}),
        Arm("rr_20", "Cross-encoder, 20 candidates", "rerank", "reduced reranking window",
            settings={"RERANK_CANDIDATES": 20}),
        Arm("rr_60", "Cross-encoder, 60 candidates (current)", "rerank",
            "wider candidate window",
            settings={"RERANK_CANDIDATES": 60}),
    ]


def speed_arms() -> list[Arm]:
    """End-to-end latency levers, measured with real answer generation
    (deterministic fact check, no judge — latency numbers stay clean).

    LLM_CONTEXT_TOP_K sets how many chunks go into the prompt; the prompt is
    the bulk of generation latency and cost. RERANK_CANDIDATES sets the
    cross-encoder window; OPENAI_MAX_OUTPUT_TOKENS caps answer length.
    """
    combined = {
        "LLM_CONTEXT_TOP_K": 10,
        "RERANK_CANDIDATES": 20,
        "OPENAI_MAX_OUTPUT_TOKENS": 450,
    }
    return [
        Arm("sp_current", "Current (ctx 20 · rerank 60 · out 600)", "speed",
            "production configuration", generate=True),
        Arm("sp_ctx12", "Context 12 chunks", "speed",
            "fewer chunks in the prompt", generate=True,
            settings={"LLM_CONTEXT_TOP_K": 12}),
        Arm("sp_ctx8", "Context 8 chunks", "speed",
            "minimal prompt context", generate=True,
            settings={"LLM_CONTEXT_TOP_K": 8}),
        Arm("sp_rr20", "Rerank 20 candidates", "speed",
            "smaller cross-encoder window", generate=True,
            settings={"RERANK_CANDIDATES": 20}),
        Arm("sp_out450", "Max 450 output tokens", "speed",
            "shorter answers", generate=True,
            settings={"OPENAI_MAX_OUTPUT_TOKENS": 450}),
        Arm("sp_combined", "Combined (ctx 10 · rerank 20 · out 450)", "speed",
            "all three levers together", generate=True, settings=dict(combined)),
    ]


# OpenAI-compatible endpoints for alternative providers, keyed by the
# Settings field that holds their API key. Each becomes a sweep arm when the
# key is configured. All have free daily quotas (as of 2026-08): Gemini
# ~250-1500 req/day, Groq ~1k req/day, DeepSeek off-peak discounts,
# OpenRouter ":free" models 50-1000 req/day.
ALT_PROVIDERS: tuple[tuple[str, str, str, str], ...] = (
    ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "https://api.anthropic.com/v1", "Anthropic"),
    ("GEMINI_API_KEY", "GEMINI_MODEL",
     "https://generativelanguage.googleapis.com/v1beta/openai", "Gemini (free tier)"),
    ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "https://api.deepseek.com/v1", "DeepSeek"),
    ("GROQ_API_KEY", "GROQ_MODEL", "https://api.groq.com/openai/v1", "Groq (free tier)"),
    ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "https://openrouter.ai/api/v1", "OpenRouter"),
)

# $/1M tokens for alternative-provider models we know (2026-08 list prices);
# anything absent reports cost 0 rather than a wrong number.
ALT_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),   # introductory pricing through 2026-08-31
    "claude-opus-5": (5.00, 25.00),
    "deepseek-chat": (0.14, 0.28),
}


def generation_arms(
    settings: Settings | None = None,
    models: tuple[str, ...] = (
        "gpt-5.4-nano",     # current production generator
        "gpt-5.6-luna",     # newer family, same price point, marketed as faster
    ),
) -> list[Arm]:
    """Answer-quality arms. These call the LLM (and the judge) — real money,
    a few minutes per arm — so they are opt-in.

    Alternative providers (Gemini / DeepSeek / Groq / OpenRouter) join the
    block automatically when their API key is set — same benchmark, same
    judge, so a free-tier model competes on exactly the same questions.
    """
    arms = [
        Arm("gen_extractive", "Extractive (no LLM)", "generation",
            "sentence extraction from the top chunks — the zero-cost floor",
            settings={"GENERATION_PROVIDER": "extractive"}, generate=True, judge=True),
    ]
    # $/1M input, $/1M output (2026-08 list prices) so each arm's run cost is
    # computed at ITS OWN rate, not the production model's.
    prices = {
        "gpt-5.4-nano": (0.20, 1.25),
        "gpt-5.4-mini": (0.75, 4.50),
        "gpt-5.6-luna": (0.20, 1.20),
        "gpt-5.6-terra": (2.00, 12.00),
    }
    for model in models:
        cost_in, cost_out = prices.get(model, (0.0, 0.0))
        overrides: dict[str, Any] = {
            "OPENAI_GENERATION_MODEL": model,
            "OPENAI_INPUT_COST_PER_1M": cost_in,
            "OPENAI_OUTPUT_COST_PER_1M": cost_out,
        }
        if model.startswith("gpt-5.6"):
            # gpt-5.6 chat completions reject tools unless effort is "none".
            overrides["OPENAI_REASONING_EFFORT"] = "none"
        arms.append(
            Arm(f"gen_{model}", model, "generation", "OpenAI-protocol generator",
                settings=overrides, generate=True, judge=True)
        )
    for key_field, model_field, base_url, label in ALT_PROVIDERS:
        key = getattr(settings, key_field, None) if settings else None
        if not key:
            continue
        model_list = str(getattr(settings, model_field, "") or "")
        for model in (m.strip() for m in model_list.split(",")):
            if not model:
                continue
            cost_in, cost_out = ALT_PRICES.get(model, (0.0, 0.0))
            arms.append(
                Arm(
                    f"gen_{model}", f"{model} · {label}", "generation",
                    f"{label} via its OpenAI-compatible endpoint",
                    settings={
                        "OPENAI_API_KEY": key,
                        "OPENAI_BASE_URL": base_url,
                        "OPENAI_GENERATION_MODEL": model,
                        "OPENAI_INPUT_COST_PER_1M": cost_in,
                        "OPENAI_OUTPUT_COST_PER_1M": cost_out,
                    },
                    generate=True, judge=True,
                )
            )
    return arms


GROUPS: dict[str, Callable[..., list[Arm]]] = {
    "pipeline": pipeline_arms,
    "top_k": top_k_arms,
    "embedding": embedding_arms,
    "rerank": rerank_arms,
    "speed": speed_arms,
    "generation": generation_arms,
}
DEFAULT_GROUPS = ("pipeline", "top_k", "rerank")

# Loading an embedding model costs seconds and GPU memory; arms share them.
_provider_cache: dict[tuple, Any] = {}
_reranker_cache: dict[tuple, Any] = {}


def _provider(settings: Settings):
    key = (settings.EMBEDDING_MODEL, settings.VECTOR_SIZE, settings.EMBEDDING_DEVICE)
    if key not in _provider_cache:
        _provider_cache[key] = build_embedding_provider(settings)
    return _provider_cache[key]


def _reranker(settings: Settings):
    key = (
        getattr(settings, "RERANK_PROVIDER", "none"),
        getattr(settings, "RERANK_MODEL", ""),
        getattr(settings, "RERANK_CANDIDATES", 0),
    )
    if key not in _reranker_cache:
        _reranker_cache[key] = build_reranker(settings)
    return _reranker_cache[key]


def build_arm_service(base: Settings, arm: Arm) -> tuple[QueryService, Settings]:
    settings = base.model_copy(update=arm.settings) if arm.settings else base
    store = QdrantVectorStore(settings, _provider(settings))
    retriever = HybridRetriever(
        store, lexical=PostgresLexicalRetriever(), reranker=_reranker(settings), **arm.retriever
    )
    gen_settings = settings if arm.generate else settings.model_copy(
        update={"GENERATION_PROVIDER": "extractive"}
    )
    # Generation arms keep the agentic search tool so they measure the same
    # answering machinery users hit, not a stripped-down variant.
    index_search = (
        IndexSearchClient(retriever, top_k=int(getattr(settings, "AGENTIC_SEARCH_TOP_K", 6)))
        if arm.generate and getattr(settings, "AGENTIC_SEARCH_ENABLED", True)
        else None
    )
    generator = build_answer_generator(gen_settings, index_search_client=index_search)
    # No cache: every arm must actually run, and arms share query text.
    service = QueryService(settings, retriever, generator, query_cache=None)
    return service, settings


def _case_scores(results: list[dict[str, Any]], metric: str) -> list[float]:
    """Per-question values, aligned by case order, for the paired test."""
    out = []
    for r in results:
        value = (r.get("retrieval") or {}).get(metric)
        out.append(float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0)
    return out


def run_sweep(
    db: Session,
    settings: Settings,
    *,
    label: str = "",
    groups: tuple[str, ...] | list[str] = DEFAULT_GROUPS,
    case_ids: list[str] | None = None,
    golden_path=GOLDEN_PATH,
    run: EvalRun | None = None,
) -> EvalRun:
    cases = load_golden(golden_path)
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.id in wanted]

    arms: list[Arm] = []
    for group in groups:
        factory = GROUPS.get(group)
        if factory is None:
            raise ValueError(f"unknown sweep group: {group}")
        # The generation block inspects settings for alternative-provider keys.
        arms.extend(factory(settings) if group == "generation" else factory())

    if run is None:
        run = EvalRun(
            label=label or "configuration sweep",
            config={"kind": "sweep", "groups": list(groups), "arms": len(arms),
                    "cases": [c.id for c in cases]},
            status="running",
            cases_total=len(cases),
        )
        db.add(run)
        db.commit()
        db.refresh(run)
    else:
        run.cases_total = len(cases)
        run.config = {**(run.config or {}), "cases": [c.id for c in cases]}
        db.commit()

    judge_client = None
    if any(a.judge for a in arms) and settings.OPENAI_API_KEY:
        from app.eval.judge import AnswerJudge

        try:
            judge_client = AnswerJudge(settings)
            run.config = {**(run.config or {}), "judge_model": judge_client.model}
            db.commit()
        except Exception as exc:  # pragma: no cover - config guard
            logger.warning("sweep_judge_init_failed", extra={"error": str(exc)[:200]})

    arm_rows: list[dict[str, Any]] = []
    per_arm_scores: dict[str, list[float]] = {}
    for index, arm in enumerate(arms, start=1):
        started = time.perf_counter()
        try:
            service, arm_settings = build_arm_service(settings, arm)
            results = execute_cases(
                db, service, arm_settings, cases,
                generate=arm.generate,
                judge_client=judge_client if arm.judge else None,
                top_k=arm.top_k,
            )
        except Exception as exc:
            logger.exception("sweep_arm_failed", extra={"arm": arm.key})
            arm_rows.append({"key": arm.key, "label": arm.label, "group": arm.group,
                             "note": arm.note, "error": str(exc)[:300]})
            run.metrics = {"arms": list(arm_rows)}
            db.commit()
            continue
        metrics = _aggregate(results)
        per_arm_scores[arm.key] = _case_scores(results, "ndcg@5")
        arm_rows.append({
            "key": arm.key,
            "label": arm.label,
            "group": arm.group,
            "note": arm.note,
            "top_k": arm.top_k,
            "config": {
                "embedding_model": arm_settings.EMBEDDING_MODEL,
                "collection": arm_settings.QDRANT_COLLECTION,
                "rerank": arm_settings.RERANK_PROVIDER,
                "rerank_candidates": getattr(arm_settings, "RERANK_CANDIDATES", None),
                "generation": (
                    getattr(arm_settings, "OPENAI_GENERATION_MODEL", None)
                    if arm.generate and arm_settings.GENERATION_PROVIDER == "openai"
                    else arm_settings.GENERATION_PROVIDER if arm.generate else None
                ),
                **{k: v for k, v in arm.retriever.items()},
            },
            "metrics": metrics,
            "wall_s": round(time.perf_counter() - started, 1),
        })
        run.metrics = {"arms": list(arm_rows)}
        run.results = [{"arm": k, "ndcg@5": v} for k, v in per_arm_scores.items()]
        db.commit()
        logger.info("sweep_arm_done", extra={"arm": arm.key, "index": index, "total": len(arms)})

    # Significance versus each group's first arm (its own control).
    controls: dict[str, str] = {}
    for row in arm_rows:
        controls.setdefault(row["group"], row["key"])
    for row in arm_rows:
        control_key = controls[row["group"]]
        row["control"] = control_key
        if row["key"] == control_key or "metrics" not in row:
            continue
        control_row = next((r for r in arm_rows if r["key"] == control_key and "metrics" in r), None)
        if not control_row:
            continue
        base_ndcg = control_row["metrics"].get("ndcg@5")
        if base_ndcg is not None and row["metrics"].get("ndcg@5") is not None:
            row["delta_ndcg@5"] = round(row["metrics"]["ndcg@5"] - base_ndcg, 4)
        base_hit = control_row["metrics"].get("hit_rate")
        if base_hit is not None and row["metrics"].get("hit_rate") is not None:
            row["delta_hit_rate"] = round(row["metrics"]["hit_rate"] - base_hit, 4)
        if control_key in per_arm_scores and row["key"] in per_arm_scores:
            row["p_value"] = paired_bootstrap_p(
                per_arm_scores[control_key], per_arm_scores[row["key"]]
            )

    run.metrics = {
        "arms": arm_rows,
        "groups": list(groups),
        "cases": len(cases),
        "baseline_key": arm_rows[0]["key"] if arm_rows else None,
    }
    run.status = "completed"
    run.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(run)
    return run
