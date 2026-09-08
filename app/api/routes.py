import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from sqlalchemy import desc, select, text
from sqlalchemy.orm import Session

import time as time_module
from urllib.parse import quote

from app.api.auth import (
    SESSION_COOKIE as AUTH_SESSION_COOKIE,
    AdminPrincipal,
    is_admin_request,
    verify_admin_login as auth_verify_admin_login,
    oauth_exchange_code as auth_oauth_exchange_code,
    public_base_url as auth_public_base_url,
    session_login as auth_session_login,
    sign_payload as auth_sign_payload,
    verify_payload as auth_verify_payload,
)
from app.api.ratelimit import RateLimited
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.ingestion.orchestrator import IngestionOrchestrator
from app.models.entities import CrawlJob, Feedback
from app.people.builder import rebuild_people
from app.schemas.api import (
    AnalyticsResponse,
    EvalPlaygroundRequest,
    EvalRunRequest,
    EvalSweepRequest,
    FeedbackCreate,
    FeedbackRead,
    IngestRunRequest,
    IngestRunResponse,
    PersonRefreshResponse,
    PopularQueriesResponse,
    QueryRequest,
    QueryResponse,
    ReindexRequest,
)
from app.services.analytics import build_analytics, build_popular_queries
from app.services.factory import build_query_service
from app.services.query import QueryService

logger = logging.getLogger(__name__)
router = APIRouter()


DbSession = Annotated[Session, Depends(get_db)]


def settings_dep() -> Settings:
    return get_settings()


_query_service_singleton: QueryService | None = None
_ingestion_singleton: IngestionOrchestrator | None = None


def query_service_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> QueryService:
    # Build once per process. The SentenceTransformer model + Qdrant client +
    # HybridRetriever are all expensive to instantiate and stateless under
    # concurrent reads, so sharing is safe and massively cheaper than per-request
    # construction.
    global _query_service_singleton
    if _query_service_singleton is None:
        _query_service_singleton = build_query_service(settings)
    return _query_service_singleton


def ingestion_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> IngestionOrchestrator:
    # Reuse the query service's vector store: building a fresh one loads a
    # SECOND copy of the embedding model into the same process, which is
    # what used to OOM-kill the container mid-ingest once both api
    # containers plus the reranker already sat on the GPU.
    global _ingestion_singleton
    if _ingestion_singleton is None:
        service = query_service_dep(settings)
        _ingestion_singleton = IngestionOrchestrator(settings, service.retriever.vector_store)
    return _ingestion_singleton


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---- GitHub SSO (ADMIN_AUTH_MODE=github_sso) ------------------------------


@router.get("/auth/login")
def auth_login(
    request: Request,
    settings: Annotated[Settings, Depends(settings_dep)],
    next: str = "/admin/dashboard",
) -> RedirectResponse:
    """Start the GitHub OAuth flow ("Sign in with GitHub")."""
    if settings.ADMIN_AUTH_MODE != "github_sso":
        raise HTTPException(status_code=404, detail="SSO not enabled")
    if not (settings.GITHUB_OAUTH_CLIENT_ID and settings.GITHUB_OAUTH_CLIENT_SECRET and settings.ADMIN_SESSION_SECRET):
        raise HTTPException(status_code=500, detail="SSO misconfigured: set GITHUB_OAUTH_CLIENT_ID/SECRET and ADMIN_SESSION_SECRET")
    # Only allow same-site relative redirect targets.
    if not next.startswith("/") or next.startswith("//"):
        next = "/admin/dashboard"
    state = auth_sign_payload(
        {"exp": time_module.time() + 600, "next": next}, settings.ADMIN_SESSION_SECRET
    )
    redirect_uri = auth_public_base_url(request, settings) + "/auth/callback"
    # Allowlist mode only needs the user's identity — request read:user so
    # the org's OAuth-App restrictions never block the flow. Org-membership
    # mode still needs read:org.
    scope = "read:user" if settings.ADMIN_GITHUB_USERS else "read:org"
    url = (
        "https://github.com/login/oauth/authorize"
        + "?client_id=" + settings.GITHUB_OAUTH_CLIENT_ID
        + "&redirect_uri=" + quote(redirect_uri, safe="")
        + "&scope=" + scope
        + "&state=" + quote(state, safe="")
    )
    return RedirectResponse(url)


@router.get("/auth/callback")
def auth_callback(
    request: Request,
    settings: Annotated[Settings, Depends(settings_dep)],
    code: str = "",
    state: str = "",
) -> RedirectResponse:
    if settings.ADMIN_AUTH_MODE != "github_sso" or not settings.ADMIN_SESSION_SECRET:
        raise HTTPException(status_code=404, detail="SSO not enabled")
    payload = auth_verify_payload(state, settings.ADMIN_SESSION_SECRET)
    if payload is None or not code:
        raise HTTPException(status_code=400, detail="invalid OAuth state")
    token = auth_oauth_exchange_code(code, settings)
    if not token:
        raise HTTPException(status_code=502, detail="GitHub token exchange failed")
    login = auth_verify_admin_login(token, settings)
    if not login:
        detail = (
            "GitHub account is not on the admin allowlist"
            if settings.ADMIN_GITHUB_USERS
            else f"GitHub account is not an active member of the '{settings.ADMIN_GITHUB_ORG}' organization"
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
    session = auth_sign_payload(
        {"login": login, "exp": time_module.time() + settings.ADMIN_SESSION_TTL_S},
        settings.ADMIN_SESSION_SECRET,
    )
    response = RedirectResponse(payload.get("next") or "/admin/dashboard")
    response.set_cookie(
        AUTH_SESSION_COOKIE,
        session,
        max_age=int(settings.ADMIN_SESSION_TTL_S),
        httponly=True,
        samesite="lax",
        secure=auth_public_base_url(request, settings).startswith("https://"),
        path="/",
    )
    return response


@router.get("/auth/logout")
def auth_logout() -> RedirectResponse:
    response = RedirectResponse("/admin/dashboard")
    response.delete_cookie(AUTH_SESSION_COOKIE, path="/")
    return response


@router.get("/auth/me")
def auth_me(
    request: Request,
    settings: Annotated[Settings, Depends(settings_dep)],
) -> dict[str, Any]:
    """Who am I, for dashboard chrome. Never raises."""
    mode = settings.ADMIN_AUTH_MODE
    login = auth_session_login(request, settings) if mode == "github_sso" else None
    return {"mode": mode, "login": login}


@router.get("/ready")
def ready(
    db: DbSession,
    service: Annotated[QueryService, Depends(query_service_dep)],
) -> dict[str, str]:
    try:
        db.execute(text("SELECT 1"))
        # Reuse the singleton's vector store — building a fresh one loads
        # the embedding model from scratch (~10 s with bge-m3) per probe.
        # ensure_collection() is memoized, so probe Qdrant directly.
        service.retriever.vector_store.client.get_collections()
        service.retriever.vector_store.ensure_collection()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "error": str(exc)},
        ) from exc
    return {"status": "ready"}


def _enforce_public_limits(request: QueryRequest, req: Request, settings: Settings) -> None:
    """Anonymous callers may only search public content, and may not set the
    privileged request flags: scope=internal reads internal chunks the moment
    any exist, eval_mode bypasses the response cache and analytics log, and
    force_rerank burns CPU on demand. Admins (validated session / token) keep
    what they asked for."""
    if request.scope == "public" and not request.eval_mode and not request.force_rerank:
        return
    if is_admin_request(req, settings):
        return
    request.scope = "public"
    request.eval_mode = False
    request.force_rerank = False


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    req: Request,
    db: DbSession,
    service: Annotated[QueryService, Depends(query_service_dep)],
    settings: Annotated[Settings, Depends(settings_dep)],
    _rl: RateLimited,
) -> QueryResponse:
    _enforce_public_limits(request, req, settings)
    return service.query(db, request)


@router.post("/query/stream")
def query_stream(
    request: QueryRequest,
    req: Request,
    db: DbSession,
    service: Annotated[QueryService, Depends(query_service_dep)],
    settings: Annotated[Settings, Depends(settings_dep)],
    _rl: RateLimited,
) -> StreamingResponse:
    """SSE version of /query: `citations` event first (fast), then `delta`
    answer fragments, then `done` with the final aligned payload."""
    _enforce_public_limits(request, req, settings)
    return StreamingResponse(
        service.query_stream(db, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering (nginx / some CDNs) so deltas flush.
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ingest/run", response_model=IngestRunResponse)
def ingest_run(
    payload: IngestRunRequest,
    request: Request,
    db: DbSession,
    orchestrator: Annotated[IngestionOrchestrator, Depends(ingestion_dep)],
    principal: AdminPrincipal,
) -> IngestRunResponse:
    requested_by = principal if principal != "anonymous" else (
        request.headers.get("x-user") or (request.client.host if request.client else None)
    )
    job_ids, stats = orchestrator.run(
        db,
        source_names=payload.source_names,
        full_reindex=payload.full_reindex,
        max_pages=payload.max_pages,
        requested_by=requested_by,
    )
    return IngestRunResponse(job_ids=job_ids, status="completed", stats=stats)


@router.get("/admin/jobs")
def admin_jobs(db: DbSession, principal: AdminPrincipal, limit: int = 50) -> list[dict]:
    jobs = db.scalars(select(CrawlJob).order_by(desc(CrawlJob.created_at)).limit(min(limit, 200))).all()
    return [
        {
            "id": job.id,
            "source_id": job.source_id,
            "connector": job.connector,
            "status": job.status,
            "requested_by": job.requested_by,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "stats": job.stats,
            "created_at": job.created_at,
        }
        for job in jobs
    ]


@router.post("/admin/reindex", response_model=IngestRunResponse)
def admin_reindex(
    payload: ReindexRequest,
    request: Request,
    db: DbSession,
    orchestrator: Annotated[IngestionOrchestrator, Depends(ingestion_dep)],
    principal: AdminPrincipal,
) -> IngestRunResponse:
    requested_by = principal if principal != "anonymous" else (
        request.headers.get("x-user") or (request.client.host if request.client else None)
    )
    job_ids, stats = orchestrator.run(
        db,
        source_names=payload.source_names,
        full_reindex=True,
        max_pages=payload.max_pages,
        requested_by=requested_by,
    )
    return IngestRunResponse(job_ids=job_ids, status="completed", stats=stats)


@router.get("/admin/analytics", response_model=AnalyticsResponse)
def admin_analytics(db: DbSession, principal: AdminPrincipal, window_days: int = 7, limit: int = 20) -> AnalyticsResponse:
    return build_analytics(db, window_days=window_days, limit=limit)


@router.get("/admin/recent")
def admin_recent_queries(
    db: DbSession,
    principal: AdminPrincipal,
    window_days: int = 7,
    limit: int = 50,
    search: str | None = None,
    answered_only: bool = False,
) -> JSONResponse:
    """Recent query logs with answer text, for the analytics dashboard."""
    from datetime import datetime, timedelta, timezone
    from app.models.entities import QueryLog

    window_days = max(1, min(window_days, 3650))
    limit = max(1, min(limit, 500))
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    stmt = (
        select(QueryLog)
        .where(QueryLog.created_at >= since)
        .order_by(desc(QueryLog.created_at))
        .limit(limit)
    )
    if answered_only:
        stmt = stmt.where(
            QueryLog.answer_generated.is_(True),
            QueryLog.answer.isnot(None),
            QueryLog.answer != "",
        )
    if search:
        stmt = stmt.where(QueryLog.query.ilike(f"%{search}%"))
    rows = db.execute(stmt).scalars().all()
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "id": row.id,
                "query": row.query,
                "scope": row.scope,
                "top_k": row.top_k,
                "answer_generated": row.answer_generated,
                "answer": row.answer,
                "result_count": row.result_count,
                "latency_ms": row.latency_ms,
                "generation_provider": row.generation_provider,
                "generation_model": row.generation_model,
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "cost_usd": row.cost_usd,
                "confidence": row.confidence,
                "top_score": row.top_score,
                "cache_hit": (row.retrieval_debug or {}).get("cache_hit"),
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "trace": _query_trace(row.retrieval_debug or {}, row.latency_ms),
            }
        )
    return JSONResponse({"window_days": window_days, "count": len(payload), "items": payload})


def _query_trace(debug: dict[str, Any], total_ms: int | None) -> dict[str, Any] | None:
    """Compact per-query trace for the dashboard: stage timings + retrieval
    funnel + tool calls, distilled from the stored retrieval_debug blob."""
    if not debug:
        return None
    rerank = debug.get("rerank") or {}
    gen = debug.get("generation") or {}
    tools = [
        {
            "name": t.get("name"),
            "ms": t.get("elapsed_ms"),
            "items": t.get("result_items"),
        }
        for t in (gen.get("tool_rounds") or [])
        if isinstance(t, dict)
    ]
    rerank_ms = rerank.get("latency_ms")
    gen_ms = gen.get("latency_ms")
    tool_ms = gen.get("tool_time_ms")
    # Retrieval = everything before rerank/generation. Only derivable when
    # the total and the later stages are known.
    retrieval_ms = None
    if total_ms is not None:
        known_later = (rerank_ms or 0) + (gen_ms or 0)
        retrieval_ms = max(0, int(total_ms) - known_later)
    return {
        "stages": {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "llm_ms": gen_ms,
            "tool_ms": tool_ms,
        },
        "funnel": {
            "lexical": debug.get("lexical_count"),
            "vector": debug.get("vector_count"),
            "fused": debug.get("fused_count"),
            "reranked": rerank.get("reranked"),
            "returned": debug.get("returned_count"),
            "llm_context": debug.get("llm_context_sent"),
            "cited": len(gen.get("cited_source_indices") or []) or None,
        },
        "queries": debug.get("queries"),
        "extra_queries": debug.get("extra_queries") or [],
        "vector_errors": debug.get("vector_errors") or [],
        "finish_reason": gen.get("finish_reason"),
        "tools": tools,
    }


@router.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard() -> HTMLResponse:
    """Human-readable analytics dashboard. Reads /admin/analytics + /admin/recent."""
    html_path = Path(__file__).resolve().parent / "static" / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/admin/eval", response_class=HTMLResponse)
def eval_dashboard() -> HTMLResponse:
    """Evaluation dashboard: golden-set runs, metric trends, per-case detail."""
    html_path = Path(__file__).resolve().parent / "static" / "eval.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.post("/eval/run")
def eval_run(
    payload: EvalRunRequest,
    db: DbSession,
    service: Annotated[QueryService, Depends(query_service_dep)],
    settings: Annotated[Settings, Depends(settings_dep)],
    principal: AdminPrincipal,
) -> dict[str, Any]:
    """Run the golden-set evaluation synchronously (~1-3 min with judge).

    Call after any retrieval/prompt/model change: `make eval`.
    """
    from app.eval.runner import run_eval

    run = run_eval(
        db,
        service,
        settings,
        label=payload.label or "",
        generate=payload.generate,
        judge=payload.judge,
        top_k=payload.top_k,
        case_ids=payload.case_ids or None,
    )
    return {"id": run.id, "status": run.status, "metrics": run.metrics}


def _is_sweep(run: Any) -> bool:
    return (run.config or {}).get("kind") == "sweep"


def _is_special(run: Any) -> bool:
    """Special-kind rows (sweep, pgrid) stay out of the normal runs lists."""
    return (run.config or {}).get("kind") is not None


@router.post("/eval/sweep")
def eval_sweep(
    payload: EvalSweepRequest,
    background: BackgroundTasks,
    db: DbSession,
    settings: Annotated[Settings, Depends(settings_dep)],
    principal: AdminPrincipal,
) -> dict[str, Any]:
    """Start a configuration sweep (ablation study) in the background.

    Each arm replays the whole golden set through a different pipeline, so a
    full sweep takes minutes; the row is written immediately with
    status="running" and filled in arm by arm. Poll GET /eval/sweeps.
    """
    from app.eval.sweep import DEFAULT_GROUPS, GROUPS
    from app.models.entities import EvalRun

    groups = tuple(payload.groups) or DEFAULT_GROUPS
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown sweep groups: {unknown}")

    run = EvalRun(
        label=payload.label or "configuration sweep",
        config={"kind": "sweep", "groups": list(groups)},
        status="running",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id
    case_ids = list(payload.case_ids or [])

    def _work() -> None:
        from app.db.session import SessionLocal
        from app.eval.sweep import run_sweep

        with SessionLocal() as session:
            row = session.get(EvalRun, run_id)
            if row is None:
                return
            try:
                run_sweep(session, settings, groups=groups, case_ids=case_ids or None, run=row)
            except Exception as exc:  # pragma: no cover - background guard
                logger.exception("eval_sweep_failed")
                row.status = "failed"
                row.metrics = {**(row.metrics or {}), "error": str(exc)[:300]}
                session.commit()

    background.add_task(_work)
    return {"id": run_id, "status": "running", "groups": list(groups)}


@router.post("/eval/playground")
def eval_playground(
    payload: EvalPlaygroundRequest,
    db: DbSession,
    settings: Annotated[Settings, Depends(settings_dep)],
    principal: AdminPrincipal,
) -> dict[str, Any]:
    """Run ONE ad-hoc configuration over the golden set, synchronously.

    Retrieval-only over all cases takes ~1 min; with answer generation each
    case adds a few seconds, so the UI defaults to a case subset there.
    Nothing is persisted — this is a scratchpad, not a benchmark record.
    """
    from app.eval.runner import _aggregate, case_failure_reasons, execute_cases, load_golden
    from app.eval.sweep import Arm, build_arm_service, generation_arms

    overrides: dict[str, Any] = {}
    if payload.rerank == "off":
        overrides["RERANK_PROVIDER"] = "none"
    else:
        overrides["RERANK_CANDIDATES"] = int(payload.rerank)
    if payload.llm_context_top_k:
        overrides["LLM_CONTEXT_TOP_K"] = payload.llm_context_top_k
    if payload.max_output_tokens:
        overrides["OPENAI_MAX_OUTPUT_TOKENS"] = payload.max_output_tokens
    if payload.generate and payload.model:
        # Reuse the sweep's model→provider wiring (base_url, key, pricing,
        # reasoning_effort) so any model the sweep can run, this can too.
        arm_settings = next(
            (a.settings for a in generation_arms(settings)
             if a.settings.get("OPENAI_GENERATION_MODEL") == payload.model),
            None,
        )
        if arm_settings is None:
            raise HTTPException(status_code=400, detail=f"unknown model: {payload.model}")
        overrides.update(arm_settings)

    arm = Arm(
        "playground", "playground", "playground",
        settings=overrides,
        retriever={
            "fusion": payload.fusion,
            "expand_queries": payload.expand_queries,
            "dedupe": payload.dedupe,
            "boosts": payload.boosts,
        },
        top_k=payload.top_k,
        generate=payload.generate,
    )
    service, arm_settings = build_arm_service(settings, arm)

    cases = load_golden()
    if payload.case_ids:
        wanted = set(payload.case_ids)
        cases = [c for c in cases if c.id in wanted]
    if payload.case_limit:
        cases = cases[: payload.case_limit]
    if not cases:
        raise HTTPException(status_code=400, detail="no matching golden cases")

    results = execute_cases(
        db, service, arm_settings, cases,
        generate=payload.generate, judge_client=None, top_k=payload.top_k,
    )
    failures = [
        {
            "id": r.get("id"),
            "query": r.get("query"),
            "reasons": case_failure_reasons(r),
            "missing_terms": (r.get("answer") or {}).get("missing_terms") or [],
        }
        for r in results
        if case_failure_reasons(r)
    ]
    return {
        "cases": len(cases),
        "metrics": _aggregate(results),
        "failures": failures,
        "config": {
            "fusion": payload.fusion,
            "expand_queries": payload.expand_queries,
            "dedupe": payload.dedupe,
            "boosts": payload.boosts,
            "rerank": payload.rerank,
            "top_k": payload.top_k,
            "model": payload.model if payload.generate else None,
            "llm_context_top_k": payload.llm_context_top_k,
            "max_output_tokens": payload.max_output_tokens,
        },
    }


@router.post("/eval/playground/grid")
def eval_playground_grid_start(
    background: BackgroundTasks,
    db: DbSession,
    settings: Annotated[Settings, Depends(settings_dep)],
    principal: AdminPrincipal,
) -> dict[str, Any]:
    """Precompute draft results for every discrete retrieval configuration
    (80 configs, retrieval-only, ~1 h in the background). The playground
    then shows an instant draft radar on any knob change."""
    from app.eval.grid import run_playground_grid
    from app.models.entities import EvalRun

    run = EvalRun(
        label="playground grid",
        config={"kind": "pgrid"},
        status="running",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id

    def _work() -> None:
        from app.db.session import SessionLocal

        with SessionLocal() as session:
            run_playground_grid(session, settings, run_id)

    background.add_task(_work)
    return {"id": run_id, "status": "running", "configs": 80}


@router.get("/eval/playground/grid")
def eval_playground_grid_get(
    db: DbSession,
    principal: AdminPrincipal,
) -> dict[str, Any]:
    """Latest precomputed grid (draft results per retrieval config), plus
    the in-flight run's progress if one is computing."""
    from app.eval.grid import latest_grid
    from app.models.entities import EvalRun

    payload = latest_grid(db) or {"grid": {}, "cases": None, "produced_at": None}
    running = db.execute(
        select(EvalRun)
        .where(EvalRun.status == "running")
        .order_by(desc(EvalRun.created_at))
        .limit(10)
    ).scalars()
    for run in running:
        if (run.config or {}).get("kind") == "pgrid":
            payload["computing"] = (run.metrics or {}).get("progress")
            break
    return payload


@router.get("/eval/playground/meta")
def eval_playground_meta(
    settings: Annotated[Settings, Depends(settings_dep)],
    principal: AdminPrincipal,
) -> dict[str, Any]:
    """Everything the playground UI needs to initialize honestly: the models
    it can run, the CURRENT production configuration (so controls start at
    what users actually get), and the real benchmark size."""
    from app.eval.runner import load_golden
    from app.eval.sweep import generation_arms

    models = [
        a.settings["OPENAI_GENERATION_MODEL"]
        for a in generation_arms(settings)
        if a.settings.get("OPENAI_GENERATION_MODEL")
    ]
    rerank_on = getattr(settings, "RERANK_PROVIDER", "none") == "cross_encoder"
    return {
        "models": models,
        "golden_cases": len(load_golden()),
        "defaults": {
            "fusion": "rrf",
            "rerank": str(settings.RERANK_CANDIDATES) if rerank_on else "off",
            "expand_queries": True,
            "dedupe": True,
            "boosts": True,
            "top_k": 8,
            "model": settings.OPENAI_GENERATION_MODEL,
            "llm_context_top_k": int(getattr(settings, "LLM_CONTEXT_TOP_K", 20) or 20),
        },
    }


@router.get("/eval/sweeps")
def eval_sweeps(db: DbSession, principal: AdminPrincipal, limit: int = 5) -> list[dict[str, Any]]:
    """Configuration-study runs, newest first (excluded from /eval/runs)."""
    from app.models.entities import EvalRun

    rows = db.scalars(
        select(EvalRun).order_by(desc(EvalRun.created_at)).limit(100)
    ).all()
    sweeps = [r for r in rows if _is_sweep(r)][: max(1, min(limit, 20))]
    return [
        {
            "id": r.id,
            "label": r.label,
            "status": r.status,
            "config": r.config,
            "metrics": r.metrics,
            "created_at": r.created_at,
            "finished_at": r.finished_at,
        }
        for r in sweeps
    ]


@router.get("/eval/runs")
def eval_runs(db: DbSession, principal: AdminPrincipal, limit: int = 30) -> list[dict[str, Any]]:
    from app.models.entities import EvalRun

    rows = [
        r
        for r in db.scalars(
            select(EvalRun).order_by(desc(EvalRun.created_at)).limit(min(limit, 200) + 20)
        ).all()
        if not _is_special(r)
    ][: min(limit, 200)]
    return [
        {
            "id": r.id,
            "label": r.label,
            "status": r.status,
            "cases_total": r.cases_total,
            "metrics": r.metrics,
            "config": r.config,
            "created_at": r.created_at,
            "finished_at": r.finished_at,
        }
        for r in rows
    ]


@router.get("/eval/cases")
def eval_cases(db: DbSession, principal: AdminPrincipal) -> dict[str, Any]:
    """The golden dataset itself + each case's latest result status."""
    from app.eval.runner import case_failure_reasons, load_golden
    from app.models.entities import EvalRun

    latest_by_case: dict[str, dict[str, Any]] = {}
    rows = [
        r
        for r in db.scalars(
            select(EvalRun)
            .where(EvalRun.status == "completed")
            .order_by(desc(EvalRun.created_at))
            .limit(30)
        ).all()
        if not _is_special(r)
    ][:10]
    for run in rows:
        for result in run.results or []:
            cid = result.get("id")
            if cid and cid not in latest_by_case:
                reasons = case_failure_reasons(result)
                latest_by_case[cid] = {
                    "status": "fail" if reasons else "pass",
                    "reasons": reasons,
                    "run_label": run.label,
                    "run_created_at": run.created_at.isoformat() if run.created_at else None,
                }
    cases = [
        {
            "id": c.id,
            "query": c.query,
            "expect_url_contains": c.expect_url_contains,
            "expect_answer_contains": c.expect_answer_contains,
            "tags": c.tags,
            "latest": latest_by_case.get(c.id),
        }
        for c in load_golden()
    ]
    return {"count": len(cases), "cases": cases}


@router.get("/eval/failures")
def eval_failures(db: DbSession, principal: AdminPrincipal, runs: int = 5) -> dict[str, Any]:
    """Standing worklist: failing cases from the latest completed runs.

    A case fixed in a newer run drops off automatically. Re-run just these
    via POST /eval/run {"case_ids": [...]} — e.g. after switching to a
    stronger generation model.
    """
    from app.eval.runner import collect_failures
    from app.models.entities import EvalRun

    rows = [
        r
        for r in db.scalars(select(EvalRun).order_by(desc(EvalRun.created_at)).limit(80)).all()
        if not _is_special(r)
    ]
    failures = collect_failures(rows, max_runs=max(1, min(runs, 20)))
    return {"count": len(failures), "failures": failures}


@router.get("/eval/runs/{run_id}")
def eval_run_detail(run_id: str, db: DbSession, principal: AdminPrincipal) -> dict[str, Any]:
    from app.models.entities import EvalRun

    run = db.get(EvalRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="eval run not found")
    return {
        "id": run.id,
        "label": run.label,
        "status": run.status,
        "cases_total": run.cases_total,
        "config": run.config,
        "metrics": run.metrics,
        "results": run.results,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
    }


@router.get("/popular", response_model=PopularQueriesResponse)
def popular(db: DbSession, _rl: RateLimited, window_days: int = 7, limit: int = 5) -> PopularQueriesResponse:
    """Top public queries from the last N days. Safe for unauthenticated clients."""
    return build_popular_queries(db, window_days=window_days, limit=limit)


@router.post("/people/refresh", response_model=PersonRefreshResponse)
def people_refresh(
    db: DbSession,
    settings: Annotated[Settings, Depends(settings_dep)],
    principal: AdminPrincipal,
) -> PersonRefreshResponse:
    """Rebuild the people graph from already-ingested sources + live Indico.

    Idempotent: re-runs are safe and merge into existing rows. Only callable
    when ``PEOPLE_LOOKUP_ENABLED=true`` to avoid populating a feature that
    is otherwise dormant.
    """
    if not getattr(settings, "PEOPLE_LOOKUP_ENABLED", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PEOPLE_LOOKUP_ENABLED must be true to refresh the people graph",
        )
    counts = rebuild_people(
        db,
        settings,
        indico_lookback_days=settings.PEOPLE_REFRESH_INDICO_LOOKBACK_DAYS,
        indico_lookahead_days=settings.PEOPLE_REFRESH_INDICO_LOOKAHEAD_DAYS,
    )
    return PersonRefreshResponse(counts=counts)


@router.post("/feedback", response_model=FeedbackRead)
def feedback(payload: FeedbackCreate, db: DbSession, _rl: RateLimited) -> FeedbackRead:
    record = Feedback(
        query_log_id=payload.query_log_id,
        query=payload.query,
        rating=payload.rating,
        comment=payload.comment,
    )
    db.add(record)
    db.commit()
    return FeedbackRead(id=record.id, created_at=record.created_at)


@router.delete("/admin/feedback/{feedback_id}")
def delete_feedback(feedback_id: str, db: DbSession, principal: AdminPrincipal) -> dict[str, str]:
    """Remove one feedback entry (spam / accidental clicks) from the dashboard."""
    record = db.get(Feedback, feedback_id)
    if not record:
        raise HTTPException(status_code=404, detail="feedback not found")
    db.delete(record)
    db.commit()
    return {"status": "deleted", "id": feedback_id}
