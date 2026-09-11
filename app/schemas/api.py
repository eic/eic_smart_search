from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


Scope = Literal["public", "internal", "all"]


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    scope: Scope = "public"
    # Previous turns of this conversation (oldest first), so follow-up
    # questions ("what about the other one?") carry their context. Capped
    # at 3 exchanges; history-bearing requests bypass the response cache.
    history: list[HistoryTurn] = Field(default_factory=list, max_length=6)
    top_k: int = Field(default=8, ge=1, le=30)
    generate_answer: bool = True
    # Evaluation harness flag: bypass the response cache (probe AND write)
    # and skip the QueryLog so eval traffic never pollutes analytics or
    # serves stale answers to real users. Stripped for non-admin callers
    # (see routes._enforce_public_limits).
    eval_mode: bool = False
    # Eval-only: apply the cross-encoder even on retrieval-only calls so
    # the measured ranking matches what answer generation actually sees.
    force_rerank: bool = False
    # Caller kind, set by the route from the User-Agent ("cli" for eic-ask)
    # and logged so the dashboard can keep CLI traffic out of popular queries.
    client: str | None = Field(default=None, max_length=32)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    url: str
    snippet: str
    source_name: str
    source_type: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval_debug: dict[str, Any] = Field(default_factory=dict)
    query_log_id: str | None = None
    # "Who to ask" referral, set when answer confidence is low: the person
    # most associated with the top sources (from the people graph), so a
    # weak answer still routes the user to a human expert.
    expert_hint: str | None = None


class IngestRunRequest(BaseModel):
    source_names: list[str] = Field(default_factory=list)
    full_reindex: bool = False
    max_pages: int | None = Field(default=None, ge=1, le=5000)


class IngestRunResponse(BaseModel):
    job_ids: list[str]
    status: str
    stats: dict[str, Any] = Field(default_factory=dict)


class FeedbackCreate(BaseModel):
    # Older widget builds also send selected_citation_ids / metadata —
    # pydantic ignores unknown fields, so they are simply dropped.
    query_log_id: str | None = None
    query: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=4000)


class FeedbackUpdate(BaseModel):
    comment: str = Field(min_length=1, max_length=4000)


class FeedbackRead(BaseModel):
    id: str
    created_at: datetime


class ReindexRequest(BaseModel):
    source_names: list[str] = Field(default_factory=list)
    max_pages: int | None = Field(default=None, ge=1, le=5000)


class AnalyticsQueryCount(BaseModel):
    query: str
    count: int
    last_asked_at: datetime
    avg_top_score: float | None = None
    answered_count: int = 0


class AnalyticsDailyPoint(BaseModel):
    day: str
    queries: int
    answered: int
    zero_result: int
    avg_latency_ms: float | None = None
    p95_latency_ms: int | None = None
    total_cost_usd: float = 0.0


class AnalyticsSummary(BaseModel):
    window_days: int
    total_queries: int
    answered_queries: int
    zero_result_queries: int
    unique_queries: int
    avg_latency_ms: float | None = None
    p50_latency_ms: int | None = None
    p95_latency_ms: int | None = None
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    confidence_breakdown: dict[str, int] = Field(default_factory=dict)
    provider_breakdown: dict[str, int] = Field(default_factory=dict)


class AnalyticsFeedbackStat(BaseModel):
    id: str | None = None
    query: str
    rating: int
    comment: str | None = None
    created_at: datetime


class AnalyticsResponse(BaseModel):
    summary: AnalyticsSummary
    daily: list[AnalyticsDailyPoint] = Field(default_factory=list)
    top_queries: list[AnalyticsQueryCount] = Field(default_factory=list)
    zero_result_queries: list[AnalyticsQueryCount] = Field(default_factory=list)
    low_confidence_queries: list[AnalyticsQueryCount] = Field(default_factory=list)
    recent_low_rated: list[AnalyticsFeedbackStat] = Field(default_factory=list)
    recent_feedback: list[AnalyticsFeedbackStat] = Field(default_factory=list)


class PopularQuery(BaseModel):
    query: str
    count: int


class PopularQueriesResponse(BaseModel):
    window_days: int
    queries: list[PopularQuery] = Field(default_factory=list)


class PersonRefreshResponse(BaseModel):
    counts: dict[str, int]


class EvalRunRequest(BaseModel):
    label: str | None = Field(default=None, max_length=200)
    generate: bool = True
    judge: bool = True
    top_k: int = Field(default=8, ge=1, le=30)
    case_ids: list[str] = Field(default_factory=list)


class EvalPlaygroundRequest(BaseModel):
    """One ad-hoc configuration run over the golden set (see /admin/eval).

    Results are returned directly and NOT stored — the playground is for
    trying parameter combinations before committing them to a sweep or prod.
    """

    fusion: str = Field(default="rrf", pattern="^(rrf|weighted)$")
    expand_queries: bool = True
    dedupe: bool = True
    boosts: bool = True
    rerank: str = Field(default="20", pattern=r"^(off|\d{1,3})$")
    top_k: int = Field(default=8, ge=1, le=30)
    generate: bool = False
    model: str | None = Field(default=None, max_length=120)
    llm_context_top_k: int | None = Field(default=None, ge=1, le=50)
    max_output_tokens: int | None = Field(default=None, ge=100, le=2000)
    case_limit: int | None = Field(default=None, ge=1, le=200)
    case_ids: list[str] = Field(default_factory=list)


class EvalSweepRequest(BaseModel):
    """Configuration study: replay the golden set through alternative
    pipelines (see app/eval/sweep.py). Runs in the background — poll
    GET /eval/sweeps."""

    label: str | None = Field(default=None, max_length=200)
    groups: list[str] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)

