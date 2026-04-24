from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


Scope = Literal["public", "internal", "all"]


class Filters(BaseModel):
    source_names: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    section: str | None = None
    repo_path_prefix: str | None = None
    filetypes: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    scope: Scope = "public"
    filters: Filters = Field(default_factory=Filters)
    top_k: int = Field(default=8, ge=1, le=30)
    generate_answer: bool = True


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


class SourceRead(BaseModel):
    id: str
    name: str
    source_type: str
    base_url: str | None = None
    visibility: str
    enabled: bool
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChunkRead(BaseModel):
    id: str
    chunk_index: int
    content: str
    heading_path: list[str] = Field(default_factory=list)
    token_count: int
    visibility: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRead(BaseModel):
    id: str
    source_id: str
    external_id: str
    source_type: str
    source_name: str
    title: str
    url: str
    repo_path: str | None = None
    filetype: str | None = None
    visibility: str
    section_path: list[str] = Field(default_factory=list)
    content_hash: str
    last_updated: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: list[ChunkRead] = Field(default_factory=list)


class IngestRunRequest(BaseModel):
    source_names: list[str] = Field(default_factory=list)
    full_reindex: bool = False
    max_pages: int | None = Field(default=None, ge=1, le=5000)


class IngestRunResponse(BaseModel):
    job_ids: list[str]
    status: str
    stats: dict[str, Any] = Field(default_factory=dict)


class FeedbackCreate(BaseModel):
    query_log_id: str | None = None
    query: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=4000)
    selected_citation_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class PopularQuery(BaseModel):
    query: str
    count: int


class PopularQueriesResponse(BaseModel):
    window_days: int
    queries: list[PopularQuery] = Field(default_factory=list)

