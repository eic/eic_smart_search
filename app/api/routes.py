from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import desc, select, text
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.ingestion.orchestrator import IngestionOrchestrator
from app.models.entities import CrawlJob, Document, Feedback, Source
from app.schemas.api import (
    AnalyticsResponse,
    ChunkRead,
    DocumentRead,
    FeedbackCreate,
    FeedbackRead,
    IngestRunRequest,
    IngestRunResponse,
    PopularQueriesResponse,
    QueryRequest,
    QueryResponse,
    ReindexRequest,
    SourceRead,
)
from app.services.analytics import build_analytics, build_popular_queries
from app.services.factory import build_ingestion_orchestrator, build_query_service, build_vector_store
from app.services.query import QueryService

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
    global _ingestion_singleton
    if _ingestion_singleton is None:
        _ingestion_singleton = build_ingestion_orchestrator(settings)
    return _ingestion_singleton


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(db: DbSession, settings: Annotated[Settings, Depends(settings_dep)]) -> dict[str, str]:
    try:
        db.execute(text("SELECT 1"))
        build_vector_store(settings).ensure_collection()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "error": str(exc)},
        ) from exc
    return {"status": "ready"}


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    db: DbSession,
    service: Annotated[QueryService, Depends(query_service_dep)],
) -> QueryResponse:
    return service.query(db, request)


@router.post("/ingest/run", response_model=IngestRunResponse)
def ingest_run(
    payload: IngestRunRequest,
    request: Request,
    db: DbSession,
    orchestrator: Annotated[IngestionOrchestrator, Depends(ingestion_dep)],
) -> IngestRunResponse:
    requested_by = request.headers.get("x-user") or (request.client.host if request.client else None)
    job_ids, stats = orchestrator.run(
        db,
        source_names=payload.source_names,
        full_reindex=payload.full_reindex,
        max_pages=payload.max_pages,
        requested_by=requested_by,
    )
    return IngestRunResponse(job_ids=job_ids, status="completed", stats=stats)


@router.get("/sources", response_model=list[SourceRead])
def sources(db: DbSession) -> list[Source]:
    return list(db.scalars(select(Source).order_by(Source.name)))


@router.get("/documents/{document_id}", response_model=DocumentRead)
def document(document_id: str, db: DbSession) -> DocumentRead:
    record = db.scalar(
        select(Document)
        .where(Document.id == document_id)
        .options(selectinload(Document.chunks))
    )
    if not record:
        raise HTTPException(status_code=404, detail="document not found")
    chunks = sorted(record.chunks, key=lambda chunk: chunk.chunk_index)
    return DocumentRead(
        id=record.id,
        source_id=record.source_id,
        external_id=record.external_id,
        source_type=record.source_type,
        source_name=record.source_name,
        title=record.title,
        url=record.url,
        repo_path=record.repo_path,
        filetype=record.filetype,
        visibility=record.visibility,
        section_path=record.section_path,
        content_hash=record.content_hash,
        last_updated=record.last_updated,
        metadata=record.doc_metadata,
        chunks=[
            ChunkRead(
                id=chunk.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                heading_path=chunk.heading_path,
                token_count=chunk.token_count,
                visibility=chunk.visibility,
                metadata=chunk.chunk_metadata,
            )
            for chunk in chunks
        ],
    )


@router.get("/admin/jobs")
def admin_jobs(db: DbSession, limit: int = 50) -> list[dict]:
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
) -> IngestRunResponse:
    requested_by = request.headers.get("x-user") or (request.client.host if request.client else None)
    job_ids, stats = orchestrator.run(
        db,
        source_names=payload.source_names,
        full_reindex=True,
        max_pages=payload.max_pages,
        requested_by=requested_by,
    )
    return IngestRunResponse(job_ids=job_ids, status="completed", stats=stats)


@router.get("/admin/analytics", response_model=AnalyticsResponse)
def admin_analytics(db: DbSession, window_days: int = 7, limit: int = 20) -> AnalyticsResponse:
    return build_analytics(db, window_days=window_days, limit=limit)


@router.get("/popular", response_model=PopularQueriesResponse)
def popular(db: DbSession, window_days: int = 7, limit: int = 5) -> PopularQueriesResponse:
    """Top public queries from the last N days. Safe for unauthenticated clients."""
    return build_popular_queries(db, window_days=window_days, limit=limit)


@router.post("/feedback", response_model=FeedbackRead)
def feedback(payload: FeedbackCreate, db: DbSession) -> FeedbackRead:
    record = Feedback(
        query_log_id=payload.query_log_id,
        query=payload.query,
        rating=payload.rating,
        comment=payload.comment,
        selected_citation_ids=payload.selected_citation_ids,
        feedback_metadata=payload.metadata,
    )
    db.add(record)
    db.commit()
    return FeedbackRead(id=record.id, created_at=record.created_at)
