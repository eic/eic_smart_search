import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.ingestion.base import Connector, DocumentPayload
from app.ingestion.chunking import HeadingAwareChunker
from app.ingestion.github import GitHubRepoConnector
from app.ingestion.github_org import GitHubOrgConnector
from app.ingestion.internal import EpicInternalCrawler
from app.ingestion.web import WebsiteCrawler
from app.ingestion.zenodo import ZenodoConnector
from app.models.entities import Chunk, CrawlJob, Document, IngestionState, PermissionMetadata, Source, new_id
from app.retrieval.qdrant_store import QdrantVectorStore
from app.services.query_cache import reset_query_cache

logger = logging.getLogger(__name__)


SOURCE_CONFIGS: dict[str, dict[str, Any]] = {
    "eic_website": {
        "source_type": "website",
        "base_url": "https://eic.github.io/",
        "visibility": "public",
    },
    "eicug_website": {
        "source_type": "website",
        "base_url": "https://www.eicug.org/",
        "visibility": "public",
    },
    "bnl_wiki_epic": {
        "source_type": "website",
        "base_url": "https://wiki.bnl.gov/EPIC/",
        "visibility": "public",
    },
    "eic_github_repo": {
        "source_type": "github_repo",
        "base_url": "https://github.com/eic/eic.github.io",
        "visibility": "public",
    },
    "eic_github_org": {
        "source_type": "github_org",
        "base_url": "https://github.com/orgs/eic/repositories",
        "visibility": "public",
    },
    "epic_internal": {
        "source_type": "internal_website",
        "base_url": "https://www.epic-eic.org/index-internal.html",
        "visibility": "public",
    },
    "zenodo_epic": {
        "source_type": "zenodo_community",
        "base_url": "https://zenodo.org/communities/epic/",
        "visibility": "public",
    },
}


class IngestionOrchestrator:
    def __init__(self, settings: Settings, vector_store: QdrantVectorStore) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.chunker = HeadingAwareChunker()

    def run(
        self,
        db: Session,
        source_names: list[str] | None = None,
        full_reindex: bool = False,
        max_pages: int | None = None,
        requested_by: str | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        names = source_names or ["eic_website", "eic_github_repo"]
        job_ids: list[str] = []
        stats: dict[str, Any] = {"sources": {}}
        any_content_changed = False
        for name in names:
            connector = self._connector(name)
            source = self._ensure_source(db, connector)
            job = CrawlJob(
                source_id=source.id,
                connector=connector.source_name,
                status="running",
                requested_by=requested_by,
                started_at=datetime.now(timezone.utc),
            )
            db.add(job)
            db.commit()
            job_ids.append(job.id)
            try:
                if full_reindex:
                    self._clear_source(db, source)
                documents = connector.iter_documents(max_items=max_pages)
                source_stats = self._ingest_documents(db, source, documents, full_reindex=full_reindex)
                self._record_state(db, source, source_stats)
                job.status = "completed"
                job.finished_at = datetime.now(timezone.utc)
                job.stats = source_stats
                stats["sources"][name] = source_stats
                db.commit()
                if full_reindex or source_stats.get("created", 0) + source_stats.get("updated", 0) > 0:
                    any_content_changed = True
            except Exception as exc:
                db.rollback()
                job = db.get(CrawlJob, job.id)
                if job:
                    job.status = "failed"
                    job.finished_at = datetime.now(timezone.utc)
                    job.error = str(exc)
                    db.commit()
                stats["sources"][name] = {"status": "failed", "error": str(exc)}
                logger.exception("ingestion_failed", extra={"source": name})
        if any_content_changed:
            reset_query_cache()
        return job_ids, stats

    def _connector(self, source_name: str) -> Connector:
        if source_name == "eic_website":
            return WebsiteCrawler(
                self.settings,
                source_name="eic_website",
                start_url=self.settings.EIC_SITE_URL,
                max_pages=self.settings.EIC_SITE_MAX_PAGES,
                exclude_prefixes=self.settings.EIC_SITE_EXCLUDE_PREFIXES,
            )
        if source_name == "eicug_website":
            return WebsiteCrawler(
                self.settings,
                source_name="eicug_website",
                start_url=self.settings.EICUG_SITE_URL,
                max_pages=self.settings.EICUG_SITE_MAX_PAGES,
                exclude_prefixes=self.settings.EICUG_SITE_EXCLUDE_PREFIXES,
            )
        if source_name == "bnl_wiki_epic":
            return WebsiteCrawler(
                self.settings,
                source_name="bnl_wiki_epic",
                start_url=self.settings.BNL_WIKI_URL,
                max_pages=self.settings.BNL_WIKI_MAX_PAGES,
                exclude_prefixes=self.settings.BNL_WIKI_EXCLUDE_PREFIXES,
                allowed_path_prefix="/EPIC/",
            )
        if source_name == "eic_github_repo":
            return GitHubRepoConnector(self.settings)
        if source_name == "eic_github_org":
            return GitHubOrgConnector(self.settings)
        if source_name == "epic_internal":
            return EpicInternalCrawler(self.settings)
        if source_name == "zenodo_epic":
            return ZenodoConnector(self.settings)
        raise ValueError(f"Unknown source_name: {source_name}")

    def _ensure_source(self, db: Session, connector: Connector) -> Source:
        config = SOURCE_CONFIGS[connector.source_name]
        source = db.scalar(select(Source).where(Source.name == connector.source_name))
        if source:
            source.source_type = connector.source_type
            source.base_url = config["base_url"]
            source.visibility = connector.visibility
            source.enabled = True
            source.config = config
        else:
            source = Source(
                name=connector.source_name,
                source_type=connector.source_type,
                base_url=config["base_url"],
                visibility=connector.visibility,
                config=config,
            )
            db.add(source)
        db.commit()
        db.refresh(source)
        return source

    def _clear_source(self, db: Session, source: Source) -> None:
        document_ids = list(db.scalars(select(Document.id).where(Document.source_id == source.id)))
        self.vector_store.delete_by_document_ids(document_ids)
        db.execute(delete(Document).where(Document.source_id == source.id))
        db.commit()

    def _ingest_documents(
        self,
        db: Session,
        source: Source,
        documents: list[DocumentPayload],
        full_reindex: bool,
    ) -> dict[str, Any]:
        stats = {"seen": len(documents), "created": 0, "updated": 0, "unchanged": 0, "chunks": 0}
        for payload in documents:
            existing = db.scalar(
                select(Document).where(
                    Document.source_id == source.id,
                    Document.external_id == payload.external_id,
                )
            )
            if existing and existing.content_hash == payload.content_hash and not full_reindex:
                stats["unchanged"] += 1
                continue
            if existing:
                self.vector_store.delete_by_document_ids([existing.id])
                db.execute(delete(Chunk).where(Chunk.document_id == existing.id))
                db.execute(delete(PermissionMetadata).where(PermissionMetadata.document_id == existing.id))
                document = existing
                stats["updated"] += 1
            else:
                document = Document(source_id=source.id, external_id=payload.external_id)
                db.add(document)
                stats["created"] += 1
            document.source_type = payload.source_type
            document.source_name = payload.source_name
            document.title = payload.title
            document.url = payload.url
            document.repo_path = payload.repo_path
            document.filetype = payload.filetype
            document.visibility = payload.visibility
            document.section_path = payload.section_path
            document.content_hash = payload.content_hash
            document.last_updated = payload.last_updated
            document.doc_metadata = payload.metadata
            db.flush()

            db.add(
                PermissionMetadata(
                    document_id=document.id,
                    visibility=payload.visibility,
                    allowed_groups=payload.metadata.get("allowed_groups", []),
                    denied_groups=payload.metadata.get("denied_groups", []),
                    policy_metadata={
                        "source_visibility": source.visibility,
                        "permission_sensitive": payload.metadata.get("permission_sensitive", False),
                    },
                )
            )
            chunks = self.chunker.chunk(payload)
            qdrant_items = []
            for chunk_payload in chunks:
                chunk_id = new_id()
                chunk = Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    source_id=source.id,
                    chunk_index=chunk_payload.chunk_index,
                    content=chunk_payload.content,
                    content_hash=chunk_payload.content_hash,
                    heading_path=chunk_payload.heading_path,
                    token_count=chunk_payload.token_count,
                    qdrant_point_id=chunk_id,
                    visibility=payload.visibility,
                    chunk_metadata=chunk_payload.metadata,
                )
                db.add(chunk)
                qdrant_items.append(
                    (
                        chunk_id,
                        chunk_payload.content,
                        self._qdrant_payload(source, document, chunk_payload.heading_path, chunk_payload.content_hash, chunk_id),
                    )
                )
            db.flush()
            self.vector_store.upsert_chunks(qdrant_items)
            stats["chunks"] += len(chunks)
            db.commit()
        return stats

    def _qdrant_payload(
        self,
        source: Source,
        document: Document,
        heading_path: list[str],
        content_hash: str,
        chunk_id: str,
    ) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "document_id": document.id,
            "source_id": source.id,
            "source_name": document.source_name,
            "source_type": document.source_type,
            "title": document.title,
            "url": document.url,
            "visibility": document.visibility,
            "section_path": document.section_path,
            "section_text": " / ".join(document.section_path + heading_path),
            "heading_path": heading_path,
            "repo_path": document.repo_path,
            "filetype": document.filetype,
            "last_updated": document.last_updated.isoformat() if document.last_updated else None,
            "content_hash": content_hash,
        }

    def _record_state(self, db: Session, source: Source, stats: dict[str, Any]) -> None:
        state = db.scalar(select(IngestionState).where(IngestionState.source_id == source.id, IngestionState.state_key == "latest"))
        value = {"stats": stats, "completed_at": datetime.now(timezone.utc).isoformat()}
        if state:
            state.state_value = value
        else:
            db.add(IngestionState(source_id=source.id, state_key="latest", state_value=value))

