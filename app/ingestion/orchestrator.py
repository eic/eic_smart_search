import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.ingestion.arxiv import ArxivConnector
from app.ingestion.base import Connector, DocumentPayload
from app.ingestion.chunking import HeadingAwareChunker
from app.ingestion.github import GitHubRepoConnector
from app.ingestion.github_discussions import GitHubDiscussionsConnector
from app.ingestion.github_issues import GitHubIssuesConnector
from app.ingestion.github_org import GitHubOrgConnector
from app.ingestion.github_releases import GitHubReleasesConnector
from app.ingestion.indico_events import IndicoEventsConnector
from app.ingestion.inspire import InspireHepConnector
from app.ingestion.epic_definitions import EpicDefinitionsConnector
from app.ingestion.internal import EpicInternalCrawler
from app.ingestion.podio_datamodel import PodioDatamodelConnector
from app.ingestion.services_registry import ServicesRegistryConnector
from app.ingestion.web import WebsiteCrawler
from app.ingestion.zenodo import ZenodoConnector
from app.models.entities import Chunk, CrawlJob, Document, Source, new_id
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
    "inspire_hep": {
        "source_type": "inspire_literature",
        "base_url": "https://inspirehep.net/",
        "visibility": "public",
    },
    "arxiv_eic": {
        "source_type": "arxiv",
        "base_url": "https://arxiv.org/",
        "visibility": "public",
    },
    "indico_events": {
        "source_type": "indico",
        "base_url": "https://indico.bnl.gov/",
        "visibility": "public",
    },
    "eic_github_issues": {
        "source_type": "github_issues",
        "base_url": "https://github.com/eic",
        "visibility": "public",
    },
    "eic_github_releases": {
        "source_type": "github_releases",
        "base_url": "https://github.com/eic",
        "visibility": "public",
    },
    "eic_github_discussions": {
        "source_type": "github_discussions",
        "base_url": "https://github.com/eic",
        "visibility": "public",
    },
    "eic_services": {
        "source_type": "service_directory",
        "base_url": "https://eic.github.io/",
        "visibility": "public",
    },
    "edm4eic_datamodel": {
        "source_type": "podio_datamodel",
        "base_url": "https://github.com/eic/EDM4eic",
        "visibility": "public",
    },
    "epic_definitions": {
        "source_type": "dd4hep_constants",
        "base_url": "https://github.com/eic/epic",
        "visibility": "public",
    },
    "eicrecon_docs": {
        "source_type": "github_repo",
        "base_url": "https://github.com/eic/EICrecon",
        "visibility": "public",
    },
    "eic_knowledge_guide": {
        "source_type": "github_repo",
        "base_url": "https://github.com/aprozo/eic",
        "visibility": "public",
    },
    "epic_prod": {
        "source_type": "website",
        "base_url": "https://eic.github.io/epic-prod/",
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
                exclude_path_contains=self.settings.EIC_SITE_EXCLUDE_PATH_CONTAINS,
            )
        if source_name == "eicug_website":
            return WebsiteCrawler(
                self.settings,
                source_name="eicug_website",
                start_url=self.settings.EICUG_SITE_URL,
                max_pages=self.settings.EICUG_SITE_MAX_PAGES,
                exclude_prefixes=self.settings.EICUG_SITE_EXCLUDE_PREFIXES,
                exclude_path_contains="",
            )
        if source_name == "bnl_wiki_epic":
            return WebsiteCrawler(
                self.settings,
                source_name="bnl_wiki_epic",
                start_url=self.settings.BNL_WIKI_URL,
                max_pages=self.settings.BNL_WIKI_MAX_PAGES,
                exclude_prefixes=self.settings.BNL_WIKI_EXCLUDE_PREFIXES,
                exclude_path_contains="",
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
        if source_name == "inspire_hep":
            return InspireHepConnector(self.settings)
        if source_name == "arxiv_eic":
            return ArxivConnector(self.settings)
        if source_name == "indico_events":
            return IndicoEventsConnector(self.settings)
        if source_name == "eic_github_issues":
            return GitHubIssuesConnector(self.settings)
        if source_name == "eic_github_releases":
            return GitHubReleasesConnector(self.settings)
        if source_name == "eic_github_discussions":
            return GitHubDiscussionsConnector(self.settings)
        if source_name == "eic_services":
            return ServicesRegistryConnector(self.settings)
        if source_name == "edm4eic_datamodel":
            return PodioDatamodelConnector(self.settings)
        if source_name == "epic_definitions":
            return EpicDefinitionsConnector(self.settings)
        if source_name == "eicrecon_docs":
            return GitHubRepoConnector(
                self.settings,
                source_name="eicrecon_docs",
                owner="eic",
                repo="EICrecon",
                ref="main",
                path_prefix="docs/",
            )
        if source_name == "eic_knowledge_guide":
            return GitHubRepoConnector(
                self.settings,
                source_name="eic_knowledge_guide",
                owner="aprozo",
                repo="eic",
                ref="main",
                path_prefix="docs/",
            )
        if source_name == "epic_prod":
            return WebsiteCrawler(
                self.settings,
                source_name="epic_prod",
                start_url="https://eic.github.io/epic-prod/",
                max_pages=300,
                exclude_prefixes=self.settings.EIC_SITE_EXCLUDE_PREFIXES,
                exclude_path_contains="",
                allowed_path_prefix="/epic-prod/",
            )
        raise ValueError(f"Unknown source_name: {source_name}")

    def _ensure_source(self, db: Session, connector: Connector) -> Source:
        config = SOURCE_CONFIGS[connector.source_name]
        source = db.scalar(select(Source).where(Source.name == connector.source_name))
        if source:
            source.source_type = connector.source_type
            source.base_url = config["base_url"]
            source.visibility = connector.visibility
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

