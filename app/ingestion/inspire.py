"""INSPIRE-HEP literature ingestion.

Fetches EIC/ePIC literature records via the open INSPIRE REST API
(https://github.com/inspirehep/rest-api-doc) and emits one document per
record: title, abstract, authors (+ORCID when present), arXiv id, DOI,
journal reference.

Peer-reviewed papers rarely land in the ePIC Zenodo community, so this
fills the biggest literature gap. Author ORCIDs are stored in metadata and
harvested by the people-graph builder.

Rate limit: 15 requests / 5 s per IP — we sleep between pages to stay
well under it.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import httpx

from app.core.config import Settings
from app.ingestion.base import DocumentPayload
from app.ingestion.pdf_text import sanitize_text

logger = logging.getLogger(__name__)

API_URL = "https://inspirehep.net/api/literature"

# Only the fields we consume — trims the payload dramatically.
_FIELDS = ",".join(
    [
        "titles",
        "abstracts",
        "authors.full_name",
        "authors.affiliations",
        "authors.ids",
        "arxiv_eprints",
        "dois",
        "publication_info",
        "earliest_date",
        "collaborations",
        "document_type",
        "keywords",
    ]
)


def _first_title(meta: dict[str, Any]) -> str:
    for entry in meta.get("titles") or []:
        if isinstance(entry, dict) and entry.get("title"):
            return str(entry["title"]).strip()
    return ""


def _first_abstract(meta: dict[str, Any]) -> str:
    for entry in meta.get("abstracts") or []:
        if isinstance(entry, dict) and entry.get("value"):
            return str(entry["value"]).strip()
    return ""


def _author_entries(meta: dict[str, Any], cap: int) -> tuple[list[str], dict[str, str]]:
    """Return (["Last, First (Affiliation)", ...], {"Last, First": orcid})."""
    authors: list[str] = []
    orcids: dict[str, str] = {}
    for author in (meta.get("authors") or [])[:cap]:
        if not isinstance(author, dict):
            continue
        name = str(author.get("full_name") or "").strip()
        if not name:
            continue
        affiliation = ""
        for aff in author.get("affiliations") or []:
            if isinstance(aff, dict) and aff.get("value"):
                affiliation = str(aff["value"]).strip()
                break
        entry = f"{name} ({affiliation})" if affiliation else name
        authors.append(entry)
        for id_entry in author.get("ids") or []:
            if isinstance(id_entry, dict) and id_entry.get("schema") == "ORCID" and id_entry.get("value"):
                orcids[name] = str(id_entry["value"]).strip()
                break
    return authors, orcids


def _arxiv_id(meta: dict[str, Any]) -> str:
    for entry in meta.get("arxiv_eprints") or []:
        if isinstance(entry, dict) and entry.get("value"):
            return str(entry["value"]).strip()
    return ""


def _doi(meta: dict[str, Any]) -> str:
    for entry in meta.get("dois") or []:
        if isinstance(entry, dict) and entry.get("value"):
            return str(entry["value"]).strip()
    return ""


def _journal_ref(meta: dict[str, Any]) -> str:
    for entry in meta.get("publication_info") or []:
        if not isinstance(entry, dict):
            continue
        title = entry.get("journal_title") or ""
        if not title:
            continue
        volume = entry.get("journal_volume") or ""
        year = entry.get("year") or ""
        pieces = [str(title)]
        if volume:
            pieces.append(str(volume))
        if year:
            pieces.append(f"({year})")
        return " ".join(pieces)
    return ""


def _keywords(meta: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for entry in meta.get("keywords") or []:
        if isinstance(entry, dict) and entry.get("value"):
            out.append(str(entry["value"]).strip())
    return out[:20]


def _parse_date(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


class InspireHepConnector:
    """Ingest INSPIRE-HEP literature matching a configurable query."""

    source_type = "inspire_literature"
    visibility = "public"

    PAGE_SIZE = 25
    # 15 req / 5 s allowed; one page per 0.5 s keeps a wide margin.
    PAGE_SLEEP_S = 0.5

    def __init__(self, settings: Settings, *, source_name: str = "inspire_hep") -> None:
        self.settings = settings
        self.source_name = source_name
        self.query = getattr(settings, "INSPIRE_QUERY", 'collaboration "ePIC"')
        self.max_records = int(getattr(settings, "INSPIRE_MAX_RECORDS", 500))
        self.author_cap = int(getattr(settings, "INSPIRE_MAX_AUTHORS_PER_PAPER", 60))

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        limit = int(max_items) if max_items else self.max_records
        headers = {
            "Accept": "application/json",
            "User-Agent": self.settings.INGEST_USER_AGENT,
        }
        documents: list[DocumentPayload] = []
        with httpx.Client(timeout=self.settings.REQUEST_TIMEOUT_SECONDS, headers=headers) as client:
            for record in self._iter_records(client, limit=limit):
                try:
                    doc = self._build_document(record)
                except Exception as exc:
                    logger.warning(
                        "inspire_record_skipped",
                        extra={"id": record.get("id"), "error": str(exc)[:200]},
                    )
                    continue
                if doc is not None:
                    documents.append(doc)
        return documents

    def _iter_records(self, client: httpx.Client, limit: int):
        page = 1
        collected = 0
        while collected < limit:
            response = client.get(
                API_URL,
                params={
                    "q": self.query,
                    "sort": "mostrecent",
                    "size": self.PAGE_SIZE,
                    "page": page,
                    "fields": _FIELDS,
                },
            )
            response.raise_for_status()
            payload = response.json()
            hits = (payload.get("hits") or {}).get("hits") or []
            if not hits:
                return
            for record in hits:
                yield record
                collected += 1
                if collected >= limit:
                    return
            if len(hits) < self.PAGE_SIZE:
                return
            page += 1
            time.sleep(self.PAGE_SLEEP_S)

    def _build_document(self, record: dict[str, Any]) -> DocumentPayload | None:
        meta = record.get("metadata") or {}
        record_id = record.get("id")
        if record_id is None:
            return None
        title = _first_title(meta) or f"INSPIRE record {record_id}"
        abstract = _first_abstract(meta)
        authors, orcids = _author_entries(meta, self.author_cap)
        arxiv_id = _arxiv_id(meta)
        doi = _doi(meta)
        journal = _journal_ref(meta)
        keywords = _keywords(meta)
        collaborations = [
            str(c.get("value")).strip()
            for c in meta.get("collaborations") or []
            if isinstance(c, dict) and c.get("value")
        ]
        doc_types = [str(t) for t in meta.get("document_type") or [] if isinstance(t, str)]
        url = f"https://inspirehep.net/literature/{record_id}"

        body_parts: list[str] = []
        if abstract:
            body_parts.append(abstract)
        if authors:
            body_parts.append("Authors: " + "; ".join(authors))
        if collaborations:
            body_parts.append("Collaboration: " + ", ".join(collaborations))
        if journal:
            body_parts.append(f"Published in: {journal}")
        if arxiv_id:
            body_parts.append(f"arXiv: {arxiv_id}")
        if doi:
            body_parts.append(f"DOI: {doi}")
        if keywords:
            body_parts.append("Keywords: " + ", ".join(keywords))
        content = sanitize_text("\n\n".join(body_parts).strip())
        if len(content.split()) < 10:
            return None

        return DocumentPayload(
            source_name=self.source_name,
            source_type=self.source_type,
            external_id=f"inspire:{record_id}",
            title=title,
            url=url,
            content=content,
            visibility=self.visibility,
            section_path=["inspire"] + (doc_types[:1] if doc_types else []),
            last_updated=_parse_date(meta.get("earliest_date")),
            metadata={
                "inspire_id": record_id,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "journal": journal,
                "authors": authors,
                "author_orcids": orcids,
                "collaborations": collaborations,
                "document_type": doc_types,
                "keywords": keywords,
            },
        )
