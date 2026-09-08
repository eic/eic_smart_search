"""arXiv preprint ingestion (metadata + abstract).

Queries the free arXiv Atom API for EIC-related preprints. Papers appear
here months before Zenodo/INSPIRE indexing catches up, so this is the
freshness source. Full-text PDFs are intentionally NOT fetched — abstracts
carry enough retrieval signal and keep ingestion fast.

API docs: https://info.arxiv.org/help/api/user-manual.html
Politeness: arXiv asks for max 1 request / 3 s — we page slowly.
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import httpx

from app.core.config import Settings
from app.ingestion.base import DocumentPayload
from app.ingestion.pdf_text import sanitize_text

logger = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def _parse_dt(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def parse_atom_feed(xml_text: str) -> list[dict[str, Any]]:
    """Parse an arXiv Atom feed into plain entry dicts (unit-testable)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("arxiv_feed_parse_failed", extra={"error": str(exc)[:200]})
        return []
    entries: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", _NS):
        abs_url = _text(entry.find("atom:id", _NS))
        # id looks like http://arxiv.org/abs/2401.12345v2 — strip version.
        arxiv_id = abs_url.rsplit("/abs/", 1)[-1] if "/abs/" in abs_url else abs_url
        base_id = arxiv_id.rsplit("v", 1)[0] if "v" in arxiv_id.split("/")[-1] else arxiv_id
        authors = [
            _text(a.find("atom:name", _NS))
            for a in entry.findall("atom:author", _NS)
            if _text(a.find("atom:name", _NS))
        ]
        categories = [
            c.get("term") or ""
            for c in entry.findall("atom:category", _NS)
            if c.get("term")
        ]
        entries.append(
            {
                "arxiv_id": base_id,
                "url": abs_url.replace("http://", "https://"),
                "title": _text(entry.find("atom:title", _NS)),
                "summary": _text(entry.find("atom:summary", _NS)),
                "authors": authors,
                "categories": categories,
                "published": _text(entry.find("atom:published", _NS)),
                "updated": _text(entry.find("atom:updated", _NS)),
                "doi": _text(entry.find("arxiv:doi", _NS)),
                "comment": _text(entry.find("arxiv:comment", _NS)),
            }
        )
    return entries


class ArxivConnector:
    """Ingest arXiv preprints matching a configurable search query."""

    source_type = "arxiv"
    visibility = "public"

    PAGE_SIZE = 100
    PAGE_SLEEP_S = 3.0  # arXiv API politeness requirement

    def __init__(self, settings: Settings, *, source_name: str = "arxiv_eic") -> None:
        self.settings = settings
        self.source_name = source_name
        self.query = getattr(
            settings,
            "ARXIV_QUERY",
            'abs:"electron-ion collider" OR abs:"ePIC detector"',
        )
        self.max_records = int(getattr(settings, "ARXIV_MAX_RECORDS", 300))

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        limit = int(max_items) if max_items else self.max_records
        headers = {"User-Agent": self.settings.INGEST_USER_AGENT}
        documents: list[DocumentPayload] = []
        start = 0
        with httpx.Client(timeout=self.settings.REQUEST_TIMEOUT_SECONDS, headers=headers) as client:
            while start < limit:
                page_size = min(self.PAGE_SIZE, limit - start)
                try:
                    response = client.get(
                        API_URL,
                        params={
                            "search_query": self.query,
                            "start": start,
                            "max_results": page_size,
                            "sortBy": "submittedDate",
                            "sortOrder": "descending",
                        },
                    )
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.warning("arxiv_fetch_failed", extra={"start": start, "error": str(exc)[:200]})
                    break
                entries = parse_atom_feed(response.text)
                if not entries:
                    break
                for entry in entries:
                    doc = self._build_document(entry)
                    if doc is not None:
                        documents.append(doc)
                if len(entries) < page_size:
                    break
                start += len(entries)
                time.sleep(self.PAGE_SLEEP_S)
        return documents

    def _build_document(self, entry: dict[str, Any]) -> DocumentPayload | None:
        arxiv_id = entry.get("arxiv_id") or ""
        title = entry.get("title") or ""
        summary = entry.get("summary") or ""
        if not arxiv_id or not title:
            return None
        authors = entry.get("authors") or []
        categories = entry.get("categories") or []

        body_parts: list[str] = []
        if summary:
            body_parts.append(summary)
        if authors:
            body_parts.append("Authors: " + "; ".join(authors))
        if categories:
            body_parts.append("Categories: " + ", ".join(categories))
        if entry.get("comment"):
            body_parts.append(f"Comments: {entry['comment']}")
        if entry.get("doi"):
            body_parts.append(f"DOI: {entry['doi']}")
        body_parts.append(f"arXiv: {arxiv_id}")
        content = sanitize_text("\n\n".join(body_parts).strip())
        if len(content.split()) < 10:
            return None

        return DocumentPayload(
            source_name=self.source_name,
            source_type=self.source_type,
            external_id=f"arxiv:{arxiv_id}",
            title=title,
            url=entry.get("url") or f"https://arxiv.org/abs/{arxiv_id}",
            content=content,
            visibility=self.visibility,
            section_path=["arxiv"] + categories[:1],
            last_updated=_parse_dt(entry.get("published") or ""),
            metadata={
                "arxiv_id": arxiv_id,
                "doi": entry.get("doi") or "",
                "authors": authors,
                "categories": categories,
                "published": entry.get("published") or "",
            },
        )
