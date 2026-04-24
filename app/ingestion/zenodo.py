"""Zenodo community ingestion.

Fetches every record in a given community via the Zenodo REST API and emits
one document per record. Each document contains:
  - Title + description (HTML-stripped)
  - Authors, publication date, keywords, resource type, DOI
  - Extracted PDF text (capped) when the record has a PDF file

PDF text extraction uses ``pypdf`` — pure Python, no native deps. PDFs above
``ZENODO_PDF_MAX_BYTES`` are skipped (metadata still indexed). Extracted text
is truncated to ``ZENODO_PDF_MAX_CHARS`` characters to keep chunks sane.
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from html import unescape
from typing import Any

import httpx

from app.core.config import Settings
from app.ingestion.base import DocumentPayload

logger = logging.getLogger(__name__)


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    # Drop script/style wholesale, then all tags.
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.I | re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _sanitize_text(cleaned).strip()


# PostgreSQL text columns reject NUL bytes, and PDFs routinely leak them
# along with other C0 control chars. Strip them everywhere before indexing.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_text(text: str) -> str:
    if not text:
        return text
    return _CONTROL_CHARS_RE.sub("", text)


def _authors(creators: list[dict[str, Any]] | None) -> list[str]:
    if not isinstance(creators, list):
        return []
    out: list[str] = []
    for c in creators:
        if isinstance(c, dict):
            name = c.get("name") or ""
            affiliation = c.get("affiliation") or ""
            if isinstance(name, str) and name.strip():
                entry = name.strip()
                if isinstance(affiliation, str) and affiliation.strip():
                    entry = f"{entry} ({affiliation.strip()})"
                out.append(entry)
    return out


def _keywords(raw: list[Any] | None) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(k).strip() for k in raw if isinstance(k, str) and k.strip()]


def _parse_date(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


class ZenodoConnector:
    """Ingest a Zenodo community as search documents."""

    source_type = "zenodo_community"
    visibility = "public"

    def __init__(
        self,
        settings: Settings,
        *,
        source_name: str = "zenodo_epic",
        community: str | None = None,
        visibility: str = "public",
    ) -> None:
        self.settings = settings
        self.source_name = source_name
        self.community = community or getattr(settings, "ZENODO_COMMUNITY", "epic")
        self.visibility = visibility
        self.api_base = "https://zenodo.org/api"
        self.max_records = int(getattr(settings, "ZENODO_MAX_RECORDS", 200))
        self.pdf_max_bytes = int(getattr(settings, "ZENODO_PDF_MAX_BYTES", 25 * 1024 * 1024))
        self.pdf_max_chars = int(getattr(settings, "ZENODO_PDF_MAX_CHARS", 120_000))
        self.include_pdf_text = bool(getattr(settings, "ZENODO_INCLUDE_PDF_TEXT", True))

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
                    doc = self._build_document(client, record)
                except Exception as exc:
                    logger.warning(
                        "zenodo_record_skipped",
                        extra={"id": record.get("id"), "error": str(exc)[:200]},
                    )
                    continue
                if doc is not None:
                    documents.append(doc)
        return documents

    # ---- paging ---------------------------------------------------------

    def _iter_records(self, client: httpx.Client, limit: int):
        # Zenodo community records endpoint caps page size at 25.
        PAGE_SIZE = 25
        page = 1
        collected = 0
        while collected < limit:
            response = client.get(
                f"{self.api_base}/communities/{self.community}/records",
                params={"size": PAGE_SIZE, "page": page, "sort": "newest"},
            )
            if response.status_code == 404:
                logger.warning("zenodo_community_not_found", extra={"community": self.community})
                return
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
            if len(hits) < PAGE_SIZE:
                return
            page += 1

    # ---- doc build ------------------------------------------------------

    def _build_document(
        self,
        client: httpx.Client,
        record: dict[str, Any],
    ) -> DocumentPayload | None:
        meta = record.get("metadata") or {}
        record_id = record.get("id")
        if record_id is None:
            return None
        title = (meta.get("title") or "").strip() or f"Zenodo record {record_id}"
        description = _strip_html(meta.get("description") or "")
        resource_type = (meta.get("resource_type") or {}).get("title") or meta.get("resource_type", {}).get("type")
        doi = meta.get("doi") or record.get("doi") or ""
        authors = _authors(meta.get("creators"))
        keywords = _keywords(meta.get("keywords"))
        publication_date = _parse_date(meta.get("publication_date"))

        links = record.get("links") or {}
        html_url = links.get("self_html") or links.get("html") or f"https://zenodo.org/records/{record_id}"

        files = record.get("files") or []
        pdf_file = next((f for f in files if isinstance(f, dict) and self._is_pdf(f)), None)
        pdf_text = ""
        pdf_meta: dict[str, Any] = {}
        if pdf_file and self.include_pdf_text:
            pdf_text, pdf_meta = self._extract_pdf_text(client, pdf_file)

        body_parts: list[str] = []
        if description:
            body_parts.append(description)
        if authors:
            body_parts.append("Authors: " + "; ".join(authors))
        if keywords:
            body_parts.append("Keywords: " + ", ".join(keywords))
        if doi:
            body_parts.append(f"DOI: {doi}")
        if pdf_text:
            body_parts.append("Full text (extracted):\n" + pdf_text)
        content = _sanitize_text("\n\n".join(part for part in body_parts if part).strip())
        if len(content.split()) < 10:
            # Skip empty metadata stubs.
            return None

        metadata: dict[str, Any] = {
            "zenodo_id": record_id,
            "doi": doi,
            "resource_type": resource_type,
            "authors": authors,
            "keywords": keywords,
            "community": self.community,
        }
        if pdf_meta:
            metadata["pdf"] = pdf_meta

        return DocumentPayload(
            source_name=self.source_name,
            source_type=self.source_type,
            external_id=f"zenodo:{record_id}",
            title=title,
            url=html_url,
            content=content,
            visibility=self.visibility,
            section_path=[self.community, resource_type] if resource_type else [self.community],
            last_updated=publication_date,
            filetype="pdf" if pdf_file else None,
            metadata=metadata,
        )

    def _is_pdf(self, file_entry: dict[str, Any]) -> bool:
        key = file_entry.get("key") or ""
        mimetype = file_entry.get("mimetype") or ""
        if isinstance(key, str) and key.lower().endswith(".pdf"):
            return True
        if isinstance(mimetype, str) and "pdf" in mimetype.lower():
            return True
        return False

    def _extract_pdf_text(
        self,
        client: httpx.Client,
        file_entry: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        size = int(file_entry.get("size") or 0)
        key = file_entry.get("key") or ""
        links = file_entry.get("links") or {}
        download_url = links.get("content") or links.get("self") or links.get("download")
        if not download_url or not isinstance(download_url, str):
            return "", {"key": key, "size": size, "skipped": "no-url"}
        if size and size > self.pdf_max_bytes:
            return "", {"key": key, "size": size, "skipped": "too-large", "max_bytes": self.pdf_max_bytes}
        try:
            response = client.get(download_url, follow_redirects=True, timeout=60)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return "", {"key": key, "size": size, "skipped": "download-failed", "error": str(exc)[:200]}

        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            return "", {"key": key, "size": size, "skipped": "pypdf-missing"}

        try:
            reader = PdfReader(io.BytesIO(response.content))
        except Exception as exc:
            return "", {"key": key, "size": size, "skipped": "pdf-parse-failed", "error": str(exc)[:200]}

        chunks: list[str] = []
        char_budget = self.pdf_max_chars
        pages_read = 0
        for page in reader.pages:
            if char_budget <= 0:
                break
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if not text:
                continue
            pages_read += 1
            if len(text) > char_budget:
                chunks.append(text[:char_budget])
                char_budget = 0
            else:
                chunks.append(text)
                char_budget -= len(text)
        full = _sanitize_text("\n\n".join(chunks))
        return full, {
            "key": key,
            "size": size,
            "total_pages": len(reader.pages),
            "pages_extracted": pages_read,
            "chars_extracted": len(full),
            "truncated": char_budget <= 0 and len(full) >= self.pdf_max_chars,
        }
