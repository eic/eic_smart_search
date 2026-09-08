"""Indico event + material ingestion into the hybrid index.

The live `search_indico_events` LLM tool stays the authority for
time-relative questions ("next DIRC meeting"). This connector complements
it by baking past/announced events INTO the retrieval index, so plain
content queries ("what was discussed about the dRICH readout") hit meeting
agendas and slide decks without any tool round-trip — much faster for
users and it works even with generation disabled.

Emits:
  * one document per event  — agenda: title, date, chairs, contributions
    with speakers
  * one document per PDF attachment (slides, minutes) — text extracted via
    ``app.ingestion.pdf_text`` with size/char caps

Content-hash idempotency in the orchestrator means unchanged events skip
re-embedding on every run, so scheduling this frequently is cheap.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from app.core.config import Settings
from app.ingestion.base import DocumentPayload
from app.ingestion.pdf_text import extract_pdf_text, extract_pptx_text, sanitize_text
from app.ingestion.safe_fetch import BlockedURLError, safe_get
from app.integrations.indico import IndicoClient, _clean_html

logger = logging.getLogger(__name__)


def _dt_iso(part: Any) -> str:
    if not isinstance(part, dict):
        return ""
    date = part.get("date") or ""
    tstr = part.get("time") or ""
    if not date:
        return ""
    return f"{date}T{tstr}" if tstr else date


def _parse_dt(part: Any) -> datetime | None:
    raw = _dt_iso(part)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _person_names(raw_list: Any) -> list[str]:
    names: list[str] = []
    for person in raw_list or []:
        if not isinstance(person, dict):
            continue
        name = str(
            person.get("fullName")
            or person.get("name")
            or " ".join(
                filter(None, [person.get("first_name"), person.get("last_name")])
            )
        ).strip()
        if name:
            affil = person.get("affiliation")
            if isinstance(affil, str) and affil.strip():
                name = f"{name} ({affil.strip()})"
            names.append(name)
    return names


def iter_attachments(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten Indico's folders → attachments structure on one node."""
    out: list[dict[str, Any]] = []
    for folder in node.get("folders") or []:
        if not isinstance(folder, dict):
            continue
        for att in folder.get("attachments") or []:
            if isinstance(att, dict):
                out.append(att)
    # Some Indico versions also expose a flat "material" list.
    for att in node.get("material") or []:
        if isinstance(att, dict):
            out.append(att)
    return out


def _attachment_kind(att: dict[str, Any]) -> str | None:
    """'pdf' | 'pptx' | None for unsupported types."""
    content_type = str(att.get("content_type") or "").lower()
    filename = str(att.get("filename") or att.get("title") or "").lower()
    if "pdf" in content_type or filename.endswith(".pdf"):
        return "pdf"
    if "presentationml" in content_type or filename.endswith((".pptx", ".ppsx")):
        return "pptx"
    return None


def _is_pdf_attachment(att: dict[str, Any]) -> bool:  # kept for tests
    return _attachment_kind(att) == "pdf"


class IndicoEventsConnector:
    """Ingest Indico events (+ material PDFs) as search documents."""

    source_type = "indico"
    visibility = "public"

    def __init__(self, settings: Settings, *, source_name: str = "indico_events") -> None:
        self.settings = settings
        self.source_name = source_name
        self.lookback_days = int(getattr(settings, "INDICO_INGEST_LOOKBACK_DAYS", 730))
        self.lookahead_days = int(getattr(settings, "INDICO_INGEST_LOOKAHEAD_DAYS", 60))
        self.include_materials = bool(getattr(settings, "INDICO_INGEST_MATERIALS", True))
        self.material_max_bytes = int(getattr(settings, "INDICO_MATERIAL_MAX_BYTES", 15 * 1024 * 1024))
        self.material_max_chars = int(getattr(settings, "INDICO_MATERIAL_MAX_CHARS", 60_000))
        self.material_max_per_event = int(getattr(settings, "INDICO_MATERIAL_MAX_PER_EVENT", 10))
        self.client = IndicoClient(
            category_url=settings.INDICO_CATEGORY_URL,
            cache_ttl_s=0,  # ingestion always wants fresh data
            timeout_s=float(getattr(settings, "PEOPLE_REFRESH_INDICO_TIMEOUT_S", 60.0)),
            user_agent=settings.INGEST_USER_AGENT,
        )

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        raw_events = self.client.fetch_raw_events(
            from_date=f"-{self.lookback_days}d",
            to_date=f"{self.lookahead_days}d",
            detail="contributions",
        )
        if max_items:
            raw_events = raw_events[: int(max_items)]
        documents: list[DocumentPayload] = []
        # Materials get their own generous timeout: plenary decks run
        # >100 MB and Indico throttles bulk crawlers to ~50 KB/s.
        material_timeout = float(getattr(self.settings, "INDICO_MATERIAL_TIMEOUT_S", 600.0))
        with httpx.Client(
            timeout=httpx.Timeout(material_timeout, connect=20),
            headers={"User-Agent": self.settings.INGEST_USER_AGENT},
            follow_redirects=True,
        ) as client:
            for ev in raw_events:
                try:
                    event_doc = self._build_event_document(ev)
                except Exception as exc:
                    logger.warning(
                        "indico_event_skipped",
                        extra={"id": ev.get("id"), "error": str(exc)[:200]},
                    )
                    continue
                if event_doc is not None:
                    documents.append(event_doc)
                if self.include_materials:
                    documents.extend(self._build_material_documents(client, ev))
        return documents

    # ---- event agenda document ------------------------------------------

    def _build_event_document(self, ev: dict[str, Any]) -> DocumentPayload | None:
        ev_id = str(ev.get("id") or "").strip()
        title = str(ev.get("title") or "").strip()
        url = str(ev.get("url") or "").strip()
        if not ev_id or not title:
            return None
        category = str(ev.get("category") or "").strip()
        start_iso = _dt_iso(ev.get("startDate"))
        chairs = _person_names(ev.get("chairs"))
        description = _clean_html(ev.get("description"))

        lines: list[str] = []
        if start_iso:
            lines.append(f"Date: {start_iso}")
        if category:
            lines.append(f"Category: {category}")
        if chairs:
            lines.append("Chairs: " + "; ".join(chairs))
        if description:
            lines.append(description)
        contributions = [c for c in ev.get("contributions") or [] if isinstance(c, dict)]
        if contributions:
            lines.append("Agenda:")
            for contrib in contributions:
                c_title = str(contrib.get("title") or "").strip()
                if not c_title:
                    continue
                speakers = _person_names(contrib.get("speakers"))
                c_when = _dt_iso(contrib.get("startDate"))
                entry = f"- {c_title}"
                if speakers:
                    entry += " — " + "; ".join(speakers)
                if c_when:
                    entry += f" ({c_when})"
                lines.append(entry)
        content = sanitize_text("\n\n".join(lines).strip())
        if len(content.split()) < 5:
            return None

        return DocumentPayload(
            source_name=self.source_name,
            source_type=self.source_type,
            external_id=f"indico:{ev_id}",
            title=title,
            url=url or self.client.category_url,
            content=content,
            visibility=self.visibility,
            section_path=["indico", category] if category else ["indico"],
            last_updated=_parse_dt(ev.get("startDate")),
            metadata={
                "indico_id": ev_id,
                "category": category,
                "start": start_iso,
                "chairs": chairs,
                "contribution_count": len(contributions),
            },
        )

    # ---- material (slides / minutes) documents --------------------------

    def _build_material_documents(
        self,
        client: httpx.Client,
        ev: dict[str, Any],
    ) -> list[DocumentPayload]:
        ev_id = str(ev.get("id") or "").strip()
        ev_title = str(ev.get("title") or "").strip()
        ev_url = str(ev.get("url") or "").strip()
        category = str(ev.get("category") or "").strip()
        ev_when = _parse_dt(ev.get("startDate"))

        # (attachment, owning contribution title) pairs, event-level first.
        candidates: list[tuple[dict[str, Any], str]] = [
            (att, "") for att in iter_attachments(ev)
        ]
        for contrib in ev.get("contributions") or []:
            if not isinstance(contrib, dict):
                continue
            c_title = str(contrib.get("title") or "").strip()
            candidates.extend((att, c_title) for att in iter_attachments(contrib))

        documents: list[DocumentPayload] = []
        taken = 0
        for att, contrib_title in candidates:
            if taken >= self.material_max_per_event:
                break
            kind = _attachment_kind(att)
            if kind is None:
                continue
            download_url = str(att.get("download_url") or att.get("url") or "").strip()
            if not download_url:
                continue
            size = int(att.get("size") or 0)
            if size and size > self.material_max_bytes:
                continue
            att_title = str(att.get("title") or att.get("filename") or "attachment").strip()
            try:
                response = safe_get(client, download_url)
                response.raise_for_status()
            except (BlockedURLError, httpx.HTTPError) as exc:
                logger.warning(
                    "indico_material_download_failed",
                    extra={"url": download_url, "error": str(exc)[:200]},
                )
                continue
            if len(response.content) > self.material_max_bytes:
                continue
            if kind == "pptx":
                text, pdf_meta = extract_pptx_text(response.content, max_chars=self.material_max_chars)
            else:
                text, pdf_meta = extract_pdf_text(response.content, max_chars=self.material_max_chars)
            if len(text.split()) < 20:
                continue
            taken += 1
            header_bits = [b for b in [ev_title, contrib_title, att_title] if b]
            title = " — ".join(header_bits[:3])
            documents.append(
                DocumentPayload(
                    source_name=self.source_name,
                    source_type=self.source_type,
                    external_id=f"indico:{ev_id}:att:{att.get('id') or att_title}",
                    title=title,
                    url=download_url,
                    content=sanitize_text(text),
                    visibility=self.visibility,
                    section_path=["indico", category, "materials"] if category else ["indico", "materials"],
                    last_updated=ev_when,
                    filetype=kind,
                    metadata={
                        "indico_id": ev_id,
                        "event_title": ev_title,
                        "event_url": ev_url,
                        "contribution": contrib_title,
                        "category": category,
                        "pdf": pdf_meta,
                    },
                )
            )
        return documents
