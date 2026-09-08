"""Curated service-directory connector.

Reads ``services.yaml`` (live EIC/ePIC tools with their direct URLs) and
emits one document per service, so "where is X / hydra monitoring / event
display" questions retrieve the entry and the answer leads with the
direct link. Zero network — the registry is the source of truth.
"""
from __future__ import annotations

import logging

import yaml

from app.core.config import Settings
from app.core.profile import services_path
from app.ingestion.base import DocumentPayload

logger = logging.getLogger(__name__)

# Curated per collaboration: profiles/<PROFILE>/services.yaml.
REGISTRY_PATH = services_path()


class ServicesRegistryConnector:
    source_type = "service_directory"
    visibility = "public"

    def __init__(self, settings: Settings, *, source_name: str = "eic_services") -> None:
        self.settings = settings
        self.source_name = source_name

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        try:
            raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or []
        except Exception as exc:
            logger.warning("services_registry_load_failed", extra={"error": str(exc)[:200]})
            return []
        documents: list[DocumentPayload] = []
        for item in raw[: max_items or len(raw)]:
            if not isinstance(item, dict) or not item.get("name") or not item.get("url"):
                continue
            name = str(item["name"]).strip()
            url = str(item["url"]).strip()
            description = " ".join(str(item.get("description") or "").split())
            aliases = [str(a).strip() for a in item.get("aliases") or []]
            content_parts = [
                f"{name} — live service. Direct link: {url}",
                description,
            ]
            if aliases:
                content_parts.append("Also known as: " + ", ".join(aliases))
            documents.append(
                DocumentPayload(
                    source_name=self.source_name,
                    source_type=self.source_type,
                    external_id=f"service:{name.lower().replace(' ', '-')}",
                    title=f"{name} (service)",
                    url=url,
                    content="\n\n".join(p for p in content_parts if p),
                    visibility=self.visibility,
                    section_path=["services"],
                    metadata={"service": True, "aliases": aliases},
                )
            )
        return documents
