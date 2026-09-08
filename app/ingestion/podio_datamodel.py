"""PODIO data-model connector: edm4eic.yaml / edm4hep.yaml → per-type docs.

The data-model YAML is the authoritative answer surface for "which
collection/member/relation" questions ("what does edm4eic::Track contain?",
"how are Cherenkov PID objects linked to reconstructed particles?"). The
crawled Doxygen renders the same information as navigation chrome; this
connector emits one clean markdown document per datatype / component / link
instead, with the Doxygen class page as the citation URL.

Sources come from ``DATAMODEL_YAML_URLS`` (comma-separated raw-file URLs,
e.g. https://raw.githubusercontent.com/eic/EDM4eic/main/edm4eic.yaml).
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx
import yaml

from app.core.config import Settings
from app.ingestion.base import DocumentPayload
from app.ingestion.safe_fetch import safe_get

logger = logging.getLogger(__name__)

# Doxygen site per model namespace (citation URLs land on the class page).
_DOXYGEN_BASE = {
    "edm4eic": "https://eic.github.io/EDM4eic/",
    "edm4hep": "https://edm4hep.web.cern.ch/",
}

# Member lines look like: "edm4hep::Vector3f position // [mm] position of the hit"
_MEMBER_RE = re.compile(r"^\s*(?P<type>[\w:<>,\s]+?)\s+(?P<name>\w+)\s*(?://\s*(?P<comment>.*))?$")


def _doxygen_class_url(qualified: str) -> str:
    """'edm4eic::TrackClusterMatch' → Doxygen class page URL.

    Doxygen file naming: CamelCase → lowercase with '_' before each upper,
    '::' → '_1_1'.
    """
    namespace, _, name = qualified.partition("::")
    base = _DOXYGEN_BASE.get(namespace)
    if not base or not name:
        return ""
    snake = re.sub(r"([A-Z])", r"_\1", name).lower()
    return f"{base}class{namespace}_1_1{snake}.html"


def _parse_member(raw: Any) -> tuple[str, str, str] | None:
    match = _MEMBER_RE.match(str(raw).strip())
    if not match:
        return None
    return (
        match.group("type").strip(),
        match.group("name").strip(),
        (match.group("comment") or "").strip(),
    )


def _members_section(heading: str, entries: list[Any]) -> list[str]:
    if not entries:
        return []
    lines = [f"### {heading}", ""]
    for raw in entries:
        parsed = _parse_member(raw)
        if parsed is None:
            lines.append(f"- `{str(raw).strip()}`")
            continue
        type_, name, comment = parsed
        line = f"- `{name}` (`{type_}`)"
        if comment:
            line += f" — {comment}"
        lines.append(line)
    lines.append("")
    return lines


class PodioDatamodelConnector:
    source_type = "podio_datamodel"
    visibility = "public"

    def __init__(self, settings: Settings, *, source_name: str = "edm4eic_datamodel") -> None:
        self.settings = settings
        self.source_name = source_name
        raw_urls = str(getattr(settings, "DATAMODEL_YAML_URLS", "") or "")
        self.yaml_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
        self.timeout = getattr(settings, "REQUEST_TIMEOUT_SECONDS", 20.0)

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        documents: list[DocumentPayload] = []
        with httpx.Client(timeout=self.timeout) as client:
            for url in self.yaml_urls:
                try:
                    response = safe_get(client, url)
                    response.raise_for_status()
                    model = yaml.safe_load(response.text) or {}
                except Exception as exc:
                    logger.warning(
                        "datamodel_fetch_failed", extra={"url": url, "error": str(exc)[:200]}
                    )
                    continue
                documents.extend(self._model_documents(model, source_url=url))
        if max_items:
            documents = documents[:max_items]
        return documents

    # ---- document building -----------------------------------------------

    def _model_documents(self, model: dict[str, Any], *, source_url: str) -> list[DocumentPayload]:
        datatypes: dict[str, Any] = model.get("datatypes") or {}
        components: dict[str, Any] = model.get("components") or {}
        links: dict[str, Any] = model.get("links") or {}
        model_name = self._model_name(datatypes, components, source_url)

        # Reverse index: which datatypes reference each type (relations).
        referenced_by: dict[str, set[str]] = {}
        for qualified, spec in datatypes.items():
            for key in ("OneToOneRelations", "OneToManyRelations"):
                for raw in (spec or {}).get(key) or []:
                    parsed = _parse_member(raw)
                    if parsed:
                        referenced_by.setdefault(parsed[0], set()).add(qualified)

        documents: list[DocumentPayload] = []
        for kind, entries in (("datatype", datatypes), ("component", components), ("link", links)):
            for qualified, spec in entries.items():
                documents.append(
                    self._type_document(
                        model_name, kind, qualified, spec or {}, referenced_by, source_url
                    )
                )
        logger.info(
            "datamodel_parsed",
            extra={"model": model_name, "documents": len(documents), "url": source_url},
        )
        return documents

    @staticmethod
    def _model_name(datatypes: dict, components: dict, source_url: str) -> str:
        for qualified in (*datatypes, *components):
            namespace = str(qualified).partition("::")[0]
            if namespace:
                return namespace
        # Fallback: file stem of the URL (edm4eic.yaml → edm4eic).
        return source_url.rsplit("/", 1)[-1].removesuffix(".yaml")

    def _type_document(
        self,
        model_name: str,
        kind: str,
        qualified: str,
        spec: dict[str, Any],
        referenced_by: dict[str, set[str]],
        source_url: str,
    ) -> DocumentPayload:
        description = " ".join(str(spec.get("Description") or "").split())
        lines: list[str] = [
            f"`{qualified}` is a {kind} in the {model_name.upper()} data model"
            + (f": {description}" if description else ".")
        ]
        if kind == "datatype":
            lines.append(
                f"Event data appears in PODIO collections of `{qualified}` "
                f"(e.g. read via the collection name in the output tree)."
            )
        lines.append("")
        lines += _members_section("Members", spec.get("Members") or [])
        lines += _members_section("One-to-one relations", spec.get("OneToOneRelations") or [])
        lines += _members_section("One-to-many relations", spec.get("OneToManyRelations") or [])
        lines += _members_section("Vector members", spec.get("VectorMembers") or [])
        if kind == "link":
            from_type = str(spec.get("From") or "").strip()
            to_type = str(spec.get("To") or "").strip()
            if from_type or to_type:
                lines.append(f"Links `{from_type}` ↔ `{to_type}`.")
                lines.append("")
        refs = sorted(referenced_by.get(qualified, ()))
        if refs:
            lines.append("### Referenced by")
            lines.append("")
            lines += [f"- `{r}`" for r in refs]
            lines.append("")
        lines.append(f"Definition source: {source_url}")

        doxygen_url = _doxygen_class_url(qualified)
        return DocumentPayload(
            source_name=self.source_name,
            source_type=self.source_type,
            external_id=f"datamodel:{qualified}",
            title=f"{qualified} — {model_name.upper()} data model {kind}",
            url=doxygen_url or source_url,
            content="\n".join(lines),
            visibility=self.visibility,
            section_path=[model_name.upper(), f"{kind}s"],
            filetype="yaml",
            metadata={"datamodel": model_name, "kind": kind},
        )
