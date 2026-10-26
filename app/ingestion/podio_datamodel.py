"""PODIO data-model connector: edm4eic.yaml / edm4hep.yaml → per-type docs.

The data-model YAML is the authoritative answer surface for "which
collection/member/relation" questions ("what does edm4eic::Track contain?",
"how are Cherenkov PID objects linked to reconstructed particles?"). The
crawled Doxygen renders the same information as navigation chrome; this
connector emits one clean markdown document per datatype / component / link
instead, with the Doxygen class page as the citation URL.

The YAML on ``main`` can be newer than the published Doxygen (built from the
latest release), so the class page is only cited when the site's
``annotated.html`` index lists it; otherwise the citation points at the
declaring line of the YAML on GitHub (``metadata["doc_page"] == "yaml"``).

Sources come from ``DATAMODEL_YAML_URLS`` (comma-separated raw-file URLs,
e.g. https://raw.githubusercontent.com/eic/EDM4eic/main/edm4eic.yaml).
"""
from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

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


_CLASS_PAGE_RE = re.compile(r"""href=["']?(class\w+?_1_1\w+?\.html)""")
# Top-level type keys in the YAML: '  edm4eic::Track:' (optionally quoted).
_TYPE_KEY_RE = re.compile(r"""^\s*["']?(?P<name>\w+::\w+)["']?\s*:""")


def _doxygen_filename(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def _parse_annotated(html: str) -> set[str]:
    """Class-page filenames referenced by a Doxygen ``annotated.html``."""
    return set(_CLASS_PAGE_RE.findall(html))


def _yaml_blob_url(source_url: str) -> str:
    """Raw/GitHub YAML URL → human GitHub blob URL (unchanged if unknown)."""
    parsed = urlparse(source_url)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    if host == "raw.githubusercontent.com" and len(parts) >= 4:
        owner, repo, rest = parts[0], parts[1], parts[2:]
        # .../refs/heads/<ref>/path or .../refs/tags/<ref>/path
        if len(rest) >= 4 and rest[0] == "refs" and rest[1] in {"heads", "tags"}:
            rest = rest[2:]
        ref, path = rest[0], "/".join(rest[1:])
        return f"https://github.com/{owner}/{repo}/blob/{ref}/{path}"
    if host == "github.com" and len(parts) >= 5 and parts[2] in {"raw", "blob"}:
        parts[2] = "blob"
        return "https://github.com/" + "/".join(parts)
    return source_url


def _type_line_numbers(yaml_text: str) -> dict[str, int]:
    """First (1-based) line declaring each ``ns::Name:`` key."""
    lines: dict[str, int] = {}
    for number, line in enumerate(yaml_text.splitlines(), start=1):
        match = _TYPE_KEY_RE.match(line)
        if match:
            lines.setdefault(match.group("name"), number)
    return lines


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
        self.user_agent = getattr(settings, "INGEST_USER_AGENT", "eic-smart-search")
        # Doxygen base → published class-page filenames (None = unknown).
        self._doxygen_index: dict[str, set[str] | None] = {}

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        documents: list[DocumentPayload] = []
        self._doxygen_index = {}
        with httpx.Client(
            timeout=self.timeout, headers={"User-Agent": self.user_agent}
        ) as client:
            for url in self.yaml_urls:
                try:
                    response = safe_get(client, url)
                    response.raise_for_status()
                    yaml_text = response.text
                    model = yaml.safe_load(yaml_text) or {}
                except Exception as exc:
                    logger.warning(
                        "datamodel_fetch_failed", extra={"url": url, "error": str(exc)[:200]}
                    )
                    continue
                documents.extend(
                    self._model_documents(
                        model, source_url=url, yaml_text=yaml_text, client=client
                    )
                )
        if max_items:
            documents = documents[:max_items]
        return documents

    # ---- Doxygen index ------------------------------------------------------

    def _published_pages(self, base: str, client: httpx.Client | None) -> set[str] | None:
        """Class pages listed in ``<base>annotated.html``; fetched once per run.

        ``None`` means unknown (no client or fetch failed) — callers then keep
        the Doxygen URL, as before this check existed.
        """
        if base in self._doxygen_index:
            return self._doxygen_index[base]
        pages: set[str] | None = None
        if client is not None:
            try:
                # Fixed, hard-coded hosts (not remote-supplied) → plain GET.
                response = client.get(
                    f"{base}annotated.html",
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                    follow_redirects=True,
                )
                response.raise_for_status()
                pages = _parse_annotated(response.text) or None
            except Exception as exc:
                logger.warning(
                    "doxygen_index_fetch_failed", extra={"base": base, "error": str(exc)[:200]}
                )
        self._doxygen_index[base] = pages
        return pages

    def _citation_url(
        self,
        qualified: str,
        source_url: str,
        type_lines: dict[str, int],
        client: httpx.Client | None,
    ) -> tuple[str, str]:
        """(url, doc_page) — Doxygen page if published, else the YAML line."""
        doxygen_url = _doxygen_class_url(qualified)
        if doxygen_url:
            base = _DOXYGEN_BASE[qualified.partition("::")[0]]
            pages = self._published_pages(base, client)
            if pages is None or _doxygen_filename(doxygen_url) in pages:
                return doxygen_url, "doxygen"
        blob = _yaml_blob_url(source_url)
        line = type_lines.get(qualified)
        return (f"{blob}#L{line}" if line else blob), "yaml"

    # ---- document building -----------------------------------------------

    def _model_documents(
        self,
        model: dict[str, Any],
        *,
        source_url: str,
        yaml_text: str = "",
        client: httpx.Client | None = None,
    ) -> list[DocumentPayload]:
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

        type_lines = _type_line_numbers(yaml_text) if yaml_text else {}
        documents: list[DocumentPayload] = []
        for kind, entries in (("datatype", datatypes), ("component", components), ("link", links)):
            for qualified, spec in entries.items():
                url, doc_page = self._citation_url(qualified, source_url, type_lines, client)
                documents.append(
                    self._type_document(
                        model_name,
                        kind,
                        qualified,
                        spec or {},
                        referenced_by,
                        source_url,
                        url=url,
                        doc_page=doc_page,
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
        *,
        url: str = "",
        doc_page: str = "",
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

        if not url:
            url = _doxygen_class_url(qualified) or source_url
            doc_page = "doxygen" if url != source_url else "yaml"
        return DocumentPayload(
            source_name=self.source_name,
            source_type=self.source_type,
            external_id=f"datamodel:{qualified}",
            title=f"{qualified} — {model_name.upper()} data model {kind}",
            url=url,
            content="\n".join(lines),
            visibility=self.visibility,
            section_path=[model_name.upper(), f"{kind}s"],
            filetype="yaml",
            metadata={"datamodel": model_name, "kind": kind, "doc_page": doc_page},
        )
