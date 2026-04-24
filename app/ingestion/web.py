import logging
import re
from collections import deque
from urllib.parse import parse_qs, urldefrag, urlparse

import httpx

from app.core.config import Settings
from app.ingestion.base import DocumentPayload
from app.ingestion.html import extract_html

logger = logging.getLogger(__name__)


def _parse_prefixes(raw: str | list[str] | None) -> tuple[str, ...]:
    if not raw:
        return ()
    if isinstance(raw, list):
        items = raw
    else:
        items = [item.strip() for item in str(raw).split(",") if item.strip()]
    return tuple(item.rstrip("/") + "/" for item in items if item)


class WebsiteCrawler:
    """BFS crawler for a single-origin HTML site.

    Same crawler, three configurations: eic.github.io, www.eicug.org, and
    wiki.bnl.gov. Start URL, max pages, and path-prefix exclusions are all
    injected so the orchestrator can spin up different `source_name` instances.
    """

    source_type = "website"

    def __init__(
        self,
        settings: Settings,
        *,
        source_name: str = "eic_website",
        start_url: str | None = None,
        max_pages: int | None = None,
        exclude_prefixes: str | list[str] | None = None,
        allowed_path_prefix: str | None = None,
        visibility: str = "public",
    ) -> None:
        self.settings = settings
        self.source_name = source_name
        self.start_url = start_url or settings.EIC_SITE_URL
        self.max_pages = int(max_pages or settings.EIC_SITE_MAX_PAGES)
        self.visibility = visibility
        self.allowed_netloc = urlparse(self.start_url).netloc
        # Optional positive scope (e.g. "/EPIC/") so a multi-wiki host only
        # follows links inside the target wiki tree. Normalized to end with "/".
        self.allowed_path_prefix = (
            allowed_path_prefix.rstrip("/") + "/"
            if allowed_path_prefix
            else None
        )
        self.exclude_prefixes = _parse_prefixes(
            exclude_prefixes
            if exclude_prefixes is not None
            else getattr(settings, "EIC_SITE_EXCLUDE_PREFIXES", "")
        )

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        limit = int(max_items) if max_items else self.max_pages
        seen: set[str] = set()
        queue: deque[str] = deque([self.start_url])
        documents: list[DocumentPayload] = []
        headers = {"User-Agent": self.settings.INGEST_USER_AGENT}
        with httpx.Client(
            timeout=self.settings.REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=headers,
        ) as client:
            while queue and len(documents) < limit:
                url = self._canonicalize(queue.popleft())
                if not url or url in seen:
                    continue
                seen.add(url)
                try:
                    response = client.get(url)
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.warning("crawl_fetch_failed", extra={"url": url, "error": str(exc)})
                    continue
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type:
                    continue
                extracted = extract_html(
                    response.text,
                    str(response.url),
                    response.headers.get("last-modified"),
                )
                for link in extracted.links:
                    next_url = self._canonicalize(link)
                    if next_url and next_url not in seen:
                        queue.append(next_url)
                if len(extracted.content.split()) < 25:
                    continue
                documents.append(
                    DocumentPayload(
                        source_name=self.source_name,
                        source_type=self.source_type,
                        external_id=str(response.url),
                        title=extracted.title,
                        url=str(response.url),
                        content=extracted.content,
                        visibility=self.visibility,
                        section_path=extracted.breadcrumbs or self._section_from_url(str(response.url)),
                        last_updated=extracted.last_updated,
                        metadata={
                            "headings": extracted.headings[:30],
                            "breadcrumbs": extracted.breadcrumbs,
                            "crawler": "website",
                            "netloc": self.allowed_netloc,
                        },
                    )
                )
        return documents

    def _canonicalize(self, url: str) -> str | None:
        clean_url, _ = urldefrag(url)
        parsed = urlparse(clean_url)
        if parsed.scheme not in {"http", "https"}:
            return None
        if parsed.netloc != self.allowed_netloc:
            return None
        lowered = clean_url.lower()
        if any(lowered.endswith(ext) for ext in [
            ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar.gz",
            ".xml", ".svg", ".root", ".obj", ".stl", ".glb", ".gltf",
            ".woff", ".woff2", ".ttf", ".ico", ".css", ".js", ".map",
        ]):
            return None
        normalized_path = re.sub(r"/+", "/", parsed.path)
        if self.allowed_path_prefix and not normalized_path.startswith(self.allowed_path_prefix):
            return None
        if self.exclude_prefixes and any(normalized_path.startswith(prefix) for prefix in self.exclude_prefixes):
            return None
        # MediaWiki produces many non-content URL variants of the same page
        # (?action=edit, ?action=history, &oldid=, etc.) — skip those. Also
        # filter query-form noise-namespace URLs like ?title=Special:Foo which
        # the path-prefix check can't catch because the path is /index.php.
        if parsed.query:
            lowered_q = parsed.query.lower()
            if any(bad in lowered_q for bad in (
                "action=edit", "action=history", "action=raw", "action=info",
                "action=delete", "action=protect", "action=watch", "action=rollback",
                "oldid=", "diff=", "printable=", "mobileaction=",
                "veaction=", "redlink=", "feed=",
            )):
                return None
            # title= with a MediaWiki "namespace" prefix indicates a
            # non-content page (Special:, User talk:, etc.). Decode URL-encoded
            # colons (%3A) too via parse_qs.
            params = parse_qs(parsed.query, keep_blank_values=False)
            title_values = params.get("title") or []
            for title_val in title_values:
                tl = title_val.lower()
                if any(
                    tl.startswith(ns)
                    for ns in (
                        "special:", "talk:", "user:", "user_talk:", "file:",
                        "file_talk:", "category:", "category_talk:", "help:",
                        "help_talk:", "mediawiki:", "mediawiki_talk:",
                        "template:", "template_talk:",
                    )
                ):
                    return None
        if parsed.netloc == self.allowed_netloc and normalized_path != parsed.path:
            clean_url = f"{parsed.scheme}://{parsed.netloc}{normalized_path}"
            if parsed.query:
                clean_url = f"{clean_url}?{parsed.query}"
        return clean_url

    def _section_from_url(self, url: str) -> list[str]:
        path = urlparse(url).path.strip("/")
        if not path:
            return ["home"]
        return [part.replace("-", " ") for part in path.split("/")[:-1] if part]
