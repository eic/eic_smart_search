from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.ingestion.base import DocumentPayload
from app.ingestion.html import extract_html
from app.ingestion.web import WebsiteCrawler


class InternalAuthProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.settings.INGEST_USER_AGENT}
        if self.settings.EPIC_INTERNAL_AUTH_HEADER:
            name, _, value = self.settings.EPIC_INTERNAL_AUTH_HEADER.partition(":")
            if name and value:
                headers[name.strip()] = value.strip()
        if self.settings.EPIC_INTERNAL_COOKIE:
            headers["Cookie"] = self.settings.EPIC_INTERNAL_COOKIE
        return headers


class EpicInternalCrawler(WebsiteCrawler):
    source_name = "epic_internal"
    source_type = "internal_website"
    visibility = "internal"

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            source_name="epic_internal",
            start_url=settings.EPIC_INTERNAL_START_URL,
            max_pages=settings.EPIC_INTERNAL_MAX_PAGES,
            visibility="internal",
        )
        self.source_type = "internal_website"  # override the parent's "website" after super().__init__
        self.auth_provider = InternalAuthProvider(settings)
        self.allowed_netloc = urlparse(settings.EPIC_INTERNAL_START_URL).netloc

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        limit = max_items or self.settings.EPIC_INTERNAL_MAX_PAGES
        seen: set[str] = set()
        queue = [self.start_url]
        documents: list[DocumentPayload] = []
        with httpx.Client(
            timeout=self.settings.REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=self.auth_provider.headers(),
        ) as client:
            while queue and len(documents) < limit:
                url = self._canonicalize(queue.pop(0))
                if not url or url in seen:
                    continue
                seen.add(url)
                response = client.get(url)
                if response.status_code in {401, 403}:
                    raise PermissionError(
                        "ePIC internal crawling requires EPIC_INTERNAL_COOKIE or EPIC_INTERNAL_AUTH_HEADER"
                    )
                response.raise_for_status()
                if "text/html" not in response.headers.get("content-type", ""):
                    continue
                extracted = extract_html(response.text, str(response.url), response.headers.get("last-modified"))
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
                        visibility="internal",
                        section_path=extracted.breadcrumbs or self._section_from_url(str(response.url)),
                        last_updated=extracted.last_updated,
                        metadata={
                            "headings": extracted.headings[:30],
                            "breadcrumbs": extracted.breadcrumbs,
                            "crawler": "epic_internal",
                            "permission_sensitive": True,
                        },
                    )
                )
                for link in extracted.links:
                    next_url = self._canonicalize(link)
                    if next_url and next_url not in seen:
                        queue.append(next_url)
        return documents

