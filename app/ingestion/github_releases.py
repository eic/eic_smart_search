"""GitHub release/changelog ingestion.

One document per release — answers "what changed in EICrecon 1.19" style
questions that no other source covers. Cheap: one API request per repo
page, no per-item follow-ups.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from app.core.config import Settings
from app.ingestion.base import DocumentPayload
from app.ingestion.github_common import API_BASE, github_headers
from app.ingestion.pdf_text import sanitize_text

logger = logging.getLogger(__name__)

def _parse_dt(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


class GitHubReleasesConnector:
    """Ingest release notes from the same repo list as the issues connector."""

    source_type = "github_releases"
    visibility = "public"

    PAGE_SIZE = 100

    def __init__(self, settings: Settings, *, source_name: str = "eic_github_releases") -> None:
        self.settings = settings
        self.source_name = source_name
        raw_repos = getattr(
            settings,
            "GITHUB_ISSUES_REPOS",  # deliberately shared — same key repos
            "eic/eic-shell,eic/epic,eic/EICrecon",
        )
        self.repos = [r.strip() for r in str(raw_repos).split(",") if r.strip()]
        self.max_per_repo = int(getattr(settings, "GITHUB_RELEASES_MAX_PER_REPO", 50))

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        limit_total = int(max_items) if max_items else self.max_per_repo * max(len(self.repos), 1)
        documents: list[DocumentPayload] = []
        headers = github_headers(self.settings)
        with httpx.Client(timeout=self.settings.REQUEST_TIMEOUT_SECONDS, headers=headers) as client:
            for repo in self.repos:
                if len(documents) >= limit_total:
                    break
                try:
                    response = client.get(
                        f"{API_BASE}/repos/{repo}/releases",
                        params={"per_page": min(self.PAGE_SIZE, self.max_per_repo)},
                    )
                    response.raise_for_status()
                    releases = response.json()
                except httpx.HTTPError as exc:
                    logger.warning(
                        "github_releases_repo_failed",
                        extra={"repo": repo, "error": str(exc)[:200]},
                    )
                    continue
                if not isinstance(releases, list):
                    continue
                for release in releases:
                    if len(documents) >= limit_total:
                        break
                    if not isinstance(release, dict) or release.get("draft"):
                        continue
                    doc = self._build_document(repo, release)
                    if doc is not None:
                        documents.append(doc)
        return documents

    def _build_document(self, repo: str, release: dict[str, Any]) -> DocumentPayload | None:
        tag = str(release.get("tag_name") or "").strip()
        name = str(release.get("name") or "").strip() or tag
        if not tag:
            return None
        body = str(release.get("body") or "").strip()
        published = _parse_dt(release.get("published_at"))
        lines = [f"Release {tag} of {repo}."]
        if release.get("prerelease"):
            lines.append("Pre-release.")
        if published:
            lines.append(f"Published: {published.date().isoformat()}")
        if body:
            lines.append(body)
        content = sanitize_text("\n\n".join(lines))
        if len(content.split()) < 5:
            return None
        return DocumentPayload(
            source_name=self.source_name,
            source_type=self.source_type,
            external_id=f"gh-release:{repo}@{tag}",
            title=f"{repo} {name}",
            url=str(release.get("html_url") or f"https://github.com/{repo}/releases/tag/{tag}"),
            content=content,
            visibility=self.visibility,
            section_path=[repo, "releases"],
            last_updated=published,
            metadata={
                "repo": repo,
                "tag": tag,
                "prerelease": bool(release.get("prerelease")),
            },
        )
