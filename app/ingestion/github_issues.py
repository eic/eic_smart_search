"""GitHub issue-thread ingestion for key EIC repos.

Issues are the closest thing to a public Q&A corpus for EIC software —
users ask "how do I fix this eic-shell error" in exactly the phrasing they
later type into search. One document per issue thread (title + body +
comments), pull requests excluded.

Unauthenticated GitHub allows 60 requests/hour, which is not enough
(one request per comment page). Set ``GITHUB_TOKEN`` for the 5000/hour
authenticated limit before enabling this source on a schedule.
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


def build_issue_content(issue: dict[str, Any], comments: list[dict[str, Any]]) -> str:
    """Render an issue thread as a plain-text conversation (unit-testable)."""
    lines: list[str] = []
    state = str(issue.get("state") or "")
    labels = [
        str(lbl.get("name"))
        for lbl in issue.get("labels") or []
        if isinstance(lbl, dict) and lbl.get("name")
    ]
    if state:
        lines.append(f"State: {state}")
    if labels:
        lines.append("Labels: " + ", ".join(labels))
    author = (issue.get("user") or {}).get("login") or "unknown"
    body = str(issue.get("body") or "").strip()
    lines.append(f"{author} (opened):")
    if body:
        lines.append(body)
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        c_author = (comment.get("user") or {}).get("login") or "unknown"
        c_body = str(comment.get("body") or "").strip()
        if not c_body:
            continue
        lines.append(f"{c_author} (comment):")
        lines.append(c_body)
    return sanitize_text("\n\n".join(lines).strip())


class GitHubIssuesConnector:
    """Ingest issue threads from a configurable list of repos."""

    source_type = "github_issues"
    visibility = "public"

    PAGE_SIZE = 100

    def __init__(self, settings: Settings, *, source_name: str = "eic_github_issues") -> None:
        self.settings = settings
        self.source_name = source_name
        raw_repos = getattr(
            settings,
            "GITHUB_ISSUES_REPOS",
            "eic/eic-shell,eic/epic,eic/EICrecon",
        )
        self.repos = [r.strip() for r in str(raw_repos).split(",") if r.strip()]
        self.max_per_repo = int(getattr(settings, "GITHUB_ISSUES_MAX_PER_REPO", 200))

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        limit_total = int(max_items) if max_items else self.max_per_repo * max(len(self.repos), 1)
        documents: list[DocumentPayload] = []
        with httpx.Client(timeout=self.settings.REQUEST_TIMEOUT_SECONDS, headers=github_headers(self.settings)) as client:
            for repo in self.repos:
                if len(documents) >= limit_total:
                    break
                try:
                    documents.extend(
                        self._iter_repo(client, repo, limit=min(self.max_per_repo, limit_total - len(documents)))
                    )
                except httpx.HTTPError as exc:
                    logger.warning(
                        "github_issues_repo_failed",
                        extra={"repo": repo, "error": str(exc)[:200]},
                    )
        return documents

    def _iter_repo(self, client: httpx.Client, repo: str, limit: int) -> list[DocumentPayload]:
        documents: list[DocumentPayload] = []
        page = 1
        while len(documents) < limit:
            response = client.get(
                f"{API_BASE}/repos/{repo}/issues",
                params={
                    "state": "all",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": self.PAGE_SIZE,
                    "page": page,
                },
            )
            response.raise_for_status()
            issues = response.json()
            if not isinstance(issues, list) or not issues:
                break
            for issue in issues:
                if len(documents) >= limit:
                    break
                if not isinstance(issue, dict) or issue.get("pull_request"):
                    continue  # the issues endpoint interleaves PRs — skip them
                doc = self._build_document(client, repo, issue)
                if doc is not None:
                    documents.append(doc)
            if len(issues) < self.PAGE_SIZE:
                break
            page += 1
        return documents

    def _build_document(
        self,
        client: httpx.Client,
        repo: str,
        issue: dict[str, Any],
    ) -> DocumentPayload | None:
        number = issue.get("number")
        title = str(issue.get("title") or "").strip()
        if number is None or not title:
            return None
        comments: list[dict[str, Any]] = []
        if int(issue.get("comments") or 0) > 0:
            try:
                response = client.get(
                    f"{API_BASE}/repos/{repo}/issues/{number}/comments",
                    params={"per_page": self.PAGE_SIZE},
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    comments = payload
            except httpx.HTTPError as exc:
                logger.warning(
                    "github_issue_comments_failed",
                    extra={"repo": repo, "issue": number, "error": str(exc)[:200]},
                )
        content = build_issue_content(issue, comments)
        if len(content.split()) < 10:
            return None
        return DocumentPayload(
            source_name=self.source_name,
            source_type=self.source_type,
            external_id=f"gh-issue:{repo}#{number}",
            title=f"{repo}#{number}: {title}",
            url=str(issue.get("html_url") or f"https://github.com/{repo}/issues/{number}"),
            content=content,
            visibility=self.visibility,
            section_path=[repo, "issues"],
            last_updated=_parse_dt(issue.get("updated_at")),
            metadata={
                "repo": repo,
                "number": number,
                "state": issue.get("state"),
                "labels": [
                    str(lbl.get("name"))
                    for lbl in issue.get("labels") or []
                    if isinstance(lbl, dict) and lbl.get("name")
                ],
                "comment_count": int(issue.get("comments") or 0),
            },
        )
