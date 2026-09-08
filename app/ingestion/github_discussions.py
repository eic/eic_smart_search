"""GitHub Discussions ingestion (GraphQL — Discussions have no REST API).

The eic org uses Discussions as its public Q&A forum; threads there match
user search phrasing even better than issues. One document per discussion:
question + answer (if marked) + top comments.

Requires ``GITHUB_TOKEN`` — the GraphQL endpoint rejects anonymous calls.
Repos without Discussions enabled are skipped silently.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from app.core.config import Settings
from app.ingestion.base import DocumentPayload
from app.ingestion.pdf_text import sanitize_text

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"

# Newest-updated first; answer + a few top comments per thread.
_DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: $first, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        url
        body
        updatedAt
        category { name }
        author { login }
        answer { body author { login } }
        comments(first: 5) {
          nodes { body author { login } }
        }
      }
    }
  }
}
"""


def _parse_dt(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def build_discussion_content(node: dict[str, Any]) -> str:
    """Render a discussion node as a plain-text thread (unit-testable)."""
    lines: list[str] = []
    category = ((node.get("category") or {}).get("name") or "").strip()
    if category:
        lines.append(f"Category: {category}")
    author = ((node.get("author") or {}).get("login")) or "unknown"
    body = str(node.get("body") or "").strip()
    lines.append(f"{author} (asked):")
    if body:
        lines.append(body)
    answer = node.get("answer") or {}
    answer_body = str(answer.get("body") or "").strip()
    if answer_body:
        answer_author = ((answer.get("author") or {}).get("login")) or "unknown"
        lines.append(f"{answer_author} (accepted answer):")
        lines.append(answer_body)
    for comment in ((node.get("comments") or {}).get("nodes")) or []:
        if not isinstance(comment, dict):
            continue
        c_body = str(comment.get("body") or "").strip()
        if not c_body or c_body == answer_body:
            continue
        c_author = ((comment.get("author") or {}).get("login")) or "unknown"
        lines.append(f"{c_author} (comment):")
        lines.append(c_body)
    return sanitize_text("\n\n".join(lines).strip())


class GitHubDiscussionsConnector:
    """Ingest Discussions threads from the configured repo list."""

    source_type = "github_discussions"
    visibility = "public"

    PAGE_SIZE = 50

    def __init__(self, settings: Settings, *, source_name: str = "eic_github_discussions") -> None:
        self.settings = settings
        self.source_name = source_name
        raw_repos = getattr(
            settings,
            "GITHUB_DISCUSSIONS_REPOS",
            "eic/eic-shell,eic/epic,eic/EICrecon",
        )
        self.repos = [r.strip() for r in str(raw_repos).split(",") if r.strip() and "/" in r]
        self.max_per_repo = int(getattr(settings, "GITHUB_DISCUSSIONS_MAX_PER_REPO", 200))

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        if not self.settings.GITHUB_TOKEN:
            logger.warning("github_discussions_skipped_no_token")
            return []
        limit_total = int(max_items) if max_items else self.max_per_repo * max(len(self.repos), 1)
        documents: list[DocumentPayload] = []
        headers = {
            "Authorization": f"Bearer {self.settings.GITHUB_TOKEN}",
            "User-Agent": self.settings.INGEST_USER_AGENT,
        }
        with httpx.Client(timeout=self.settings.REQUEST_TIMEOUT_SECONDS, headers=headers) as client:
            for repo in self.repos:
                if len(documents) >= limit_total:
                    break
                owner, name = repo.split("/", 1)
                documents.extend(
                    self._iter_repo(
                        client, owner, name, limit=min(self.max_per_repo, limit_total - len(documents))
                    )
                )
        return documents

    def _iter_repo(self, client: httpx.Client, owner: str, name: str, limit: int) -> list[DocumentPayload]:
        documents: list[DocumentPayload] = []
        cursor: str | None = None
        repo = f"{owner}/{name}"
        while len(documents) < limit:
            try:
                response = client.post(
                    GRAPHQL_URL,
                    json={
                        "query": _DISCUSSIONS_QUERY,
                        "variables": {
                            "owner": owner,
                            "name": name,
                            "first": min(self.PAGE_SIZE, limit - len(documents)),
                            "after": cursor,
                        },
                    },
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                logger.warning(
                    "github_discussions_repo_failed",
                    extra={"repo": repo, "error": str(exc)[:200]},
                )
                return documents
            repository = ((payload.get("data") or {}).get("repository")) or {}
            discussions = repository.get("discussions") or {}
            nodes = discussions.get("nodes") or []
            if not nodes:
                # Repo without Discussions enabled (or GraphQL errors) — skip.
                if payload.get("errors"):
                    logger.info(
                        "github_discussions_unavailable",
                        extra={"repo": repo, "error": str(payload["errors"])[:200]},
                    )
                return documents
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                doc = self._build_document(repo, node)
                if doc is not None:
                    documents.append(doc)
                if len(documents) >= limit:
                    break
            page_info = discussions.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
        return documents

    def _build_document(self, repo: str, node: dict[str, Any]) -> DocumentPayload | None:
        number = node.get("number")
        title = str(node.get("title") or "").strip()
        if number is None or not title:
            return None
        content = build_discussion_content(node)
        if len(content.split()) < 10:
            return None
        return DocumentPayload(
            source_name=self.source_name,
            source_type=self.source_type,
            external_id=f"gh-discussion:{repo}#{number}",
            title=f"{repo} discussion: {title}",
            url=str(node.get("url") or f"https://github.com/{repo}/discussions/{number}"),
            content=content,
            visibility=self.visibility,
            section_path=[repo, "discussions"],
            last_updated=_parse_dt(node.get("updatedAt")),
            metadata={
                "repo": repo,
                "number": number,
                "category": ((node.get("category") or {}).get("name")) or None,
                "answered": bool(node.get("answer")),
            },
        )
