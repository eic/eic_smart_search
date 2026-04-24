"""Crawl every public repo in a GitHub org for Markdown/RST docs only.

Intentionally excludes source code: we use the same file-name filter as
`GitHubRepoConnector` (TEXT_EXTENSIONS + SKIP rules) but scoped to `.md`,
`.markdown`, `.rst`, and `CHANGELOG`/`CONTRIBUTING` files. Per-repo budget
keeps one large repo from starving the others.
"""
from __future__ import annotations

import base64
import datetime as dt
import logging
import re
from pathlib import PurePosixPath
from typing import Any

import httpx
import yaml

from app.core.config import Settings
from app.ingestion.base import DocumentPayload

logger = logging.getLogger(__name__)


ORG_TEXT_EXTENSIONS = {".md", ".markdown", ".mdown", ".rst"}
ORG_SKIP_PATH_PARTS = {
    ".git",
    ".github",
    "_site",
    "vendor",
    "node_modules",
    "__pycache__",
    "third_party",
    "external",
}
ORG_SKIP_SUFFIXES = (".lock",)
# Per-repo cap so one mega-repo with 500 mardown pages can't blow the budget.
MAX_DOCS_PER_REPO = 60


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value


class GitHubOrgConnector:
    """Iterate every non-archived public repo in an org, markdown only."""

    source_type = "github_org"

    def __init__(
        self,
        settings: Settings,
        *,
        source_name: str = "eic_github_org",
        org: str | None = None,
        visibility: str = "public",
    ) -> None:
        self.settings = settings
        self.source_name = source_name
        self.org = org or getattr(settings, "GITHUB_ORG", "eic")
        self.visibility = visibility
        self.api_base = "https://api.github.com"

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        limit = int(max_items or getattr(self.settings, "GITHUB_ORG_MAX_FILES", 1500))
        max_repos = int(getattr(self.settings, "GITHUB_ORG_MAX_REPOS", 200))
        headers = self._headers()
        documents: list[DocumentPayload] = []
        with httpx.Client(timeout=self.settings.REQUEST_TIMEOUT_SECONDS, headers=headers) as client:
            repos = self._list_org_repos(client, max_repos=max_repos)
            logger.info(
                "github_org_repos_listed",
                extra={"org": self.org, "count": len(repos)},
            )
            for repo in repos:
                if len(documents) >= limit:
                    break
                try:
                    repo_docs = self._index_repo(client, repo, per_repo_limit=MAX_DOCS_PER_REPO)
                except httpx.HTTPError as exc:
                    logger.warning(
                        "github_org_repo_skipped",
                        extra={"repo": repo.get("name"), "error": str(exc)[:200]},
                    )
                    continue
                documents.extend(repo_docs)
        return documents[:limit]

    # ---- helpers ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.settings.INGEST_USER_AGENT,
        }
        if self.settings.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {self.settings.GITHUB_TOKEN}"
        return headers

    def _list_org_repos(self, client: httpx.Client, max_repos: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page = 1
        while len(out) < max_repos:
            response = client.get(
                f"{self.api_base}/orgs/{self.org}/repos",
                params={"per_page": 100, "page": page, "type": "public"},
            )
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list) or not batch:
                break
            for repo in batch:
                if repo.get("archived") or repo.get("disabled"):
                    continue
                if repo.get("fork"):
                    # Forks usually mirror upstream docs; skip to avoid noise.
                    continue
                out.append(repo)
                if len(out) >= max_repos:
                    break
            if len(batch) < 100:
                break
            page += 1
        return out

    def _index_repo(
        self,
        client: httpx.Client,
        repo: dict[str, Any],
        per_repo_limit: int,
    ) -> list[DocumentPayload]:
        name = repo.get("name")
        default_branch = repo.get("default_branch") or "main"
        description = repo.get("description") or ""
        if not isinstance(name, str):
            return []
        repo_api = f"{self.api_base}/repos/{self.org}/{name}"
        tree_response = client.get(
            f"{repo_api}/git/trees/{default_branch}",
            params={"recursive": "1"},
        )
        if tree_response.status_code == 404:
            return []
        tree_response.raise_for_status()
        tree_payload = tree_response.json()
        tree = tree_payload.get("tree")
        if not isinstance(tree, list):
            return []
        commit_sha = None
        commit_response = client.get(f"{repo_api}/commits/{default_branch}")
        if commit_response.is_success:
            commit_sha = commit_response.json().get("sha")

        documents: list[DocumentPayload] = []
        for item in tree:
            if len(documents) >= per_repo_limit:
                break
            if item.get("type") != "blob":
                continue
            path = item.get("path")
            if not isinstance(path, str) or not self._should_index(path):
                continue
            try:
                content = self._fetch_text(client, repo_api, default_branch, path)
            except httpx.HTTPError as exc:
                logger.warning(
                    "github_org_file_skipped",
                    extra={"repo": name, "path": path, "error": str(exc)[:200]},
                )
                continue
            normalized, front_matter = self._normalize_content(path, content)
            if len(normalized.split()) < 20:
                continue
            documents.append(
                DocumentPayload(
                    source_name=self.source_name,
                    source_type=self.source_type,
                    external_id=f"{self.org}/{name}/{default_branch}/{path}",
                    title=self._title(name, path, front_matter, normalized),
                    url=f"https://github.com/{self.org}/{name}/blob/{default_branch}/{path}",
                    content=normalized,
                    visibility=self.visibility,
                    section_path=[name] + self._section_path(path, front_matter),
                    repo_path=path,
                    filetype=PurePosixPath(path).suffix.lstrip(".") or None,
                    last_updated=None,
                    metadata={
                        "repo": f"{self.org}/{name}",
                        "repo_owner": self.org,
                        "repo_name": name,
                        "repo_description": description,
                        "ref": default_branch,
                        "commit_hash": commit_sha,
                        "front_matter": _json_safe(front_matter),
                        "raw_url": f"https://raw.githubusercontent.com/{self.org}/{name}/{default_branch}/{path}",
                    },
                )
            )
        return documents

    def _fetch_text(self, client: httpx.Client, repo_api: str, ref: str, path: str) -> str:
        response = client.get(f"{repo_api}/contents/{path}", params={"ref": ref})
        response.raise_for_status()
        payload = response.json()
        encoded = payload.get("content")
        if not isinstance(encoded, str):
            raise ValueError(f"github contents response for {path} missing content")
        return base64.b64decode(encoded).decode("utf-8", errors="replace")

    def _should_index(self, path: str) -> bool:
        lowered = path.lower()
        if lowered.endswith(ORG_SKIP_SUFFIXES):
            return False
        if any(part in lowered for part in ORG_SKIP_PATH_PARTS):
            return False
        name = PurePosixPath(path).name.lower()
        if name in {"license", "license.md", "license.txt", "code_of_conduct.md"}:
            return False
        suffix = PurePosixPath(path).suffix.lower()
        return suffix in ORG_TEXT_EXTENSIONS

    def _normalize_content(self, path: str, content: str) -> tuple[str, dict[str, Any]]:
        front_matter: dict[str, Any] = {}
        body = content
        if content.startswith("---"):
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, flags=re.S)
            if match:
                try:
                    parsed = yaml.safe_load(match.group(1)) or {}
                    if isinstance(parsed, dict):
                        front_matter = parsed
                except yaml.YAMLError:
                    front_matter = {}
                body = match.group(2)
        return body.strip(), front_matter

    def _title(
        self,
        repo_name: str,
        path: str,
        front_matter: dict[str, Any],
        content: str,
    ) -> str:
        for key in ("title", "name"):
            value = front_matter.get(key)
            if isinstance(value, str) and value.strip():
                return f"{value.strip()} — {repo_name}"
        heading = re.search(r"^#\s+(.+)$", content, flags=re.M)
        if heading:
            return f"{heading.group(1).strip()} — {repo_name}"
        stem = PurePosixPath(path).stem.replace("-", " ").replace("_", " ").title()
        return f"{stem} — {repo_name}"

    def _section_path(self, path: str, front_matter: dict[str, Any]) -> list[str]:
        parent = PurePosixPath(path).parent
        if str(parent) == ".":
            return []
        return [part.replace("-", " ").replace("_", " ") for part in parent.parts]
