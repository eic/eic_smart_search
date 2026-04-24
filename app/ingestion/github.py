import base64
import datetime as dt
import logging
import re
from pathlib import PurePosixPath
from typing import Any

import httpx
import yaml


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value

from app.core.config import Settings
from app.ingestion.base import DocumentPayload

logger = logging.getLogger(__name__)


TEXT_EXTENSIONS = {".md", ".markdown", ".mdown", ".rst", ".txt", ".yml", ".yaml"}
SKIP_PATH_PARTS = {
    ".git",
    ".github",
    "_site",
    "vendor",
    "node_modules",
    "__pycache__",
    "assets/images",
    "assets/css",
    "assets/bootstrap",
}
SKIP_SUFFIXES = (".min.js", ".map", ".lock")


class GitHubRepoConnector:
    source_name = "eic_github_repo"
    source_type = "github_repo"
    visibility = "public"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.owner = settings.GITHUB_REPO_OWNER
        self.repo = settings.GITHUB_REPO_NAME
        self.ref = settings.GITHUB_REPO_REF
        self.api_base = f"https://api.github.com/repos/{self.owner}/{self.repo}"

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        limit = max_items or self.settings.GITHUB_MAX_FILES
        headers = self._headers()
        with httpx.Client(timeout=self.settings.REQUEST_TIMEOUT_SECONDS, headers=headers) as client:
            commit_sha = self._commit_sha(client)
            tree = self._tree(client)
            documents: list[DocumentPayload] = []
            for item in tree:
                if len(documents) >= limit:
                    break
                if item.get("type") != "blob":
                    continue
                path = item.get("path")
                if not isinstance(path, str) or not self._should_index(path):
                    continue
                try:
                    content = self._fetch_text(client, path)
                except httpx.HTTPError as exc:
                    logger.warning("github_file_fetch_failed", extra={"path": path, "error": str(exc)})
                    continue
                normalized, front_matter = self._normalize_content(path, content)
                if len(normalized.split()) < 20:
                    continue
                documents.append(
                    DocumentPayload(
                        source_name=self.source_name,
                        source_type=self.source_type,
                        external_id=f"{self.owner}/{self.repo}/{self.ref}/{path}",
                        title=self._title(path, front_matter, normalized),
                        url=f"https://github.com/{self.owner}/{self.repo}/blob/{self.ref}/{path}",
                        content=normalized,
                        visibility="public",
                        section_path=self._section_path(path, front_matter),
                        repo_path=path,
                        filetype=PurePosixPath(path).suffix.lstrip(".") or None,
                        last_updated=None,
                        metadata={
                            "repo": f"{self.owner}/{self.repo}",
                            "repo_owner": self.owner,
                            "repo_name": self.repo,
                            "ref": self.ref,
                            "commit_hash": commit_sha,
                            "front_matter": front_matter,
                            "raw_url": f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.ref}/{path}",
                        },
                    )
                )
            return documents

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.settings.INGEST_USER_AGENT,
        }
        if self.settings.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {self.settings.GITHUB_TOKEN}"
        return headers

    def _commit_sha(self, client: httpx.Client) -> str | None:
        response = client.get(f"{self.api_base}/commits/{self.ref}")
        response.raise_for_status()
        sha = response.json().get("sha")
        return sha if isinstance(sha, str) else None

    def _tree(self, client: httpx.Client) -> list[dict[str, Any]]:
        response = client.get(f"{self.api_base}/git/trees/{self.ref}", params={"recursive": "1"})
        response.raise_for_status()
        payload = response.json()
        tree = payload.get("tree")
        if not isinstance(tree, list):
            raise ValueError("GitHub tree response did not include a tree list")
        return tree

    def _fetch_text(self, client: httpx.Client, path: str) -> str:
        response = client.get(f"{self.api_base}/contents/{path}", params={"ref": self.ref})
        response.raise_for_status()
        payload = response.json()
        encoded = payload.get("content")
        if not isinstance(encoded, str):
            raise ValueError(f"GitHub contents response for {path} did not include content")
        return base64.b64decode(encoded).decode("utf-8", errors="replace")

    def _should_index(self, path: str) -> bool:
        lowered = path.lower()
        if lowered.endswith(SKIP_SUFFIXES):
            return False
        if any(part in lowered for part in SKIP_PATH_PARTS):
            return False
        suffix = PurePosixPath(path).suffix.lower()
        if suffix not in TEXT_EXTENSIONS:
            return False
        if PurePosixPath(path).name.lower() in {"license", "gemfile.lock"}:
            return False
        return True

    def _normalize_content(self, path: str, content: str) -> tuple[str, dict[str, Any]]:
        front_matter: dict[str, Any] = {}
        body = content
        if content.startswith("---"):
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, flags=re.S)
            if match:
                try:
                    parsed = yaml.safe_load(match.group(1)) or {}
                    if isinstance(parsed, dict):
                        front_matter = _json_safe(parsed)
                except yaml.YAMLError:
                    front_matter = {}
                body = match.group(2)
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in {".yml", ".yaml"}:
            body = f"# {path}\n\n```yaml\n{body}\n```"
        return body.strip(), front_matter

    def _title(self, path: str, front_matter: dict[str, Any], content: str) -> str:
        for key in ("title", "name"):
            value = front_matter.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        heading = re.search(r"^#\s+(.+)$", content, flags=re.M)
        if heading:
            return heading.group(1).strip()
        return PurePosixPath(path).stem.replace("-", " ").replace("_", " ").title()

    def _section_path(self, path: str, front_matter: dict[str, Any]) -> list[str]:
        categories = front_matter.get("categories")
        if isinstance(categories, list):
            return [str(category) for category in categories]
        collection = PurePosixPath(path).parts[0] if PurePosixPath(path).parts else ""
        collection = collection.lstrip("_").replace("-", " ").replace("_", " ")
        parent = str(PurePosixPath(path).parent)
        if parent == ".":
            return [collection] if collection else []
        return [collection] + [part.replace("-", " ").replace("_", " ") for part in PurePosixPath(parent).parts[1:]]
