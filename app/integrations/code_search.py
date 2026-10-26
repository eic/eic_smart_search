"""GitHub code search + file fetch as LLM tools, scoped to one org.

Makes the assistant "code aware" without indexing source trees: exact
identifier lookups ("where is StPicoEvent defined", "what is PID type 90")
go straight to GitHub's code-search index, and the model can then pull the
actual file to read the definition. Always tracks HEAD — no re-ingestion.

Scope is a single GitHub org (``CODE_SEARCH_ORG``: eic / star-bnl); the
fetch tool refuses paths outside it. Code search costs a separate, small
rate budget (10 requests/min authenticated) — on throttle the tools return
an error message the model can relay honestly.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

import httpx

from app.ingestion.safe_fetch import safe_get

logger = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"
_RAW_BASE = "https://raw.githubusercontent.com"

_MAX_FILE_LINES = 400
_MAX_FILE_CHARS = 15_000
_MAX_FRAGMENTS = 3

# GitHub search qualifiers (``repo:x``, ``org:x``, ``user:x``, ``-repo:x``,
# ``path:x`` …). The query comes from the LLM, which can be steered by an
# injected document into widening the search beyond the configured org with
# the server's token — so every ``word:value`` token is dropped and scope is
# set only by us. ``(?!:)`` keeps C++ scoped names (``std::vector``) intact.
_QUALIFIER_RE = re.compile(r"(?<![\w:])-?\w+:(?!:)\S*")
_REPO_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def strip_search_qualifiers(query: str) -> str:
    """Remove every GitHub ``qualifier:value`` token, keep plain words."""
    return " ".join(_QUALIFIER_RE.sub(" ", query or "").split())


class CodeSearchClient:
    def __init__(
        self,
        org: str,
        token: str | None,
        *,
        cache_ttl_s: float = 300.0,
        timeout_s: float = 10.0,
        user_agent: str = "eic-smart-search/1.0 (code-search)",
    ) -> None:
        if not org:
            raise ValueError("CodeSearchClient: org is required")
        self.org = org
        self.token = token
        self.cache_ttl_s = max(0.0, cache_ttl_s)
        self.timeout_s = timeout_s
        self.user_agent = user_agent
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    # ---- cache ----------------------------------------------------------

    def _cached(self, key: tuple) -> Any | None:
        now = time.time()
        with self._lock:
            entry = self._cache.get(key)
            if entry and entry[0] > now:
                return entry[1]
        return None

    def _store(self, key: tuple, value: Any) -> None:
        if self.cache_ttl_s <= 0:
            return
        with self._lock:
            self._cache[key] = (time.time() + self.cache_ttl_s, value)
            if len(self._cache) > 200:
                self._cache.pop(next(iter(self._cache)))

    def _headers(self, *, text_match: bool = False) -> dict[str, str]:
        accept = "application/vnd.github.text-match+json" if text_match else "application/vnd.github+json"
        headers = {"Accept": accept, "User-Agent": self.user_agent}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _valid_repo(self, repo: str) -> bool:
        """True for exactly ``<org>/<name>`` with a plain repo name."""
        org, sep, name = (repo or "").partition("/")
        return (
            sep == "/"
            and org == self.org
            and bool(_REPO_NAME_RE.match(name))
            and name.strip(".") != ""
        )

    # ---- tools ----------------------------------------------------------

    def search_code(self, query: str, *, repo: str = "", limit: int = 6) -> dict[str, Any]:
        query = strip_search_qualifiers(query)
        if not query:
            return {"error": "empty query"}
        limit = max(1, min(int(limit or 6), 10))
        repo = (repo or "").strip()
        scope = f"repo:{repo}" if self._valid_repo(repo) else f"org:{self.org}"
        cache_key = ("search", query, scope, limit)
        cached = self._cached(cache_key)
        if cached is not None:
            return cached
        try:
            response = httpx.get(
                f"{_API_BASE}/search/code",
                params={"q": f"{query} {scope}", "per_page": str(limit)},
                headers=self._headers(text_match=True),
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            logger.warning("code_search_failed", extra={"query": query, "error": str(exc)[:200]})
            return {"error": f"code search unavailable: {str(exc)[:120]}"}
        if response.status_code in (403, 429):
            return {"error": "GitHub code-search rate limit hit; retry in a minute"}
        if response.status_code != 200:
            return {"error": f"GitHub code search returned HTTP {response.status_code}"}
        items = response.json().get("items") or []
        hits = []
        for item in items[:limit]:
            fragments = [
                " ".join((m.get("fragment") or "").split())[:400]
                for m in (item.get("text_matches") or [])[:_MAX_FRAGMENTS]
            ]
            hits.append(
                {
                    "repo": (item.get("repository") or {}).get("full_name"),
                    "path": item.get("path"),
                    "url": item.get("html_url"),
                    "fragments": [f for f in fragments if f],
                }
            )
        result = {"count": len(hits), "org": self.org, "hits": hits}
        self._store(cache_key, result)
        return result

    def fetch_code_file(
        self, repo: str, path: str, *, start_line: int = 0, end_line: int = 0
    ) -> dict[str, Any]:
        repo = (repo or "").strip().strip("/")
        path = (path or "").strip().lstrip("/")
        if not self._valid_repo(repo):
            return {"error": f"repo must belong to the {self.org} org (got {repo!r})"}
        # Reject percent-encoding outright (``%2e%2e`` would bypass the
        # ``..`` check once the URL is decoded), plus backslashes, ``?``/``#``
        # (would rewrite the raw URL) and control characters.
        if (
            not path
            or ".." in path
            or "%" in path
            or any(ch in path for ch in "\\?#")
            or any(ord(c) < 32 or ord(c) == 127 for c in path)
        ):
            return {"error": "invalid path"}
        cache_key = ("file", repo, path)
        text = self._cached(cache_key)
        if text is None:
            url = f"{_RAW_BASE}/{repo}/HEAD/{path}"
            try:
                with httpx.Client(timeout=self.timeout_s, headers={"User-Agent": self.user_agent}) as client:
                    response = safe_get(client, url)
            except Exception as exc:
                logger.warning("code_fetch_failed", extra={"url": url, "error": str(exc)[:200]})
                return {"error": f"fetch failed: {str(exc)[:120]}"}
            if response.status_code != 200:
                return {"error": f"file not found (HTTP {response.status_code}): {repo}/{path}"}
            text = response.text
            self._store(cache_key, text)
        lines = text.splitlines()
        total = len(lines)
        start = max(1, int(start_line or 1))
        end = int(end_line or 0) or min(total, start + _MAX_FILE_LINES - 1)
        end = min(end, total, start + _MAX_FILE_LINES - 1)
        numbered = [f"{i}: {line}" for i, line in enumerate(lines[start - 1 : end], start=start)]
        content = "\n".join(numbered)[:_MAX_FILE_CHARS]
        return {
            "repo": repo,
            "path": path,
            "url": f"https://github.com/{repo}/blob/HEAD/{path}#L{start}",
            "total_lines": total,
            "start_line": start,
            "end_line": end,
            "content": content,
        }


SEARCH_CODE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": (
            "Search the collaboration's GitHub source code (exact identifier "
            "match, like grep across all repos). Call this for code-level "
            "questions the docs don't answer: 'where is <Class/function> "
            "defined', 'what does get<X>() return', numeric enum/type values "
            "('PID type 90'), configuration flag defaults, 'which file "
            "implements <algorithm>'. Follow up with fetch_code_file on the "
            "best hit to read the actual definition before answering. Cite "
            "the hit's url as a Markdown link."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Code identifiers or exact strings to find (class, "
                        "method, constant, flag name). Avoid natural-language "
                        "filler — GitHub code search is literal."
                    ),
                },
                "repo": {
                    "type": "string",
                    "description": "Optional org/repo to restrict to (e.g. eic/EICrecon).",
                },
                "limit": {"type": "integer", "description": "Max hits (default 6, max 10)."},
            },
            "required": ["query"],
        },
    },
}

FETCH_CODE_FILE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_code_file",
        "description": (
            "Read a source file (or a line range) from the collaboration's "
            "GitHub org, to quote and explain the actual code found via "
            "search_code. Returns numbered lines. Large files are clipped to "
            "400 lines — pass start_line/end_line to window."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "org/repo, e.g. eic/EICrecon."},
                "path": {"type": "string", "description": "File path inside the repo."},
                "start_line": {"type": "integer", "description": "First line to return (1-based)."},
                "end_line": {"type": "integer", "description": "Last line to return."},
            },
            "required": ["repo", "path"],
        },
    },
}

__all__ = [
    "CodeSearchClient",
    "SEARCH_CODE_TOOL_SCHEMA",
    "FETCH_CODE_FILE_TOOL_SCHEMA",
]
