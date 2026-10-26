"""Live Zenodo search for the configured ePIC community.

Why a tool and not just rely on the indexed Zenodo connector?
    - Indexer runs on a schedule; freshly published records won't be in the
      vector store until the next ingestion.
    - The community has > MAX_RECORDS papers; the indexer caps coverage,
      while ad-hoc searches benefit from the full Zenodo backing index.
    - When the user explicitly asks "find a paper on X", a direct Zenodo
      query with their phrasing tends to outperform our hybrid retrieval
      on title-heavy short queries.

The tool returns metadata only (title, authors, DOI, link, date) — the
LLM cites the URL directly, so we don't need to re-extract PDF text.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from html import unescape
from typing import Any

import httpx

logger = logging.getLogger(__name__)


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Elasticsearch query-string syntax Zenodo passes `q` through to. An
# unbalanced "/" (regex delimiter, e.g. "J/psi") or similar makes Zenodo
# answer HTTP 500, so these become plain word separators.
_ES_SPECIAL_RE = re.compile(r'[/:^~*?\[\]{}()"\\!|&<>=]')
# Leading -/+ on a token are must-not / must operators.
_ES_PREFIX_OP_RE = re.compile(r"(^|\s)[-+]+")


def sanitize_query(query: str) -> str:
    """Strip ES query-string operators: "J/psi photoproduction" -> "J psi photoproduction"."""
    query = _ES_SPECIAL_RE.sub(" ", query or "")
    query = _ES_PREFIX_OP_RE.sub(r"\1", query)
    return _WS_RE.sub(" ", query).strip()


def _strip_html(raw: str | None, max_len: int = 400) -> str:
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


@dataclass(slots=True)
class ZenodoHit:
    id: str
    title: str
    url: str
    doi: str
    resource_type: str
    publication_date: str
    authors: list[str] = field(default_factory=list)
    description: str = ""
    keywords: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "doi": self.doi,
            "resource_type": self.resource_type,
            "publication_date": self.publication_date,
            "authors": self.authors[:8],  # Author lists for HEP papers can have hundreds.
            "keywords": self.keywords[:8],
            "description": self.description,
        }


@dataclass(slots=True)
class _CacheEntry:
    hits: list[ZenodoHit]
    expires_at: float


class ZenodoSearchClient:
    """Hits the Zenodo REST search endpoint, scoped to a community.

    The ``cache_ttl_s`` is intentionally short — Zenodo's search index is
    cheap to hit and freshness matters when a user just published a paper.
    Cache exists only to absorb burst duplicates within a single chat turn.
    """

    BASE_URL = "https://zenodo.org/api/records"

    def __init__(
        self,
        community: str,
        *,
        cache_ttl_s: float = 120.0,
        timeout_s: float = 10.0,
        user_agent: str = "eic-smart-search/1.0 (zenodo-search)",
    ) -> None:
        if not community:
            raise ValueError("ZenodoSearchClient: community is required")
        self.community = community
        self.cache_ttl_s = max(0.0, cache_ttl_s)
        self.timeout_s = timeout_s
        self.user_agent = user_agent
        self._cache: dict[tuple[str, int, str], _CacheEntry] = {}
        self._lock = threading.Lock()

    # Filter prefixes Zenodo accepts in `q`. We already scope by community in
    # the request params, so an LLM-helpfully-added `community:ePIC` collapses
    # the search to nothing. Strip them defensively.
    _NEUTRALIZE_FILTERS_RE = re.compile(
        r"\b(community|communities|owners|status|access_right):\S+",
        flags=re.IGNORECASE,
    )

    def search(
        self,
        query: str,
        *,
        size: int = 10,
        sort: str = "bestmatch",
    ) -> list[ZenodoHit]:
        """Run a Zenodo search and return hits scoped to ``self.community``.

        ``sort`` accepts Zenodo's values: 'bestmatch', 'mostrecent', 'mostviewed'.
        """
        query = self._NEUTRALIZE_FILTERS_RE.sub(" ", query or "")
        query = sanitize_query(query)
        if not query:
            # Empty after stripping (e.g. "community:ePIC" was the entire q).
            # Fall back to a wildcard so the user still gets community results.
            query = "*"
        size = max(1, min(int(size or 10), 25))
        if sort not in {"bestmatch", "mostrecent", "mostviewed"}:
            sort = "bestmatch"
        cache_key = (query, size, sort)
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and cached.expires_at > now:
                return list(cached.hits)
        hits = self._fetch(query, size=size, sort=sort)
        if self.cache_ttl_s > 0:
            with self._lock:
                self._cache[cache_key] = _CacheEntry(hits=hits, expires_at=now + self.cache_ttl_s)
        return hits

    # On a 5xx, retry once with only this many leading terms — shorter
    # queries hit Zenodo's flaky search backend far less often.
    RETRY_TERMS = 4

    def _fetch(self, query: str, *, size: int, sort: str) -> list[ZenodoHit]:
        hits, server_error = self._fetch_once(query, size=size, sort=sort)
        if server_error:
            short = " ".join(query.split()[: self.RETRY_TERMS])
            if short and short != query:
                hits, _ = self._fetch_once(short, size=size, sort=sort)
        return hits

    def _fetch_once(self, query: str, *, size: int, sort: str) -> tuple[list[ZenodoHit], bool]:
        """One request; returns (hits, server_error) where server_error is a 5xx."""
        params = {
            "q": query,
            "communities": self.community,
            "size": str(size),
            "sort": sort,
        }
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        try:
            response = httpx.get(self.BASE_URL, params=params, headers=headers, timeout=self.timeout_s)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("zenodo_search_failed", extra={"query": query, "error": str(exc)[:200]})
            status = getattr(getattr(exc, "response", None), "status_code", None)
            return [], isinstance(status, int) and status >= 500
        try:
            payload = response.json()
        except ValueError:
            logger.warning("zenodo_search_non_json", extra={"query": query})
            return [], False
        raw_hits = (payload.get("hits") or {}).get("hits") or []
        if not isinstance(raw_hits, list):
            return [], False
        return [hit for hit in (self._build(record) for record in raw_hits) if hit is not None], False

    @staticmethod
    def _build(record: dict[str, Any]) -> ZenodoHit | None:
        if not isinstance(record, dict):
            return None
        meta = record.get("metadata") or {}
        record_id = record.get("id")
        if record_id is None:
            return None
        links = record.get("links") or {}
        url = links.get("self_html") or links.get("html") or f"https://zenodo.org/records/{record_id}"
        resource_type_raw = meta.get("resource_type") or {}
        if isinstance(resource_type_raw, dict):
            resource_type = str(resource_type_raw.get("title") or resource_type_raw.get("type") or "").strip()
        else:
            resource_type = str(resource_type_raw or "").strip()
        creators = meta.get("creators") or []
        authors: list[str] = []
        if isinstance(creators, list):
            for c in creators:
                if isinstance(c, dict):
                    name = c.get("name")
                    if isinstance(name, str) and name.strip():
                        authors.append(name.strip())
        keywords_raw = meta.get("keywords")
        keywords = (
            [str(k).strip() for k in keywords_raw if isinstance(k, str) and k.strip()]
            if isinstance(keywords_raw, list)
            else []
        )
        return ZenodoHit(
            id=str(record_id),
            title=str(meta.get("title") or "").strip() or f"Zenodo record {record_id}",
            url=str(url),
            doi=str(meta.get("doi") or record.get("doi") or "").strip(),
            resource_type=resource_type,
            publication_date=str(meta.get("publication_date") or "").strip(),
            authors=authors,
            description=_strip_html(meta.get("description")),
            keywords=keywords,
        )


SEARCH_ZENODO_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_zenodo",
        "description": (
            "Search the live Zenodo community for papers, technical notes, "
            "presentations, datasets, or proceedings. Call this when the user "
            "asks for a paper / preprint / proposal / technical note on a "
            "topic ('any paper on calorimetry?', 'find the TDR draft', "
            "'recent Zenodo upload about tracking') OR asks 'is there a "
            "publication that…' / 'show me the proposal for…'. Returns "
            "metadata only (title, DOI, authors, link). Quote the result "
            "URL as a Markdown link in the answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search keywords. Use the user's own physics / detector "
                        "vocabulary; Zenodo's full-text search handles synonyms "
                        "poorly so prefer concrete nouns ('hpDIRC', 'EICRecon', "
                        "'TDR', 'silicon vertex tracker') over verbose phrases. "
                        "Do NOT include 'community:ePIC' or similar filter "
                        "prefixes — community scope is already set server-side."
                    ),
                },
                "size": {
                    "type": "integer",
                    "description": "Max records to return. Default 10, max 25.",
                    "default": 10,
                },
                "sort": {
                    "type": "string",
                    "enum": ["bestmatch", "mostrecent", "mostviewed"],
                    "description": (
                        "How to order. 'bestmatch' for topical questions, "
                        "'mostrecent' when the user asks for new / recent / "
                        "latest papers."
                    ),
                    "default": "bestmatch",
                },
            },
            "required": ["query"],
        },
    },
}


__all__ = ["ZenodoSearchClient", "ZenodoHit", "SEARCH_ZENODO_TOOL_SCHEMA"]
