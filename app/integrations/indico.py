"""Live Indico meeting lookup for the ePIC category.

Meeting data is inherently time-relative ("next meeting", "this week") and
changes up to the last minute, so we treat it as a live stream rather than
trying to bake it into the hybrid index. The generator exposes the lookup
as an OpenAI tool; the LLM decides when to call it.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndicoEvent:
    id: str
    title: str
    url: str
    category: str
    start: str  # ISO-ish "2026-05-22T11:00:00-04:00"
    end: str
    description: str
    location: str

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "category": self.category,
            "start": self.start,
            "end": self.end,
            "location": self.location,
            "description": self.description[:600],
        }


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_html(raw: str | None) -> str:
    if not raw:
        return ""
    stripped = _TAG_RE.sub(" ", raw)
    return _WS_RE.sub(" ", stripped).strip()


def _combine_dt(part: dict[str, Any] | None) -> str:
    if not isinstance(part, dict):
        return ""
    date = part.get("date") or ""
    tstr = part.get("time") or ""
    tz = part.get("tz") or ""
    out = date
    if tstr:
        out = f"{out}T{tstr}"
    if tz:
        out = f"{out} {tz}"
    return out


@dataclass(slots=True)
class _CacheEntry:
    events: list[IndicoEvent]
    expires_at: float


class IndicoClient:
    """Thin wrapper around Indico's category JSON export endpoint.

    Supports multiple category URLs so one Indico instance can surface
    events spanning several sibling categories (e.g. ePIC WGs + ePIC
    Collaboration Meetings — these live under different category IDs
    even though they belong to the same project).

    Public API:
        search(query=None, from_date="-7d", to_date="+30d", limit=20)
    """

    def __init__(
        self,
        category_url: str | list[str],
        *,
        cache_ttl_s: float = 300.0,
        timeout_s: float = 10.0,
        user_agent: str = "eic-smart-search/1.0 (indico-integration)",
    ) -> None:
        raw_urls: list[str]
        if isinstance(category_url, str):
            raw_urls = [u.strip() for u in category_url.split(",") if u.strip()]
        else:
            raw_urls = [u.strip() for u in category_url if u and u.strip()]
        if not raw_urls:
            raise ValueError("IndicoClient: at least one category URL is required")
        self.category_urls = [u.rstrip("/") for u in raw_urls]
        # Export path per category: "/category/402/" -> "/export/categ/402.json".
        self._export_urls = [self._to_export_url(u) for u in self.category_urls]
        # Keep a human-readable summary for answer citations.
        self.category_url = ", ".join(self.category_urls)
        self.cache_ttl_s = max(0.0, cache_ttl_s)
        self.timeout_s = timeout_s
        self.user_agent = user_agent
        self._cache: dict[tuple[str, str], _CacheEntry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _to_export_url(category_url: str) -> str:
        match = re.match(r"(https?://[^/]+)/category/(\d+)/?", category_url)
        if not match:
            raise ValueError(f"Not a category URL: {category_url!r}")
        host, category_id = match.group(1), match.group(2)
        return f"{host}/export/categ/{category_id}.json"

    # If the caller's window comes up empty (e.g. LLM guessed a 30-day
    # lookahead but the event is 2.5 months out), try again with a much
    # wider window. Many important ePIC events — annual collaboration
    # meetings, joint ePIC/EICUG meetings — sit months ahead of now.
    _FALLBACK_WINDOWS: tuple[tuple[str, str], ...] = (
        ("-30d", "365d"),
    )

    def search(
        self,
        query: str | None = None,
        from_date: str = "-7d",
        to_date: str = "30d",
        limit: int = 20,
    ) -> list[IndicoEvent]:
        """Return upcoming/recent events optionally matched by a query string.

        Date args are Indico's relative format: "-7d", "today", "30d", "2026-05-01".
        `limit` caps the returned slice; the cache is always the full window.
        """
        attempts: list[tuple[str, str]] = [(from_date, to_date)]
        for fallback in self._FALLBACK_WINDOWS:
            if fallback not in attempts:
                attempts.append(fallback)
        events: list[IndicoEvent] = []
        for attempt_from, attempt_to in attempts:
            window_events = self._fetch_window((attempt_from, attempt_to))
            if query:
                window_events = self._filter(window_events, query)
            if window_events:
                events = window_events
                break
        events.sort(key=lambda e: e.start)
        return events[:limit]

    def _fetch_window(self, cache_key: tuple[str, str]) -> list[IndicoEvent]:
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and cached.expires_at > now:
                return list(cached.events)
        from_date, to_date = cache_key
        events = self._fetch_remote(from_date, to_date)
        if self.cache_ttl_s > 0:
            with self._lock:
                self._cache[cache_key] = _CacheEntry(events=events, expires_at=now + self.cache_ttl_s)
        return events

    def _fetch_remote(self, from_date: str, to_date: str) -> list[IndicoEvent]:
        """Fetch all configured categories and dedupe the union by event id."""
        seen: set[str] = set()
        merged: list[IndicoEvent] = []
        for export_url in self._export_urls:
            for ev in self._fetch_single(export_url, from_date, to_date):
                if ev.id and ev.id not in seen:
                    seen.add(ev.id)
                    merged.append(ev)
        return merged

    def _fetch_single(self, export_url: str, from_date: str, to_date: str) -> list[IndicoEvent]:
        params = {"from": from_date, "to": to_date, "detail": "events", "pretty": "no"}
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        try:
            response = httpx.get(export_url, params=params, headers=headers, timeout=self.timeout_s)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("indico_fetch_failed", extra={"url": export_url, "error": str(exc)})
            return []
        try:
            payload = response.json()
        except ValueError:
            logger.warning("indico_non_json_response", extra={"url": export_url})
            return []
        if isinstance(payload, dict) and payload.get("message") and not payload.get("results"):
            logger.warning("indico_api_message", extra={"indico_message": payload.get("message"), "url": export_url})
            return []
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            return []
        events: list[IndicoEvent] = []
        for raw in results:
            if not isinstance(raw, dict):
                continue
            events.append(
                IndicoEvent(
                    id=str(raw.get("id") or ""),
                    title=str(raw.get("title") or "").strip(),
                    url=str(raw.get("url") or "").strip(),
                    category=str(raw.get("category") or "").strip(),
                    start=_combine_dt(raw.get("startDate")),
                    end=_combine_dt(raw.get("endDate")),
                    description=_clean_html(raw.get("description")),
                    location=str(raw.get("location") or raw.get("room") or "").strip(),
                )
            )
        return events

    # Tokens that are too generic to discriminate between Indico events.
    # "meeting" matches almost every event; "week"/"today"/"next" are
    # time deictics already handled by from_date/to_date.
    _STOP_TOKENS = frozenset({
        "meeting", "meetings", "agenda", "agendas", "session", "sessions",
        "event", "events", "week", "weeks", "today", "tomorrow", "next",
        "upcoming", "last", "previous", "this", "the", "for", "and",
        "of", "a", "an", "to", "in", "on",
    })

    @staticmethod
    def _token_matches(token: str, haystack: str) -> bool:
        """Match a search token against a field with prefix tolerance.

        Full substring match first; for longer tokens (≥5 chars) a 5-char
        prefix also counts so "calorimetry" still finds "calorimeter" and
        "tracking" finds "tracker". Avoids a hard morphology lib.
        """
        if token in haystack:
            return True
        if len(token) >= 5 and token[:5] in haystack:
            return True
        return False

    @classmethod
    def _filter(cls, events: list[IndicoEvent], query: str) -> list[IndicoEvent]:
        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 1]
        tokens = [t for t in tokens if t not in cls._STOP_TOKENS]
        if not tokens:
            return events
        scored: list[tuple[int, IndicoEvent]] = []
        for ev in events:
            title = ev.title.lower()
            category = ev.category.lower()
            description = ev.description.lower()
            title_hits = sum(1 for t in tokens if cls._token_matches(t, title))
            category_hits = sum(1 for t in tokens if cls._token_matches(t, category))
            description_hits = sum(1 for t in tokens if cls._token_matches(t, description))
            if title_hits + category_hits + description_hits == 0:
                continue
            # Require a discriminating hit (title or category) when the
            # query had more than one meaningful token; for single-token
            # queries be more permissive and let description hits count.
            if title_hits == 0 and category_hits == 0 and len(tokens) > 1:
                continue
            score = title_hits * 3 + category_hits * 2 + description_hits
            scored.append((score, ev))
        scored.sort(key=lambda s: (-s[0], s[1].start))
        return [e for _, e in scored]

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()


# Tool schema consumed by OpenAIAnswerGenerator when INDICO is enabled.
INDICO_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_indico_events",
        "description": (
            "Search live meeting schedule data for the ePIC Indico category. "
            "Call this whenever the user asks about meetings, schedules, agendas, "
            "upcoming events, when a working group meets, or anything time-relative "
            "('next week', 'today', 'this month'). Do NOT call it for general "
            "physics/software questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional keyword filter (working group name, topic). Leave empty for all events.",
                },
                "from_date": {
                    "type": "string",
                    "description": "Start of window — Indico relative form ('-7d', 'today') or ISO date ('2026-05-01'). Default '-7d'.",
                    "default": "-7d",
                },
                "to_date": {
                    "type": "string",
                    "description": (
                        "End of window — Indico relative form ('30d', '365d') or ISO date. "
                        "Default '30d' for weekly/regular working-group meetings. For "
                        "Collaboration Meetings / Joint ePIC-EICUG Meetings / Annual reviews "
                        "(infrequent, may be months away) use '365d' to avoid missing them."
                    ),
                    "default": "30d",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max events to return. Default 10, max 25.",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
}


__all__ = ["IndicoClient", "IndicoEvent", "INDICO_TOOL_SCHEMA"]
