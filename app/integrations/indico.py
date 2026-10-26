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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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
    # False when the entry came from a failed fetch (no earlier data to
    # fall back on); such entries expire sooner so a retry comes quickly.
    ok: bool = True


class IndicoClient:
    """Thin wrapper around Indico's category JSON export endpoint.

    Supports multiple category URLs so one Indico instance can surface
    events spanning several sibling categories (e.g. ePIC WGs + ePIC
    Collaboration Meetings — these live under different category IDs
    even though they belong to the same project).

    Public API:
        search(query=None, from_date="-7d", to_date="+30d", limit=20)
        warm(windows=None)  # prefetch the tool's default windows

    Caching is stale-while-revalidate: an expired window is still served
    immediately while a background daemon thread refreshes it (one
    in-flight refresh per window). A caller blocks on the network only
    when the window has never been fetched, and at most once per search().
    A failed refresh keeps the previous data.
    """

    # Windows the search_indico_events tool actually requests (see
    # OpenAIAnswerGenerator._run_tool and _FALLBACK_WINDOWS): tool defaults,
    # the "today" upcoming-only form the prompt asks for, their wide 365d
    # fallbacks, and the recent-past window used when nothing is upcoming.
    DEFAULT_WARM_WINDOWS: tuple[tuple[str, str], ...] = (
        ("-7d", "30d"),
        ("-30d", "365d"),
        ("today", "30d"),
        ("today", "365d"),
        ("-90d", "0d"),
    )
    # A refresh that failed while stale data exists postpones the next
    # attempt by this much (capped by the TTL) instead of retrying on
    # every query against a struggling Indico.
    FAILURE_RETRY_S = 60.0

    def __init__(
        self,
        category_url: str | list[str],
        *,
        cache_ttl_s: float = 300.0,
        timeout_s: float = 10.0,
        user_agent: str = "eic-smart-search/1.0 (indico-integration)",
        warm_on_init: bool = True,
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
        # Cache keys with a refresh currently running (background or warm()).
        self._inflight: set[tuple[str, str]] = set()
        # Keys whose last refresh failed — warn once per failure streak.
        self._failing: set[tuple[str, str]] = set()
        if self.cache_ttl_s > 0 and warm_on_init:
            self._spawn(self.warm)

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
            fb = fallback
            # Preserve future-only intent: if the caller asked from "today"
            # (or later), the fallback must not silently re-introduce past
            # events — "when is the NEXT meeting" answered with July is
            # worse than "nothing scheduled yet".
            if not str(from_date).startswith("-"):
                fb = (from_date, fallback[1])
            if fb not in attempts:
                attempts.append(fb)
        events: list[IndicoEvent] = []
        # At most one blocking network round per search(): once a window had
        # to be fetched synchronously, later fallback windows are served from
        # cache only (and fetched in the background if missing).
        allow_block = True
        for attempt_from, attempt_to in attempts:
            key = (attempt_from, attempt_to)
            if allow_block and self.cache_ttl_s > 0:
                with self._lock:
                    will_block = key not in self._cache
            else:
                will_block = allow_block
            window_events = self._fetch_window(key, allow_block=allow_block)
            if will_block:
                allow_block = False
            if window_events is None:
                continue
            if query:
                window_events = self._filter(window_events, query)
            if window_events:
                events = window_events
                break
        events.sort(key=lambda e: e.start)
        return events[:limit]

    def _fetch_window(
        self, cache_key: tuple[str, str], *, allow_block: bool = True
    ) -> list[IndicoEvent] | None:
        """Events for one window, stale-while-revalidate.

        Fresh entry -> returned. Expired entry -> returned as-is while a
        background refresh is scheduled. No entry -> fetched synchronously
        when ``allow_block``; otherwise a background fetch is scheduled and
        ``None`` is returned (the caller already spent its one fetch round).
        """
        if self.cache_ttl_s <= 0:
            if not allow_block:
                return None
            return self._fetch_remote(*cache_key)
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            if cached.expires_at <= now:
                self._refresh_async(cache_key)
            return list(cached.events)
        if not allow_block:
            self._refresh_async(cache_key)
            return None
        self._refresh(cache_key)
        with self._lock:
            cached = self._cache.get(cache_key)
        return list(cached.events) if cached is not None else []

    def _spawn(self, target, *args) -> None:
        """Start ``target(*args)`` in a daemon thread (patched in tests)."""
        threading.Thread(target=target, args=args, daemon=True, name="indico-refresh").start()

    def _refresh_async(self, cache_key: tuple[str, str]) -> None:
        with self._lock:
            if cache_key in self._inflight:
                return
            self._inflight.add(cache_key)
        try:
            self._spawn(self._refresh, cache_key, True)
        except Exception:
            with self._lock:
                self._inflight.discard(cache_key)
            raise

    def _refresh(self, cache_key: tuple[str, str], claimed: bool = False) -> None:
        """Fetch one window and store it, keeping stale data on failure.

        ``claimed`` means the caller already put the key in ``_inflight``.
        """
        if not claimed:
            with self._lock:
                self._inflight.add(cache_key)
        try:
            events, ok = self._fetch_remote_checked(*cache_key)
            now = time.time()
            with self._lock:
                previous = self._cache.get(cache_key)
                if ok:
                    self._cache[cache_key] = _CacheEntry(events=events, expires_at=now + self.cache_ttl_s)
                    self._failing.discard(cache_key)
                    return
                retry_at = now + min(self.cache_ttl_s, self.FAILURE_RETRY_S)
                first_failure = cache_key not in self._failing
                self._failing.add(cache_key)
                if previous is not None and previous.ok:
                    # Keep serving the last good data; just pace the retry.
                    previous.expires_at = retry_at
                else:
                    self._cache[cache_key] = _CacheEntry(events=events, expires_at=retry_at, ok=False)
            if first_failure:
                logger.warning(
                    "indico_refresh_failed_serving_stale" if previous is not None and previous.ok
                    else "indico_refresh_failed",
                    extra={"window": f"{cache_key[0]}..{cache_key[1]}"},
                )
        except Exception as exc:  # never let a background refresh die loudly
            logger.warning("indico_refresh_error", extra={"error": str(exc)[:200]})
        finally:
            with self._lock:
                self._inflight.discard(cache_key)

    def warm(self, windows: list[tuple[str, str]] | None = None) -> int:
        """Prefetch windows (default: those the live tool uses) into the cache.

        Blocking; meant for a background/keepalive thread. Windows that are
        still fresh or already being refreshed are skipped. Returns the
        number of windows fetched. Thread-safe; never raises.
        """
        if self.cache_ttl_s <= 0:
            return 0
        fetched = 0
        for window in windows or self.DEFAULT_WARM_WINDOWS:
            key = (str(window[0]), str(window[1]))
            now = time.time()
            with self._lock:
                cached = self._cache.get(key)
                if key in self._inflight or (cached is not None and cached.expires_at > now):
                    continue
                self._inflight.add(key)
            self._refresh(key, claimed=True)
            fetched += 1
        return fetched

    def _fetch_categories_parallel(self, fetch_one) -> list[list]:
        """Run ``fetch_one(export_url)`` for every category concurrently.

        A tool call sits directly on the user's answer latency; with 2+
        categories the sequential fetch doubled it. Order of results
        matches ``self._export_urls`` so dedup stays deterministic.
        """
        if len(self._export_urls) == 1:
            return [fetch_one(self._export_urls[0])]
        with ThreadPoolExecutor(max_workers=len(self._export_urls)) as pool:
            return list(pool.map(fetch_one, self._export_urls))

    def _fetch_remote(self, from_date: str, to_date: str) -> list[IndicoEvent]:
        """Fetch all configured categories and dedupe the union by event id."""
        return self._fetch_remote_checked(from_date, to_date)[0]

    def _fetch_remote_checked(self, from_date: str, to_date: str) -> tuple[list[IndicoEvent], bool]:
        """Like ``_fetch_remote`` but also report whether every category succeeded."""
        seen: set[str] = set()
        merged: list[IndicoEvent] = []
        per_category = self._fetch_categories_parallel(
            lambda url: self._fetch_single_checked(url, from_date, to_date)
        )
        ok = all(events is not None for events in per_category)
        for events in per_category:
            for ev in events or []:
                if ev.id and ev.id not in seen:
                    seen.add(ev.id)
                    merged.append(ev)
        return merged, ok

    # indico.bnl.gov 504s when a detail=contributions export spans much
    # more than a few months (the payload is generated synchronously).
    # Wide requests are sliced into windows of this many days and merged.
    RAW_FETCH_WINDOW_DAYS = 60

    def fetch_raw_events(
        self,
        from_date: str,
        to_date: str,
        *,
        detail: str = "contributions",
    ) -> list[dict]:
        """Return the raw event dicts (with chairs/contributions populated).

        Used by the people-graph builder and the indico_events connector.
        ``detail=contributions`` includes speakers per talk; the heavier
        ``subcontributions`` is rarely worth the bytes for ePIC events.
        """
        seen: set[str] = set()
        out: list[dict] = []
        for span_from, span_to in self._split_window(from_date, to_date):
            per_category = self._fetch_categories_parallel(
                lambda url, f=span_from, t=span_to: self._fetch_raw_single(url, f, t, detail)
            )
            for results in per_category:
                for raw in results:
                    ev_id = str(raw.get("id") or "")
                    if ev_id and ev_id not in seen:
                        seen.add(ev_id)
                        out.append(raw)
        return out

    _REL_DATE_RE = re.compile(r"^(-?\d+)d$")

    @classmethod
    def _parse_window_date(cls, raw: str) -> "date | None":
        raw = (raw or "").strip()
        match = cls._REL_DATE_RE.match(raw)
        if match:
            return date.today() + timedelta(days=int(match.group(1)))
        if raw == "today":
            return date.today()
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None

    @classmethod
    def _split_window(cls, from_date: str, to_date: str) -> list[tuple[str, str]]:
        """Slice [from_date, to_date] into RAW_FETCH_WINDOW_DAYS spans.

        Falls back to the single original window when either bound uses a
        format we can't parse (Indico accepts more variants than we slice).
        Boundary days overlap by design — the caller dedupes by event id.
        """
        start = cls._parse_window_date(from_date)
        end = cls._parse_window_date(to_date)
        if start is None or end is None or start >= end:
            return [(from_date, to_date)]
        spans: list[tuple[str, str]] = []
        cursor = start
        step = timedelta(days=cls.RAW_FETCH_WINDOW_DAYS)
        while cursor < end:
            span_end = min(cursor + step, end)
            spans.append((cursor.isoformat(), span_end.isoformat()))
            cursor = span_end
        return spans

    def fetch_timetable(self, event_id: str) -> list[dict]:
        """Flat timetable of one event: sessions + contributions with rooms.

        Backs the get_indico_event_timetable tool — room/time-slot detail
        ("what room is the tracking workfest?") lives only here; the
        category export carries event-level location at best.
        """
        event_id = re.sub(r"[^0-9]", "", str(event_id))
        if not event_id:
            return []
        host = self._export_urls[0].split("/export/")[0]
        url = f"{host}/export/timetable/{event_id}.json"
        try:
            response = httpx.get(
                url,
                params={"pretty": "no"},
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("indico_timetable_fetch_failed", extra={"event": event_id, "error": str(exc)[:200]})
            return []
        days = (payload.get("results") or {}).get(event_id) or {}
        entries: list[dict] = []

        def _add(entry: dict) -> None:
            title = str(entry.get("title") or "").strip()
            if not title:
                return
            start = entry.get("startDate") or {}
            entries.append(
                {
                    "title": title,
                    "type": str(entry.get("entryType") or entry.get("_type") or ""),
                    "date": start.get("date") or "",
                    "time": (start.get("time") or "")[:5],
                    "room": str(entry.get("room") or entry.get("roomFullname") or "").strip(),
                    "location": str(entry.get("location") or "").strip(),
                }
            )

        for _day, day_entries in sorted(days.items()):
            if not isinstance(day_entries, dict):
                continue
            for _k, entry in sorted(day_entries.items()):
                if not isinstance(entry, dict):
                    continue
                _add(entry)
                for _k2, nested in sorted((entry.get("entries") or {}).items()):
                    if isinstance(nested, dict):
                        _add(nested)
        return entries

    def _fetch_raw_single(
        self,
        export_url: str,
        from_date: str,
        to_date: str,
        detail: str,
    ) -> list[dict]:
        params = {"from": from_date, "to": to_date, "detail": detail, "pretty": "no"}
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        try:
            response = httpx.get(export_url, params=params, headers=headers, timeout=self.timeout_s)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("indico_raw_fetch_failed", extra={"url": export_url, "error": str(exc)})
            return []
        try:
            payload = response.json()
        except ValueError:
            return []
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            return []
        return [r for r in results if isinstance(r, dict)]

    def _fetch_single(self, export_url: str, from_date: str, to_date: str) -> list[IndicoEvent]:
        return self._fetch_single_checked(export_url, from_date, to_date) or []

    def _fetch_single_checked(self, export_url: str, from_date: str, to_date: str) -> list[IndicoEvent] | None:
        """Events of one category, or ``None`` when the fetch failed."""
        params = {"from": from_date, "to": to_date, "detail": "events", "pretty": "no"}
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        try:
            response = httpx.get(export_url, params=params, headers=headers, timeout=self.timeout_s)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("indico_fetch_failed", extra={"url": export_url, "error": str(exc)})
            return None
        try:
            payload = response.json()
        except ValueError:
            logger.warning("indico_non_json_response", extra={"url": export_url})
            return None
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


INDICO_TIMETABLE_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_indico_event_timetable",
        "description": (
            "Fetch the detailed timetable of ONE specific Indico event: every "
            "session and talk with its date, start time, ROOM and building. "
            "Call this for questions about where/when a specific session, "
            "workfest, or talk happens inside a known meeting (e.g. 'what "
            "room is the tracking workfest at the Glasgow collaboration "
            "meeting?'). First identify the event id — from a source URL "
            "like indico.bnl.gov/event/31808/ the id is 31808 — then call "
            "this tool with it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "Numeric Indico event id (digits only, e.g. '31808').",
                }
            },
            "required": ["event_id"],
        },
    },
}
