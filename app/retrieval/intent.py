"""Shared query-intent vocabularies.

Single source of truth for the small keyword sets used by BOTH the cache /
service layer and the answer generator — they were byte-identical copies in
two files, which drift silently. Import from here.
"""
from __future__ import annotations

# Live-schedule intent: force the Indico tool AND bypass the response cache
# (a cached "next meeting" goes stale). Intentionally narrow.
TIME_SENSITIVE_KEYWORDS: tuple[str, ...] = (
    " meeting", "meetings", "agenda", "agendas", "schedule", "scheduled",
    "when is", "when's", "when are", "when will", "what's next", "what is next",
    "upcoming", "next week", "this week", "next month", "today's",
)


def is_time_sensitive(query: str) -> bool:
    q = f" {query.lower()} "
    return any(k in q for k in TIME_SENSITIVE_KEYWORDS)
