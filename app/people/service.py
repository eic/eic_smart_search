"""Lookup helpers for the lookup_person LLM tool and the /people API.

Match strategy, in order:
    1. Exact normalized_name match — short-circuit, single result.
    2. Trigram similarity over canonical_name + normalized_name (pg_trgm).
       Filtered by a minimum similarity floor so "alex" doesn't match
       everyone with "Alex" in any alias.
    3. Token-prefix substring fallback for Postgres without pg_trgm in
       the test environment (sqlite-style — kept defensive, not core).

Output is always a list of dicts shaped for both the LLM tool result and
the /people/search REST endpoint.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import Text as _SAText
from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import Session

from app.models.entities import Person
from app.people.normalize import normalize_name

logger = logging.getLogger(__name__)


# Trigram similarity floor. 0.3 is pg_trgm's own default; we sit slightly
# above to avoid noisy matches on common short tokens.
_MIN_SIMILARITY = 0.32

# Cap activity included in the tool card. The LLM only needs enough recent
# context to form a sentence; padding the prompt with 50 talks is wasteful.
_MAX_ACTIVITY_IN_CARD = 8


def lookup(
    db: Session,
    query: str,
    *,
    limit: int = 5,
    min_similarity: float = _MIN_SIMILARITY,
) -> list[Person]:
    """Return up to ``limit`` Person rows matching ``query``, best-first."""
    nq = normalize_name(query)
    if not nq:
        return []

    # 1) exact hit on normalized form
    exact = db.scalar(select(Person).where(Person.normalized_name == nq))
    if exact is not None:
        return [exact]

    # 2) trigram similarity (requires pg_trgm — enabled in 0001_initial)
    try:
        sim_canonical = func.similarity(Person.canonical_name, query)
        sim_normalized = func.similarity(Person.normalized_name, nq)
        score = func.greatest(sim_canonical, sim_normalized).label("score")
        rows = db.execute(
            select(Person, score)
            .where(score >= min_similarity)
            .order_by(desc(score), desc(Person.activity_count))
            .limit(limit)
        ).all()
        if rows:
            return [r[0] for r in rows]
    except Exception as exc:  # pragma: no cover — only hit when pg_trgm is unavailable
        logger.debug("people_trgm_lookup_failed", extra={"error": str(exc)[:200]})

    # 2.5) Topic fallback: the query is not a person's name — find people BY
    # role/topic instead ("jet convener", "dRICH expert") via working_groups
    # and activity titles. Ranked by activity so active people surface first.
    topic_tokens = [t for t in nq.split() if len(t) > 2 and t not in {"the", "who", "now"}]
    if topic_tokens:
        conditions = [
            or_(
                func.lower(Person.working_groups.cast(_SAText)).contains(t),
                func.lower(Person.activity.cast(_SAText)).contains(t),
            )
            for t in topic_tokens
        ]
        try:
            topical = db.execute(
                select(Person)
                .where(*conditions)
                .order_by(desc(Person.activity_count))
                .limit(limit)
            ).scalars().all()
            if topical:
                return list(topical)
        except Exception as exc:
            logger.debug("people_topic_lookup_failed", extra={"error": str(exc)[:200]})

    # 3) Substring fallback. Splits the query on whitespace and requires every
    # token to appear somewhere in canonical_name OR normalized_name. Keeps the
    # service usable on a non-Postgres test backend.
    tokens = [t for t in nq.split() if t]
    if not tokens:
        return []
    conditions = [
        or_(
            func.lower(Person.canonical_name).contains(t),
            Person.normalized_name.contains(t),
        )
        for t in tokens
    ]
    fallback = db.execute(
        select(Person)
        .where(*conditions)
        .order_by(desc(Person.activity_count))
        .limit(limit)
    ).scalars().all()
    return list(fallback)


def to_card(person: Person, *, max_activity: int = _MAX_ACTIVITY_IN_CARD) -> dict[str, Any]:
    """Render a Person as the dict the LLM tool / REST endpoint emits."""
    activity = list(person.activity or [])[:max_activity]
    return {
        "name": person.canonical_name,
        "aliases": list(person.aliases or []),
        "affiliations": list(person.affiliations or []),
        "orcid": person.orcid,
        "email": person.email,
        "working_groups": list(person.working_groups or []),
        "activity_count": int(person.activity_count or 0),
        "last_seen": person.last_seen.isoformat() if person.last_seen else None,
        "recent_activity": activity,
    }
