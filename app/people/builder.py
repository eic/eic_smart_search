"""Aggregate the people graph from already-ingested sources.

Run via ``POST /people/refresh`` or ``rebuild_people()`` directly.
The graph is dedup-keyed on ``normalize_name(full_name)`` — so
"Müller, Hans (BNL)" and "Hans Müller" collapse to the same row,
and a future ORCID-aware pass can promote real identity matches.

Currently aggregates from:
    1. Zenodo author lists already stored in ``documents.doc_metadata``.
    2. INSPIRE-HEP author lists (+ORCID) from ingested literature docs.
    3. Indico chairs + contribution speakers (live API call when enabled).
    4. Website convener/email mentions from crawled internal pages.

GitHub commit authors are NOT pulled here yet — adding that requires
new API calls per repo and is best done on its own cron.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings
from app.integrations.indico import IndicoClient
from app.models.entities import Document, Person
from app.people.normalize import (
    ParsedAuthor,
    join_indico_person,
    merge_unique,
    normalize_name,
    parse_author_string,
)

logger = logging.getLogger(__name__)


# Cap stored activity per person so a prolific author doesn't bloat one row.
# Newest entries win when we exceed it.
MAX_ACTIVITY_PER_PERSON = 60


# Match "Title-cased Name(s) ( email@domain )" inside crawled markdown chunks.
# Requires 2-4 capitalised tokens for the name to skip false positives like
# bare role labels ("Conveners", "Members") sitting before the parenthesis.
_NAME_EMAIL_RE = re.compile(
    r"([A-Z][\w.\-']+(?:\s+[A-Z][\w.\-']+){1,3})\s*\(\s*"
    r"([A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\s*\)"
)

# Role labels that the regex sometimes drags in as the first token of the
# captured name when the chunk reads "Conveners Sasha Prozorov ( … )".
# We strip them after the match instead of complicating the regex.
_NAME_PREFIX_STOPWORDS = frozenset(
    {
        "convener",
        "conveners",
        "co-convener",
        "co-conveners",
        "contact",
        "contacts",
        "lead",
        "leads",
        "member",
        "members",
        "chair",
        "chairs",
        "co-chair",
        "co-chairs",
    }
)


def rebuild_people(
    db: Session,
    settings: Settings,
    *,
    indico_lookback_days: int = 365,
    indico_lookahead_days: int = 30,
) -> dict[str, int]:
    """Rebuild persons from Zenodo + Indico in place. Idempotent.

    Does NOT delete unseen rows: a person who appeared in last quarter's
    Zenodo dump but is missing now (record withdrawn, etc.) stays in the
    table — disappearing is rarely real, usually a transient. Stale-row
    cleanup belongs in a separate explicit reset endpoint.
    """
    counts = {
        "zenodo_authors": 0,
        "inspire_authors": 0,
        "indico_chairs": 0,
        "indico_speakers": 0,
        "website_conveners": 0,
        "email_merges": 0,
        "persons_upserted": 0,
    }
    upserts: dict[str, "_PersonAccumulator"] = {}

    _collect_from_zenodo(db, upserts, counts)
    _collect_from_inspire(db, upserts, counts)
    _collect_from_website_conveners(db, upserts, counts)
    if getattr(settings, "INDICO_ENABLED", False):
        try:
            _collect_from_indico(
                upserts,
                counts,
                category_url=settings.INDICO_CATEGORY_URL,
                lookback_days=indico_lookback_days,
                lookahead_days=indico_lookahead_days,
                cache_ttl_s=settings.INDICO_CACHE_TTL_S,
                # Use the dedicated, longer timeout — the
                # detail=contributions payload routinely exceeds the
                # 10s default budget intended for the live tool.
                timeout_s=getattr(settings, "PEOPLE_REFRESH_INDICO_TIMEOUT_S", 60.0),
            )
        except Exception as exc:  # don't let Indico flakiness fail the whole rebuild
            logger.warning("people_indico_collect_failed", extra={"error": str(exc)[:300]})

    counts["email_merges"] = _merge_by_email(db, upserts)
    counts["persons_upserted"] = _flush(db, upserts)
    db.commit()
    return counts


# ---- accumulator ---------------------------------------------------------


class _PersonAccumulator:
    """In-memory aggregate before we hit the DB.

    Keyed by ``normalized_name``; holds the longest seen display name as
    the canonical, plus merged aliases / affiliations / activity. Avoids
    one round-trip per author.
    """

    __slots__ = (
        "normalized",
        "canonical_name",
        "aliases",
        "affiliations",
        "activity",
        "last_seen",
        "email",
        "orcid",
        "working_groups",
    )

    def __init__(self, normalized: str, canonical_name: str) -> None:
        self.normalized = normalized
        self.canonical_name = canonical_name
        self.aliases: list[str] = []
        self.affiliations: list[str] = []
        self.activity: list[dict[str, Any]] = []
        self.last_seen: datetime | None = None
        self.email: str | None = None
        self.orcid: str | None = None
        self.working_groups: list[str] = []

    def add_name_variant(self, display: str) -> None:
        if display and display != self.canonical_name:
            self.aliases = merge_unique(self.aliases, [display])
        # Prefer the longest display string — usually carries the middle
        # initial / accents that the shorter forms drop.
        if display and len(display) > len(self.canonical_name):
            self.aliases = merge_unique(self.aliases, [self.canonical_name])
            self.canonical_name = display

    def add_affiliation(self, affiliation: str | None) -> None:
        if affiliation:
            self.affiliations = merge_unique(self.affiliations, [affiliation])

    def add_email(self, email: str | None) -> None:
        if not email:
            return
        normalized = email.strip().lower()
        if "@" not in normalized:
            return
        # Keep the first email we see — emails on these pages are stable
        # primary contacts and we don't want a misfired match later in the
        # crawl to overwrite a known-good address.
        if not self.email:
            self.email = normalized

    def add_orcid(self, orcid: str | None) -> None:
        # First ORCID wins — it's a stable global identifier, later
        # sightings of the same person can't legitimately change it.
        if orcid and not self.orcid:
            self.orcid = orcid.strip()

    def add_working_group(self, working_group: str | None) -> None:
        if working_group:
            self.working_groups = merge_unique(self.working_groups, [working_group])

    def absorb(self, other: "_PersonAccumulator") -> None:
        """Fold ``other`` into self. Used by the email-merge pass."""
        self.add_name_variant(other.canonical_name)
        for alias in other.aliases:
            self.add_name_variant(alias)
        self.affiliations = merge_unique(self.affiliations, other.affiliations)
        self.working_groups = merge_unique(self.working_groups, other.working_groups)
        for entry in other.activity:
            self.add_activity(entry)
        if not self.email:
            self.email = other.email
        if not self.orcid:
            self.orcid = other.orcid

    def add_activity(self, entry: dict[str, Any]) -> None:
        self.activity.append(entry)
        when_raw = entry.get("when")
        if isinstance(when_raw, str) and when_raw:
            try:
                when = datetime.fromisoformat(when_raw.replace("Z", "+00:00"))
            except ValueError:
                when = None
            if when is not None:
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                if self.last_seen is None or when > self.last_seen:
                    self.last_seen = when


def _accumulate(
    upserts: dict[str, _PersonAccumulator],
    parsed: ParsedAuthor,
) -> _PersonAccumulator | None:
    normalized = normalize_name(parsed.full_name)
    if not normalized:
        return None
    acc = upserts.get(normalized)
    if acc is None:
        acc = _PersonAccumulator(normalized=normalized, canonical_name=parsed.full_name)
        upserts[normalized] = acc
    else:
        acc.add_name_variant(parsed.full_name)
    acc.add_affiliation(parsed.affiliation)
    return acc


# ---- Zenodo --------------------------------------------------------------


def _collect_from_zenodo(
    db: Session,
    upserts: dict[str, _PersonAccumulator],
    counts: dict[str, int],
) -> None:
    """Iterate already-indexed Zenodo documents and pull their authors.

    The Zenodo connector stores ``authors`` (list[str], pre-formatted as
    ``"Name (Affiliation)"``) under ``documents.doc_metadata``. We don't
    re-fetch from Zenodo — this just reshapes what's already in Postgres.
    """
    rows = db.execute(
        select(Document).where(Document.source_type == "zenodo_community")
    ).scalars()
    for doc in rows:
        meta = doc.doc_metadata or {}
        raw_authors = meta.get("authors")
        if not isinstance(raw_authors, list):
            continue
        when_iso = doc.last_updated.isoformat() if doc.last_updated else None
        for raw in raw_authors:
            if not isinstance(raw, str):
                continue
            parsed = parse_author_string(raw)
            if parsed is None:
                continue
            acc = _accumulate(upserts, parsed)
            if acc is None:
                continue
            counts["zenodo_authors"] += 1
            acc.add_activity(
                {
                    "kind": "paper",
                    "role": "author",
                    "title": doc.title,
                    "url": doc.url,
                    "when": when_iso,
                    "source": "zenodo",
                    "doi": meta.get("doi"),
                }
            )


# ---- INSPIRE-HEP ---------------------------------------------------------


def _collect_from_inspire(
    db: Session,
    upserts: dict[str, _PersonAccumulator],
    counts: dict[str, int],
) -> None:
    """Pull authors (+ORCIDs) from already-indexed INSPIRE literature docs.

    The INSPIRE connector stores ``authors`` as ``"Last, First (Affil)"``
    strings and ``author_orcids`` as a ``{"Last, First": orcid}`` map in
    ``documents.doc_metadata``. ORCID is the only cross-source identifier
    we get for free, so this is what upgrades name-string matching into
    real identity matching.
    """
    rows = db.execute(
        select(Document).where(Document.source_type == "inspire_literature")
    ).scalars()
    for doc in rows:
        meta = doc.doc_metadata or {}
        raw_authors = meta.get("authors")
        if not isinstance(raw_authors, list):
            continue
        orcid_map = meta.get("author_orcids")
        if not isinstance(orcid_map, dict):
            orcid_map = {}
        when_iso = doc.last_updated.isoformat() if doc.last_updated else None
        for raw in raw_authors:
            if not isinstance(raw, str):
                continue
            parsed = parse_author_string(raw)
            if parsed is None:
                continue
            acc = _accumulate(upserts, parsed)
            if acc is None:
                continue
            counts["inspire_authors"] += 1
            # orcid_map keys are the bare "Last, First" without affiliation.
            bare_name = raw.split(" (", 1)[0].strip()
            orcid = orcid_map.get(bare_name)
            if isinstance(orcid, str):
                acc.add_orcid(orcid)
            acc.add_activity(
                {
                    "kind": "paper",
                    "role": "author",
                    "title": doc.title,
                    "url": doc.url,
                    "when": when_iso,
                    "source": "inspire",
                    "doi": meta.get("doi") or None,
                }
            )


# ---- Indico --------------------------------------------------------------


def _collect_from_indico(
    upserts: dict[str, _PersonAccumulator],
    counts: dict[str, int],
    *,
    category_url: str,
    lookback_days: int,
    lookahead_days: int,
    cache_ttl_s: float,
    timeout_s: float,
) -> None:
    client = IndicoClient(
        category_url=category_url,
        cache_ttl_s=cache_ttl_s,
        timeout_s=timeout_s,
        warm_on_init=False,  # one-off bulk export; the query-time cache is not used here
    )
    raw_events = client.fetch_raw_events(
        from_date=f"-{int(lookback_days)}d",
        to_date=f"{int(lookahead_days)}d",
        detail="contributions",
    )
    for ev in raw_events:
        ev_title = str(ev.get("title") or "").strip()
        ev_url = str(ev.get("url") or "").strip()
        ev_when = _start_iso(ev.get("startDate"))
        category = str(ev.get("category") or "").strip()

        for chair_raw in ev.get("chairs") or []:
            parsed = join_indico_person(chair_raw if isinstance(chair_raw, dict) else {})
            if parsed is None:
                continue
            acc = _accumulate(upserts, parsed)
            if acc is None:
                continue
            counts["indico_chairs"] += 1
            acc.add_activity(
                {
                    "kind": "meeting",
                    "role": "chair",
                    "title": ev_title,
                    "url": ev_url,
                    "when": ev_when,
                    "source": "indico",
                    "category": category,
                }
            )

        for contrib in ev.get("contributions") or []:
            if not isinstance(contrib, dict):
                continue
            talk_title = str(contrib.get("title") or "").strip() or ev_title
            talk_url = str(contrib.get("url") or "").strip() or ev_url
            talk_when = _start_iso(contrib.get("startDate")) or ev_when
            for speaker_raw in contrib.get("speakers") or []:
                parsed = join_indico_person(speaker_raw if isinstance(speaker_raw, dict) else {})
                if parsed is None:
                    continue
                acc = _accumulate(upserts, parsed)
                if acc is None:
                    continue
                counts["indico_speakers"] += 1
                acc.add_activity(
                    {
                        "kind": "talk",
                        "role": "speaker",
                        "title": talk_title,
                        "url": talk_url,
                        "when": talk_when,
                        "source": "indico",
                        "category": category,
                        "event_title": ev_title,
                    }
                )


def _start_iso(part: Any) -> str | None:
    if not isinstance(part, dict):
        return None
    date = part.get("date") or ""
    tstr = part.get("time") or ""
    if not date:
        return None
    if tstr:
        return f"{date}T{tstr}"
    return date


# ---- website conveners --------------------------------------------------


def _collect_from_website_conveners(
    db: Session,
    upserts: dict[str, _PersonAccumulator],
    counts: dict[str, int],
) -> None:
    """Mine working-group convener listings from epic-eic.org/sc/* pages.

    The crawler already indexes these pages as ``internal_website`` docs;
    each chunk's body has lines like ``- Sasha Prozorov ( … @cvut.cz )``.
    We pluck name + email + working-group label per match and feed them
    into the accumulator. Email is the merge key downstream — the same
    person often appears in Zenodo as their formal name and on the
    website as their preferred (e.g. nickname) form.
    """
    rows = (
        db.execute(
            select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.source_type == "internal_website")
            .where(Document.url.contains("epic-eic.org/sc/"))
        )
        .scalars()
        .all()
    )
    for doc in rows:
        wg_label = _wg_label_from_doc(doc)
        for chunk in doc.chunks:
            text = chunk.content or ""
            # Restrict to chunks that *talk about* conveners, otherwise we
            # match stray "Name ( email )" mentions in unrelated paragraphs.
            if "convener" not in text.lower():
                continue
            chunk_wg_label = _wg_label_from_chunk(chunk) or wg_label
            for match in _NAME_EMAIL_RE.finditer(text):
                raw_name = _strip_role_prefix(match.group(1))
                email = match.group(2)
                if not raw_name or len(raw_name.split()) < 2:
                    continue
                parsed = ParsedAuthor(full_name=raw_name, affiliation=None)
                acc = _accumulate(upserts, parsed)
                if acc is None:
                    continue
                acc.add_email(email)
                acc.add_working_group(chunk_wg_label)
                counts["website_conveners"] += 1
                acc.add_activity(
                    {
                        "kind": "working_group_role",
                        "role": "convener",
                        "title": chunk_wg_label or doc.title,
                        "url": doc.url,
                        "when": doc.last_updated.isoformat() if doc.last_updated else None,
                        "source": "epic_internal",
                    }
                )


def _strip_role_prefix(raw: str) -> str:
    """Drop a leading role label ("Conveners", "Chair") from a captured name.

    The regex deliberately accepts 2-4 capitalised tokens, which lets
    "Conveners Sasha Prozorov" match. Strip the role token here so the
    person's actual name is what we register.
    """
    tokens = raw.split()
    while tokens and tokens[0].lower() in _NAME_PREFIX_STOPWORDS:
        tokens.pop(0)
    return " ".join(tokens).strip()


def _wg_label_from_chunk(chunk: Any) -> str | None:
    heading_path = getattr(chunk, "heading_path", None)
    if isinstance(heading_path, list) and heading_path:
        deepest = str(heading_path[-1] or "").strip()
        if deepest:
            return deepest
    return None


def _wg_label_from_doc(doc: Document) -> str | None:
    """Pick a stable WG name from the document title.

    Crawled SC pages title themselves like
    ``"User Learning Working Group · The ePIC Collaboration Website"`` —
    take the segment before the bullet so the working_groups list reads
    cleanly.
    """
    title = (doc.title or "").strip()
    if not title:
        return None
    head = title.split("·", 1)[0].strip()
    return head or title


# ---- email-based merge --------------------------------------------------


_EMAIL_LOCAL_SEP_RE = re.compile(r"[._\-+]+")


def _email_local_fingerprint(email: str | None) -> str:
    """Turn ``alexandr.prozorov@cvut.cz`` → ``"alexandr prozorov"`` (token-sorted).

    Used as a soft identity signal: when an accumulator has an email but no
    corresponding DB row carries it yet, we check whether the email's local
    part matches a known person by name. False positives are possible
    (institutional aliases, generic locals like "info@") — but for ePIC
    style addresses ``first.last@institution`` the hit rate is high and
    the alternative is leaving Sasha and Alexandr as two rows forever.
    """
    if not email or "@" not in email:
        return ""
    local = email.split("@", 1)[0]
    cleaned = _EMAIL_LOCAL_SEP_RE.sub(" ", local)
    fingerprint = normalize_name(cleaned)
    # Single-token fingerprints ("info", "asmith") aren't discriminating
    # enough to merge across sources. Require ≥2 tokens.
    if len(fingerprint.split()) < 2:
        return ""
    return fingerprint


def _merge_by_email(db: Session, upserts: dict[str, _PersonAccumulator]) -> int:
    """Collapse accumulators that share an email — with an existing row OR
    each other — using both the email itself and an email-local-part
    fingerprint as identity signals. Returns the number of merges performed.
    """
    if not upserts:
        return 0
    merges = 0

    def _redirect_acc(acc: _PersonAccumulator, target_normalized: str, target_canonical: str) -> None:
        nonlocal merges
        if acc.normalized == target_normalized:
            return
        upserts.pop(acc.normalized, None)
        target = upserts.get(target_normalized)
        if target is None:
            target = _PersonAccumulator(target_normalized, target_canonical)
            upserts[target_normalized] = target
        target.absorb(acc)
        merges += 1

    # 1) An accumulator's email matches a row already in the DB. Re-key the
    # accumulator under that row's normalized_name so the flush below merges
    # them. Without this step we'd insert a duplicate row.
    emails = sorted({a.email for a in upserts.values() if a.email})
    rows_by_email: dict[str, Person] = {}
    if emails:
        existing_rows = (
            db.execute(select(Person).where(Person.email.in_(emails)))
            .scalars()
            .all()
        )
        rows_by_email = {p.email.lower(): p for p in existing_rows if p.email}
        for acc in list(upserts.values()):
            if not acc.email:
                continue
            target_row = rows_by_email.get(acc.email)
            if target_row is None:
                continue
            _redirect_acc(acc, target_row.normalized_name, target_row.canonical_name)

    # 2) Email-local fingerprint vs DB row. Catches the case where the
    # formal row exists (e.g. "Prozorov, Alexandr" from Indico) but has
    # no stored email yet, while the new website-derived accumulator
    # ("Sasha Prozorov") carries the email.
    fingerprints: dict[str, list[_PersonAccumulator]] = {}
    for acc in list(upserts.values()):
        fp = _email_local_fingerprint(acc.email)
        if fp:
            fingerprints.setdefault(fp, []).append(acc)
    if fingerprints:
        existing_by_norm = (
            db.execute(select(Person).where(Person.normalized_name.in_(list(fingerprints.keys()))))
            .scalars()
            .all()
        )
        for row in existing_by_norm:
            for acc in fingerprints.get(row.normalized_name, []):
                # Skip if the accumulator was already redirected away.
                if acc.normalized not in upserts:
                    continue
                _redirect_acc(acc, row.normalized_name, row.canonical_name)

    # 3) In-batch merge by exact email. Two new accumulators sharing the
    # same email — pick the one with more activity as the keeper.
    seen_email: dict[str, _PersonAccumulator] = {}
    for acc in list(upserts.values()):
        if not acc.email:
            continue
        prev = seen_email.get(acc.email)
        if prev is None:
            seen_email[acc.email] = acc
            continue
        if len(acc.activity) > len(prev.activity):
            keeper, victim = acc, prev
            seen_email[acc.email] = acc
        else:
            keeper, victim = prev, acc
        _redirect_acc(victim, keeper.normalized, keeper.canonical_name)

    # 4) In-batch merge by fingerprint vs another accumulator's normalized
    # name. ("alexandr.prozorov" matches the "Prozorov, Alexandr" Indico
    # accumulator that has no email of its own.)
    by_norm = {a.normalized: a for a in upserts.values()}
    for acc in list(upserts.values()):
        if acc.normalized not in upserts:
            continue
        fp = _email_local_fingerprint(acc.email)
        if not fp or fp == acc.normalized:
            continue
        target = by_norm.get(fp)
        if target is None or target is acc or target.normalized not in upserts:
            continue
        _redirect_acc(acc, target.normalized, target.canonical_name)

    return merges


# ---- flush ---------------------------------------------------------------


def _flush(db: Session, upserts: dict[str, _PersonAccumulator]) -> int:
    """Upsert each accumulator to the persons table, merging into existing rows."""
    if not upserts:
        return 0
    existing_rows = db.execute(
        select(Person).where(Person.normalized_name.in_(list(upserts.keys())))
    ).scalars().all()
    by_norm: dict[str, Person] = {p.normalized_name: p for p in existing_rows}
    upserted = 0
    for normalized, acc in upserts.items():
        person = by_norm.get(normalized)
        if person is None:
            person = Person(
                canonical_name=acc.canonical_name,
                normalized_name=normalized,
                aliases=acc.aliases,
                affiliations=acc.affiliations,
                email=acc.email,
                orcid=acc.orcid,
                working_groups=acc.working_groups,
                activity=_dedupe_and_sort_activity(acc.activity),
                activity_count=len(acc.activity),
                last_seen=acc.last_seen,
            )
            db.add(person)
        else:
            # Prefer the longer canonical_name we just observed; demote the
            # old one to an alias.
            if len(acc.canonical_name) > len(person.canonical_name):
                person.aliases = merge_unique(person.aliases, [person.canonical_name] + acc.aliases)
                person.canonical_name = acc.canonical_name
            else:
                person.aliases = merge_unique(person.aliases, acc.aliases + [acc.canonical_name])
            person.affiliations = merge_unique(person.affiliations, acc.affiliations)
            person.working_groups = merge_unique(person.working_groups, acc.working_groups)
            if acc.email and not person.email:
                person.email = acc.email
            if acc.orcid and not person.orcid:
                person.orcid = acc.orcid
            merged_activity = _dedupe_and_sort_activity((person.activity or []) + acc.activity)
            person.activity = merged_activity
            person.activity_count = len(merged_activity)
            if acc.last_seen and (person.last_seen is None or acc.last_seen > person.last_seen):
                person.last_seen = acc.last_seen
        upserted += 1
    return upserted


def _dedupe_and_sort_activity(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate (kind, url, title) tuples; sort newest-first; cap length."""
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for entry in entries:
        key = (
            str(entry.get("kind") or ""),
            str(entry.get("url") or ""),
            str(entry.get("title") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    deduped.sort(key=lambda e: str(e.get("when") or ""), reverse=True)
    return deduped[:MAX_ACTIVITY_PER_PERSON]
