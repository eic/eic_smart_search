"""Name normalization + parsing helpers for the people graph.

The aggregator pulls names from heterogeneous sources (Zenodo's
"Last, First (Affiliation)" strings, Indico's already-decomposed
first/last/affiliation objects, free-form text). These helpers
collapse all of that into one comparable shape so we can dedupe.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_PUNCT_TO_SPACE_RE = re.compile(r"[._/]+")
_KEEP_CHARS_RE = re.compile(r"[^a-z0-9\s\-']")
_WS_RE = re.compile(r"\s+")
_AFFIL_PAREN_RE = re.compile(r"\(([^)]+)\)\s*$")


def normalize_name(raw: str) -> str:
    """Diacritic-fold, lowercase, strip punctuation. Used as the dedup key.

    Examples:
        "Müller, Hans (BNL)" -> "muller hans"  (after parse_author_string)
        "O'Brien" -> "o'brien"
        "Jean-Luc"  -> "jean-luc"
    """
    if not raw:
        return ""
    nfkd = unicodedata.normalize("NFKD", raw)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    lowered = ascii_only.lower()
    spaced = _PUNCT_TO_SPACE_RE.sub(" ", lowered)
    cleaned = _KEEP_CHARS_RE.sub(" ", spaced)
    result = _WS_RE.sub(" ", cleaned).strip()
    # Punctuation-only inputs ("---", "...") must round-trip to "" so the
    # caller can treat them as "no name". Otherwise we'd seed garbage rows.
    if not any(c.isalnum() for c in result):
        return ""
    # Token-sort so "Markus Diefenthaler" and "Diefenthaler, Markus"
    # collapse to the same dedup key. Indico's `fullName` ships in both
    # orders depending on the conference template, and Zenodo always
    # uses "Last, First". The canonical_name stores the readable form;
    # this output is only used for matching.
    tokens = sorted(result.split())
    return " ".join(tokens)


@dataclass(slots=True)
class ParsedAuthor:
    full_name: str
    affiliation: str | None


def parse_author_string(raw: str) -> ParsedAuthor | None:
    """Split a Zenodo-style author string into (name, affiliation).

    Zenodo formats authors as either ``"Last, First"`` or, after our
    own ``_authors`` helper, ``"Last, First (Affiliation)"``. We accept
    either and also tolerate ``"First Last"`` for safety.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    affiliation: str | None = None
    match = _AFFIL_PAREN_RE.search(text)
    if match:
        affiliation = match.group(1).strip() or None
        text = text[: match.start()].strip()
    if not text:
        return None
    if "," in text:
        last, first = (part.strip() for part in text.split(",", 1))
        if first and last:
            full = f"{first} {last}"
        else:
            full = (first or last).strip()
    else:
        full = text
    full = _WS_RE.sub(" ", full).strip(" -'")
    if not full:
        return None
    return ParsedAuthor(full_name=full, affiliation=affiliation)


def join_indico_person(payload: dict) -> ParsedAuthor | None:
    """Turn an Indico ``chairs[i]`` / ``speakers[i]`` dict into a name.

    Indico nodes carry a pre-built ``fullName`` plus first/last fields;
    fall back through them in order, then to ``name``.
    """
    if not isinstance(payload, dict):
        return None
    candidates = [
        payload.get("fullName"),
        payload.get("name"),
        " ".join(filter(None, [payload.get("first_name"), payload.get("last_name")])).strip() or None,
    ]
    full = next((str(c).strip() for c in candidates if isinstance(c, str) and c.strip()), None)
    if not full:
        return None
    affil = payload.get("affiliation") or payload.get("affiliation_link") or None
    if isinstance(affil, str):
        affil = affil.strip() or None
    else:
        affil = None
    full = _WS_RE.sub(" ", full).strip(" -'")
    if not full:
        return None
    return ParsedAuthor(full_name=full, affiliation=affil)


def merge_unique(existing: list[str] | None, new_values: list[str]) -> list[str]:
    """Append `new_values` to `existing` skipping case-insensitive duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for value in (existing or []) + list(new_values):
        if not isinstance(value, str):
            continue
        v = value.strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out
