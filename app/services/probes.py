"""Screening of requests that are not questions.

Three kinds of traffic reach /query without a person behind them:

* package sandboxes that run every new PyPI release's console script with
  canned subcommands (``eic-ask init``, ``eic-ask login`` …) — one burst per
  release, each word an LLM call;
* GraphQL fuzzers posting ``query { __typename }`` shapes into any JSON body;
* symbol-only strings with no letters at all.

They are still retrieved and logged (bursts stay visible in the raw log) but
tagged ``client="probe"`` so the answer generator is skipped, nothing is
cached, and analytics keep them out of every ranked list.
"""

from __future__ import annotations

import re

MIN_CLI_WORDS = 3

_GRAPHQL = re.compile(r"__typename|\b(?:query|mutation|subscription)\b\s*[!@{(]", re.IGNORECASE)
_LETTER = re.compile(r"[^\W\d_]")


def classify_probe(query: str, client: str | None) -> str | None:
    """Return a short reason when the request looks automated, else None."""
    text = query.strip()
    if not _LETTER.search(text):
        return "no-letters"
    if _GRAPHQL.search(text):
        return "graphql"
    if client == "cli" and len(text.split()) < MIN_CLI_WORDS:
        return "short-cli"
    return None
