"""LLM tool surface for the people graph.

The actual data lives in PostgreSQL (``persons`` table) and is built by
``app.people.builder``. This module is just the OpenAI tool schema and a
small client object that the answer generator can hold without dragging
SQLAlchemy types into ``app.llm.generation``.

The client is a thin shim because every lookup needs a fresh SQLAlchemy
``Session`` (the LLM tool may be called mid-request, well after the
request-bound session would have closed). We open a short-lived session
from the global ``SessionLocal`` for each tool call.
"""
from __future__ import annotations

import logging
from typing import Any

from app.db.session import SessionLocal
from app.people.service import lookup, to_card

logger = logging.getLogger(__name__)


class PeopleLookupClient:
    """Adapter the answer generator calls. Takes no constructor args by
    design — wiring stays simple in ``build_answer_generator``.
    """

    def lookup(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        limit = max(1, min(int(limit or 5), 10))
        with SessionLocal() as session:
            try:
                people = lookup(session, query, limit=limit)
            except Exception as exc:
                logger.warning("people_lookup_failed", extra={"query": query, "error": str(exc)[:200]})
                return []
            return [to_card(p) for p in people]


LOOKUP_PERSON_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup_person",
        "description": (
            "Look up a person mentioned in the EIC / ePIC ecosystem and return a "
            "card with their affiliations, working-group involvement, recent papers, "
            "talks, and meeting roles. Call this whenever the user asks 'who is X', "
            "'who works on X', 'expert on X', 'contact for X', 'who chairs X', or "
            "any question that hinges on identifying or describing a specific "
            "collaborator. Pass either the person's name (preferred) OR a topic "
            "keyword to find experts for. Do NOT call this for general physics or "
            "software questions where the answer doesn't depend on a specific "
            "person."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Full or partial name of the person to look up. Surnames "
                        "and first names both work — the matcher does fuzzy / "
                        "diacritic-insensitive lookup (e.g. 'Muller' matches 'Müller')."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max candidates to return when the name is ambiguous. Default 5, max 10.",
                    "default": 5,
                },
            },
            "required": ["name"],
        },
    },
}


__all__ = ["PeopleLookupClient", "LOOKUP_PERSON_TOOL_SCHEMA"]
