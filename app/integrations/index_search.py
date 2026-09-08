"""`search_index` LLM tool — agentic follow-up retrieval mid-answer.

The generator's first round already sees the top chunks for the user's
query. This tool lets the model run *additional* searches with refined
phrasing when the initial context doesn't cover part of the question
(multi-hop: "compare dRICH and pfRICH radiators" → search each).

Same session pattern as PeopleLookupClient: tool calls happen mid-request,
so each lookup opens a short-lived SessionLocal.
"""
from __future__ import annotations

import logging
from typing import Any

from app.db.session import SessionLocal
from app.retrieval.types import RetrievalFilters

logger = logging.getLogger(__name__)


class IndexSearchClient:
    def __init__(self, retriever, *, top_k: int = 6) -> None:
        self._retriever = retriever
        self.top_k = max(1, min(int(top_k), 12))

    def search(self, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        k = max(1, min(int(top_k or self.top_k), 12))
        filters = RetrievalFilters(scope="public")
        try:
            with SessionLocal() as session:
                # rerank=False: the tool call sits inside answer latency and
                # the model reads snippets, not ranks — RRF order is enough.
                chunks, _ = self._retriever.search(session, query, filters, k, rerank=False)
        except Exception as exc:
            logger.warning("index_search_tool_failed", extra={"query": query[:120], "error": str(exc)[:200]})
            return []
        return [
            {
                "title": chunk.title,
                "url": chunk.url,
                "source": chunk.source_name,
                "content": chunk.content[:1200],
            }
            for chunk in chunks
        ]


SEARCH_INDEX_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_index",
        "description": (
            "Run an additional search over the indexed EIC/ePIC knowledge base "
            "(documentation, wikis, GitHub, papers, meeting slides). Call this "
            "when the pre-fetched sources do not cover part of the question — "
            "e.g. a comparison needs facts about a second detector, or a "
            "follow-up fact is missing. Use a short, specific query phrased "
            "like a document heading ('pfRICH radiator design'), not a full "
            "sentence. Do NOT repeat the user's original query verbatim — the "
            "pre-fetched sources already are its results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Refined search query for the missing information.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Results to return (default 6, max 12).",
                    "default": 6,
                },
            },
            "required": ["query"],
        },
    },
}


__all__ = ["IndexSearchClient", "SEARCH_INDEX_TOOL_SCHEMA"]
