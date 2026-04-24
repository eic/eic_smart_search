from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.entities import Chunk
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParentExpansionConfig:
    enabled: bool
    filetypes: frozenset[str]
    heading_depth: int
    max_tokens: int

    @classmethod
    def from_settings(cls, settings: Settings) -> ParentExpansionConfig:
        raw = getattr(settings, "PARENT_EXPANSION_FILETYPES", ["md", "markdown"])
        if isinstance(raw, str):
            raw = [raw]
        normalized = frozenset(item.lower().lstrip(".") for item in raw if item)
        return cls(
            enabled=bool(getattr(settings, "PARENT_EXPANSION_ENABLED", False)),
            filetypes=normalized,
            heading_depth=max(1, int(getattr(settings, "PARENT_EXPANSION_HEADING_DEPTH", 2))),
            max_tokens=max(100, int(getattr(settings, "PARENT_EXPANSION_MAX_TOKENS", 1200))),
        )


def expand_parents(
    db: Session,
    chunks: list[RetrievedChunk],
    config: ParentExpansionConfig,
) -> tuple[list[RetrievedChunk], dict]:
    if not config.enabled or not chunks:
        return chunks, {"enabled": config.enabled, "expanded": 0}

    output: list[RetrievedChunk] = []
    seen_groups: set[tuple[str, tuple[str, ...]]] = set()
    expanded = 0
    merged_duplicates = 0

    for chunk in chunks:
        normalized_ft = (chunk.filetype or "").lower().lstrip(".")
        if normalized_ft not in config.filetypes:
            output.append(chunk)
            continue

        parent_key = tuple(chunk.heading_path[: config.heading_depth])
        group_key = (chunk.document_id, parent_key)
        if group_key in seen_groups:
            merged_duplicates += 1
            continue
        seen_groups.add(group_key)

        stitched, sibling_count = _stitch_parent(db, chunk.document_id, parent_key, config)
        if stitched:
            chunk.content = stitched
            chunk.metadata = {
                **chunk.metadata,
                "parent_expansion": {
                    "parent_key": list(parent_key),
                    "merged_siblings": sibling_count,
                    "filetype": normalized_ft,
                },
            }
            expanded += 1
        output.append(chunk)

    return output, {
        "enabled": True,
        "expanded": expanded,
        "merged_duplicates": merged_duplicates,
        "filetypes": sorted(config.filetypes),
        "heading_depth": config.heading_depth,
        "max_tokens": config.max_tokens,
    }


def _stitch_parent(
    db: Session,
    document_id: str,
    parent_key: tuple[str, ...],
    config: ParentExpansionConfig,
) -> tuple[str, int]:
    rows = db.scalars(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    ).all()
    if not rows:
        return "", 0

    depth = config.heading_depth
    matching = [row for row in rows if _heading_prefix_matches(row.heading_path, parent_key, depth)]
    if not matching:
        matching = rows

    pieces: list[str] = []
    token_total = 0
    for row in matching:
        row_tokens = int(row.token_count or 0) or max(1, len(row.content.split()))
        if token_total and token_total + row_tokens > config.max_tokens:
            break
        pieces.append(row.content.strip())
        token_total += row_tokens

    if not pieces:
        return "", 0
    return "\n\n".join(pieces), len(pieces)


def _heading_prefix_matches(
    heading_path: list[str] | None,
    parent_key: tuple[str, ...],
    depth: int,
) -> bool:
    if not parent_key:
        return True
    if not heading_path:
        return False
    return tuple(heading_path[:depth]) == parent_key
