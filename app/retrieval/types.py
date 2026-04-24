from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class RetrievalFilters:
    scope: str = "public"
    source_names: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)
    section: str | None = None
    repo_path_prefix: str | None = None
    filetypes: list[str] = field(default_factory=list)

    @property
    def allowed_visibilities(self) -> list[str]:
        if self.scope == "public":
            return ["public"]
        # Internal/all scopes are explicit API choices. Public-only queries never
        # include these labels, which is the non-negotiable safety boundary.
        return ["public", "internal"]


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source_name: str
    source_type: str
    title: str
    url: str
    content: str
    score: float
    vector_score: float = 0.0
    lexical_score: float = 0.0
    visibility: str = "public"
    repo_path: str | None = None
    filetype: str | None = None
    section_path: list[str] = field(default_factory=list)
    heading_path: list[str] = field(default_factory=list)
    last_updated: datetime | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

