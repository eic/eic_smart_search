import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass(slots=True)
class DocumentPayload:
    source_name: str
    source_type: str
    external_id: str
    title: str
    url: str
    content: str
    visibility: str = "public"
    section_path: list[str] = field(default_factory=list)
    repo_path: str | None = None
    filetype: str | None = None
    last_updated: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class Connector(Protocol):
    source_name: str
    source_type: str
    visibility: str

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]: ...

