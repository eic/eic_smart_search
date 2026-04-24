import hashlib
import re
from dataclasses import dataclass, field

from app.ingestion.base import DocumentPayload


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
TOKEN_RE = re.compile(r"\S+")


@dataclass(slots=True)
class ChunkPayload:
    chunk_index: int
    content: str
    content_hash: str
    heading_path: list[str] = field(default_factory=list)
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


class HeadingAwareChunker:
    def __init__(self, target_tokens: int = 360, overlap_tokens: int = 60) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, document: DocumentPayload) -> list[ChunkPayload]:
        blocks = self._split_blocks(document.content)
        chunks: list[ChunkPayload] = []
        buffer: list[str] = []
        current_heading_path = list(document.section_path)
        buffer_heading_path = list(current_heading_path)

        def flush() -> None:
            nonlocal buffer, buffer_heading_path
            text = "\n\n".join(part.strip() for part in buffer if part.strip()).strip()
            if not text:
                buffer = []
                return
            chunks.append(
                ChunkPayload(
                    chunk_index=len(chunks),
                    content=text,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    heading_path=list(buffer_heading_path),
                    token_count=len(TOKEN_RE.findall(text)),
                    metadata={"source_document_hash": document.content_hash},
                )
            )
            if self.overlap_tokens > 0:
                words = text.split()
                overlap = " ".join(words[-self.overlap_tokens :])
                buffer = [overlap] if overlap else []
                buffer_heading_path = list(current_heading_path)
            else:
                buffer = []

        for block in blocks:
            heading_match = HEADING_RE.match(block)
            if heading_match:
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                base = list(document.section_path)
                relative = current_heading_path[len(base) :]
                relative = relative[: max(level - 1, 0)]
                current_heading_path = base + relative + [heading]
                if buffer and self._token_count(buffer) >= self.target_tokens // 2:
                    flush()
                buffer_heading_path = list(current_heading_path)
                buffer.append(block)
                continue

            if not buffer:
                buffer_heading_path = list(current_heading_path)
            buffer.append(block)
            if self._token_count(buffer) >= self.target_tokens:
                flush()

        flush()
        return [chunk for chunk in chunks if chunk.token_count > 0]

    def _split_blocks(self, text: str) -> list[str]:
        normalized = re.sub(r"\r\n?", "\n", text)
        return [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]

    def _token_count(self, blocks: list[str]) -> int:
        return sum(len(TOKEN_RE.findall(block)) for block in blocks)

