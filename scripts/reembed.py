"""Re-embed every chunk into the currently configured Qdrant collection.

Use after switching EMBEDDING_MODEL / VECTOR_SIZE / QDRANT_COLLECTION:
no re-crawling — reads chunks straight from Postgres and upserts vectors
with the same payload shape the orchestrator writes.

Run inside the api container:
    docker compose exec api python scripts/reembed.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/app")

from sqlalchemy import func, select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models.entities import Chunk, Document, Source  # noqa: E402
from app.services.factory import build_vector_store  # noqa: E402

BATCH = 256


def main() -> None:
    settings = get_settings()
    store = build_vector_store(settings)
    store.ensure_collection()
    print(f"collection={settings.QDRANT_COLLECTION} model={settings.EMBEDDING_MODEL} dim={settings.VECTOR_SIZE}")

    with SessionLocal() as db:
        total = db.scalar(select(func.count()).select_from(Chunk))
        print(f"chunks to embed: {total}")
        done = 0
        started = time.time()
        offset = 0
        while True:
            rows = db.execute(
                select(Chunk, Document, Source)
                .join(Document, Document.id == Chunk.document_id)
                .join(Source, Source.id == Chunk.source_id)
                .order_by(Chunk.id)
                .offset(offset)
                .limit(BATCH)
            ).all()
            if not rows:
                break
            offset += len(rows)
            items = []
            for chunk, document, source in rows:
                payload = {
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "source_id": source.id,
                    "source_name": document.source_name,
                    "source_type": document.source_type,
                    "title": document.title,
                    "url": document.url,
                    "visibility": chunk.visibility,
                    "section_path": document.section_path,
                    "section_text": " / ".join((document.section_path or []) + (chunk.heading_path or [])),
                    "heading_path": chunk.heading_path,
                    "repo_path": document.repo_path,
                    "filetype": document.filetype,
                    "last_updated": document.last_updated.isoformat() if document.last_updated else None,
                    "content_hash": chunk.content_hash,
                }
                items.append((chunk.id, chunk.content, payload))
            store.upsert_chunks(items)
            done += len(items)
            rate = done / max(time.time() - started, 1e-9)
            print(f"\r{done}/{total}  ({rate:.0f} chunks/s)", end="", flush=True)
    print(f"\ndone in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
