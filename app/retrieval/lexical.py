from datetime import datetime
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.retrieval.types import RetrievedChunk, RetrievalFilters


class PostgresLexicalRetriever:
    def search(self, db: Session, query: str, filters: RetrievalFilters, limit: int) -> list[RetrievedChunk]:
        sql = """
        SELECT
            c.id AS chunk_id,
            c.document_id,
            d.source_name,
            d.source_type,
            d.title,
            d.url,
            c.content,
            c.visibility,
            d.repo_path,
            d.filetype,
            d.section_path,
            c.heading_path,
            d.last_updated,
            c.content_hash,
            d.doc_metadata,
            c.chunk_metadata,
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', :query)) AS lexical_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE
            c.visibility IN :visibilities
            AND to_tsvector('english', c.content) @@ plainto_tsquery('english', :query)
        """
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "visibilities": filters.allowed_visibilities,
        }
        bindparams = [bindparam("visibilities", expanding=True)]
        if filters.source_names:
            sql += " AND d.source_name IN :source_names"
            params["source_names"] = filters.source_names
            bindparams.append(bindparam("source_names", expanding=True))
        if filters.source_types:
            sql += " AND d.source_type IN :source_types"
            params["source_types"] = filters.source_types
            bindparams.append(bindparam("source_types", expanding=True))
        if filters.section:
            sql += " AND lower(d.section_path::text) LIKE :section"
            params["section"] = f"%{filters.section.lower()}%"
        if filters.repo_path_prefix:
            sql += " AND d.repo_path LIKE :repo_path_prefix"
            params["repo_path_prefix"] = f"{filters.repo_path_prefix}%"
        if filters.filetypes:
            sql += " AND d.filetype IN :filetypes"
            params["filetypes"] = filters.filetypes
            bindparams.append(bindparam("filetypes", expanding=True))
        sql += " ORDER BY lexical_score DESC LIMIT :limit"

        statement = text(sql).bindparams(*bindparams)
        rows = db.execute(statement, params).mappings().all()
        return [self._row_to_chunk(row) for row in rows]

    def fetch_by_chunk_ids(self, db: Session, chunk_scores: dict[str, tuple[float, float]]) -> list[RetrievedChunk]:
        if not chunk_scores:
            return []
        sql = text(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.source_name,
                d.source_type,
                d.title,
                d.url,
                c.content,
                c.visibility,
                d.repo_path,
                d.filetype,
                d.section_path,
                c.heading_path,
                d.last_updated,
                c.content_hash,
                d.doc_metadata,
                c.chunk_metadata
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN :chunk_ids
            """
        ).bindparams(bindparam("chunk_ids", expanding=True))
        rows = db.execute(sql, {"chunk_ids": list(chunk_scores)}).mappings().all()
        chunks = []
        for row in rows:
            vector_score, lexical_score = chunk_scores[row["chunk_id"]]
            chunk = self._row_to_chunk(row)
            chunk.vector_score = vector_score
            chunk.lexical_score = lexical_score
            chunks.append(chunk)
        return chunks

    def _row_to_chunk(self, row: Any) -> RetrievedChunk:
        metadata = dict(row.get("doc_metadata") or {})
        metadata.update(row.get("chunk_metadata") or {})
        lexical_score = float(row.get("lexical_score") or 0.0)
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            source_name=row["source_name"],
            source_type=row["source_type"],
            title=row["title"],
            url=row["url"],
            content=row["content"],
            score=lexical_score,
            lexical_score=lexical_score,
            visibility=row["visibility"],
            repo_path=row["repo_path"],
            filetype=row["filetype"],
            section_path=list(row["section_path"] or []),
            heading_path=list(row["heading_path"] or []),
            last_updated=row["last_updated"] if isinstance(row["last_updated"], datetime) else None,
            content_hash=row["content_hash"],
            metadata=metadata,
        )

