from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import Settings
from app.llm.embeddings import EmbeddingProvider
from app.retrieval.types import RetrievalFilters

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(self, settings: Settings, embeddings: EmbeddingProvider) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.collection_name = settings.QDRANT_COLLECTION
        self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    def ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            self._validate_collection_dimension()
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.embeddings.dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info("created_qdrant_collection", extra={"collection": self.collection_name})

    def _validate_collection_dimension(self) -> None:
        collection = self.client.get_collection(self.collection_name)
        vectors_config = collection.config.params.vectors
        actual_size = getattr(vectors_config, "size", None)
        if actual_size and actual_size != self.embeddings.dimension:
            raise ValueError(
                f"Qdrant collection {self.collection_name} has vector size {actual_size}, "
                f"but the configured embedding provider uses {self.embeddings.dimension}. "
                "Recreate the collection or reset the qdrant_data volume, then reingest."
            )

    def upsert_chunks(self, chunks: Sequence[tuple[str, str, dict[str, Any]]]) -> None:
        if not chunks:
            return
        self.ensure_collection()
        texts = [content for _, content, _ in chunks]
        vectors = self.embeddings.embed_texts(texts)
        points = [
            qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
            for (point_id, _, payload), vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, filters: RetrievalFilters, limit: int) -> list[dict[str, Any]]:
        self.ensure_collection()
        query_vector = self.embeddings.embed_query(query)
        qdrant_filter = self._build_filter(filters)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "chunk_id": result.payload.get("chunk_id") if result.payload else None,
                "score": float(result.score),
                "payload": result.payload or {},
            }
            for result in results
            if result.payload and result.payload.get("chunk_id")
        ]

    def delete_by_document_ids(self, document_ids: Sequence[str]) -> None:
        if not document_ids:
            return
        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchAny(any=list(document_ids)),
                        )
                    ]
                )
            ),
        )

    def _build_filter(self, filters: RetrievalFilters) -> qmodels.Filter:
        must: list[Any] = [
            qmodels.FieldCondition(
                key="visibility",
                match=qmodels.MatchAny(any=filters.allowed_visibilities),
            )
        ]
        if filters.source_names:
            must.append(qmodels.FieldCondition(key="source_name", match=qmodels.MatchAny(any=filters.source_names)))
        if filters.source_types:
            must.append(qmodels.FieldCondition(key="source_type", match=qmodels.MatchAny(any=filters.source_types)))
        if filters.section:
            must.append(qmodels.FieldCondition(key="section_text", match=qmodels.MatchText(text=filters.section)))
        if filters.repo_path_prefix:
            must.append(
                qmodels.FieldCondition(
                    key="repo_path",
                    match=qmodels.MatchText(text=filters.repo_path_prefix),
                )
            )
        if filters.filetypes:
            must.append(qmodels.FieldCondition(key="filetype", match=qmodels.MatchAny(any=filters.filetypes)))
        return qmodels.Filter(must=must)
