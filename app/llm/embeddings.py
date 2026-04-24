import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

from app.core.config import Settings


TOKEN_RE = re.compile(r"[A-Za-z0-9_./+-]+")


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbeddingProvider:
    """Zero-credential deterministic embeddings for local development.

    This is intentionally simple. It gives Qdrant a stable vector signal for
    smoke tests and cheap deployments, while the provider interface lets teams
    swap in a higher quality embedding service without changing retrieval code.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformersEmbeddingProvider:
    """Local sentence-transformers adapter.

    The default model is sentence-transformers/all-MiniLM-L6-v2, which produces
    384-dimensional normalized sentence embeddings and runs comfortably on CPU.
    """

    def __init__(self, model_name: str, dimension: int, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "sentence-transformers is required for EMBEDDING_PROVIDER=sentence_transformers. "
                "Install backend/requirements.txt or rebuild the Docker image."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self._dimension = dimension

        actual_dimension = int(self.model.get_sentence_embedding_dimension() or 0)
        if actual_dimension and actual_dimension != dimension:
            raise ValueError(
                f"Embedding model {model_name} returns {actual_dimension} dimensions, "
                f"but VECTOR_SIZE is {dimension}."
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class HttpEmbeddingProvider:
    """Generic HTTP embedding adapter.

    Expected request:
        {"texts": ["..."], "model": "optional-model"}

    Expected response:
        {"embeddings": [[...], [...]]}
    """

    def __init__(self, url: str, dimension: int, model: str | None = None, timeout: float = 30.0) -> None:
        self.url = url
        self.model = model
        self.timeout = timeout
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        payload: dict[str, object] = {"texts": list(texts)}
        if self.model:
            payload["model"] = self.model
        response = httpx.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Embedding HTTP provider did not return an embeddings list")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class TeiEmbeddingProvider:
    """Hugging Face Text Embeddings Inference adapter.

    TEI expects {"inputs": [...]} at /embed and returns the raw embeddings list.
    Qwen3 embedding models benefit from an instruction on query text only.
    """

    def __init__(
        self,
        url: str,
        dimension: int,
        query_instruction: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.query_instruction = query_instruction
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_inputs(list(texts))

    def embed_query(self, text: str) -> list[float]:
        if self.query_instruction:
            text = f"Instruct: {self.query_instruction}\nQuery: {text}"
        return self._embed_inputs([text])[0]

    def _embed_inputs(self, inputs: list[str]) -> list[list[float]]:
        response = httpx.post(f"{self.url}/embed", json={"inputs": inputs}, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("TEI embedding provider did not return an embeddings list")
        return data


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.EMBEDDING_PROVIDER == "sentence_transformers":
        return SentenceTransformersEmbeddingProvider(
            model_name=settings.EMBEDDING_MODEL,
            dimension=settings.VECTOR_SIZE,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    if settings.EMBEDDING_PROVIDER == "http":
        if not settings.EMBEDDING_HTTP_URL:
            raise ValueError("EMBEDDING_HTTP_URL is required when EMBEDDING_PROVIDER=http")
        return HttpEmbeddingProvider(
            url=str(settings.EMBEDDING_HTTP_URL),
            dimension=settings.VECTOR_SIZE,
            model=settings.EMBEDDING_HTTP_MODEL,
        )
    if settings.EMBEDDING_PROVIDER == "tei":
        if not settings.EMBEDDING_HTTP_URL:
            raise ValueError("EMBEDDING_HTTP_URL is required when EMBEDDING_PROVIDER=tei")
        return TeiEmbeddingProvider(
            url=str(settings.EMBEDDING_HTTP_URL),
            dimension=settings.VECTOR_SIZE,
            query_instruction=settings.EMBEDDING_QUERY_INSTRUCTION,
        )
    return HashingEmbeddingProvider(dimension=settings.VECTOR_SIZE)
