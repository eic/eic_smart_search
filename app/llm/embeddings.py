import logging
from collections.abc import Sequence
from typing import Protocol


from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...



class SentenceTransformersEmbeddingProvider:
    """Local sentence-transformers adapter.

    The default model is sentence-transformers/all-MiniLM-L6-v2, which produces
    384-dimensional normalized sentence embeddings and runs comfortably on CPU.
    """

    def __init__(
        self,
        model_name: str,
        dimension: int,
        batch_size: int = 64,
        device: str = "auto",
        max_seq_length: int | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "sentence-transformers is required for EMBEDDING_PROVIDER=sentence_transformers. "
                "Install backend/requirements.txt or rebuild the Docker image."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Explicit device avoids the sentence-transformers>=5 meta-tensor load bug.
        # CUDA init is flaky under WSL2 (intermittent "Cannot copy out of meta
        # tensor" during model load) — fall back to CPU instead of crashing.
        model_kwargs = {}
        if device == "cuda":
            # fp16 halves VRAM (bge-m3: 2.3 GB -> 1.2 GB) so the embedding
            # model and the cross-encoder reranker fit a 6 GB GPU together;
            # embedding quality impact is negligible.
            import torch

            model_kwargs = {"torch_dtype": torch.float16}
        try:
            self.model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)
        except (NotImplementedError, RuntimeError, TypeError):
            if device == "cpu":
                raise
            logger.warning(
                "Failed to load %s on %s; falling back to CPU.", model_name, device
            )
            self.model = SentenceTransformer(model_name, device="cpu")
        if max_seq_length:
            self.model.max_seq_length = int(max_seq_length)
        self._dimension = dimension
        # Tiny query-embedding cache: acronym expansion embeds two variants
        # per query, and popular queries repeat — an encode is ~10-30 ms on
        # GPU, worse on CPU, for a byte-identical result.
        self._query_cache: dict[str, list[float]] = {}
        self._query_cache_max = 256

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
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        vector = self.embed_texts([text])[0]
        if len(self._query_cache) >= self._query_cache_max:
            # FIFO eviction is fine at this size; no LRU bookkeeping needed.
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[text] = vector
        return vector




def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    return SentenceTransformersEmbeddingProvider(
        model_name=settings.EMBEDDING_MODEL,
        dimension=settings.VECTOR_SIZE,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        device=settings.EMBEDDING_DEVICE,
        max_seq_length=getattr(settings, "EMBEDDING_MAX_SEQ_LENGTH", None),
    )
