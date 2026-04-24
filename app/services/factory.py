from app.core.config import Settings, get_settings
from app.ingestion.orchestrator import IngestionOrchestrator
from app.llm.embeddings import build_embedding_provider
from app.llm.generation import build_answer_generator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import PostgresLexicalRetriever
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.query_rewrite import build_query_rewriter
from app.retrieval.rerank import build_reranker
from app.services.query import QueryService
from app.services.query_cache import get_query_cache


def build_vector_store(settings: Settings | None = None) -> QdrantVectorStore:
    settings = settings or get_settings()
    embeddings = build_embedding_provider(settings)
    return QdrantVectorStore(settings, embeddings)


def build_query_service(settings: Settings | None = None) -> QueryService:
    settings = settings or get_settings()
    vector_store = build_vector_store(settings)
    retriever = HybridRetriever(
        vector_store,
        lexical=PostgresLexicalRetriever(),
        reranker=build_reranker(settings),
    )
    cache = (
        get_query_cache(max_size=settings.QUERY_CACHE_MAX_SIZE, ttl_s=settings.QUERY_CACHE_TTL_S)
        if getattr(settings, "QUERY_CACHE_ENABLED", True)
        else None
    )
    return QueryService(
        settings,
        retriever,
        build_answer_generator(settings),
        query_rewriter=build_query_rewriter(settings),
        query_cache=cache,
    )


def build_ingestion_orchestrator(settings: Settings | None = None) -> IngestionOrchestrator:
    settings = settings or get_settings()
    return IngestionOrchestrator(settings, build_vector_store(settings))

