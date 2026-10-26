import logging
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


def _warm_once(service) -> None:
    """Same cheap warm-up as build_query_service, plus one rerank pair."""
    from app.retrieval.rerank import NullReranker
    from app.retrieval.types import RetrievedChunk

    retriever = service.retriever
    retriever.vector_store.embeddings.embed_query("warm up")
    if not isinstance(retriever.reranker, NullReranker):
        chunk = RetrievedChunk(
            chunk_id="warm", document_id="warm", source_name="warm", source_type="warm",
            title="warm", url="", content="warm up", score=0.0,
        )
        retriever.reranker.rerank("warm up", [chunk])
    # Indico tool: refresh the event windows the tool queries so the first
    # "when is the next ... meeting" of the day never waits on indico.bnl.gov.
    indico = getattr(getattr(service, "answer_generator", None), "_indico", None)
    if indico is not None and hasattr(indico, "warm"):
        indico.warm()


def _keepalive_loop(settings, interval_s: float) -> None:
    from app.api.routes import query_service_dep

    while True:  # daemon thread: dies with the process, never blocks shutdown
        time.sleep(interval_s)
        try:
            _warm_once(query_service_dep(settings))
        except Exception as exc:  # keepalive must never crash or spam logs
            logger.debug("warm_keepalive_failed", extra={"error": str(exc)})


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)

    # Fail closed: a non-local deployment left on ADMIN_AUTH_MODE=none would
    # expose every mutating endpoint to the internet. Refuse to boot.
    if settings.APP_ENV != "local" and settings.ADMIN_AUTH_MODE == "none":
        raise RuntimeError(
            "ADMIN_AUTH_MODE=none is not allowed when APP_ENV != local. "
            "Set ADMIN_AUTH_MODE to token / github_org / github_sso."
        )

    app = FastAPI(title=settings.APP_NAME, version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix=settings.API_PREFIX)

    @app.on_event("startup")
    def _cap_threadpool() -> None:
        # Sync route handlers run in anyio's threadpool (default 40). Cap it
        # below the DB pool ceiling (20+10) so a burst can never queue on
        # connection checkout — it fails fast at the handler instead.
        try:
            import anyio.to_thread

            limiter = anyio.to_thread.current_default_thread_limiter()
            limiter.total_tokens = 24
        except Exception:  # pragma: no cover
            pass

    @app.on_event("startup")
    def _warm_query_service() -> None:
        # Build the QueryService singleton (embedding model load + CUDA
        # warm-up, ~10 s for bge-m3) at process start instead of on the
        # first user's query.
        from app.api.routes import query_service_dep

        try:
            query_service_dep(settings)
        except Exception:  # pragma: no cover — warm-up must never block boot
            pass

    @app.on_event("startup")
    def _start_warm_keepalive() -> None:
        # Idle keepalive (WARM_INTERVAL_S, 0 disables): first queries after
        # hours of idle took 15-40 s on a cold embedding/GPU path.
        interval_s = float(getattr(settings, "WARM_INTERVAL_S", 0) or 0)
        if interval_s > 0:
            threading.Thread(
                target=_keepalive_loop, args=(settings, interval_s),
                name="warm-keepalive", daemon=True,
            ).start()

    return app


app = create_app()

