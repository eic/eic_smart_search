from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging


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

    return app


app = create_app()

