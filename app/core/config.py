from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = "EIC Smart Search Backend"
    APP_ENV: str = "local"
    LOG_LEVEL: str = "INFO"
    API_PREFIX: str = ""

    DATABASE_URL: str = "postgresql+psycopg://smartsearch:smartsearch@postgres:5432/smartsearch"
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "collaboration_knowledge_chunks"
    VECTOR_SIZE: int = 384

    CORS_ALLOW_ORIGINS: str | list[str] = Field(
        default_factory=lambda: [
            "http://localhost:4000",
            "http://127.0.0.1:4000",
            "https://eic.github.io",
        ]
    )

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64
    # Cap tokens per text at embed time. bge-m3 accepts 8192, but long-
    # sequence attention OOMs small GPUs; 1024 covers our chunk sizes.
    EMBEDDING_MAX_SEQ_LENGTH: int = 1024
    EMBEDDING_DEVICE: Literal["auto", "cpu", "cuda"] = "auto"
    # Idle keepalive: every WARM_INTERVAL_S seconds a background thread runs
    # one tiny embed (+ one rerank pair when a reranker is configured) so the
    # first query after hours of idle doesn't pay a cold GPU/model path.
    # 0 disables.
    WARM_INTERVAL_S: float = 600.0

    GENERATION_PROVIDER: Literal["extractive", "openai"] = "extractive"
    MIN_SUPPORT_SCORE: float = 0.16
    # Cross-encoder floor for the answer support gate: below it the service
    # abstains ("couldn't find enough support"). 0.0 = ms-marco "on-topic"
    # midpoint; lower (negative) to abstain less on vague-but-answerable
    # queries, raise to abstain more aggressively.
    MIN_RERANK_SCORE: float = 0.0
    # Confidence buckets for the query log, judged on the best raw cosine
    # similarity (vector_score) among the first 3 returned chunks: < LOW →
    # "low", < HIGH → "medium", else "high" (0 → "none"; answers built from
    # Indico tool results → "high").
    CONFIDENCE_VEC_LOW: float = 0.30
    CONFIDENCE_VEC_HIGH: float = 0.55
    # Number of chunks fetched and passed to the LLM when generate_answer=True.
    # The user-facing citation list is still capped at request.top_k. At query
    # time this is additionally clamped to RERANK_CANDIDATES when the
    # cross-encoder runs, so the LLM never sees unreranked tail chunks.
    LLM_CONTEXT_TOP_K: int = 20

    RERANK_PROVIDER: Literal["none", "cross_encoder"] = "none"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_CANDIDATES: int = 20
    RERANK_SCORE_WEIGHT: float = 0.7
    RERANK_MAX_PASSAGE_CHARS: int = 1500
    RERANK_TIMEOUT_S: float = 3.0

    QUERY_CACHE_ENABLED: bool = True
    QUERY_CACHE_MAX_SIZE: int = 500
    QUERY_CACHE_TTL_S: float = 432000.0

    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_GENERATION_MODEL: str = "gpt-5.4-nano"
    OPENAI_REQUEST_TIMEOUT_S: float = 30.0
    OPENAI_MAX_OUTPUT_TOKENS: int = 600
    OPENAI_TEMPERATURE: float | None = None
    # gpt-5.6-family models require "none" to combine function tools with
    # /v1/chat/completions; leave unset for models that pick their own.
    OPENAI_REASONING_EFFORT: str | None = None
    # Extra JSON merged into every chat-completions request body, for
    # OpenAI-compatible servers with their own knobs (vLLM's
    # chat_template_kwargs to switch a model's thinking mode, ...).
    OPENAI_EXTRA_BODY: dict[str, Any] | None = None
    # Disable for providers whose tool-round messages can't be reconstructed
    # by the streaming accumulator (Gemini): answers then arrive as one delta.
    OPENAI_STREAM_ENABLED: bool = True
    OPENAI_MAX_CONTEXT_CHARS: int = 1800
    OPENAI_INPUT_COST_PER_1M: float = 0.0
    OPENAI_OUTPUT_COST_PER_1M: float = 0.0

    # Alternative OpenAI-compatible providers, benchmarked as extra arms in
    # the configuration sweep's "generation" block when their key is set
    # (see app/eval/sweep.py). All have free daily quotas as of 2026-08.
    # To make one the production generator, point OPENAI_BASE_URL /
    # OPENAI_API_KEY / OPENAI_GENERATION_MODEL at it instead.
    # The *_MODEL fields accept a comma-separated list — one sweep arm each.
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-haiku-4-5,claude-sonnet-5,claude-opus-5"
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_MODEL: str = "deepseek-chat"
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile,llama-3.1-8b-instant"
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
    # National Research Platform (nrp.ai) managed open-weights LLMs — free
    # for non-commercial research; vLLM behind an OpenAI-compatible gateway.
    NRP_API_KEY: str | None = None
    NRP_MODEL: str = "qwen3,gpt-oss,deepseek-v4-flash,glm-5,gemma"

    EIC_SITE_URL: str = "https://eic.github.io/"
    EIC_SITE_MAX_PAGES: int = 500
    EIC_SITE_EXCLUDE_PREFIXES: str = "/epic/artifacts,/epic/build-,/epic/geoviewer,/epic/craterlake_views,/epic-prod/artifacts,/epic-prod/build-,/EDM4eic/"
    EIC_SITE_EXCLUDE_PATH_CONTAINS: str = "/aio/,/aio.html"  # Carpentry all-in-one pages duplicate every episode

    EICUG_SITE_URL: str = "https://www.eicug.org/"
    EICUG_SITE_MAX_PAGES: int = 300
    EICUG_SITE_EXCLUDE_PREFIXES: str = ""

    BNL_WIKI_URL: str = "https://wiki.bnl.gov/EPIC/index.php?title=Main_Page"
    BNL_WIKI_MAX_PAGES: int = 500
    BNL_WIKI_EXCLUDE_PREFIXES: str = (
        "/EPIC/index.php/Special:,/EPIC/index.php/Talk:,/EPIC/index.php/User:,"
        "/EPIC/index.php/User_talk:,/EPIC/index.php/File:,/EPIC/index.php/Category:,"
        "/EPIC/index.php/Help:,/EPIC/index.php/MediaWiki:,/EPIC/index.php/Template:,"
        "/EPIC/index.php/Template_talk:,/EPIC/skins/,/EPIC/resources/,/EPIC/extensions/,"
        "/EPIC/load.php,/EPIC/images/"
    )

    GITHUB_ORG: str = "eic"
    GITHUB_ORG_MAX_FILES: int = 1500
    GITHUB_ORG_MAX_REPOS: int = 210

    ZENODO_COMMUNITY: str = "epic"
    ZENODO_MAX_RECORDS: int = 200
    ZENODO_INCLUDE_PDF_TEXT: bool = True
    ZENODO_PDF_MAX_BYTES: int = 25 * 1024 * 1024
    ZENODO_PDF_MAX_CHARS: int = 120_000

    GITHUB_REPO_OWNER: str = "eic"
    GITHUB_REPO_NAME: str = "eic.github.io"
    GITHUB_REPO_REF: str = "master"
    GITHUB_TOKEN: str | None = None
    GITHUB_MAX_FILES: int = 600

    # PODIO data-model YAMLs (comma-separated raw URLs) rendered into one
    # clean document per datatype/component/link — the authoritative answer
    # surface for "which collection / member / relation" questions.
    DATAMODEL_YAML_URLS: str = ""

    # ePIC DD4hep constants (system IDs = ParticleID.getType() values).
    EPIC_DEFINITIONS_URL: str = ""

    INDICO_ENABLED: bool = False
    INDICO_CATEGORY_URL: str = "https://indico.bnl.gov/category/402/"
    INDICO_CACHE_TTL_S: float = 300.0
    INDICO_TIMEOUT_S: float = 10.0

    PEOPLE_LOOKUP_ENABLED: bool = False
    PEOPLE_REFRESH_INDICO_LOOKBACK_DAYS: int = 365
    PEOPLE_REFRESH_INDICO_LOOKAHEAD_DAYS: int = 30
    # Indico's contributions-detail export is much heavier than the
    # plain events form (often >1MB). Give the rebuild path its own
    # generous timeout so a slow connection doesn't quietly skip it.
    PEOPLE_REFRESH_INDICO_TIMEOUT_S: float = 60.0

    ZENODO_SEARCH_ENABLED: bool = False
    ZENODO_SEARCH_CACHE_TTL_S: float = 120.0
    ZENODO_SEARCH_TIMEOUT_S: float = 10.0

    # GitHub code search + file fetch as LLM tools, scoped to one org
    # (eic / star-bnl). Needs GITHUB_TOKEN for useful rate limits.
    CODE_SEARCH_ENABLED: bool = False
    CODE_SEARCH_ORG: str = ""

    # INSPIRE-HEP literature (peer-reviewed papers the Zenodo community lacks).
    INSPIRE_QUERY: str = 'collaboration "ePIC"'
    INSPIRE_MAX_RECORDS: int = 500
    INSPIRE_MAX_AUTHORS_PER_PAPER: int = 60

    # arXiv preprints (freshness source; metadata + abstract only).
    ARXIV_QUERY: str = 'abs:"electron-ion collider" OR abs:"ePIC detector"'
    ARXIV_MAX_RECORDS: int = 300

    # Indico events baked into the retrieval index (live tool stays for
    # time-relative questions). Materials = attached slide/minute PDFs.
    INDICO_INGEST_LOOKBACK_DAYS: int = 730
    INDICO_INGEST_LOOKAHEAD_DAYS: int = 60
    INDICO_INGEST_MATERIALS: bool = True
    # Plenary decks with embedded photos run >100 MB (the ePIC status
    # pptx is 155 MB) — cap protects memory, not bandwidth.
    INDICO_MATERIAL_MAX_BYTES: int = 200 * 1024 * 1024
    INDICO_MATERIAL_TIMEOUT_S: float = 600.0
    INDICO_MATERIAL_MAX_CHARS: int = 60_000
    INDICO_MATERIAL_MAX_PER_EVENT: int = 40

    # GitHub issue threads (needs GITHUB_TOKEN — 60 req/h unauthenticated).
    GITHUB_ISSUES_REPOS: str = "eic/eic-shell,eic/epic,eic/EICrecon"
    GITHUB_ISSUES_MAX_PER_REPO: int = 200

    # GitHub release notes (same repo list as issues).
    GITHUB_RELEASES_MAX_PER_REPO: int = 50

    # GitHub Discussions threads (GraphQL — GITHUB_TOKEN required).
    GITHUB_DISCUSSIONS_REPOS: str = "eic/eic-shell,eic/epic,eic/EICrecon"
    GITHUB_DISCUSSIONS_MAX_PER_REPO: int = 200

    # Prefer trafilatura for website main-content extraction (bs4 fallback).
    WEB_EXTRACT_TRAFILATURA: bool = True

    EPIC_INTERNAL_START_URL: str = "https://www.epic-eic.org/index-internal.html"
    EPIC_INTERNAL_MAX_PAGES: int = 100
    EPIC_INTERNAL_COOKIE: str | None = None
    EPIC_INTERNAL_AUTH_HEADER: str | None = None

    REQUEST_TIMEOUT_SECONDS: float = 20.0
    INGEST_USER_AGENT: str = "eic-smart-search/0.1 (+https://eic.github.io/)"

    # Auth for mutating endpoints (/ingest/run, /admin/reindex, /people/refresh,
    # /eval/run).
    #   none        open (local dev)
    #   token       shared ADMIN_TOKEN secret
    #   github_org  caller's GitHub PAT must belong to an active member of
    #               ADMIN_GITHUB_ORG (needs read:org scope)
    #   github_sso  browser OAuth flow ("Sign in with GitHub") — org-membership
    #               checked at login, session kept in a signed cookie. Bearer
    #               PATs still work (github_org rules) so curl/Make keep working.
    ADMIN_AUTH_MODE: Literal["none", "token", "github_org", "github_sso"] = "none"
    ADMIN_TOKEN: str | None = None
    ADMIN_GITHUB_ORG: str = "eic"
    # Comma-separated GitHub logins allowed as admins. When set, this REPLACES
    # the org-membership check (works around org OAuth-App access restrictions
    # — only the user's identity is needed, no read:org).
    ADMIN_GITHUB_USERS: str = ""
    ADMIN_AUTH_CACHE_TTL_S: float = 300.0
    # github_sso: register a GitHub OAuth App (callback: <base>/auth/callback).
    GITHUB_OAUTH_CLIENT_ID: str | None = None
    GITHUB_OAUTH_CLIENT_SECRET: str | None = None
    # Secret for signing session cookies + OAuth state. REQUIRED for github_sso.
    ADMIN_SESSION_SECRET: str | None = None
    ADMIN_SESSION_TTL_S: float = 7 * 86400.0
    # Absolute public base URL for OAuth redirects (e.g. https://api.aprozo.space).
    # Empty = derived from the request (respects X-Forwarded-Proto).
    PUBLIC_BASE_URL: str | None = None

    # Public endpoint rate limiting (per client IP, sliding minute window).
    QUERY_RATE_LIMIT_PER_MINUTE: int = 60

    # Agentic retrieval: expose search_index as an LLM tool.
    AGENTIC_SEARCH_ENABLED: bool = True
    AGENTIC_SEARCH_TOP_K: int = 6

    # Evaluation harness. Judge model defaults to the generation model;
    # point it at a cheaper one if the answer model is expensive.
    EVAL_JUDGE_MODEL: str | None = None

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "QDRANT_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_TEMPERATURE",
        "OPENAI_EXTRA_BODY",
        "GITHUB_TOKEN",
        "EPIC_INTERNAL_COOKIE",
        "EPIC_INTERNAL_AUTH_HEADER",
        mode="before",
    )
    @classmethod
    def empty_string_as_none(cls, value: str | None) -> str | None:
        if value == "":
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
