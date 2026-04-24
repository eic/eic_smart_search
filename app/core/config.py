from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
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

    EMBEDDING_PROVIDER: Literal["sentence_transformers", "hashing", "http", "tei"] = "sentence_transformers"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_HTTP_URL: AnyHttpUrl | None = None
    EMBEDDING_HTTP_MODEL: str | None = None
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_QUERY_INSTRUCTION: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    GENERATION_PROVIDER: Literal["extractive", "http", "openai"] = "extractive"
    GENERATION_HTTP_URL: AnyHttpUrl | None = None
    GENERATION_HTTP_MODEL: str | None = None
    MIN_SUPPORT_SCORE: float = 0.16

    RERANK_PROVIDER: Literal["none", "cross_encoder"] = "none"
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_CANDIDATES: int = 20
    RERANK_SCORE_WEIGHT: float = 0.7
    RERANK_MAX_PASSAGE_CHARS: int = 1500
    RERANK_TIMEOUT_S: float = 3.0

    PARENT_EXPANSION_ENABLED: bool = False
    PARENT_EXPANSION_FILETYPES: str | list[str] = Field(default_factory=lambda: ["md", "markdown"])
    PARENT_EXPANSION_HEADING_DEPTH: int = 2
    PARENT_EXPANSION_MAX_TOKENS: int = 1200

    QUERY_CACHE_ENABLED: bool = True
    QUERY_CACHE_MAX_SIZE: int = 500
    QUERY_CACHE_TTL_S: float = 3600.0

    QUERY_REWRITE_PROVIDER: Literal["none", "openai"] = "none"
    QUERY_REWRITE_MODEL: str | None = None
    QUERY_REWRITE_MAX_VARIANTS: int = 2
    QUERY_REWRITE_TRIGGER_MAX_WORDS: int = 8
    QUERY_REWRITE_MAX_OUTPUT_TOKENS: int = 200
    QUERY_REWRITE_TIMEOUT_S: float = 15.0
    QUERY_REWRITE_CACHE_SIZE: int = 512

    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_GENERATION_MODEL: str = "gpt-5.4-nano"
    OPENAI_REQUEST_TIMEOUT_S: float = 30.0
    OPENAI_MAX_OUTPUT_TOKENS: int = 600
    OPENAI_TEMPERATURE: float | None = None
    OPENAI_MAX_CONTEXT_CHARS: int = 1800
    OPENAI_INPUT_COST_PER_1M: float = 0.0
    OPENAI_OUTPUT_COST_PER_1M: float = 0.0

    EIC_SITE_URL: str = "https://eic.github.io/"
    EIC_SITE_MAX_PAGES: int = 500
    EIC_SITE_EXCLUDE_PREFIXES: str = "/epic/artifacts,/epic/build-,/epic/geoviewer,/epic/craterlake_views,/epic-prod/artifacts,/epic-prod/build-,/EDM4eic/"

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

    EPIC_INTERNAL_START_URL: str = "https://www.epic-eic.org/index-internal.html"
    EPIC_INTERNAL_MAX_PAGES: int = 100
    EPIC_INTERNAL_COOKIE: str | None = None
    EPIC_INTERNAL_AUTH_HEADER: str | None = None

    REQUEST_TIMEOUT_SECONDS: float = 20.0
    INGEST_USER_AGENT: str = "eic-smart-search/0.1 (+https://eic.github.io/)"

    @field_validator("CORS_ALLOW_ORIGINS", "PARENT_EXPANSION_FILETYPES", mode="before")
    @classmethod
    def split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "QDRANT_API_KEY",
        "EMBEDDING_HTTP_URL",
        "EMBEDDING_HTTP_MODEL",
        "GENERATION_HTTP_URL",
        "GENERATION_HTTP_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_TEMPERATURE",
        "QUERY_REWRITE_MODEL",
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
