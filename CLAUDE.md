# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working with this repository.

## Development commands

Run from the project root.

```bash
make dev        # Start full local stack with hot reload (docker compose up --build)
make up         # Start in background
make down       # Stop all services
make logs       # Tail API logs
make migrate    # Run database migrations (alembic upgrade head)
make reindex    # Force full reindex of eic_website
make clean-data # DESTRUCTIVE: wipe postgres_data/ + qdrant_data/
make test       # Run tests (pytest -q)
make lint       # Check style (ruff check app)
make format     # Auto-format (ruff format app)
make ingest     # Trigger sample ingestion via API
make query      # Send example query via API
```

**Initial setup:**
```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY if you want LLM answers;
# otherwise leave GENERATION_PROVIDER=extractive for local-only mode.
make up
make migrate
# API: http://localhost:8000
# Readiness: curl http://localhost:8000/ready
```

**Run a single test file:**
```bash
docker compose run --rm api pytest app/tests/test_query_service.py -v
```

**Add a migration after changing models:**
```bash
docker compose run --rm api alembic revision --autogenerate -m "description"
docker compose run --rm api alembic upgrade head
```

## Architecture

Python 3.12 FastAPI application for EIC-ecosystem semantic search with hybrid retrieval. Deployed as a single container + managed PostgreSQL + Qdrant.

### Layer structure

```
API Layer        app/api/routes.py          FastAPI endpoints
Service Layer    app/services/              QueryService, QueryCache, analytics, factory
Retrieval        app/retrieval/             HybridRetriever, rerank, query_rewrite, parent_expand
Ingestion        app/ingestion/             Per-source connectors + orchestrator
LLM Providers    app/llm/                   Embedding + generation (pluggable)
Data Access      app/db/session.py          SQLAlchemy sessions
Models           app/models/entities.py     ORM entities
Schemas          app/schemas/api.py         Pydantic request/response types
Config           app/core/config.py         Pydantic settings from env
```

### Query flow

`POST /query` → `QueryService.query()`:

1. **Cache probe**: if the (query, scope, top_k, generate_answer, filters) tuple is cached within TTL, return immediately.
2. **Query rewrite** (optional): mini-LLM produces paraphrase variants for short queries; LRU-cached.
3. **Hybrid retrieve**: `HybridRetriever.search()` runs lexical (Postgres FTS) and vector (Qdrant + MiniLM) **in parallel**, fuses with 0.62·vec + 0.38·lex, applies freshness + instructional boosts, dedupes.
4. **Rerank** (optional): local cross-encoder, thread-pool with timeout fallback.
5. **Parent expansion** (optional, md-scoped): replaces hits with their heading-section parent.
6. **Answer generation**: OpenAI chat completions with citation-anchored prompt (`[1][2]`). Falls back to extractive generator on error.
7. **QueryLog + cache insert** for analytics.

### Ingestion flow

`POST /ingest/run` → `IngestionOrchestrator.run()` → connector yields `DocumentPayload` items → `HeadingAwareChunker` splits → chunks saved to PostgreSQL → embedded and stored in Qdrant → job/state tables updated. Content-hash idempotency: unchanged documents skip re-embedding. `reset_query_cache()` fires on any content change.

**Six connectors** (`source_name` → class):

- `eic_website` / `eicug_website` / `bnl_wiki_epic` → `WebsiteCrawler` (same class, per-source config)
- `epic_internal` → `EpicInternalCrawler` (auth-aware subclass)
- `eic_github_repo` → `GitHubRepoConnector` (single repo)
- `eic_github_org` → `GitHubOrgConnector` (all public repos in an org, markdown only)
- `zenodo_epic` → `ZenodoConnector` (metadata + PDF text extraction via pypdf)

### Pluggable providers

Built in `app/services/factory.py`:

| Provider | Env var | Default | Alternatives |
|---|---|---|---|
| Embedding | `EMBEDDING_PROVIDER` | `sentence_transformers` (free, 384-d MiniLM) | `hashing`, `http`, `tei` |
| Generation | `GENERATION_PROVIDER` | `extractive` | `openai`, `http` |
| Reranker | `RERANK_PROVIDER` | `none` | `cross_encoder` |
| Query rewrite | `QUERY_REWRITE_PROVIDER` | `none` | `openai` |

### Visibility / permission model

Chunks carry `visibility` metadata (`public` / `internal`). Enforced at both Postgres (WHERE) and Qdrant (filter) query level. The `/query` endpoint accepts a `scope` parameter (`public`, `internal`, `all`). Public widget should always send `scope: "public"`.

### Database schema key tables

`sources`, `documents`, `chunks` (core content + hashes), `crawl_jobs` (job tracking), `ingestion_state` (cursor per connector), `query_logs` (telemetry + tokens + cost + confidence), `feedback` (user ratings).

### Local stack (docker-compose.yml)

- **api** — FastAPI + uvicorn, runs `alembic upgrade head` on startup, `--reload` in dev
- **postgres** — PostgreSQL 16 (port 5432), data in `./postgres_data/`
- **qdrant** — Qdrant v1.12.5 (ports 6333/6334), data in `./qdrant_data/`

### Key files when extending

- `app/api/routes.py` — new endpoints
- `app/services/query.py` — cache handling + per-request flow
- `app/retrieval/hybrid.py` — scoring boost tuning, fusion weights, parallel retrieval
- `app/ingestion/orchestrator.py` — new `source_name` → connector mapping
- `app/ingestion/<connector>.py` — source-specific crawl logic
- `app/models/entities.py` — schema changes (follow with `alembic revision`)
- `.env.example` — document new config options
