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
Retrieval        app/retrieval/             HybridRetriever, rerank, lexical, qdrant_store
Ingestion        app/ingestion/             Per-source connectors + orchestrator
LLM Providers    app/llm/                   Embedding + generation (pluggable)
Data Access      app/db/session.py          SQLAlchemy sessions
Models           app/models/entities.py     ORM entities
Schemas          app/schemas/api.py         Pydantic request/response types
Config           app/core/config.py         Pydantic settings from env
```

### Query flow

`POST /query` → `QueryService.query()`:

0. **Probe screen** (`app/services/probes.py`): automated non-questions (CLI subcommand guesses from package sandboxes, GraphQL fuzz, letterless strings) are tagged `client="probe"` — retrieved and logged, but no LLM call, no cache, and `analytics.human_traffic()` hides them (and `client="cli"`) from every ranked list.
1. **Cache probe**: if the (query, scope, top_k, generate_answer, filters) tuple is cached within TTL, return immediately.
2. **Acronym expansion**: domain acronyms (TDR, SVT, …) add an expanded query variant.
3. **Hybrid retrieve**: `HybridRetriever.search()` runs lexical (Postgres FTS) and vector (Qdrant + MiniLM) **in parallel**, fuses with 0.62·vec + 0.38·lex, applies freshness + instructional boosts, dedupes.
4. **Rerank** (optional): local cross-encoder, thread-pool with timeout fallback.
6. **Answer generation**: OpenAI chat completions with citation-anchored prompt (`[1][2]`). Falls back to extractive generator on error.
7. **QueryLog + cache insert** for analytics.

`POST /query/stream` is the SSE variant (used for perceived speed): emits a `citations` event as soon as retrieval returns, `delta` events per answer fragment, then `done` with the final aligned answer + citations. Same cache/log semantics via the shared `QueryService._finalize`.

### Ingestion flow

`POST /ingest/run` → `IngestionOrchestrator.run()` → connector yields `DocumentPayload` items → `HeadingAwareChunker` splits → chunks saved to PostgreSQL → embedded and stored in Qdrant → job/state tables updated. Content-hash idempotency: unchanged documents skip re-embedding. `reset_query_cache()` fires on any content change.

**Connectors** (`source_name` → class):

- `eic_website` / `eicug_website` / `bnl_wiki_epic` → `WebsiteCrawler` (same class, per-source config; trafilatura extraction with bs4 fallback)
- `epic_internal` → `EpicInternalCrawler` (auth-aware subclass)
- `eic_github_repo` → `GitHubRepoConnector` (single repo)
- `eic_github_org` → `GitHubOrgConnector` (all public repos in an org, markdown only)
- `zenodo_epic` → `ZenodoConnector` (metadata + PDF text via PyMuPDF/pypdf)
- `inspire_hep` → `InspireHepConnector` (INSPIRE-HEP literature; authors + ORCID feed the people graph)
- `arxiv_eic` → `ArxivConnector` (arXiv preprints, metadata + abstract)
- `indico_events` → `IndicoEventsConnector` (event agendas + attached slide/minutes PDFs)
- `eic_github_issues` → `GitHubIssuesConnector` (issue threads incl. comments; needs `GITHUB_TOKEN`)
- `eic_github_releases` → `GitHubReleasesConnector` (release notes / changelogs)
- `eic_github_discussions` → `GitHubDiscussionsConnector` (Q&A threads via GraphQL; `GITHUB_TOKEN` required)

### Pluggable providers

Built in `app/services/factory.py`:

| Provider | Env var | Default | Alternatives |
|---|---|---|---|
| Embedding | `EMBEDDING_PROVIDER` | `sentence_transformers` (MiniLM or bge-m3) | — |
| Generation | `GENERATION_PROVIDER` | `extractive` | `openai` (incl. any OpenAI-compatible server) |
| Reranker | `RERANK_PROVIDER` | `none` | `cross_encoder` |

### Evaluation harness

`make eval` (or `make eval-retrieval` for the free 10-second variant) runs the golden set in [profiles/eic/golden.yaml](profiles/eic/golden.yaml) through the live stack with `eval_mode=true` (bypasses cache + analytics log). Per-case retrieval metrics (hit@k, MRR, vector score, latency) + answer checks + LLM-as-a-judge scores (relevance/faithfulness/completeness) land in the `eval_runs` table; dashboard at `/admin/eval`. **Run it after any retrieval/prompt/model change** and add a golden case whenever a real query fails.

`make sweep` runs the **configuration study** ([app/eval/sweep.py](app/eval/sweep.py)): the same golden set replayed through alternative pipelines — the reconstructed v0 baseline (MiniLM-384 index, weighted fusion, no reranker), each upgrade added one at a time, several `top_k` values, reranker settings, embedding models, and (opt-in) generation models. Arms are built by copying `Settings` and flipping the `HybridRetriever` ablation switches (`fusion`, `expand_queries`, `dedupe`, `boosts`), so no restart or re-index is needed; the 384-d MiniLM collection is kept in Qdrant precisely to keep that arm runnable. Results land in one `eval_runs` row with `config.kind="sweep"` (filtered out of the normal runs list) and render as charts on `/admin/eval`. `POST /eval/sweep` takes `groups` and an optional `arms` list of arm keys (first key = control; `make sweep GROUPS=generation ARMS=gen_gpt-5.4-nano,gen_qwen3`). Alternative OpenAI-compatible providers (Gemini, Groq, NRP …) become generation arms when their key is set; per-model request extras such as vLLM thinking switches go through `OPENAI_EXTRA_BODY` (see `NRP_MODEL_OVERRIDES`). Metrics live in [app/eval/metrics.py](app/eval/metrics.py): nDCG@k / precision@k / success@k, bootstrap 95% CIs, and paired-bootstrap p-values against each block's control.

### Admin auth

Mutating endpoints (`/ingest/run`, `/admin/reindex`, `/people/refresh`, `/eval/run`) are guarded by `require_admin` ([app/api/auth.py](app/api/auth.py)). `ADMIN_AUTH_MODE`: `none` (dev), `token` (shared `ADMIN_TOKEN`), or `github_org` — callers authenticate with their own GitHub PAT and must be an active member of `ADMIN_GITHUB_ORG` (default `eic`).

### Visibility / permission model

Chunks carry `visibility` metadata (`public` / `internal`). Enforced at both Postgres (WHERE) and Qdrant (filter) query level. The `/query` endpoint accepts a `scope` parameter (`public`, `internal`, `all`). Public widget should always send `scope: "public"`.

### Database schema key tables

`sources`, `documents`, `chunks` (core content + hashes), `crawl_jobs` (job tracking), `ingestion_state` (cursor per connector), `query_logs` (telemetry + tokens + cost + confidence), `feedback` (user ratings; the widget POSTs the rating on click and PATCHes the optional report text onto the same row).

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
