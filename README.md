# EIC Smart Search

AI search for the EIC / ePIC ecosystem. Hybrid retrieval (Postgres FTS +
Qdrant vectors, RRF fusion, GPU cross-encoder rerank) over 11 sources,
streamed LLM answers with inline citations, a live meeting / people /
publication tool layer, a built-in evaluation harness, and a drop-in
widget for [eic.github.io](https://eic.github.io).

## What's indexed

| Source | Connector | Scope |
| --- | --- | --- |
| `eic_website` | `WebsiteCrawler` | [eic.github.io](https://eic.github.io) (trafilatura extraction) |
| `eicug_website` | `WebsiteCrawler` | [www.eicug.org](https://www.eicug.org) |
| `bnl_wiki_epic` | `WebsiteCrawler` | [wiki.bnl.gov/EPIC/](https://wiki.bnl.gov/EPIC/), MediaWiki-aware |
| `epic_internal` | `EpicInternalCrawler` | [www.epic-eic.org](https://www.epic-eic.org/index-internal.html) |
| `eic_github_repo` | `GitHubRepoConnector` | Single repo (eic.github.io sources) |
| `eic_github_org` | `GitHubOrgConnector` | All public repos under [github.com/eic](https://github.com/orgs/eic/repositories), markdown |
| `eic_github_issues` | `GitHubIssuesConnector` | Issue threads incl. comments (real user Q&A) |
| `eic_github_releases` | `GitHubReleasesConnector` | Release notes / changelogs |
| `eic_github_discussions` | `GitHubDiscussionsConnector` | Discussions Q&A (GraphQL; activates when repos enable it) |
| `zenodo_epic` | `ZenodoConnector` | [Zenodo ePIC community](https://zenodo.org/communities/epic) + PDF text (PyMuPDF) |
| `inspire_hep` | `InspireHepConnector` | INSPIRE-HEP literature; authors + ORCID feed the people graph |
| `arxiv_eic` | `ArxivConnector` | arXiv preprints (metadata + abstract, freshness source) |
| `indico_events` | `IndicoEventsConnector` | Meeting agendas + attached slide/minutes **PDFs** |

Live LLM tools (no index round-trip): `search_indico_events` (upcoming
meetings), `lookup_person` (people graph: affiliations, ORCID, recent
talks/papers), `search_zenodo` (fresh records beyond the snapshot).

## Quick start

```bash
cp .env.example .env      # set OPENAI_API_KEY, or keep extractive local mode
make up                   # CPU-only; make up-gpu for CUDA embeddings + rerank
make migrate
curl http://localhost:8000/ready
make ingest               # sample crawl; see "Keeping the index fresh" below
make query
```

API at `http://localhost:8000`. Dashboards: `/admin/dashboard`
(analytics), `/admin/eval` (evaluation).

## Query API

```bash
# JSON, single response
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query":"How do I run simulation tutorials?","scope":"public","top_k":8,"generate_answer":true}'

# Server-Sent Events: citations arrive at retrieval speed (~0.5 s),
# answer streams token by token, final event carries aligned citations.
curl -N -X POST localhost:8000/query/stream -H 'Content-Type: application/json' -d '{...}'
```

Flow: cache probe → hybrid retrieve (lexical FTS with AND→OR fallback ∥
vector, acronym expansion, RRF fusion, freshness/instructional boosts,
test-beam demotion, cross-source dedup by DOI/arXiv/title) →
cross-encoder rerank → answer generation with tool calls → citation
alignment → query log + cache.

## Evaluation — run after every change

```bash
make eval LABEL=after-my-change   # golden set + retrieval metrics + LLM judge (~2 min)
make eval-retrieval               # retrieval-only, free, ~10 s
make sweep                        # configuration study (ablation), background, ~20 min
```

Golden set lives in `profiles/<profile>/golden.yaml` — **add a case whenever a
real query fails**. Results land in the `eval_runs` table; `/admin/eval`
shows metric trends, per-case detail, the golden dataset, and a standing
**failing-cases worklist** with one-click re-run (e.g. after switching
generation model). Eval traffic runs with `eval_mode=true` — it never
touches the response cache or analytics.

`make sweep` answers the other question — *which decision bought which
point*. It replays the same benchmark through the reconstructed v0
baseline (MiniLM-384 index, weighted fusion, no reranker), then each
upgrade one at a time, plus `top_k`, reranker, embedding-model and
(opt-in) generation-model variants. Every arm reports nDCG@5 with a
bootstrap 95% CI and a paired-bootstrap p-value against its control, so
improvements are separated from noise. Charts render on `/admin/eval`;
method notes in [docs/EVALUATION.md](docs/EVALUATION.md).

## Access control

`ADMIN_AUTH_MODE` guards mutating endpoints (`/ingest/run`,
`/admin/reindex`, `/people/refresh`, `/eval/run`):

- `none` — open (local dev default)
- `token` — shared `ADMIN_TOKEN` bearer secret
- `github_org` — callers use their own GitHub PAT (`read:org`); active
  membership in `ADMIN_GITHUB_ORG` is the ACL
- `github_sso` — **"Sign in with GitHub"** browser flow (register an
  OAuth App, set `GITHUB_OAUTH_CLIENT_ID/SECRET` + `ADMIN_SESSION_SECRET`);
  7-day signed session cookie; PATs still work for curl/Make

Public endpoints (`/query`, `/query/stream`, `/popular`) are rate-limited
per client IP (`QUERY_RATE_LIMIT_PER_MINUTE`).

## Keeping the index fresh

```bash
make refresh-all          # every source + people graph, sequentially
```

Cron example (host with the stack running):

```cron
# crawl websites + github nightly, papers + indico + people weekly, eval weekly
15 3 * * *  cd /path/to/eic_smart_search && make ingest-full ingest-issues >/dev/null
30 4 * * 0  cd /path/to/eic_smart_search && make ingest-papers ingest-indico people-refresh >/dev/null
0  6 * * 0  cd /path/to/eic_smart_search && make eval LABEL=weekly >/dev/null
# monthly FULL website reindex: incremental crawls never delete pages that
# vanished from the site (e.g. tutorial URL-format changes leave dead links)
0  5 1 * *  cd /path/to/eic_smart_search && make reindex >/dev/null
```

Content-hash idempotency makes re-runs cheap: unchanged documents skip
re-embedding, and the response cache resets automatically when content
changes.

## Local / private LLM

Any OpenAI-compatible server works — no code changes:

```bash
# e.g. vLLM on the same GPU box
vllm serve Qwen/Qwen3-32B-AWQ --port 8001
# .env
GENERATION_PROVIDER=openai
OPENAI_BASE_URL=http://host.docker.internal:8001/v1
OPENAI_GENERATION_MODEL=Qwen/Qwen3-32B-AWQ
```

Then `make eval LABEL=qwen3-32b` and compare runs on `/admin/eval` —
the judge scores answer the "is local good enough?" question with data.

## Development

```bash
make test        # pytest inside the api container
make lint        # ruff
make format
docker compose run --rm api pytest app/tests/test_query_service.py -v
```

See `CLAUDE.md` for the layer map and `ARCHITECTURE.md` for the data
flow. Migrations: `docker compose run --rm api alembic revision
--autogenerate -m "..."` after model changes.

## Deployment notes

- Single API container + PostgreSQL 16 + Qdrant. `docker-compose.gpu.yml`
  adds CUDA (embeddings + rerank ~10× faster).
- The public widget (`assets/js/smart-search.js` in the eic.github.io
  repo) needs only `search.backend_url` in `_config.yml` pointed at this
  API; CORS origins via `CORS_ALLOW_ORIGINS`.
- Behind a proxy/tunnel set `PUBLIC_BASE_URL` so OAuth redirects and
  cookie security resolve correctly.
- **Memory**: keep `/ready` probes pointed at the singleton (never build a
  fresh embedding provider per request — a 2 GB model per probe reads as a
  memory leak). Heavy full reindexes run inside the API worker and briefly
  saturate it; prefer running them off-peak (cron) and treat multi-GB RSS
  during ingest as transient, not a leak.
