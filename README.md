# EIC Smart Search

AI-powered search across the EIC documentation ecosystem. Fuses lexical +
vector retrieval over 1500+ documents from 6 sources, generates grounded
LLM answers with inline citations, ships a drop-in widget for
[eic.github.io](https://eic.github.io).

## What's indexed

| Source | Connector | Scope |
| --- | --- | --- |
| `eic_website` | `WebsiteCrawler` | [eic.github.io](https://eic.github.io) |
| `eicug_website` | `WebsiteCrawler` | [www.eicug.org](https://www.eicug.org) |
| `bnl_wiki_epic` | `WebsiteCrawler` | [wiki.bnl.gov/EPIC/](https://wiki.bnl.gov/EPIC/), MediaWiki-aware |
| `epic_internal` | `EpicInternalCrawler` | [www.epic-eic.org](https://www.epic-eic.org/index-internal.html) |
| `eic_github_org` | `GitHubOrgConnector` | All public repos under [github.com/eic](https://github.com/orgs/eic/repositories), markdown only |
| `zenodo_epic` | `ZenodoConnector` | [Zenodo ePIC community](https://zenodo.org/communities/epic), includes PDF text extraction |

## Architecture

```text
             ┌──────────────┐     ┌───────────┐
             │  Ingestion   │────▶│ Postgres  │ (full-text, chunks, docs, analytics)
             │ Orchestrator │     └───────────┘
             └──────┬───────┘     ┌───────────┐
                    │             │  Qdrant   │ (vectors)
                    └────────────▶└───────────┘
                                       │
┌────────────────┐     ┌───────────┐   │
│  Smart-search  │────▶│  /query   │───┴──▶  Hybrid retriever:
│  widget (JS)   │     │  endpoint │         lexical || vector → fuse
│  on            │     └─────┬─────┘                  │
│  eic.github.io │           │                        ▼
└────────────────┘           ▼                  Rerank (optional)
                      QueryService:                   │
                      cache hit? ──▶ return           ▼
                         │  miss                Parent-expand (md)
                         ▼                            │
                      rewriter ──── mini LLM          ▼
                         │                       AnswerGenerator
                         ▼                       (gpt-5.4-nano)
                      retrieve + rerank ──────────────┤
                                                      ▼
                                       QueryResponse  (answer + citations + debug)
```

### Key components

- [`app/retrieval/hybrid.py`](app/retrieval/hybrid.py) — weighted fusion (vector 0.62 + lexical 0.38), parallel lexical|vector via thread pool, boosts for freshness + instructional intent, dedup
- [`app/retrieval/rerank.py`](app/retrieval/rerank.py) — optional local cross-encoder with timeout fallback
- [`app/retrieval/query_rewrite.py`](app/retrieval/query_rewrite.py) — LLM-based paraphrase expansion (short queries only), LRU-cached
- [`app/retrieval/parent_expand.py`](app/retrieval/parent_expand.py) — markdown-scoped heading-tree expansion
- [`app/llm/generation.py`](app/llm/generation.py) — OpenAI chat completions with citation-anchored prompt, cost tracking, fallback to extractive on error
- [`app/services/query_cache.py`](app/services/query_cache.py) — LRU + TTL cache, auto-invalidated on ingest
- [`app/services/analytics.py`](app/services/analytics.py) — `/admin/analytics` + public `/popular` endpoints

## Quick start

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY if you want LLM answers; otherwise leave
# GENERATION_PROVIDER=extractive for local-only mode.
make up            # brings up postgres, qdrant, api
make migrate       # alembic upgrade head
curl -X POST http://localhost:8000/ingest/run \
     -H 'content-type: application/json' \
     -d '{"source_names":["eic_website","eicug_website","bnl_wiki_epic","epic_internal","zenodo_epic","eic_github_org"]}'
```

Then query:

```bash
curl -X POST http://localhost:8000/query \
     -H 'content-type: application/json' \
     -d '{"query":"what are Rucio metadata tags?","top_k":5,"generate_answer":true}' | jq
```

## Frontend integration

A production patch for `eic.github.io` is maintained separately (not in this
repo). It replaces the navbar Lunr search with this backend while keeping
the static index as fallback.

## Providers & config

All pluggable, set via `.env` (see `.env.example` for the full list).

| Concern | Default | Alternatives |
| --- | --- | --- |
| Embeddings | `sentence_transformers` (free, CPU, 384-d MiniLM) | `http`, `tei`, `hashing` |
| Generation | `openai` (gpt-5.4-nano) | `extractive` (free, no LLM), `http` (generic POST adapter) |
| Reranker | `none` | `cross_encoder` (BAAI/bge-reranker-base, local, free) |
| Query rewrite | `openai` (2 paraphrase variants for short queries) | `none` |
| Cache | in-memory LRU, 500 entries, 1h TTL | `QUERY_CACHE_ENABLED=false` to disable |

## Observability

- `GET /admin/analytics?window_days=7` — top queries, zero-result queries,
  low-confidence queries, recent low-rated feedback, cost + p50/p95 latency
- `GET /popular?window_days=7&limit=5` — public sanitized popular list,
  consumed by the frontend widget for empty-state suggestions
- `POST /feedback` — user ratings, joined with `query_logs` for eval

## Development

```bash
make test         # pytest — 55 tests as of this commit
make lint         # ruff
make format       # ruff format
```

Key tests:

- [`test_query_service.py`](app/tests/test_query_service.py) — scope isolation + cache-hit end-to-end
- [`test_query_cache.py`](app/tests/test_query_cache.py) — LRU, TTL, singleton
- [`test_rerank.py`](app/tests/test_rerank.py) — promotion, timeout, error fallback
- [`test_ingestion_regressions.py`](app/tests/test_ingestion_regressions.py) — regression suite for the bugs we hit during multi-source rollout (NUL bytes, YAML dates, MediaWiki URL filters, short-content link extraction, epic_internal source_name)
- [`test_query_rewrite.py`](app/tests/test_query_rewrite.py) — paraphrase parsing, caching, error fallback
- [`test_parent_expand.py`](app/tests/test_parent_expand.py) — markdown scope, dedup, token budget

## Known limitations

- **`sentence-transformers` meta-tensor issue on some PyTorch 2.6 hosts** — reranker `bge-reranker-base` can fail to load on first call; the timeout fallback (3s) catches it and returns unreranked. Workaround: bump `RERANK_TIMEOUT_S` or disable rerank entirely.
- **PDF extraction is best-effort** — `pypdf` copes with most Zenodo uploads but occasionally returns partial text on heavily-encoded/scanned PDFs.
- **Cache is in-memory, not shared across workers** — run a single uvicorn worker or accept cache misses across worker boundaries.

## License

Apache 2.0 — see [LICENSE](LICENSE).
