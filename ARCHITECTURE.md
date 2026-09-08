# Smart Search Backend Architecture

## Goal

Index the EIC/ePIC knowledge ecosystem (docs, wikis, GitHub, literature,
meetings) and serve grounded, cited, streamed answers — with a live tool
layer for the things an index can't know (upcoming meetings, people,
fresh publications) and a built-in evaluation harness so every retrieval
or prompt change is measured, not guessed.

## Runtime components

- **FastAPI API** — query (JSON + SSE), ingestion, admin/analytics,
  evaluation, people graph, GitHub SSO auth. Single container.
- **PostgreSQL 16** — sources, documents, chunks (+ full-text GIN
  indexes = the lexical retrieval leg), crawl jobs, ingestion state,
  query logs, feedback, persons, eval runs.
- **Qdrant** — one vector point per chunk with payload filters
  (visibility, source).
- **Embedding provider** — local sentence-transformers (bge-m3 on GPU in
  production, fp16; MiniLM anywhere; automatic CPU fallback).
- **Answer provider** — OpenAI-protocol client (any compatible server:
  OpenAI, vLLM, Ollama) with tool calling and token streaming; falls
  back to the extractive generator on any error.
- **Reranker** — local cross-encoder over the fused candidate pool.

## Ingestion

Thirteen connectors emit normalized `DocumentPayload` records (see
README for the table): website crawlers (trafilatura extraction with a
heading-aware bs4 fallback), GitHub repo/org/issues/releases/discussions,
Zenodo (+PDF text), INSPIRE-HEP, arXiv, and Indico events with attached
slide/minutes PDFs (PyMuPDF, pypdf fallback). Heading-aware chunking;
per-chunk content hashes give idempotent re-ingestion; any content
change resets the response cache. Wide Indico windows are fetched in
60-day slices (the export endpoint 504s beyond that).

## Retrieval

1. Visibility-safe filters from request scope (enforced in both engines).
2. **Lexical**: Postgres FTS, strict-AND first, automatic OR fallback
   when AND matches nothing (Postgres' stopword list is small — filler
   words like "many" otherwise zero out good questions).
3. **Vector**: Qdrant + embedded query, run concurrently with lexical.
4. **RRF fusion** (rank-based, K=60) — no hand-tuned weights; a chunk
   one engine ranks well stays visible regardless of the other engine's
   score scale. Instructional + freshness boosts on top.
5. **Cross-encoder rerank** of the top candidates (submit path only —
   previews skip it to keep typing fast).
6. **Dedup + diversity**: content-hash dedup; cross-source *work*
   identity (DOI / arXiv id / normalized title for paper sources) so the
   same paper via Zenodo+INSPIRE+arXiv appears once; per-document
   diversity decay; test-beam/prototype pages demoted for queries that
   aren't about test beams.
7. Citations with query-focused snippets.

## Answering

OpenAI-protocol generation with a citation-anchored prompt (`[1][2]`),
LaTeX-aware formatting, and design-vs-prototype guardrails. Live tools
the model may call: `search_indico_events`, `lookup_person`,
`search_zenodo` (forced on round 1 for obviously time-sensitive /
person queries; static context withheld in those cases). Tool rounds are
capped; per-tool latency is recorded. `/query/stream` emits SSE:
`citations` (at retrieval speed) → `delta` per token → `done` with the
final aligned answer + citations. After generation, cited sources are
reordered to the front and `[N]` markers remapped. Answers, tokens,
cost, latency and confidence (raw top cosine) land in `query_logs`; an
in-memory LRU+TTL cache serves repeats and is bypassed for
time-sensitive queries and eval traffic.

## Evaluation

`app/eval/golden.yaml` → runner (`eval_mode` bypasses cache + logs) →
per-case retrieval metrics (nDCG@k, precision@k, hit@k, MRR, latency),
answer checks (expected substrings, citation-marker validity),
LLM-as-a-judge (relevance / faithfulness / completeness) → `eval_runs`
table → `/admin/eval` dashboard (trends, per-case detail, golden dataset
view, failing-cases worklist with one-click re-run).

`app/eval/sweep.py` adds the configuration study: the same benchmark
replayed through alternative pipelines (v0 baseline, each upgrade, top_k,
reranker, embedding model, generation model) built by copying `Settings`
and flipping `HybridRetriever` ablation switches. One `eval_runs` row per
sweep (`config.kind="sweep"`), charted on the same dashboard, with
bootstrap CIs and paired-bootstrap significance from `app/eval/metrics.py`.

## Access control

`require_admin` guards all mutating endpoints. Modes: open (dev), shared
token, GitHub PAT org-membership check, or full GitHub OAuth SSO with
HMAC-signed session cookies — org membership is the ACL in both GitHub
modes. Public query endpoints are rate-limited per client IP.

## Deployment shape

Docker Compose: `api` + `postgres` + `qdrant`; `docker-compose.gpu.yml`
overlays CUDA. Scheduled freshness via cron hitting the Make targets
(see README). The public widget on eic.github.io talks to `/query/stream`
with a plain-`/query` fallback; CORS pins the allowed origins.
