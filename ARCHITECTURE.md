# Smart Search Backend Architecture

## Goal

This service indexes public EIC collaboration knowledge and permission-sensitive internal ePIC content, then serves grounded search answers with citations. The backend is intentionally independent of the current `eic.github.io` frontend search implementation so the site can adopt it in a later phase.

## Runtime Components

- **FastAPI API** exposes query, ingestion, source, document, job, health, readiness, reindex, and feedback endpoints.
- **PostgreSQL** stores source metadata, documents, chunks, crawl jobs, ingestion state, permission metadata, query logs, and feedback. It also provides the lexical retrieval leg through full-text search.
- **Qdrant** stores one vector point per chunk with metadata payloads used for filtering and source attribution.
- **Embedding provider adapter** defaults to local `sentence-transformers/all-MiniLM-L6-v2` embeddings and can be replaced by HTTP, TEI, or hashing providers.
- **Answer provider adapter** defaults to extractive grounded answers and can be replaced by any HTTP generation service.

## Ingestion

Connectors emit normalized `DocumentPayload` records:

- `eic_website`: crawls HTML from `https://eic.github.io/`, extracts main content, title, headings, breadcrumbs, links, and HTTP last-modified metadata.
- `eic_github_repo`: reads docs-like files from `eic/eic.github.io`, preserving path, ref, commit hash, front matter, and raw/blob URLs.
- `epic_internal`: crawls `https://www.epic-eic.org/index-internal.html` with optional cookie/header auth. All records from this connector are labeled `internal`.

Documents are split with heading-aware chunking. Each chunk stores hierarchy, content hash, source metadata, visibility, and Qdrant point ID.

## Retrieval

The query flow is:

1. Build a visibility-safe filter from request scope.
2. Run PostgreSQL full-text search against chunk content.
3. Run Qdrant vector search using the same visibility and metadata filters.
4. Fuse normalized lexical and vector scores.
5. Boost instructional/tutorial-like documents for "how do I" queries.
6. Add small freshness boosts when dates are known.
7. Reduce duplicated mirrored content, especially site page plus repo markdown.
8. Return citations with exact snippets and retrieval metadata.

Public queries only allow `visibility = public`. Internal/all scopes include `public` and `internal` labels. The separation is enforced in both PostgreSQL and Qdrant filters.

## Answering

The default answer generator is extractive. If retrieved support is weak, it returns:

> I couldn't find enough support in the indexed sources.

It still returns best matching citations so users can inspect the nearest evidence.

## Deployment Shape

The local stack is Docker Compose with `api`, `postgres`, and `qdrant`. The same containers can be deployed cheaply on a VM or container platform. Scheduled reindexing can be done by calling `/admin/reindex` from cron or GitHub Actions.
