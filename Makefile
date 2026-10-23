.PHONY: dev up up-gpu down logs migrate ingest ingest-full ingest-papers ingest-indico ingest-issues refresh-all reindex query eval eval-retrieval eval-ci sweep backup tunnel-status test lint format clean-data people-refresh

dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

up:
	docker compose up -d --build

# Same as `up` but with GPU access for the api service (needs NVIDIA runtime).
up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

# The public hostnames (api.aprozo.com -> :8000, star-api.aprozo.com -> :8001)
# are served by a dashboard-managed Cloudflare tunnel running as a systemd
# service; routes are edited in Cloudflare Zero Trust, not in a local file.
tunnel-status:
	systemctl status cloudflared --no-pager | head -5

migrate:
	docker compose run --rm api alembic upgrade head

ingest:
	curl -X POST http://localhost:8000/ingest/run \
		-H 'Content-Type: application/json' \
		-d '{"source_names":["eic_website"],"max_pages":25}'

ingest-full:
	docker compose run --rm api python -m app.ingestion.cli --sources eic_website --full

reindex:
	docker compose run --rm api python -m app.ingestion.cli --sources eic_website --full

# Literature sources: INSPIRE-HEP (peer-reviewed, ORCID for people graph)
# + arXiv preprints.
ingest-papers:
	docker compose run --rm api python -m app.ingestion.cli --sources inspire_hep,arxiv_eic

# Indico event agendas + slide/minutes PDFs into the search index.
ingest-indico:
	docker compose run --rm api python -m app.ingestion.cli --sources indico_events

# GitHub issue threads (set GITHUB_TOKEN in .env first).
ingest-issues:
	docker compose run --rm api python -m app.ingestion.cli --sources eic_github_issues

query:
	curl -X POST http://localhost:8000/query \
		-H 'Content-Type: application/json' \
		-d '{"query":"How do I run simulation tutorials?","scope":"public","top_k":8,"generate_answer":true}'

# Rebuild the people graph (Zenodo authors + Indico chairs/speakers).
# Run after each ingestion that adds Zenodo records or whenever you want
# fresh meeting attendance data. Requires PEOPLE_LOOKUP_ENABLED=true.
people-refresh:
	curl -X POST http://localhost:8000/people/refresh

# Run the golden-set evaluation (retrieval metrics + LLM judge).
# Run after ANY retrieval/prompt/model change; results at /admin/eval.
# Pass a label: make eval LABEL=after-rerank-change
LABEL ?= manual
eval:
	curl -m 600 -X POST http://localhost:8000/eval/run \
		-H 'Content-Type: application/json' -H "Authorization: Bearer $$ADMIN_TOKEN" \
		-d '{"label":"$(LABEL)","generate":true,"judge":true}'

# Configuration study (ablation): replay the benchmark through the original
# baseline pipeline, each upgrade, several top_k values and reranker settings.
# Runs in the background (~15 min); watch it fill in at /admin/eval.
# GROUPS=pipeline,top_k,rerank,embedding,generation
GROUPS ?= pipeline,top_k,rerank,embedding
# ARMS=gen_gpt-5.4-nano,gen_qwen3 narrows the run to those arm keys (first = control).
sweep:
	@GROUPS_JSON=$$(printf '"%s",' $$(echo $(GROUPS) | tr ',' ' ') | sed 's/,$$//'); \
	ARMS_JSON=$$(printf '"%s",' $$(echo $(ARMS) | tr ',' ' ') | sed 's/,$$//; s/^""$$//'); \
	curl -m 60 -X POST http://localhost:8000/eval/sweep \
		-H 'Content-Type: application/json' -H "Authorization: Bearer $$ADMIN_TOKEN" \
		-d "{\"label\":\"$(LABEL)\",\"groups\":[$$GROUPS_JSON],\"arms\":[$$ARMS_JSON]}"

# CI quality gate: retrieval benchmark with pass/fail thresholds.
eval-ci:
	ADMIN_TOKEN=$$(grep '^ADMIN_TOKEN=' .env | cut -d= -f2) python3 scripts/eval_gate.py --api http://localhost:8000

# Fast retrieval-only eval (no LLM cost, ~10 s).
eval-retrieval:
	curl -m 120 -X POST http://localhost:8000/eval/run \
		-H 'Content-Type: application/json' -H "Authorization: Bearer $$ADMIN_TOKEN" \
		-d '{"label":"$(LABEL)","generate":false,"judge":false}'

# Refresh every source + the people graph. Content-hash idempotency makes
# unchanged documents cheap; safe to run from cron (see README).
refresh-all:
	docker compose run --rm api python -m app.ingestion.cli --sources eic_website,eicug_website,bnl_wiki_epic,epic_internal,eic_github_repo,eic_github_org,eic_github_issues,eic_github_releases,zenodo_epic,inspire_hep,arxiv_eic,indico_events,eic_services,epic_prod
	curl -m 600 -X POST http://localhost:8000/people/refresh -H "Authorization: Bearer $$ADMIN_TOKEN"

# Test/lint run inside the api container — the host doesn't need
# pytest/ruff installed and the environment matches CI/production.
test:
	docker compose run --rm api pytest -q

lint:
	docker compose run --rm api ruff check app

format:
	docker compose run --rm api ruff format app

# DESTRUCTIVE: wipes the local postgres + qdrant bind-mount directories.
# Use only when you want to start completely fresh. Stops containers first
# because Postgres will not release its lock while running.
# Back up Postgres + Qdrant outside the WSL vhdx (see scripts/backup.sh).
backup:
	./scripts/backup.sh

clean-data:
	docker compose down
	rm -rf postgres_data qdrant_data
	mkdir -p postgres_data qdrant_data

