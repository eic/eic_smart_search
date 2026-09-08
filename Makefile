.PHONY: dev up up-gpu up-all down logs migrate ingest ingest-full ingest-papers ingest-indico ingest-issues refresh-all reindex query eval eval-retrieval eval-ci sweep backup tunnel tunnel-bg tunnel-stop tunnel-logs test lint format clean-data people-refresh

# Cloudflare tunnel name (from ~/.cloudflared/config.yml). Used to clear
# stale edge connectors before (re)starting.
CF_TUNNEL := eic_ai_search

dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

up:
	docker compose up -d --build

# Same as `up` but with GPU access for the api service (needs NVIDIA runtime).
up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Start the stack (background) AND the Cloudflare tunnel (background).
# Both survive closing the terminal. Stop everything with `make down`.
up-all: up tunnel-bg

down:
	docker compose down
	-pkill -x cloudflared

# Expose http://localhost:8000 publicly via the cloudflared tunnel
# configured in ~/.cloudflared/config.yml (api.aprozo.space → :8000).
# Foreground process — run in a dedicated terminal, Ctrl-C to stop.
tunnel:
	cloudflared tunnel cleanup $(CF_TUNNEL) || true
	cloudflared tunnel run

# Background tunnel (used by up-all). Clears stale edge connectors first,
# then runs fully detached (setsid = own session, survives make/terminal exit).
# Logs: /tmp/cloudflared.log. Stop with `make tunnel-stop`.
tunnel-bg:
	-pkill -x cloudflared
	cloudflared tunnel cleanup $(CF_TUNNEL) || true
	setsid cloudflared tunnel run >/tmp/cloudflared.log 2>&1 </dev/null &
	@echo "cloudflared starting, waiting for edge registration..."
	@for i in $$(seq 1 12); do \
		if grep -q "Registered tunnel connection" /tmp/cloudflared.log; then \
			echo "cloudflared up (log: /tmp/cloudflared.log)"; exit 0; fi; \
		sleep 3; \
	done; \
	echo "cloudflared FAILED to register in 36s — see /tmp/cloudflared.log"; \
	tail -5 /tmp/cloudflared.log

tunnel-stop:
	-pkill -x cloudflared

tunnel-logs:
	tail -f /tmp/cloudflared.log

logs:
	docker compose logs -f api

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
sweep:
	@GROUPS_JSON=$$(printf '"%s",' $$(echo $(GROUPS) | tr ',' ' ') | sed 's/,$$//'); \
	curl -m 60 -X POST http://localhost:8000/eval/sweep \
		-H 'Content-Type: application/json' -H "Authorization: Bearer $$ADMIN_TOKEN" \
		-d "{\"label\":\"$(LABEL)\",\"groups\":[$$GROUPS_JSON]}"

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

