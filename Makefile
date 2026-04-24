.PHONY: dev up down logs migrate ingest ingest-full reindex query test lint format clean-data

dev:
	docker compose up --build

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

migrate:
	docker compose run --rm api alembic upgrade head

ingest:
	curl -X POST http://localhost:8000/ingest/run \
		-H 'Content-Type: application/json' \
		-d '{"source_names":["eic_website"],"max_pages":25}'

ingest-full:
	curl -X POST http://localhost:8000/ingest/run \
		-H 'Content-Type: application/json' \
		-d '{"source_names":["eic_website"]}'

reindex:
	curl -X POST http://localhost:8000/admin/reindex \
		-H 'Content-Type: application/json' \
		-d '{"source_names":["eic_website"]}'

query:
	curl -X POST http://localhost:8000/query \
		-H 'Content-Type: application/json' \
		-d '{"query":"How do I run simulation tutorials?","scope":"public","top_k":8,"generate_answer":true}'

test:
	pytest -q

lint:
	ruff check app

format:
	ruff format app

# DESTRUCTIVE: wipes the local postgres + qdrant bind-mount directories.
# Use only when you want to start completely fresh. Stops containers first
# because Postgres will not release its lock while running.
clean-data:
	docker compose down
	rm -rf postgres_data qdrant_data
	mkdir -p postgres_data qdrant_data

