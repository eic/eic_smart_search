"""Run ingestion as a one-shot process, OUT of the API worker.

The `/ingest/run` endpoint executes the orchestrator on the request thread,
so a large refresh competes with live queries for the GPU, the DB pool and
memory (the known event-loop-starvation + OOM incident). Run it as a
sibling container instead — its own RSS, its own session, killable without
touching the API:

    docker compose run --rm api python -m app.ingestion.cli --sources inspire_hep,arxiv_eic
    docker compose run --rm api python -m app.ingestion.cli --all --full

Cron should point at this, not `curl /ingest/run`.
"""
from __future__ import annotations

import argparse
import sys

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.ingestion.orchestrator import SOURCE_CONFIGS
from app.services.factory import build_ingestion_orchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot EIC ingestion")
    parser.add_argument("--sources", default="", help="comma-separated source_names")
    parser.add_argument("--all", action="store_true", help="every configured source")
    parser.add_argument("--full", action="store_true", help="full reindex (clears source first)")
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        names = list(SOURCE_CONFIGS.keys())
    else:
        names = [s.strip() for s in args.sources.split(",") if s.strip()]
    if not names:
        parser.error("give --sources <a,b,...> or --all")

    settings = get_settings()
    orchestrator = build_ingestion_orchestrator(settings)
    with SessionLocal() as db:
        job_ids, stats = orchestrator.run(
            db,
            source_names=names,
            full_reindex=args.full,
            max_pages=args.max_pages,
            requested_by="cli",
        )
    failed = 0
    for name, s in stats.get("sources", {}).items():
        if s.get("status") == "failed":
            failed += 1
            print(f"FAILED {name}: {str(s.get('error'))[:200]}")
        else:
            print(f"ok {name}: seen={s.get('seen')} created={s.get('created')} "
                  f"updated={s.get('updated')} unchanged={s.get('unchanged')}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
