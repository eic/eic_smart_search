#!/usr/bin/env bash
# Backup Postgres + Qdrant to a directory OUTSIDE the WSL vhdx (the classic
# corruption path). Keeps the last N days. Wire into cron:
#   30 2 * * *  cd /path/to/eic_smart_search && ./scripts/backup.sh >> /tmp/backup.log 2>&1
set -euo pipefail

DEST="${BACKUP_DIR:-/mnt/c/eic_backups}"
KEEP_DAYS="${BACKUP_KEEP_DAYS:-7}"
STAMP="$(date +%Y%m%d-%H%M%S)"
COLLECTION="${QDRANT_COLLECTION:-collaboration_knowledge_bgem3}"
mkdir -p "$DEST"

echo "[$(date)] backup -> $DEST/$STAMP"

# Postgres: logical dump (small, portable).
docker compose exec -T postgres pg_dump -U smartsearch -d smartsearch \
  | gzip > "$DEST/pg-$STAMP.sql.gz"

# Qdrant: snapshot the collection, then copy it out of the container.
SNAP=$(curl -s -X POST "http://localhost:6333/collections/$COLLECTION/snapshots" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['result']['name'])" 2>/dev/null || true)
if [ -n "${SNAP:-}" ]; then
  docker compose cp "qdrant:/qdrant/snapshots/$COLLECTION/$SNAP" "$DEST/qdrant-$STAMP.snapshot"
  # Free the in-container snapshot so they don't accumulate.
  curl -s -X DELETE "http://localhost:6333/collections/$COLLECTION/snapshots/$SNAP" >/dev/null || true
else
  echo "WARN: qdrant snapshot failed (collection $COLLECTION?)"
fi

# Retention.
find "$DEST" -name 'pg-*.sql.gz' -mtime +"$KEEP_DAYS" -delete 2>/dev/null || true
find "$DEST" -name 'qdrant-*.snapshot' -mtime +"$KEEP_DAYS" -delete 2>/dev/null || true

echo "[$(date)] backup done: $(ls -1 "$DEST"/pg-$STAMP.sql.gz "$DEST"/qdrant-$STAMP.snapshot 2>/dev/null | wc -l) files"
