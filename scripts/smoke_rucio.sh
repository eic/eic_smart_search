#!/usr/bin/env bash
# End-to-end smoke test:
#   1. Make sure the stack is up and ready.
#   2. Ingest eic.github.io (idempotent — skipped if already indexed).
#   3. Ask: "what metadata tags Rucio uses?"
#   4. Verify the top citations include the Rucio tutorial page.
#
# Usage:
#   bash scripts/smoke_rucio.sh                # default: assume stack already up
#   bash scripts/smoke_rucio.sh --boot         # also run `make up && make migrate` first
#   bash scripts/smoke_rucio.sh --reindex      # force a full reindex before querying
#
# Exit code 0 = passed (target URL in top 5 citations), 1 = failed.

set -euo pipefail

API=${API:-http://localhost:8000}
TARGET_URL="https://eic.github.io/tutorial-file-access/02-rucio_usage/index.html"
QUESTION="what metadata tags Rucio uses?"

BOOT=0
REINDEX=0
for arg in "$@"; do
  case "$arg" in
    --boot) BOOT=1 ;;
    --reindex) REINDEX=1 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

if ! command -v jq >/dev/null; then
  echo "ERROR: jq is required (apt install jq)"; exit 2
fi

if [ "$BOOT" -eq 1 ]; then
  echo ">> bringing stack up"
  make up >/dev/null
  make migrate >/dev/null
fi

echo ">> waiting for $API/ready"
for i in $(seq 1 60); do
  if curl -sf "$API/ready" >/dev/null; then
    echo "   ready"
    break
  fi
  sleep 2
  if [ "$i" -eq 60 ]; then
    echo "ERROR: API did not become ready in 120s"; exit 1
  fi
done

if [ "$REINDEX" -eq 1 ]; then
  echo ">> forcing full reindex of eic_website"
  curl -sf -X POST "$API/admin/reindex" \
    -H 'Content-Type: application/json' \
    -d '{"source_names":["eic_website"]}' | jq '.stats.sources.eic_website'
else
  echo ">> running idempotent ingest of eic_website (250 pages max)"
  curl -sf -X POST "$API/ingest/run" \
    -H 'Content-Type: application/json' \
    -d '{"source_names":["eic_website"]}' | jq '.stats.sources.eic_website'
fi

echo ">> asking: $QUESTION"
RESPONSE=$(curl -sf -X POST "$API/query" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg q "$QUESTION" '{query:$q, scope:"public", top_k:5, generate_answer:true}')")

echo ""
echo "===== ANSWER ====="
echo "$RESPONSE" | jq -r '.answer'
echo ""
echo "===== TOP 5 CITATIONS ====="
echo "$RESPONSE" | jq -r '.citations[] | "  [\(.score)] \(.title)\n    \(.url)"'
echo ""

HITS=$(echo "$RESPONSE" | jq -r --arg url "$TARGET_URL" '[.citations[] | select(.url == $url)] | length')
RANK=$(echo "$RESPONSE" | jq -r --arg url "$TARGET_URL" '[.citations[].url] | index($url)')

if [ "$HITS" = "0" ] || [ "$RANK" = "null" ]; then
  echo "FAIL: target URL not in top 5 citations"
  echo "  target: $TARGET_URL"
  exit 1
fi

echo "PASS: target URL found at rank $((RANK + 1)) of 5"
echo "  $TARGET_URL"
