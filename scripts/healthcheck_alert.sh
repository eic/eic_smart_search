#!/usr/bin/env bash
# Ping /ready; on failure, notify. Cron every few minutes:
#   */5 * * * *  /path/to/eic_smart_search/scripts/healthcheck_alert.sh
#
# Set ALERT_WEBHOOK (Mattermost/Slack incoming webhook) to get a message;
# otherwise it just logs. Auto-restarts the stack once before alerting.
set -uo pipefail

URL="${HEALTH_URL:-http://localhost:8000/ready}"
WEBHOOK="${ALERT_WEBHOOK:-}"

notify() {
  local msg="$1"
  echo "[$(date)] ALERT: $msg"
  if [ -n "$WEBHOOK" ]; then
    curl -s -X POST "$WEBHOOK" -H 'Content-Type: application/json' \
      -d "{\"text\": \"🔴 EIC Smart Search: $msg\"}" >/dev/null || true
  fi
}

if curl -sf -m 15 "$URL" | grep -q '"ready"'; then
  exit 0
fi

# One self-heal attempt before shouting.
cd "$(dirname "$0")/.." || exit 1
docker compose up -d >/dev/null 2>&1 || true
sleep 30
if curl -sf -m 15 "$URL" | grep -q '"ready"'; then
  notify "was down, auto-restart recovered it"
  exit 0
fi
notify "DOWN and auto-restart failed — manual intervention needed"
exit 1
