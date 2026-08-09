#!/usr/bin/env bash
# Ensure NutriCoachAI backend (:8000) and frontend (:5173) are reachable.
# Starts missing services as daemons (nohup + pidfile); does NOT stop them on exit.
# Does not kill an occupied port — use ./scripts/restart_backend.sh for that.
#
# Usage:
#   source scripts/ensure_stack.sh
#   # or
#   ./scripts/ensure_stack.sh && node scripts/entry_038_food_choice_ui.mjs
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/lib_stack_daemon.sh"

STARTED_BACKEND=0
STARTED_FRONTEND=0

if stack_backend_healthy; then
  echo "[ok] Backend already running at ${BACKEND_URL}"
else
  stack_clear_stale_pidfile "$BACKEND_PIDFILE" "backend"
  if lsof -ti:"$BACKEND_PORT" >/dev/null 2>&1; then
    echo "[warn] Port ${BACKEND_PORT} is in use but /health failed — run ./scripts/restart_backend.sh" >&2
    exit 1
  fi
  stack_start_backend_daemon
  STARTED_BACKEND=1
fi

if stack_frontend_healthy; then
  echo "[ok] Frontend already running at ${FRONTEND_URL}"
else
  stack_clear_stale_pidfile "$FRONTEND_PIDFILE" "frontend"
  stack_start_frontend_daemon
  STARTED_FRONTEND=1
fi

# Verify Ollama reachability reported by backend health.
FOOD_CHOICE_OK="$(
  curl -sf --max-time 3 "${BACKEND_URL}/health" \
    | "$ROOT_DIR/.venv/bin/python" -c "import json,sys; d=json.load(sys.stdin); print('yes' if d.get('ollama_reachable') else 'no')"
)"
if [ "$FOOD_CHOICE_OK" != "yes" ]; then
  echo "[warn] Ollama not reachable — chat and food-choice live tests will fail until Ollama is running."
fi

if [ "$STARTED_BACKEND" = 1 ] || [ "$STARTED_FRONTEND" = 1 ]; then
  echo "[info] Stack started by ensure_stack.sh (left running via nohup/pidfile)."
fi

export NUTRICOACH_BACKEND_URL="$BACKEND_URL"
export NUTRICOACH_FRONTEND_URL="$FRONTEND_URL"
