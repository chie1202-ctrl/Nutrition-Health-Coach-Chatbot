#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-deepseek-r1:8b}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

echo "== NutriCoachAI launcher =="
echo "Project root: $ROOT_DIR"

if ! command -v ollama >/dev/null 2>&1; then
  echo "[error] Ollama is not installed. Install from https://ollama.com and retry."
  exit 1
fi

if ! curl -sf "${OLLAMA_BASE_URL%/}/api/tags" >/dev/null 2>&1; then
  echo "[info] Starting Ollama in the background..."
  ollama serve >/tmp/nutricoach-ollama.log 2>&1 &
  OLLAMA_PID=$!
  for _ in $(seq 1 30); do
    if curl -sf "${OLLAMA_BASE_URL%/}/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if ! curl -sf "${OLLAMA_BASE_URL%/}/api/tags" >/dev/null 2>&1; then
    echo "[error] Ollama did not become ready. See /tmp/nutricoach-ollama.log"
    exit 1
  fi
  echo "[ok] Ollama ready (pid ${OLLAMA_PID})"
else
  echo "[ok] Ollama already running at ${OLLAMA_BASE_URL}"
fi

if ! ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL}"; then
  echo "[info] Pulling model ${OLLAMA_MODEL}..."
  ollama pull "${OLLAMA_MODEL}"
fi

if [ ! -d "$ROOT_DIR/.venv" ]; then
  echo "[info] Creating Python virtual environment..."
  python3 -m venv "$ROOT_DIR/.venv"
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"
pip install -q -r "$ROOT_DIR/backend/requirements.txt"

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  echo "[info] Installing frontend dependencies..."
  (cd "$ROOT_DIR/frontend" && npm install)
fi

cleanup() {
  if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then kill "$BACKEND_PID" || true; fi
  if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then kill "$FRONTEND_PID" || true; fi
}
trap cleanup EXIT INT TERM

ENV_FILE="$ROOT_DIR/backend/.env"
if [ ! -f "$ENV_FILE" ]; then
  ENV_FILE="$ROOT_DIR/backend/.env.example"
fi
echo "[info] Loading env from ${ENV_FILE##*/}"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

echo "[info] Starting backend on :${BACKEND_PORT} (model: ${OLLAMA_MODEL})"
(
  cd "$ROOT_DIR/backend"
  uvicorn main:app --host 127.0.0.1 --port "$BACKEND_PORT"
) >/tmp/nutricoach-backend.log 2>&1 &
BACKEND_PID=$!

echo "[info] Starting frontend on :${FRONTEND_PORT}"
(
  cd "$ROOT_DIR/frontend"
  npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT"
) >/tmp/nutricoach-frontend.log 2>&1 &
FRONTEND_PID=$!

for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo
echo "NutriCoachAI is starting."
echo "  Frontend: http://127.0.0.1:${FRONTEND_PORT}"
echo "  Backend:  http://127.0.0.1:${BACKEND_PORT}/health"
echo "  Logs:     /tmp/nutricoach-backend.log , /tmp/nutricoach-frontend.log"
echo
echo "Press Ctrl+C to stop backend and frontend."

wait
