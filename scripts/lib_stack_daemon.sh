#!/usr/bin/env bash
# Shared daemon helpers for NutriCoachAI backend/frontend.
# Sourced by ensure_stack.sh and restart_backend.sh — not meant to be run alone.
#
# shellcheck shell=bash

STACK_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT}}"
FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:${FRONTEND_PORT}}"
BACKEND_LOG="${BACKEND_LOG:-/tmp/nutricoach-backend.log}"
FRONTEND_LOG="${FRONTEND_LOG:-/tmp/nutricoach-frontend.log}"
BACKEND_PIDFILE="${BACKEND_PIDFILE:-/tmp/nutricoach-backend.pid}"
FRONTEND_PIDFILE="${FRONTEND_PIDFILE:-/tmp/nutricoach-frontend.pid}"

stack_wait_for_url() {
  local url="$1"
  local label="$2"
  local attempts="${3:-60}"
  for _ in $(seq 1 "$attempts"); do
    if curl -sf --max-time 2 "$url" >/dev/null 2>&1; then
      echo "[ok] ${label} ready at ${url}"
      return 0
    fi
    sleep 1
  done
  echo "[error] ${label} not ready at ${url}" >&2
  return 1
}

stack_backend_healthy() {
  curl -sf --max-time 3 "${BACKEND_URL}/health" >/dev/null 2>&1
}

stack_frontend_healthy() {
  curl -sf --max-time 3 "${FRONTEND_URL}" >/dev/null 2>&1
}

stack_pid_alive() {
  local pid="$1"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

stack_backend_pid_matches() {
  local pid="$1"
  local cmd
  cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$cmd" == *"uvicorn"* && "$cmd" == *"main:app"* ]]
}

stack_frontend_pid_matches() {
  local pid="$1"
  local cmd
  cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$cmd" == *"vite"* || "$cmd" == *"npm run dev"* || "$cmd" == *"node"* ]]
}

stack_read_pidfile() {
  local pidfile="$1"
  if [ -f "$pidfile" ]; then
    tr -d '[:space:]' <"$pidfile"
  fi
}

stack_clear_stale_pidfile() {
  local pidfile="$1"
  local kind="$2"
  local pid
  pid="$(stack_read_pidfile "$pidfile")"
  if [ -z "$pid" ]; then
    rm -f "$pidfile" 2>/dev/null || true
    return 0
  fi
  if ! stack_pid_alive "$pid"; then
    echo "[info] Clearing stale ${kind} pidfile (pid ${pid} not running)"
    rm -f "$pidfile" 2>/dev/null || true
    return 0
  fi
  if [ "$kind" = "backend" ] && ! stack_backend_pid_matches "$pid"; then
    echo "[info] Clearing stale backend pidfile (pid ${pid} is not uvicorn)"
    rm -f "$pidfile" 2>/dev/null || true
  fi
}

stack_stop_backend() {
  local pid
  stack_clear_stale_pidfile "$BACKEND_PIDFILE" "backend"
  pid="$(stack_read_pidfile "$BACKEND_PIDFILE")"
  if [ -n "$pid" ] && stack_pid_alive "$pid" && stack_backend_pid_matches "$pid"; then
    echo "[info] Stopping backend pid ${pid} (from ${BACKEND_PIDFILE})"
    kill "$pid" 2>/dev/null || true
    sleep 2
    if stack_pid_alive "$pid"; then
      kill -9 "$pid" 2>/dev/null || true
      sleep 1
    fi
  elif lsof -ti:"$BACKEND_PORT" >/dev/null 2>&1; then
    echo "[info] Stopping process(es) on :${BACKEND_PORT} (no valid pidfile)"
    # Prefer graceful; only used by explicit restart.
    lsof -ti:"$BACKEND_PORT" | xargs kill 2>/dev/null || true
    sleep 2
  fi
  rm -f "$BACKEND_PIDFILE" 2>/dev/null || true
}

stack_start_backend_daemon() {
  if [ ! -x "$STACK_ROOT_DIR/.venv/bin/uvicorn" ]; then
    echo "[error] Missing $STACK_ROOT_DIR/.venv/bin/uvicorn — run ./start.sh once to bootstrap." >&2
    return 1
  fi
  local env_file="$STACK_ROOT_DIR/backend/.env"
  if [ ! -f "$env_file" ]; then
    env_file="$STACK_ROOT_DIR/backend/.env.example"
  fi
  echo "[info] Starting backend daemon on :${BACKEND_PORT} (log: ${BACKEND_LOG}, pidfile: ${BACKEND_PIDFILE})"
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
  (
    cd "$STACK_ROOT_DIR/backend"
    # nohup + background so the process survives the parent shell exiting
    nohup "$STACK_ROOT_DIR/.venv/bin/uvicorn" main:app --host 127.0.0.1 --port "$BACKEND_PORT" \
      >>"$BACKEND_LOG" 2>&1 &
    echo $! >"$BACKEND_PIDFILE"
  )
  stack_wait_for_url "${BACKEND_URL}/health" "Backend"
}

stack_start_frontend_daemon() {
  if [ ! -d "$STACK_ROOT_DIR/frontend/node_modules" ]; then
    echo "[error] Missing frontend/node_modules — run ./start.sh once to bootstrap." >&2
    return 1
  fi
  echo "[info] Starting frontend daemon on :${FRONTEND_PORT} (log: ${FRONTEND_LOG}, pidfile: ${FRONTEND_PIDFILE})"
  (
    cd "$STACK_ROOT_DIR/frontend"
    nohup npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" >>"$FRONTEND_LOG" 2>&1 &
    echo $! >"$FRONTEND_PIDFILE"
  )
  stack_wait_for_url "${FRONTEND_URL}" "Frontend"
}
