#!/usr/bin/env bash
# Explicitly restart the backend (e.g. after logic.py changes). Frontend left running.
# Stops via pidfile when valid; falls back to port-based kill only for restart.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/lib_stack_daemon.sh"

stack_stop_backend
stack_start_backend_daemon
echo "[ok] Backend restarted (pidfile ${BACKEND_PIDFILE})"
