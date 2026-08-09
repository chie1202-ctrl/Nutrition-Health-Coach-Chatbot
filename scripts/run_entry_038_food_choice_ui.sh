#!/usr/bin/env bash
# Run Entry 038 Playwright UI validation with stack preflight.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

chmod +x scripts/ensure_stack.sh
./scripts/ensure_stack.sh

echo "[info] Running Entry 038 food-choice UI validation..."
node scripts/entry_038_food_choice_ui.mjs
