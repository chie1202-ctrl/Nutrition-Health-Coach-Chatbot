#!/usr/bin/env python3
"""M2 vs M3 memory token comparison under growing cross-session history.

Design (redesign 2026-07-30):
  - MEMORY_BUDGET_ENABLED=false: M2 bounded by per-component caps only
  - Fixed turns/session; vary closed-session count only
  - Also report M3_MATCH (recency transcript matched to M2 length)

Usage (from repo root):
  MEMORY_BUDGET_ENABLED=false TOKEN_EVAL_FAST=false \\
    backend/.venv/bin/python backend/eval/run_long_context_token_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import logic  # noqa: E402
import long_context_scenarios as lcs  # noqa: E402
from eval_db import add_eval_db_arguments, eval_db_metadata, setup_eval_database  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _apply_fast_mode(fast: bool) -> None:
    if fast:
        logic._invoke_summary_llm = lambda _prompt: ""  # type: ignore[attr-defined]


def run_scenario(num_closed_sessions: int, turns_per_session: int) -> Dict[str, Any]:
    seeded = lcs.seed_scenario(num_closed_sessions, turns_per_session, label_prefix="LongToken")
    user_id = seeded["user_id"]
    active_session_id = seeded["active_session_id"]

    m2 = lcs.measure_memory_tokens(user_id, active_session_id, "M2")
    m3 = lcs.measure_memory_tokens(user_id, active_session_id, "M3")
    m3_match = lcs.measure_memory_tokens(user_id, active_session_id, "M3_MATCH")

    return {
        "scenario": seeded["scenario"],
        "M2": m2,
        "M3": m3,
        "M3_MATCH": m3_match,
        "M2_tokens_lt_M3": (m2["estimated_memory_tokens"] or 0) < (m3["estimated_memory_tokens"] or 0),
        "token_delta_M3_minus_M2": (m3["estimated_memory_tokens"] or 0) - (m2["estimated_memory_tokens"] or 0),
        "m3_match_chars_near_m2": abs((m3_match["memory_chars"] or 0) - (m2["memory_chars"] or 0)) <= 64,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-context M2 vs M3 token eval (component-bounded M2)")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use empty fallback summaries (no Ollama); not production-realistic for M2",
    )
    add_eval_db_arguments(parser)
    args = parser.parse_args()

    os.environ.setdefault("MEMORY_BUDGET_ENABLED", "false")
    fast = args.fast or os.getenv("TOKEN_EVAL_FAST", "false").strip().lower() in ("true", "1", "yes")
    _apply_fast_mode(fast)

    setup_eval_database(args)
    rows: List[Dict[str, Any]] = []
    for num_sessions, turns in lcs.DEFAULT_SCENARIOS:
        print(f"Scenario: {num_sessions} closed sessions x {turns} turns...")
        rows.append(run_scenario(num_sessions, turns))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_fast" if fast else "_live"
    out_path = os.path.join(RESULTS_DIR, f"long_context_token_eval_{stamp}{suffix}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": stamp,
        "eval": "long_context_token",
        "design": "component_caps_no_global_budget",
        "token_eval_fast": fast,
        "ollama_model": os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
        "memory_budget_enabled": logic.memory_budget_enabled(),
        "memory_budget_chars": logic.memory_budget_chars(),
        "session_summary_max_chars": logic.session_summary_max_chars(),
        "cumulative_summary_max_chars": logic.cumulative_summary_max_chars(),
        "active_turn_max_chars": logic.active_turn_max_chars(),
        **eval_db_metadata(),
        "note": (
            "Tokens = len(memory_text)//4. M2 bounded by per-component caps only "
            "(MEMORY_BUDGET_ENABLED=false). M3 full transcript. M3_MATCH = recency "
            "transcript truncated to M2 memory length."
        ),
        "scenarios": rows,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"\nWrote {out_path}\n")
    print(f"{'sessions':>8} {'turns':>6} {'M2':>8} {'M3':>8} {'M3m':>8} {'M2<M3':>8}")
    for row in rows:
        sc = row["scenario"]
        m2 = row["M2"]["estimated_memory_tokens"]
        m3 = row["M3"]["estimated_memory_tokens"]
        m3m = row["M3_MATCH"]["estimated_memory_tokens"]
        win = "yes" if row["M2_tokens_lt_M3"] else "no"
        print(f"{sc['closed_sessions']:>8} {sc['turns_per_closed_session']:>6} {m2:>8} {m3:>8} {m3m:>8} {win:>8}")


if __name__ == "__main__":
    main()
