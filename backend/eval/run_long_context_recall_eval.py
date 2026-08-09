#!/usr/bin/env python3
"""Long-context M2 vs M3 keyword recall (cross-session memory quality).

Requires Ollama online. RAG disabled.

Usage:
  backend/.venv/bin/python backend/eval/run_long_context_recall_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
MODES = ["M2", "M3"]

if os.getenv("SEED_EVAL_FAST", "true").strip().lower() in ("true", "1", "yes"):
    logic._invoke_summary_llm = lambda _prompt: ""  # type: ignore[attr-defined]


def run_mode(user_id: int, active_session_id: int, mode: str) -> Dict[str, Any]:
    tokens = lcs.measure_memory_tokens(user_id, active_session_id, mode)
    started = time.perf_counter()
    response = logic.process_chat_message(
        user_id,
        lcs.RECALL_PROMPT,
        rag_store=None,
        force_new_session=False,
        memory_mode=mode,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    reply = response.get("reply") or ""
    kw = lcs.keyword_pass_rate(reply, lcs.RECALL_KEYWORDS)
    return {
        "memory_mode": mode,
        "keyword_pass_rate": kw["keyword_pass_rate"],
        "keyword_checks": kw["keyword_checks"],
        "latency_ms": latency_ms,
        "estimated_memory_tokens": tokens.get("estimated_memory_tokens"),
        "reply_excerpt": reply[:240],
    }


def run_scenario(num_closed_sessions: int, turns_per_session: int) -> Dict[str, Any]:
    by_mode: Dict[str, Dict[str, Any]] = {}
    scenario_meta: Dict[str, Any] = {}
    for mode in MODES:
        seeded = lcs.seed_scenario(
            num_closed_sessions,
            turns_per_session,
            label_prefix=f"LongRecall_{mode}",
        )
        scenario_meta = seeded["scenario"]
        by_mode[mode] = run_mode(seeded["user_id"], seeded["active_session_id"], mode)
    return {
        "scenario": scenario_meta,
        "keywords": lcs.RECALL_KEYWORDS,
        "M2": by_mode["M2"],
        "M3": by_mode["M3"],
        "M2_recall_gte_M3": by_mode["M2"]["keyword_pass_rate"] >= by_mode["M3"]["keyword_pass_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-context M2 vs M3 keyword recall")
    add_eval_db_arguments(parser)
    args = parser.parse_args()

    if not logic.check_ollama_reachable():
        print("ERROR: Ollama not reachable. Start ./start.sh or ollama serve.")
        sys.exit(1)

    setup_eval_database(args)
    rows: List[Dict[str, Any]] = []
    for num_sessions, turns in lcs.DEFAULT_SCENARIOS:
        print(f"Scenario: {num_sessions}s x {turns}t...", flush=True)
        rows.append(run_scenario(num_sessions, turns))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"long_context_recall_eval_{stamp}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": stamp,
        "eval": "long_context_recall",
        "recall_prompt": lcs.RECALL_PROMPT,
        "keywords": lcs.RECALL_KEYWORDS,
        "rag_enabled": False,
        "ollama_model": os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
        **eval_db_metadata(),
        "scenarios": rows,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"\nWrote {out_path}\n")
    print(f"{'sessions':>8} {'M2 kpr':>8} {'M3 kpr':>8}")
    for row in rows:
        sc = row["scenario"]
        print(
            f"{sc['closed_sessions']:>8} "
            f"{row['M2']['keyword_pass_rate']:>8.3f} "
            f"{row['M3']['keyword_pass_rate']:>8.3f}"
        )


if __name__ == "__main__":
    main()
