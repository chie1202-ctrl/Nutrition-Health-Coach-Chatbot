#!/usr/bin/env python3
"""LOCOMO token curve: injection tokens vs closed-session count for five memory modes.

Measures build_memory_context() only (no chat LLM). Uses official LOCOMO dialogues.

Usage:
  backend/.venv/bin/python backend/eval/run_locomo_token_curve.py --fast
  LOCOMO_EVAL_FAST=false backend/.venv/bin/python backend/eval/run_locomo_token_curve.py
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

from locomo.load_locomo import get_subset_summary, load_locomo_samples, select_stratified_subset  # noqa: E402
from locomo.seed_locomo import measure_injection_tokens, seed_locomo_conversation  # noqa: E402
from eval_db import add_eval_db_arguments, eval_db_metadata, setup_eval_database  # noqa: E402

RESULTS_DIR = os.path.join(EVAL_DIR, "results")
SECOM_MEMORY_MODES = ["M0", "RECURSUM", "SESSION_RET", "M2", "M3"]
SESSION_CUTOFFS = [1, 2, 4, 6, 8]
TOKEN_CURVE_QUERY = "What happened in our previous conversations?"


def _apply_fast_mode(fast: bool) -> None:
    if fast:
        logic._invoke_summary_llm = lambda _prompt: ""  # type: ignore[attr-defined]


def run_cutoff(
    sample: Dict[str, Any],
    mode: str,
    closed_sessions: int,
    *,
    fast_summaries: bool,
) -> Dict[str, Any]:
    seeded = seed_locomo_conversation(
        sample,
        mode,
        max_closed_sessions=closed_sessions,
        fast_summaries=fast_summaries,
    )
    metrics = measure_injection_tokens(
        seeded["user_id"],
        seeded["qa_session_id"],
        mode,
        query=TOKEN_CURVE_QUERY if mode == "SESSION_RET" else None,
    )
    return {
        "sample_id": seeded["sample_id"],
        "memory_mode": mode,
        "closed_sessions": closed_sessions,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LOCOMO token curve for five memory modes")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--cutoffs",
        default=",".join(str(value) for value in SESSION_CUTOFFS),
        help="Comma-separated closed-session cutoffs",
    )
    add_eval_db_arguments(parser)
    args = parser.parse_args()

    cutoffs = [int(value.strip()) for value in args.cutoffs.split(",") if value.strip()]
    fast = args.fast or os.getenv("LOCOMO_EVAL_FAST", "true").strip().lower() in ("true", "1", "yes")
    _apply_fast_mode(fast)

    setup_eval_database(args)
    all_samples = load_locomo_samples()
    subset = select_stratified_subset(all_samples, target_n=args.samples)

    rows: List[Dict[str, Any]] = []
    for sample in subset:
        sample_id = sample.get("sample_id")
        for closed_sessions in cutoffs:
            for mode in SECOM_MEMORY_MODES:
                print(f"Token curve {sample_id} mode={mode} closed={closed_sessions}", flush=True)
                rows.append(run_cutoff(sample, mode, closed_sessions, fast_summaries=fast))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"locomo_token_curve_{stamp}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": stamp,
        "eval": "locomo_token_curve",
        "locomo_eval_fast": fast,
        "session_cutoffs": cutoffs,
        "token_curve_query": TOKEN_CURVE_QUERY,
        "modes": SECOM_MEMORY_MODES,
        "subset": get_subset_summary(all_samples, target_n=args.samples),
        "memory_budget_chars": logic.memory_budget_chars(),
        "session_ret_max_tokens": logic.session_ret_max_tokens(),
        "session_ret_top_k": logic.session_ret_top_k(),
        **eval_db_metadata(),
        "rows": rows,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
