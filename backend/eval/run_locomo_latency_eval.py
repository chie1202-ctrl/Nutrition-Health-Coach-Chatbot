#!/usr/bin/env python3
"""LOCOMO latency eval: TTFT + total latency for five memory modes (Method A).

Uses production SSE path (iter_chat_sse_events). RAG disabled.

Usage:
  backend/.venv/bin/python backend/eval/run_locomo_latency_eval.py --repeats 2 --max-samples 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import logic  # noqa: E402

from locomo.load_locomo import get_subset_summary, load_locomo_samples, select_stratified_subset  # noqa: E402
from locomo.seed_locomo import seed_locomo_conversation  # noqa: E402
from eval_db import add_eval_db_arguments, eval_db_metadata, setup_eval_database  # noqa: E402

RESULTS_DIR = os.path.join(EVAL_DIR, "results")
SECOM_MEMORY_MODES = ["M0", "RECURSUM", "SESSION_RET", "M2", "M3"]
SESSION_CUTOFFS = [1, 2, 4, 6, 8]
LATENCY_PROMPT = (
    "What was discussed in our previous sessions? Reply in English only."
)


def _apply_fast_mode(fast: bool) -> None:
    if fast:
        logic._invoke_summary_llm = lambda _prompt: ""  # type: ignore[attr-defined]


def _estimate_output_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _chat_latency_sse(user_id: int, mode: str, prompt: str) -> Dict[str, Any]:
    started = time.perf_counter()
    ttft_ms: Optional[float] = None
    memory_tokens: Optional[int] = None
    final_reply = ""
    error: Optional[str] = None

    for event in logic.iter_chat_sse_events(
        user_id,
        prompt,
        rag_store=None,
        force_new_session=False,
        memory_mode=mode,
    ):
        kind = event.get("event")
        data = event.get("data") or {}
        if kind == "meta":
            used = data.get("memory_used") or {}
            memory_tokens = used.get("estimated_memory_tokens")
        elif kind == "token":
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - started) * 1000, 2)
            token_text = data.get("text") or ""
            if token_text:
                final_reply += token_text
        elif kind == "done":
            done_reply = data.get("reply") or ""
            if len(done_reply) > len(final_reply):
                final_reply = done_reply
            used = data.get("memory_used") or {}
            if memory_tokens is None:
                memory_tokens = used.get("estimated_memory_tokens")
        elif kind == "error":
            error = str(data.get("detail") or data)

    total_ms = round((time.perf_counter() - started) * 1000, 2)
    if ttft_ms is None:
        ttft_ms = total_ms
    visible = logic.strip_think_tags(final_reply)
    output_tokens = _estimate_output_tokens(visible)
    decode_ms = round(max(0.0, total_ms - ttft_ms), 2)
    return {
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_ms,
        "decode_ms": decode_ms,
        "ms_per_output_token": round(decode_ms / output_tokens, 2) if output_tokens else None,
        "estimated_memory_tokens": memory_tokens,
        "estimated_output_tokens": output_tokens,
        "error": error,
    }


def _aggregate_runs(runs: List[Dict[str, Any]], mode: str, repeats: int) -> Dict[str, Any]:
    def med(key: str) -> Optional[float]:
        values = [row[key] for row in runs if row.get(key) is not None and row.get("error") is None]
        return round(statistics.median(values), 2) if values else None

    mem = [row["estimated_memory_tokens"] for row in runs if row.get("estimated_memory_tokens") is not None]
    return {
        "memory_mode": mode,
        "repeats": repeats,
        "runs": runs,
        "ttft_ms_median": med("ttft_ms"),
        "total_latency_ms_median": med("total_latency_ms"),
        "decode_ms_median": med("decode_ms"),
        "estimated_memory_tokens": round(statistics.median(mem), 1) if mem else None,
    }


def run_scenario(
    sample: Dict[str, Any],
    mode: str,
    closed_sessions: int,
    repeats: int,
    *,
    fast_summaries: bool,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for rep in range(repeats):
        seeded = seed_locomo_conversation(
            sample,
            mode,
            user_label=f"LOCOMO_LAT_{sample.get('sample_id')}_{mode}_{closed_sessions}_r{rep}",
            max_closed_sessions=closed_sessions,
            fast_summaries=fast_summaries,
        )
        runs.append(_chat_latency_sse(seeded["user_id"], mode, LATENCY_PROMPT))
    return _aggregate_runs(runs, mode, repeats)


def main() -> None:
    parser = argparse.ArgumentParser(description="LOCOMO five-mode latency eval (TTFT + total)")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for smoke tests")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--cutoffs",
        default=",".join(str(value) for value in SESSION_CUTOFFS),
    )
    add_eval_db_arguments(parser)
    args = parser.parse_args()

    if not logic.check_ollama_reachable():
        print("ERROR: Ollama not reachable.")
        sys.exit(1)

    cutoffs = [int(value.strip()) for value in args.cutoffs.split(",") if value.strip()]
    fast = args.fast or os.getenv("LOCOMO_EVAL_FAST", "true").strip().lower() in ("true", "1", "yes")
    _apply_fast_mode(fast)

    setup_eval_database(args)
    all_samples = load_locomo_samples()
    subset = select_stratified_subset(all_samples, target_n=args.samples)
    if args.max_samples is not None:
        subset = subset[: max(0, int(args.max_samples))]

    rows: List[Dict[str, Any]] = []
    for sample in subset:
        sample_id = sample.get("sample_id")
        for closed_sessions in cutoffs:
            by_mode = {
                mode: run_scenario(sample, mode, closed_sessions, args.repeats, fast_summaries=fast)
                for mode in SECOM_MEMORY_MODES
            }
            rows.append(
                {
                    "sample_id": sample_id,
                    "closed_sessions": closed_sessions,
                    "repeats": args.repeats,
                    "modes": by_mode,
                }
            )
            print(f"Latency {sample_id} closed={closed_sessions} done", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"locomo_latency_{stamp}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": stamp,
        "eval": "locomo_latency",
        "method": "A",
        "ollama_model": os.getenv("OLLAMA_MODEL", logic.get_ollama_chat_model_name()),
        "locomo_eval_fast": fast,
        "latency_prompt": LATENCY_PROMPT,
        "session_cutoffs": cutoffs,
        "modes": SECOM_MEMORY_MODES,
        "subset": get_subset_summary(all_samples, target_n=args.samples),
        "scenarios": rows,
        **eval_db_metadata(),
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
