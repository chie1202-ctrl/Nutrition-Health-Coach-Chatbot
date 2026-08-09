#!/usr/bin/env python3
"""Long-context M2 vs M3 latency benchmark (production 8B path).

Hardened design (2026-07-31):
  - Default n=7 measured repeats (local LLM TTFT is high-variance)
  - Per-condition warm-up (1 discarded chat per mode × scenario)
  - Alternating M2/M3 order across repeats
  - Client TTFT (first streamed token) + Ollama raw timing metrics
  - Report median + IQR (not median alone)
  - Fixed num_predict (default 768 via OLLAMA_NUM_PREDICT)

Requires Ollama online. RAG disabled to isolate memory modes.

Usage:
  backend/.venv/bin/python backend/eval/run_long_context_latency_eval.py
  backend/.venv/bin/python backend/eval/run_long_context_latency_eval.py --repeats 7 --sessions 1,4,8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

# Latency seeding uses fast summaries by default; token axis uses live summaries separately.
if os.getenv("SEED_EVAL_FAST", "true").strip().lower() in ("true", "1", "yes"):
    logic._invoke_summary_llm = lambda _prompt: ""  # type: ignore[attr-defined]


def _estimate_output_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _ns_to_ms(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value) / 1_000_000.0, 2)
    except (TypeError, ValueError):
        return None


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        raise ValueError("empty")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _iqr(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    ordered = sorted(values)
    return round(_percentile(ordered, 0.75) - _percentile(ordered, 0.25), 2)


def _ollama_num_ctx() -> int:
    try:
        return max(2048, int(os.getenv("OLLAMA_NUM_CTX", "16384")))
    except (TypeError, ValueError):
        return 16384


def _unload_ollama_model() -> None:
    """Drop model weights/KV cache so the next run is not a prompt-cache hit."""
    base = logic.get_ollama_base_url().rstrip("/")
    model = logic.get_ollama_chat_model_name()
    payload = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    time.sleep(0.5)


def _stream_ollama_generate(prompt: str) -> Dict[str, Any]:
    """Stream /api/generate; return client TTFT + Ollama raw timing fields."""
    base = logic.get_ollama_base_url().rstrip("/")
    model = logic.get_ollama_chat_model_name()
    num_predict = logic.get_ollama_num_predict()
    temperature = logic.get_ollama_temperature()
    reasoning = logic.get_ollama_reasoning()
    num_ctx = _ollama_num_ctx()

    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    if reasoning is not None:
        body["think"] = bool(reasoning)

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    ttft_ms: Optional[float] = None
    chunks: List[str] = []
    metrics: Dict[str, Any] = {}
    error: Optional[str] = None

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                try:
                    event = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                text = event.get("response") or ""
                if text:
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started) * 1000, 2)
                    chunks.append(text)
                if event.get("done"):
                    metrics = {
                        "prompt_eval_count": event.get("prompt_eval_count"),
                        "eval_count": event.get("eval_count"),
                        "total_duration_ms": _ns_to_ms(event.get("total_duration")),
                        "load_duration_ms": _ns_to_ms(event.get("load_duration")),
                        "prompt_eval_duration_ms": _ns_to_ms(event.get("prompt_eval_duration")),
                        "eval_duration_ms": _ns_to_ms(event.get("eval_duration")),
                    }
                    break
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = str(exc)

    total_ms = round((time.perf_counter() - started) * 1000, 2)
    if ttft_ms is None:
        ttft_ms = total_ms

    return {
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_ms,
        "raw_reply": "".join(chunks),
        "ollama": metrics,
        "error": error,
    }


def _chat_latency(user_id: int, mode: str) -> Dict[str, Any]:
    """Measure latency using production prompt + Ollama raw /api/generate stream."""
    blocked, turn_context = logic.prepare_chat_turn(
        user_id,
        lcs.RECALL_PROMPT,
        rag_store=None,
        force_new_session=False,
        memory_mode=mode,
    )
    if blocked:
        return {
            "ttft_ms": None,
            "total_latency_ms": None,
            "error": "safety_blocked_or_prepare_failed",
        }

    memory_context = turn_context["memory_context"]
    memory_text = memory_context.get("memory_text") or ""
    memory_used = memory_context.get("memory_used") or {}
    prompt = turn_context["prompt"]

    logic.save_chat(user_id, "user", lcs.RECALL_PROMPT, session_id=turn_context["session_id"])
    stream = _stream_ollama_generate(prompt)

    raw_reply = stream.get("raw_reply") or ""
    visible = logic.strip_think_tags(raw_reply)
    final_reply = logic.safety_filter(visible)
    logic.save_chat(user_id, "assistant", final_reply, session_id=turn_context["session_id"])

    ollama = stream.get("ollama") or {}
    ttft_ms = stream.get("ttft_ms")
    total_ms = stream.get("total_latency_ms")
    prompt_eval_ms = ollama.get("prompt_eval_duration_ms")
    eval_ms = ollama.get("eval_duration_ms")
    output_tokens = ollama.get("eval_count") or _estimate_output_tokens(visible)
    decode_ms = eval_ms if eval_ms is not None else round(max(0.0, (total_ms or 0) - (ttft_ms or 0)), 2)

    return {
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_ms,
        "decode_ms": decode_ms,
        "prompt_eval_duration_ms": prompt_eval_ms,
        "eval_duration_ms": eval_ms,
        "load_duration_ms": ollama.get("load_duration_ms"),
        "ollama_total_duration_ms": ollama.get("total_duration_ms"),
        "prompt_eval_count": ollama.get("prompt_eval_count"),
        "eval_count": ollama.get("eval_count"),
        "ms_per_output_token": round(decode_ms / output_tokens, 2) if output_tokens else None,
        "prompt_chars": len(prompt),
        "memory_chars": len(memory_text),
        "estimated_memory_tokens": memory_used.get("estimated_memory_tokens"),
        "reply_chars": len(visible),
        "estimated_output_tokens": output_tokens,
        "ollama_num_predict": logic.get_ollama_num_predict(),
        "error": stream.get("error"),
    }


def _aggregate_runs(runs: List[Dict[str, Any]], mode: str, repeats: int) -> Dict[str, Any]:
    def values(key: str) -> List[float]:
        out: List[float] = []
        for row in runs:
            if row.get("error"):
                continue
            val = row.get(key)
            if val is not None:
                out.append(float(val))
        return out

    def med(key: str) -> Optional[float]:
        vals = values(key)
        return round(statistics.median(vals), 2) if vals else None

    def mean(key: str) -> Optional[float]:
        vals = values(key)
        return round(statistics.mean(vals), 2) if vals else None

    mem = values("estimated_memory_tokens")
    prompt_chars = values("prompt_chars")
    memory_chars = values("memory_chars")

    return {
        "memory_mode": mode,
        "repeats": repeats,
        "runs": runs,
        "ttft_ms_median": med("ttft_ms"),
        "ttft_ms_iqr": _iqr(values("ttft_ms")),
        "ttft_ms_mean": mean("ttft_ms"),
        "prompt_eval_duration_ms_median": med("prompt_eval_duration_ms"),
        "prompt_eval_duration_ms_iqr": _iqr(values("prompt_eval_duration_ms")),
        "total_latency_ms_median": med("total_latency_ms"),
        "total_latency_ms_iqr": _iqr(values("total_latency_ms")),
        "ollama_total_duration_ms_median": med("ollama_total_duration_ms"),
        "ollama_total_duration_ms_iqr": _iqr(values("ollama_total_duration_ms")),
        "decode_ms_median": med("decode_ms"),
        "ms_per_output_token_median": med("ms_per_output_token"),
        "prompt_chars_median": med("prompt_chars"),
        "memory_chars_median": med("memory_chars"),
        "prompt_eval_count_median": med("prompt_eval_count"),
        "estimated_memory_tokens": round(statistics.median(mem), 1) if mem else None,
        "prompt_chars": round(statistics.median(prompt_chars), 1) if prompt_chars else None,
        "memory_chars": round(statistics.median(memory_chars), 1) if memory_chars else None,
        "latency_ms_median": med("total_latency_ms"),
        "latency_ms_mean": mean("total_latency_ms"),
    }


def _parse_sessions(raw: Optional[str]) -> List[Tuple[int, int]]:
    if not raw:
        return list(lcs.DEFAULT_SCENARIOS)
    wanted = {int(part.strip()) for part in raw.split(",") if part.strip()}
    selected = [(s, t) for s, t in lcs.DEFAULT_SCENARIOS if s in wanted]
    if not selected:
        raise SystemExit(f"No matching scenarios for --sessions {raw}")
    return selected


def run_scenario(
    num_closed_sessions: int,
    turns_per_session: int,
    repeats: int,
    *,
    warmup_per_condition: bool = True,
) -> Dict[str, Any]:
    """Warm up each mode once, then alternate M2/M3 across measured repeats."""
    if warmup_per_condition:
        for mode in MODES:
            print(f"  warmup {mode} ({num_closed_sessions}s)...", flush=True)
            warm = lcs.seed_scenario(
                num_closed_sessions,
                turns_per_session,
                label_prefix=f"LongLat_warm_{mode}",
                unique_salt=f"warm-{mode}-{uuid.uuid4().hex[:8]}",
            )
            _ = _chat_latency(warm["user_id"], mode)

    runs_by_mode: Dict[str, List[Dict[str, Any]]] = {"M2": [], "M3": []}
    for rep in range(repeats):
        order = ["M2", "M3"] if rep % 2 == 0 else ["M3", "M2"]
        for mode in order:
            print(f"  measure r{rep + 1}/{repeats} {mode}...", flush=True)
            seeded = lcs.seed_scenario(
                num_closed_sessions,
                turns_per_session,
                label_prefix=f"LongLat_{mode}_r{rep}",
                unique_salt=f"{mode}-r{rep}-{uuid.uuid4().hex[:8]}",
            )
            runs_by_mode[mode].append(_chat_latency(seeded["user_id"], mode))

    by_mode = {mode: _aggregate_runs(runs_by_mode[mode], mode, repeats) for mode in MODES}

    def _pref(mode: str) -> float:
        row = by_mode[mode]
        return float(
            row.get("prompt_eval_duration_ms_median")
            or row.get("ttft_ms_median")
            or 0
        )

    m2_prefill = _pref("M2")
    m3_prefill = _pref("M3")
    m2_ttft = by_mode["M2"]["ttft_ms_median"] or 0
    m3_ttft = by_mode["M3"]["ttft_ms_median"] or 0
    m2_total = by_mode["M2"]["total_latency_ms_median"] or 0
    m3_total = by_mode["M3"]["total_latency_ms_median"] or 0

    return {
        "scenario": {
            "closed_sessions": num_closed_sessions,
            "turns_per_closed_session": turns_per_session,
        },
        "warmup_per_condition": warmup_per_condition,
        "M2": by_mode["M2"],
        "M3": by_mode["M3"],
        "M2_prefill_lt_M3": m2_prefill < m3_prefill,
        "prefill_delta_M3_minus_M2_ms": round(m3_prefill - m2_prefill, 2),
        "M2_ttft_lt_M3": m2_ttft < m3_ttft,
        "ttft_delta_M3_minus_M2_ms": round(m3_ttft - m2_ttft, 2),
        "M2_total_latency_lt_M3": m2_total < m3_total,
        "total_latency_delta_M3_minus_M2_ms": round(m3_total - m2_total, 2),
        "M2_latency_lt_M3": m2_total < m3_total,
        "latency_delta_M3_minus_M2_ms": round(m3_total - m2_total, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-context M2 vs M3 latency eval (TTFT + Ollama metrics)")
    parser.add_argument("--repeats", type=int, default=7, help="Measured chat repeats per mode per scenario")
    parser.add_argument(
        "--sessions",
        type=str,
        default="",
        help="Comma-separated closed-session counts (default: all DEFAULT_SCENARIOS)",
    )
    parser.add_argument(
        "--no-warmup-per-condition",
        action="store_true",
        help="Skip per-condition warm-up (not recommended)",
    )
    add_eval_db_arguments(parser)
    args = parser.parse_args()

    os.environ.setdefault("MEMORY_BUDGET_ENABLED", "false")
    os.environ.setdefault("OLLAMA_REASONING", "false")

    if not logic.check_ollama_reachable():
        print("ERROR: Ollama not reachable. Start ./start.sh or ollama serve.")
        sys.exit(1)

    setup_eval_database(args)
    scenarios = _parse_sessions(args.sessions or None)
    warmup = not args.no_warmup_per_condition

    rows: List[Dict[str, Any]] = []
    for num_sessions, turns in scenarios:
        print(f"Scenario: {num_sessions}s x {turns}t (repeats={args.repeats}, warmup={warmup})...", flush=True)
        rows.append(
            run_scenario(
                num_sessions,
                turns,
                args.repeats,
                warmup_per_condition=warmup,
            )
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"long_context_latency_eval_{stamp}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": stamp,
        "eval": "long_context_latency",
        "design": "component_caps_no_global_budget_hardened_v3",
        "methodology": (
            "Ollama /api/generate stream after prepare_chat_turn; "
            "per-condition warm-up discarded; alternating M2/M3; "
            "median+IQR; client TTFT + prompt_eval_duration; "
            "unique seed salts to avoid Ollama prompt-cache hits; "
            f"num_ctx={_ollama_num_ctx()}"
        ),
        "repeats_per_mode": args.repeats,
        "warmup_per_condition": warmup,
        "recall_prompt": lcs.RECALL_PROMPT,
        "rag_enabled": False,
        "memory_budget_enabled": logic.memory_budget_enabled(),
        "ollama_model": os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
        "ollama_reasoning": os.getenv("OLLAMA_REASONING", "false"),
        "ollama_num_predict": logic.get_ollama_num_predict(),
        "ollama_num_ctx": _ollama_num_ctx(),
        "cache_busting": "unique_seed_salt",
        **eval_db_metadata(),
        "note": (
            "Prefer prompt_eval_duration_ms_median for prefill comparison; "
            "report ttft_ms_median with IQR. High IQR => descriptive only. "
            "Flag invalid if prompt_eval_count collapses vs longer prompts."
        ),
        "scenarios": rows,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"\nWrote {out_path}\n")

    def _fmt(val: Any, width: int = 9) -> str:
        if val is None:
            return f"{'—':>{width}}"
        return f"{val:>{width}}"

    header = (
        f"{'sessions':>8} {'M2 pref':>9} {'M3 pref':>9} {'M2 TTFT':>9} {'M3 TTFT':>9} "
        f"{'M2 IQR':>8} {'M3 IQR':>8} {'pref M2<M3':>11}"
    )
    print(header)
    for row in rows:
        sc = row["scenario"]
        m2 = row["M2"]
        m3 = row["M3"]
        m2p = m2.get("prompt_eval_duration_ms_median") or m2.get("ttft_ms_median")
        m3p = m3.get("prompt_eval_duration_ms_median") or m3.get("ttft_ms_median")
        win = "yes" if row["M2_prefill_lt_M3"] else "no"
        print(
            f"{sc['closed_sessions']:>8} {_fmt(m2p)} {_fmt(m3p)} "
            f"{_fmt(m2.get('ttft_ms_median'))} {_fmt(m3.get('ttft_ms_median'))} "
            f"{_fmt(m2.get('ttft_ms_iqr'), 8)} {_fmt(m3.get('ttft_ms_iqr'), 8)} {win:>11}"
        )


if __name__ == "__main__":
    main()
