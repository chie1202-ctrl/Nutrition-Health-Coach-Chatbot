#!/usr/bin/env python3
"""C9 — Multi-session rollup evaluation (S04).

Seeds >=4 closed sessions with distinct facts, triggers cumulative rollup,
then measures M2 recall of rolled-up (cumulative) and recent session facts.

Requires Ollama online. RAG disabled (matches M1/M1.5 harness).

Usage:
  backend/.venv/bin/python backend/eval/run_rollup_eval.py
  EVAL_MEMORY_MODES=M2,M1 backend/.venv/bin/python backend/eval/run_rollup_eval.py
"""

from __future__ import annotations

import csv
import json
import os
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
from eval_db import add_eval_db_arguments, setup_eval_database  # noqa: E402

RESULTS_DIR = os.path.join(EVAL_DIR, "results")
SCRIPTS_PATH = os.path.join(EVAL_DIR, "rollup_scripts.json")
DEFAULT_MODES = ["M2"]
EVAL_REPLY_ENGLISH_SUFFIX = "\n\nReply in English only."


def _eval_user_prompt(prompt: str) -> str:
    if os.getenv("EVAL_REPLY_ENGLISH", "true").strip().lower() in ("false", "0", "no"):
        return prompt
    return f"{prompt}{EVAL_REPLY_ENGLISH_SUFFIX}"


def _load_scripts() -> List[Dict[str, Any]]:
    with open(SCRIPTS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _reset_eval_user(name: str) -> int:
    for user in logic.get_all_users():
        if user.get("name") == name:
            logic.delete_user(int(user["user_id"]))
    return logic.create_user_profile(
        name=name,
        gender="female",
        birth_date="1990-01-01",
        height_cm=165.0,
        initial_weight_kg=68.0,
        goal="lose_weight",
        diet_preference="balanced",
        allergies=["shellfish"],
    )


def _keyword_hits(text: str, keywords: List[str]) -> Dict[str, bool]:
    low = (text or "").lower()
    return {keyword: keyword.lower() in low for keyword in keywords}


def _keyword_pass_rate(checks: Dict[str, bool]) -> float:
    if not checks:
        return 0.0
    return round(sum(1 for ok in checks.values() if ok) / len(checks), 3)


def _summary_fidelity(summary: str, expected_facts: List[str]) -> Dict[str, Any]:
    low = (summary or "").lower()
    hits = [fact for fact in expected_facts if fact.lower() in low]
    total = len(expected_facts)
    return {
        "recall": round(len(hits) / total, 3) if total else 0.0,
        "hits": hits,
        "missed": [fact for fact in expected_facts if fact.lower() not in low],
    }


def _seed_closed_session(user_id: int, turns: List[Dict[str, str]]) -> int:
    session_id, _ = logic.resolve_session(user_id, force_new=True)
    for turn in turns:
        logic.save_chat(user_id, turn["role"], turn["content"], session_id=session_id)
    logic.close_session(session_id, user_id, trigger_summarization=False)
    if len(turns) >= 2:
        logic.summarize_session(session_id, user_id)
        logic.maybe_rollup_memory(user_id)
    return session_id


def _memory_snapshot(user_id: int) -> Dict[str, Any]:
    state = logic.get_user_memory_state(user_id)
    summaries = logic.list_user_summaries(user_id, limit=50)
    session_summaries = [s for s in summaries if s.get("summary_type") == "session"]
    archived_sessions = [s for s in session_summaries if s.get("archived")]
    active_sessions = [s for s in session_summaries if not s.get("archived")]
    rollup_records = [s for s in summaries if s.get("summary_type") == "rollup"]
    return {
        "rollup_session_threshold": logic.rollup_session_threshold(),
        "rollup_count": len(rollup_records),
        "rollup_triggered": len(rollup_records) >= 1,
        "cumulative_summary_chars": len(state.get("cumulative_summary") or ""),
        "cumulative_summary_excerpt": (state.get("cumulative_summary") or "")[:240],
        "archived_session_summary_count": len(archived_sessions),
        "active_session_summary_count": len(active_sessions),
        "total_session_summary_count": len(session_summaries),
    }


def run_script_for_mode(script: Dict[str, Any], mode: str) -> Dict[str, Any]:
    user_id = _reset_eval_user(f"Rollup_{script['script_id']}_{mode}")
    logic.ensure_user_memory_state(user_id)

    closed_sessions = script["sessions"][:-1]
    final_session = script["sessions"][-1]
    session_ids: List[int] = []
    summary_records: List[Dict[str, Any]] = []
    rollup_snapshots: List[Dict[str, Any]] = []

    for index, session in enumerate(closed_sessions):
        session_id = _seed_closed_session(user_id, session["turns"])
        session_ids.append(session_id)
        snapshot = _memory_snapshot(user_id)
        rollup_snapshots.append(
            {
                "after_session": index + 1,
                "session_label": session.get("session_label", f"session_{index + 1}"),
                **snapshot,
            }
        )
        summaries = logic.list_user_summaries(user_id, limit=10)
        latest = next((item for item in summaries if item.get("session_id") == session_id), None)
        if latest and session.get("expected_key_facts"):
            summary_records.append(
                {
                    "session_label": session.get("session_label", f"session_{index + 1}"),
                    "fidelity": _summary_fidelity(latest.get("content", ""), session["expected_key_facts"]),
                }
            )

    post_seed_snapshot = _memory_snapshot(user_id)
    logic.resolve_session(user_id, force_new=True)
    started = time.perf_counter()
    response = logic.process_chat_message(
        user_id,
        _eval_user_prompt(final_session["user_prompt"]),
        rag_store=None,
        force_new_session=False,
        memory_mode=mode,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    reply = response.get("reply") or ""
    memory_used = response.get("memory_used") or {}
    keyword_checks = _keyword_hits(reply, final_session.get("expected_keywords", []))
    cumulative_checks = _keyword_hits(reply, final_session.get("cumulative_keywords", []))
    recent_checks = _keyword_hits(reply, final_session.get("recent_keywords", []))

    min_closed = len(closed_sessions)
    rollup_ok = (
        post_seed_snapshot["rollup_triggered"]
        and post_seed_snapshot["cumulative_summary_chars"] > 0
        and min_closed >= 4
    )

    return {
        "script_id": script["script_id"],
        "memory_mode": mode,
        "user_id": user_id,
        "closed_session_count": min_closed,
        "rollup_verification": {
            "min_closed_sessions_met": min_closed >= 4,
            "rollup_triggered": post_seed_snapshot["rollup_triggered"],
            "cumulative_populated": post_seed_snapshot["cumulative_summary_chars"] > 0,
            "rollup_ok": rollup_ok,
        },
        "memory_state_after_seed": post_seed_snapshot,
        "rollup_progression": rollup_snapshots,
        "memory_used": memory_used,
        "cumulative_summary_included": bool(memory_used.get("cumulative_summary_included")),
        "recent_session_summaries_count": memory_used.get("recent_session_summaries_count"),
        "keyword_checks": keyword_checks,
        "keyword_pass_rate": _keyword_pass_rate(keyword_checks),
        "cumulative_keyword_checks": cumulative_checks,
        "cumulative_keyword_pass_rate": _keyword_pass_rate(cumulative_checks),
        "recent_keyword_checks": recent_checks,
        "recent_keyword_pass_rate": _keyword_pass_rate(recent_checks),
        "summary_fidelity": summary_records,
        "latency_ms": latency_ms,
        "reply_excerpt": reply[:320],
        "session_ids": session_ids,
    }


def write_results(rows: List[Dict[str, Any]]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"rollup_eval_{stamp}.json")
    csv_path = os.path.join(RESULTS_DIR, f"rollup_eval_{stamp}.csv")

    payload = {
        "generated_at": stamp,
        "eval": "C9_multi_session_rollup",
        "entry": "039",
        "script": "S04_multi_session_rollup",
        "rag_enabled": False,
        "ollama_model": os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
        "rollup_session_threshold": logic.rollup_session_threshold(),
        "results": rows,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "script_id",
                "memory_mode",
                "rollup_ok",
                "keyword_pass_rate",
                "cumulative_keyword_pass_rate",
                "recent_keyword_pass_rate",
                "cumulative_summary_included",
                "latency_ms",
            ],
        )
        writer.writeheader()
        for row in rows:
            verification = row.get("rollup_verification") or {}
            writer.writerow(
                {
                    "script_id": row["script_id"],
                    "memory_mode": row["memory_mode"],
                    "rollup_ok": verification.get("rollup_ok"),
                    "keyword_pass_rate": row["keyword_pass_rate"],
                    "cumulative_keyword_pass_rate": row["cumulative_keyword_pass_rate"],
                    "recent_keyword_pass_rate": row["recent_keyword_pass_rate"],
                    "cumulative_summary_included": row.get("cumulative_summary_included"),
                    "latency_ms": row["latency_ms"],
                }
            )

    return json_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-session rollup evaluation (C9)")
    add_eval_db_arguments(parser)
    args = parser.parse_args()

    if not logic.check_ollama_reachable():
        print("ERROR: Ollama not reachable. Start ./start.sh or ollama serve.")
        sys.exit(1)

    setup_eval_database(args)
    scripts = _load_scripts()
    modes = os.getenv("EVAL_MEMORY_MODES", ",".join(DEFAULT_MODES)).split(",")
    modes = [mode.strip().upper() for mode in modes if mode.strip()]

    rows: List[Dict[str, Any]] = []
    for script in scripts:
        for mode in modes:
            print(f"Running {script['script_id']} ({mode})...", flush=True)
            rows.append(run_script_for_mode(script, mode))

    output_path = write_results(rows)
    print(f"\nWrote evaluation results to {output_path}\n")
    for row in rows:
        verification = row["rollup_verification"]
        print(
            f"{row['memory_mode']}: rollup_ok={verification['rollup_ok']} "
            f"kpr={row['keyword_pass_rate']} "
            f"cumulative_kpr={row['cumulative_keyword_pass_rate']} "
            f"recent_kpr={row['recent_keyword_pass_rate']} "
            f"cumulative_in_prompt={row['cumulative_summary_included']}"
        )


if __name__ == "__main__":
    main()
