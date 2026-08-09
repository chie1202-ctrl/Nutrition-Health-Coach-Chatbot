#!/usr/bin/env python3
"""Batch evaluation runner for Cross-Session Memory ablation (M0–M3)."""

from __future__ import annotations

import csv
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
from eval_db import add_eval_db_arguments, setup_eval_database  # noqa: E402

SCRIPTS_PATH = os.path.join(EVAL_DIR, "scripts.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
MEMORY_MODES = ["M0", "M1", "M2", "M3"]
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


def _seed_session_turns(user_id: int, turns: List[Dict[str, str]]) -> int:
    session_id, _ = logic.resolve_session(user_id, force_new=True)
    for turn in turns:
        logic.save_chat(user_id, turn["role"], turn["content"], session_id=session_id)
    # Summarize synchronously so M1/M2 have summaries before the eval chat turn.
    # close_session() starts a background thread; 0.2s is insufficient when Ollama is online.
    logic.close_session(session_id, user_id, trigger_summarization=False)
    if len(turns) >= 2:
        logic.summarize_session(session_id, user_id)
        logic.maybe_rollup_memory(user_id)
    return session_id


def _keyword_hits(text: str, keywords: List[str]) -> Dict[str, bool]:
    low = (text or "").lower()
    return {keyword: keyword.lower() in low for keyword in keywords}


def _summary_fidelity(summary: str, expected_facts: List[str]) -> Dict[str, Any]:
    low = (summary or "").lower()
    hits = [fact for fact in expected_facts if fact.lower() in low]
    total = len(expected_facts)
    return {
        "recall": round(len(hits) / total, 3) if total else 0.0,
        "hits": hits,
        "missed": [fact for fact in expected_facts if fact.lower() not in low],
    }


def run_script_for_mode(script: Dict[str, Any], mode: str) -> Dict[str, Any]:
    user_id = _reset_eval_user(f"Eval_{script['script_id']}_{mode}")
    logic.ensure_user_memory_state(user_id)
    session_ids: List[int] = []
    summary_records: List[Dict[str, Any]] = []

    for index, session in enumerate(script["sessions"][:-1]):
        session_id = _seed_session_turns(user_id, session["turns"])
        session_ids.append(session_id)
        summaries = logic.list_user_summaries(user_id, limit=5)
        latest = next((item for item in summaries if item.get("session_id") == session_id), None)
        if latest and session.get("expected_key_facts"):
            summary_records.append(
                {
                    "session_label": session.get("session_label", f"session_{index + 1}"),
                    "fidelity": _summary_fidelity(latest.get("content", ""), session["expected_key_facts"]),
                }
            )

    final_session = script["sessions"][-1]
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
    keyword_checks = _keyword_hits(response.get("reply", ""), final_session.get("expected_keywords", []))

    return {
        "script_id": script["script_id"],
        "memory_mode": mode,
        "user_id": user_id,
        "reply_excerpt": (response.get("reply") or "")[:240],
        "memory_used": response.get("memory_used", {}),
        "keyword_checks": keyword_checks,
        "keyword_pass_rate": round(
            sum(1 for ok in keyword_checks.values() if ok) / max(1, len(keyword_checks)),
            3,
        ),
        "summary_fidelity": summary_records,
        "latency_ms": latency_ms,
        "session_ids": session_ids,
    }


def write_results(rows: List[Dict[str, Any]]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"memory_eval_{stamp}.json")
    csv_path = os.path.join(RESULTS_DIR, f"memory_eval_{stamp}.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "script_id",
                "memory_mode",
                "keyword_pass_rate",
                "estimated_memory_tokens",
                "latency_ms",
                "reply_excerpt",
            ],
        )
        writer.writeheader()
        for row in rows:
            memory_used = row.get("memory_used") or {}
            writer.writerow(
                {
                    "script_id": row["script_id"],
                    "memory_mode": row["memory_mode"],
                    "keyword_pass_rate": row["keyword_pass_rate"],
                    "estimated_memory_tokens": memory_used.get("estimated_memory_tokens"),
                    "latency_ms": row["latency_ms"],
                    "reply_excerpt": row["reply_excerpt"],
                }
            )

    return json_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cross-session memory ablation eval")
    add_eval_db_arguments(parser)
    args = parser.parse_args()

    setup_eval_database(args)
    scripts = _load_scripts()
    modes = os.getenv("EVAL_MEMORY_MODES", ",".join(MEMORY_MODES)).split(",")
    modes = [mode.strip().upper() for mode in modes if mode.strip().upper() in MEMORY_MODES]

    rows: List[Dict[str, Any]] = []
    for script in scripts:
        for mode in modes:
            print(f"Running {script['script_id']} ({mode})...")
            rows.append(run_script_for_mode(script, mode))

    output_path = write_results(rows)
    print(f"Wrote evaluation results to {output_path}")


if __name__ == "__main__":
    main()
