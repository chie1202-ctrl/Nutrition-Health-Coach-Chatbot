#!/usr/bin/env python3
"""Batch evaluation runner for food-choice comparison (O8 / Entry 038).

Does NOT require the HTTP server — calls logic.process_chat_message() directly.
Requires: Ollama running (logic.assert_ollama_ready).

For Playwright UI validation, use ./scripts/run_entry_038_food_choice_ui.sh instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import logic  # noqa: E402
from eval_db import add_eval_db_arguments, eval_db_metadata, setup_eval_database  # noqa: E402

SCRIPTS_PATH = os.path.join(EVAL_DIR, "food_choice_scripts.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def _load_scripts(case_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    with open(SCRIPTS_PATH, "r", encoding="utf-8") as handle:
        scripts = json.load(handle)
    if not case_filter:
        return scripts
    wanted = {case_id.strip() for case_id in case_filter}
    return [script for script in scripts if script["case_id"] in wanted]


def _create_eval_user(case_id: str, profile: Dict[str, Any]) -> int:
    stamp = int(time.time() * 1000)
    name = f"FCEval_{case_id}_{stamp}"
    for user in logic.get_all_users():
        if str(user.get("name", "")).startswith(f"FCEval_{case_id}_"):
            logic.delete_user(int(user["user_id"]))
    kwargs = dict(profile)
    kwargs.pop("name", None)
    user_id = logic.create_user_profile(
        name=name,
        gender=kwargs.pop("gender", "female"),
        birth_date=kwargs.pop("birth_date", "19900101"),
        height_cm=kwargs.pop("height_cm", 165),
        **kwargs,
    )
    initial_weight = profile.get("initial_weight_kg")
    if initial_weight is not None:
        logic.add_new_weight_entry(user_id, float(initial_weight))
    return user_id


def _profile_checks_pass(
    validation: Dict[str, Any],
    comparison: Dict[str, Any],
    checks: Dict[str, Any],
) -> Dict[str, Any]:
    scores = validation.get("scores") or {}
    recommendation = str(comparison.get("recommendation") or "").lower()
    results: Dict[str, Any] = {}

    min_dims = int(checks.get("min_dims_filled", 0) or 0)
    if min_dims:
        results["min_dims_filled"] = scores.get("dims_filled", 0) >= min_dims
    if checks.get("require_allergy_note"):
        results["allergy_note"] = bool(scores.get("allergy_note_present"))
    if checks.get("require_diabetes_signal"):
        results["diabetes_signal"] = bool(scores.get("diabetes_signal"))
    forbidden = checks.get("forbidden_in_recommendation") or []
    if forbidden:
        hits = [token for token in forbidden if token.lower() in recommendation]
        results["forbidden_in_recommendation"] = not hits
        results["forbidden_hits"] = hits
    return results


def _evaluate_run(
    script: Dict[str, Any],
    *,
    repeat: int,
    rag_store,
    rag_enabled: bool,
) -> Dict[str, Any]:
    case_id = script["case_id"]
    user_id = _create_eval_user(case_id, script.get("user_profile") or {})
    user = logic.get_user_profile(user_id)
    prompt = script["prompt"]
    expect_trigger = bool(script.get("expect_trigger", True))
    expect_sources = bool(script.get("expect_sources", False))
    checks = script.get("checks") or {}

    intent_detected = logic.detect_food_choice_intent(prompt)
    t0 = time.perf_counter()
    error = None
    response: Dict[str, Any] = {}
    try:
        response = logic.process_chat_message(user_id, prompt, rag_store=rag_store if rag_enabled else None)
    except Exception as exc:
        error = str(exc)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    food_choice = response.get("food_choice")
    triggered = bool(food_choice)
    sources = response.get("sources") or []
    validation: Dict[str, Any] = {}
    profile_check_results: Dict[str, Any] = {}
    if food_choice:
        validation = logic.validate_food_choice(food_choice, user)
        profile_check_results = _profile_checks_pass(validation, food_choice, checks)

    routing_ok = triggered == expect_trigger
    structure_ok = True
    if triggered:
        scores = validation.get("scores") or {}
        min_dims = int(checks.get("min_dims_filled", 0) or 0)
        structure_ok = (
            scores.get("dims_filled", 0) >= min_dims
            and bool(scores.get("has_recommendation"))
        )
    profile_ok = (
        all(value for key, value in profile_check_results.items() if key != "forbidden_hits")
        if profile_check_results
        else True
    )
    sources_ok = (len(sources) >= 1) if expect_sources else True
    case_pass = routing_ok and (not triggered or (structure_ok and profile_ok and sources_ok))

    run = {
        "case_id": case_id,
        "repeat": repeat,
        "user_id": user_id,
        "prompt": prompt,
        "intent_detected": intent_detected,
        "expect_trigger": expect_trigger,
        "food_choice_triggered": triggered,
        "latency_ms": latency_ms,
        "latency_s": round(latency_ms / 1000, 2),
        "error": error,
        "sources": sources,
        "routing_ok": routing_ok,
        "structure_ok": structure_ok,
        "profile_ok": profile_ok,
        "sources_ok": sources_ok,
        "case_pass": case_pass and error is None,
        "validation": validation,
        "profile_checks": profile_check_results,
        "reply_preview": (response.get("reply") or "")[:240],
        "food_choice": food_choice,
    }
    logic.delete_user(user_id)
    return run


def _summarize_case(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return {}
    case_id = runs[0]["case_id"]
    pass_count = sum(1 for run in runs if run.get("case_pass"))
    routing_count = sum(1 for run in runs if run.get("routing_ok"))
    structure_runs = [run for run in runs if run.get("food_choice_triggered")]
    structure_count = sum(1 for run in structure_runs if run.get("structure_ok"))
    allergy_safe_runs = [
        run for run in structure_runs if (run.get("validation") or {}).get("scores", {}).get("allergy_safe") is not None
    ]
    allergy_safe_count = sum(
        1 for run in allergy_safe_runs if (run.get("validation") or {}).get("scores", {}).get("allergy_safe")
    )
    dims = [
        (run.get("validation") or {}).get("scores", {}).get("dims_filled", 0)
        for run in structure_runs
    ]
    latencies = [run.get("latency_s", 0) for run in runs]
    return {
        "case_id": case_id,
        "runs": len(runs),
        "pass_rate": round(pass_count / len(runs), 3),
        "routing_accuracy": round(routing_count / len(runs), 3),
        "structure_pass_rate": round(structure_count / max(1, len(structure_runs)), 3) if structure_runs else None,
        "allergy_safe_rate": round(allergy_safe_count / max(1, len(allergy_safe_runs)), 3) if allergy_safe_runs else None,
        "avg_dims_filled": round(sum(dims) / max(1, len(dims)), 2) if dims else 0,
        "mean_latency_s": round(sum(latencies) / len(latencies), 2),
    }


def _load_baseline_comparison() -> Dict[str, Any]:
    baseline_path = os.path.join(RESULTS_DIR, "food_choice_live_probe_20260630_143344.json")
    if not os.path.isfile(baseline_path):
        return {}
    with open(baseline_path, "r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    runs = baseline.get("runs") or []
    triggered = [run for run in runs if run.get("food_choice_triggered")]
    return {
        "artifact": baseline_path,
        "timestamp": baseline.get("timestamp"),
        "routing_accuracy": round(
            sum(1 for run in runs if run.get("intent_detected") == run.get("food_choice_triggered", False) or (
                run.get("intent_detected") and not run.get("food_choice_triggered")
            )) / max(1, len(runs)),
            3,
        ),
        "triggered_runs": len(triggered),
        "avg_dims_filled": round(
            sum((run.get("score") or {}).get("comparison_dims_filled", 0) for run in triggered) / max(1, len(triggered)),
            2,
        ),
        "notes": [
            "Ad-hoc probe n=3; sodium/glycemic sometimes empty",
            "P2 recommended sushi despite shellfish allergy (note-only warning)",
        ],
    }


def write_results(report: Dict[str, Any]) -> tuple[str, str]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"food_choice_eval_{stamp}.json")
    csv_path = os.path.join(RESULTS_DIR, f"food_choice_eval_{stamp}.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "repeat",
                "case_pass",
                "routing_ok",
                "structure_ok",
                "profile_ok",
                "sources_ok",
                "food_choice_triggered",
                "dims_filled",
                "allergy_safe",
                "latency_s",
                "sources_count",
                "error",
            ],
        )
        writer.writeheader()
        for run in report.get("runs", []):
            scores = (run.get("validation") or {}).get("scores") or {}
            writer.writerow(
                {
                    "case_id": run["case_id"],
                    "repeat": run["repeat"],
                    "case_pass": run.get("case_pass"),
                    "routing_ok": run.get("routing_ok"),
                    "structure_ok": run.get("structure_ok"),
                    "profile_ok": run.get("profile_ok"),
                    "sources_ok": run.get("sources_ok"),
                    "food_choice_triggered": run.get("food_choice_triggered"),
                    "dims_filled": scores.get("dims_filled"),
                    "allergy_safe": scores.get("allergy_safe"),
                    "latency_s": run.get("latency_s"),
                    "sources_count": len(run.get("sources") or []),
                    "error": run.get("error") or "",
                }
            )

    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run food-choice comparison evaluation harness.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per case (default: 3)")
    parser.add_argument("--rag", choices=("on", "off"), default="on", help="RAG on/off (default: on)")
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Comma-separated case IDs (e.g. FC01_pizza_vs_stirfry,FC02_burger_vs_sushi)",
    )
    add_eval_db_arguments(parser)
    args = parser.parse_args()

    case_filter = [part.strip() for part in args.cases.split(",") if part.strip()] or None
    scripts = _load_scripts(case_filter)
    if not scripts:
        print("No matching food-choice scripts found.", file=sys.stderr)
        return 1

    setup_eval_database(args)
    logic.assert_ollama_ready()
    rag_enabled = args.rag == "on"
    rag_store = None
    rag_init_s = None
    if rag_enabled:
        t0 = time.perf_counter()
        rag_store = logic.initialize_rag()
        rag_init_s = round(time.perf_counter() - t0, 2)

    all_runs: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for script in scripts:
        case_runs: List[Dict[str, Any]] = []
        for repeat in range(1, args.repeats + 1):
            print(f"Running {script['case_id']} repeat {repeat}/{args.repeats}...", flush=True)
            case_runs.append(
                _evaluate_run(script, repeat=repeat, rag_store=rag_store, rag_enabled=rag_enabled)
            )
        all_runs.extend(case_runs)
        summaries.append(_summarize_case(case_runs))

    total_pass = sum(1 for run in all_runs if run.get("case_pass"))
    routing_pass = sum(1 for run in all_runs if run.get("routing_ok"))
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ollama_reachable": logic.check_ollama_reachable(),
        "rag_enabled": rag_enabled,
        "rag_init_s": rag_init_s,
        "repeats_per_case": args.repeats,
        "cases_run": [script["case_id"] for script in scripts],
        "baseline_comparison": _load_baseline_comparison(),
        **eval_db_metadata(),
        "summary": {
            "total_runs": len(all_runs),
            "case_pass_rate": round(total_pass / max(1, len(all_runs)), 3),
            "routing_accuracy": round(routing_pass / max(1, len(all_runs)), 3),
            "per_case": summaries,
        },
        "runs": all_runs,
    }
    json_path, csv_path = write_results(report)
    print(f"\nWROTE {json_path}")
    print(f"WROTE {csv_path}")
    print(
        f"PASS {total_pass}/{len(all_runs)} | routing {routing_pass}/{len(all_runs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
