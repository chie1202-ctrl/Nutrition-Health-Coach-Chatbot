#!/usr/bin/env python3
"""Probe live meal-plan LLM reliability; capture raw failures for analysis."""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import logic  # noqa: E402

OUT_DIR = ROOT / "backend/eval/results"
RAW_DIR = OUT_DIR / "meal_plan_raw_failures"


def classify_failure(
    *,
    invoke_error: str | None,
    raw: str,
    stripped: str,
    extracted: str,
    parse_error: str | None,
    parsed: dict | None,
    validation: dict | None,
) -> str:
    if invoke_error:
        if "timeout" in invoke_error.lower() or "abort" in invoke_error.lower():
            return "timeout"
        return "invoke_error"
    if not raw.strip():
        return "empty_response"
    if parse_error:
        if len(extracted) < len(stripped) * 0.85 and not extracted.rstrip().endswith("}"):
            return "truncation"
        if "```" in raw:
            return "markdown_fences_or_parse"
        return "invalid_json"
    if not isinstance(parsed, dict) or not isinstance(parsed.get("days"), list):
        return "schema_mismatch"
    if validation and not validation.get("valid"):
        issues = validation.get("issues") or []
        if any("allergen" in i for i in issues):
            return "allergy_validation"
        if any("distinct" in i for i in issues):
            return "distinct_meals"
        if any("7 days" in i for i in issues):
            return "day_count"
        return "validation_other"
    return "unknown"


def build_b1_user() -> tuple[int, dict, dict | None]:
    name = f"MealProbe_{int(time.time() * 1000)}"
    user_id = logic.create_user_profile(
        name=name,
        gender="female",
        birth_date="19920618",
        height_cm=165,
        initial_weight_kg=64,
        goal="lose_weight",
        allergies=["shellfish"],
    )
    logic.add_new_weight_entry(user_id, 61.8)
    user = logic.get_user_profile(user_id)
    latest = logic.get_latest_metrics_bundle(user_id)
    return user_id, user, latest


def probe_attempt(user: dict, latest: dict | None, attempt: int) -> dict:
    t0 = time.perf_counter()
    try:
        plan, llm_degraded = logic.generate_meal_plan(user, latest, rag_store=None)
        validation = logic.validate_meal_plan(plan, user)
        live_ok = (not llm_degraded) and bool(validation.get("valid"))
        invoke_error = None
        failure_class = "success" if live_ok else ("template_fallback" if llm_degraded else "validation")
        raw_len = 0
        parse_error = None
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "attempt": attempt,
            "elapsed_ms": elapsed_ms,
            "live_ok": False,
            "failure_class": "invoke_error",
            "invoke_error": str(exc),
        }

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "attempt": attempt,
        "elapsed_ms": elapsed_ms,
        "live_ok": live_ok,
        "failure_class": failure_class,
        "invoke_error": invoke_error,
        "llm_degraded": llm_degraded,
        "validation": validation,
        "summary_start": str(plan.get("summary", ""))[:120],
        "is_template": "template fallback" in str(plan.get("summary", "")).lower(),
    }


def main() -> int:
    logic.assert_ollama_ready()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    user_id, user, latest = build_b1_user()
    attempts = []
    for i in range(1, n + 1):
        print(f"Attempt {i}/{n}...", flush=True)
        attempts.append(probe_attempt(user, latest, i))

    success = sum(1 for a in attempts if a["live_ok"])
    by_class: dict[str, int] = {}
    for a in attempts:
        by_class[a["failure_class"]] = by_class.get(a["failure_class"], 0) + 1

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "meal_plan_num_predict": logic.get_meal_plan_num_predict(),
        "meal_plan_temperature": logic.get_meal_plan_temperature(),
        "attempts": n,
        "success_count": success,
        "success_rate": round(success / n, 3) if n else 0,
        "failure_classes": by_class,
        "trials": attempts,
    }
    out = OUT_DIR / "meal_plan_reliability_probe.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"success": success, "attempts": n, "rate": report["success_rate"], "classes": by_class}, indent=2))
    print(f"WROTE {out}")
    logic.delete_user(user_id)
    return 0 if success == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
