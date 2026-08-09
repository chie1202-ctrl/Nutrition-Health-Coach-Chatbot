#!/usr/bin/env python3
"""Automated delivery verification for B2/B4/B5/B6 acceptance metrics."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_CSS = ROOT / "frontend" / "src" / "styles.css"
APP_JSX = ROOT / "frontend" / "src" / "App.jsx"


def run_pytest() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT / "backend",
        capture_output=True,
        text=True,
    )
    passed = failed = 0
    match = re.search(r"(\d+) passed", result.stdout)
    if match:
        passed = int(match.group(1))
    fail_match = re.search(r"(\d+) failed", result.stdout)
    if fail_match:
        failed = int(fail_match.group(1))
    return {
        "exit_code": result.returncode,
        "passed": passed,
        "failed": failed,
        "stdout_tail": result.stdout.strip()[-400:],
    }


def run_frontend_build() -> dict:
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=ROOT / "frontend",
        capture_output=True,
        text=True,
    )
    return {
        "exit_code": result.returncode,
        "ok": result.returncode == 0,
        "stdout_tail": (result.stdout + result.stderr).strip()[-400:],
    }


def check_mobile_css() -> dict:
    css = FRONTEND_CSS.read_text(encoding="utf-8")
    jsx = APP_JSX.read_text(encoding="utf-8")
    checks = {
        "media_480px": "@media (max-width: 480px)" in css,
        "sidebar_backdrop": ".sidebar-backdrop" in css,
        "medical_disclaimer_css": ".medical-disclaimer" in css,
        "mobile_default_sidebar_closed": "isMobileViewport" in jsx,
        "source_citations_ui": "source-citations" in jsx,
        "message_row_user_padding_reset": ".message-row.user" in css and "padding-left: 0" in css,
    }
    return {"checks": checks, "all_pass": all(checks.values())}


def check_backend_logic() -> dict:
    sys.path.insert(0, str(ROOT / "backend"))
    import logic  # noqa: WPS433

    user = {
        "name": "Verify User",
        "allergies": ["shellfish"],
        "goal": "lose_weight",
    }
    offline = logic.build_offline_meal_plan(user)
    validation = logic.validate_meal_plan(offline, user)
    blob = " ".join(
        f"{day['breakfast']} {day['lunch']} {day['dinner']} {day['snack']}".lower()
        for day in offline["days"]
    )
    safety_blocked = logic.safety_check_input("I want to starve and eat nothing")
    return {
        "meal_plan_day_count": validation["day_count"],
        "meal_plan_distinct_main_meals": validation["distinct_main_meals"],
        "meal_plan_valid": validation["valid"],
        "allergen_shellfish_absent": "shellfish" not in blob,
        "safety_terms_count": len(logic.UNSAFE_HEALTH_TERMS),
        "safety_input_blocked": safety_blocked == logic.SAFETY_BLOCK_REPLY,
    }


def main() -> int:
    report = {
        "batch": "B2_B4_B5_B6_delivery_verification",
        "pytest": run_pytest(),
        "frontend_build": run_frontend_build(),
        "mobile_css": check_mobile_css(),
        "backend_metrics": check_backend_logic(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    ok = (
        report["pytest"]["exit_code"] == 0
        and report["frontend_build"]["ok"]
        and report["mobile_css"]["all_pass"]
        and report["backend_metrics"]["meal_plan_valid"]
        and report["backend_metrics"]["safety_input_blocked"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
