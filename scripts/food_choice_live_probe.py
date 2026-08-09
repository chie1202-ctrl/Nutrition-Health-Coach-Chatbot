#!/usr/bin/env python3
"""Quick smoke probe for food-choice comparison (FC01, FC02, FC05 × 1 repeat)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "backend" / "eval" / "run_food_choice_eval.py"


def main() -> int:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--cases",
        "FC01_pizza_vs_stirfry,FC02_burger_vs_sushi,FC05_negative_breakfast",
        "--repeats",
        "1",
        "--rag",
        "on",
    ]
    print("Running food-choice smoke probe:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
