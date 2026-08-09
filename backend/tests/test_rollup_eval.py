"""Unit tests for C9 rollup eval seeding (no Ollama chat required)."""

from __future__ import annotations

import json
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BACKEND_DIR, "eval")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import logic  # noqa: E402
import run_rollup_eval as rollup_eval  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    db_path = tmp_path / "rollup_eval_test.db"
    monkeypatch.setattr(logic, "DB_PATH", str(db_path))
    logic.init_db()
    monkeypatch.setattr(logic, "_invoke_summary_llm", lambda _prompt: "")
    yield


def test_s04_triggers_rollup_after_four_closes():
    with open(os.path.join(EVAL_DIR, "rollup_scripts.json"), "r", encoding="utf-8") as handle:
        scripts = json.load(handle)
    script = next(item for item in scripts if item["script_id"] == "S04_multi_session_rollup")

    user_id = rollup_eval._reset_eval_user("Rollup_S04_unit")
    logic.ensure_user_memory_state(user_id)

    for session in script["sessions"][:-1]:
        rollup_eval._seed_closed_session(user_id, session["turns"])

    snapshot = rollup_eval._memory_snapshot(user_id)
    assert snapshot["rollup_triggered"] is True
    assert snapshot["cumulative_summary_chars"] > 0
    assert snapshot["archived_session_summary_count"] >= 2
    assert snapshot["active_session_summary_count"] >= 2


def test_rollup_scripts_schema():
    with open(os.path.join(EVAL_DIR, "rollup_scripts.json"), "r", encoding="utf-8") as handle:
        scripts = json.load(handle)
    script = scripts[0]
    closed = script["sessions"][:-1]
    recall = script["sessions"][-1]
    assert script["script_id"] == "S04_multi_session_rollup"
    assert len(closed) >= 4
    assert recall.get("user_prompt")
    assert recall.get("expected_keywords")
    assert recall.get("cumulative_keywords")
    assert recall.get("recent_keywords")
