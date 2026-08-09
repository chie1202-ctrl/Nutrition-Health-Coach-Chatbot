"""Unit tests for persona QA eval retry / empty-reply honesty."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_persona_qa.py"


def _load_persona_qa():
    spec = importlib.util.spec_from_file_location("run_persona_qa", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chat_with_retries_marks_empty_reply_as_failure():
    mod = _load_persona_qa()

    def fake_chat(user_id, message, force_new_session=False):
        return {"reply": "", "sources": ["NHS.pdf"], "safety_blocked": False}, 1.2

    with mock.patch.object(mod, "chat", side_effect=fake_chat):
        outcome = mod.chat_with_retries(1, "How many calories?", max_attempts=2)

    assert outcome["ok"] is False
    assert outcome["error"] == "empty_reply"
    assert outcome["attempt_count"] == 2
    assert all(a.get("error") == "empty_reply" for a in outcome["attempts"])


def test_chat_with_retries_recovers_on_second_attempt():
    mod = _load_persona_qa()
    calls = {"n": 0}

    def fake_chat(user_id, message, force_new_session=False):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"reply": "   ", "sources": []}, 0.5
        return {"reply": "Aim for 1800 kcal.", "sources": ["x.pdf"], "session_id": 9}, 0.8

    with mock.patch.object(mod, "chat", side_effect=fake_chat):
        outcome = mod.chat_with_retries(1, "calories?", max_attempts=2)

    assert outcome["ok"] is True
    assert outcome["attempt_count"] == 2
    assert "1800" in outcome["reply"]
    assert outcome["error"] is None
