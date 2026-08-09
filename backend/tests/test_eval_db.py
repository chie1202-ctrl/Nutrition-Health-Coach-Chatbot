"""Tests for isolated evaluation database configuration."""

from __future__ import annotations

import argparse
import os

import pytest


def test_configure_eval_database_uses_isolated_path(tmp_path, monkeypatch):
    import logic

    eval_db = tmp_path / "isolated_eval.db"
    monkeypatch.delenv("EVAL_DB_PATH", raising=False)
    monkeypatch.delenv("DB_PATH", raising=False)

    path = logic.configure_eval_database(str(eval_db), fresh=True, init=True)
    assert path == str(eval_db.resolve())
    assert logic.get_db_path() == path
    assert eval_db.exists()
    assert os.path.abspath(path) != os.path.abspath(logic.DEFAULT_DB_PATH)

    user_id = logic.create_user_profile(
        name="EVAL_ONLY_USER",
        gender="female",
        birth_date="1990-01-01",
        height_cm=165.0,
        initial_weight_kg=68.0,
        goal="lose_weight",
        diet_preference="balanced",
        allergies=[],
    )
    assert user_id > 0
    names = {u["name"] for u in logic.get_all_users()}
    assert "EVAL_ONLY_USER" in names


def test_configure_eval_database_refuses_production_path(monkeypatch):
    import logic

    monkeypatch.delenv("EVAL_DB_PATH", raising=False)
    with pytest.raises(ValueError, match="production database"):
        logic.configure_eval_database(logic.DEFAULT_DB_PATH, fresh=False, init=False)


def test_setup_eval_database_helper(tmp_path, monkeypatch):
    import sys

    import logic

    eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from eval_db import eval_db_metadata, setup_eval_database

    monkeypatch.delenv("EVAL_DB_PATH", raising=False)
    eval_db = tmp_path / "helper_eval.db"
    args = argparse.Namespace(eval_db=str(eval_db), fresh_eval_db=True, use_main_db=False)
    path = setup_eval_database(args)
    assert path == str(eval_db.resolve())
    meta = eval_db_metadata()
    assert meta["isolated_eval_db"] is True
    assert meta["db_path"] == path
    assert logic.get_db_path() == path
