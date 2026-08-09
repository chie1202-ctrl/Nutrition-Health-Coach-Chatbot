#!/usr/bin/env python3
"""Isolated SQLite helpers for evaluation runners.

Eval runners should call ``setup_eval_database(args)`` before ``logic.init_db()``
(or let this helper call init). By default they use:

  backend/database/eval/health_coach_eval.db

via ``EVAL_DB_PATH`` / ``--eval-db``, not the production ``health_coach.db``.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import logic


def add_eval_db_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eval-db",
        type=str,
        default="",
        help="Isolated SQLite path for this eval (overrides EVAL_DB_PATH)",
    )
    parser.add_argument(
        "--fresh-eval-db",
        action="store_true",
        help="Delete the eval DB file before init (clean slate)",
    )
    parser.add_argument(
        "--use-main-db",
        action="store_true",
        help="Use production health_coach.db (not recommended; pollutes app data)",
    )


def setup_eval_database(args: Any = None, **kwargs: Any) -> str:
    """Configure logic.DB_PATH for an eval run and initialize schema.

    Returns the absolute DB path in use.
    """
    use_main = bool(getattr(args, "use_main_db", False) if args is not None else kwargs.get("use_main_db", False))
    fresh = bool(getattr(args, "fresh_eval_db", False) if args is not None else kwargs.get("fresh", False))
    explicit = ""
    if args is not None:
        explicit = (getattr(args, "eval_db", None) or "").strip()
    explicit = explicit or (kwargs.get("path") or "").strip()

    if use_main:
        if explicit or (os.getenv("EVAL_DB_PATH") or "").strip():
            print(
                "WARNING: --use-main-db ignores --eval-db / EVAL_DB_PATH; "
                f"using {logic.DEFAULT_DB_PATH}",
                flush=True,
            )
        logic.set_db_path(logic.DEFAULT_DB_PATH)
        logic.init_db()
        print(f"Eval DB: {logic.get_db_path()} (PRODUCTION / main)", flush=True)
        return logic.get_db_path()

    path = logic.configure_eval_database(explicit or None, fresh=fresh, init=True)
    print(f"Eval DB: {path} (isolated)", flush=True)
    return path


def eval_db_metadata() -> dict:
    path = logic.get_db_path()
    return {
        "db_path": path,
        "isolated_eval_db": os.path.abspath(path) != os.path.abspath(logic.DEFAULT_DB_PATH),
        "eval_db_env": (os.getenv("EVAL_DB_PATH") or "").strip() or None,
    }
