import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import logic  # noqa: E402


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_health_coach.db"
    vector_path = tmp_path / "vector_db"
    monkeypatch.setattr(logic, "DB_PATH", str(db_path))
    monkeypatch.setattr(logic, "VECTOR_DB_DIR", str(vector_path))
    logic.init_db()
    return db_path


@pytest.fixture()
def api_client(isolated_db, monkeypatch):
    import main

    monkeypatch.setattr(main.app.state, "rag_store", None, raising=False)
    main.startup_event()
    from fastapi.testclient import TestClient

    return TestClient(main.app)
