"""
Test fixtures. Env is set here, before any `app.*` import, so the cached
settings and the module-level DB engine pick up the throwaway SQLite file and
the mock model backend.
"""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="compactllm-tests-"))
os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP / 'test.db'}")
os.environ.setdefault("BENCHMARK_JSON_PATH", str(_TMP / "no-benchmark.json"))
os.environ.setdefault("DATASET_DIR", str(_TMP / "no-splits"))

import pytest
from fastapi.testclient import TestClient

from app.db.database import Base, engine
from app.main import app


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
