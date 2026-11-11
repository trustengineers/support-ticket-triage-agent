from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from app import db
from app.config import ROOT, Price, Settings
from app.kb import KnowledgeBase
from app.tools import ToolContext

TICKET_ID = "T-100"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(model_a="model-a", model_b="model-b", prices={"model-a": Price(10.0, 50.0)}, db_path=tmp_path / "test.db")


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_dir(ROOT / "kb")


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    connection = db.connect(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def ctx(conn: sqlite3.Connection, kb: KnowledgeBase) -> ToolContext:
    db.upsert_ticket(conn, TICKET_ID, "Refund", "Can I get a refund?")
    run_id = db.start_run(conn, TICKET_ID, "model-a")
    return ToolContext(conn=conn, kb=kb, run_id=run_id, ticket_id=TICKET_ID)
