"""SQLite storage: tickets, agent runs, logged steps, drafted replies and escalations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id          TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    customer    TEXT,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id      TEXT NOT NULL REFERENCES tickets(id),
    model          TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'running',
    category       TEXT,
    priority       TEXT,
    sentiment      TEXT,
    steps          INTEGER NOT NULL DEFAULT 0,
    tool_calls     INTEGER NOT NULL DEFAULT 0,
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL,
    latency_ms     INTEGER,
    created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS steps (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES runs(id),
    step       INTEGER NOT NULL,
    thought    TEXT,
    tool       TEXT,
    arguments  TEXT,
    result     TEXT,
    is_error   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS replies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    ticket_id   TEXT NOT NULL REFERENCES tickets(id),
    reply_text  TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS escalations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    ticket_id   TEXT NOT NULL REFERENCES tickets(id),
    reason      TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database and make sure the schema exists. Use ":memory:" in tests."""
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def upsert_ticket(conn: sqlite3.Connection, ticket_id: str, subject: str, body: str, customer: str | None = None) -> None:
    """Insert a ticket, or update its text if the id already exists."""
    conn.execute(
        "INSERT INTO tickets (id, subject, body, customer) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET subject = excluded.subject, body = excluded.body, customer = excluded.customer",
        (ticket_id, subject, body, customer),
    )
    conn.commit()


def start_run(conn: sqlite3.Connection, ticket_id: str, model: str) -> int:
    """Create a run row for one agent pass over one ticket and return its id."""
    cursor = conn.execute("INSERT INTO runs (ticket_id, model) VALUES (?, ?)", (ticket_id, model))
    conn.commit()
    return int(cursor.lastrowid)


def save_classification(conn: sqlite3.Connection, run_id: int, category: str, priority: str, sentiment: str) -> None:
    """Store the agent's classification on the run."""
    conn.execute("UPDATE runs SET category = ?, priority = ?, sentiment = ? WHERE id = ?", (category, priority, sentiment, run_id))
    conn.commit()


def save_reply(conn: sqlite3.Connection, run_id: int, ticket_id: str, reply_text: str) -> int:
    """Store a suggested reply. A later draft in the same run replaces the earlier one."""
    conn.execute("DELETE FROM replies WHERE run_id = ?", (run_id,))
    cursor = conn.execute("INSERT INTO replies (run_id, ticket_id, reply_text) VALUES (?, ?, ?)", (run_id, ticket_id, reply_text))
    conn.commit()
    return int(cursor.lastrowid)


def save_escalation(conn: sqlite3.Connection, run_id: int, ticket_id: str, reason: str) -> int:
    """Flag the ticket for a human."""
    cursor = conn.execute("INSERT INTO escalations (run_id, ticket_id, reason) VALUES (?, ?, ?)", (run_id, ticket_id, reason))
    conn.commit()
    return int(cursor.lastrowid)


def log_step(
    conn: sqlite3.Connection,
    run_id: int,
    step: int,
    thought: str,
    tool: str | None,
    arguments: Any,
    result: str | None,
    is_error: bool,
) -> None:
    """Append one agent step to the audit log."""
    conn.execute(
        "INSERT INTO steps (run_id, step, thought, tool, arguments, result, is_error) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, step, thought, tool, None if arguments is None else json.dumps(arguments, ensure_ascii=False), result, int(is_error)),
    )
    conn.commit()


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: str,
    steps: int,
    tool_calls: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None,
    latency_ms: int,
) -> None:
    """Record the outcome and the usage totals of a run."""
    conn.execute(
        "UPDATE runs SET status = ?, steps = ?, tool_calls = ?, input_tokens = ?, output_tokens = ?, cost_usd = ?, latency_ms = ? WHERE id = ?",
        (status, steps, tool_calls, input_tokens, output_tokens, cost_usd, latency_ms, run_id),
    )
    conn.commit()


def get_reply(conn: sqlite3.Connection, run_id: int) -> str | None:
    """Return the reply drafted in a run, if any."""
    row = conn.execute("SELECT reply_text FROM replies WHERE run_id = ?", (run_id,)).fetchone()
    return row["reply_text"] if row else None


def get_escalation(conn: sqlite3.Connection, run_id: int) -> str | None:
    """Return the first escalation reason of a run, if any."""
    row = conn.execute("SELECT reason FROM escalations WHERE run_id = ? ORDER BY id LIMIT 1", (run_id,)).fetchone()
    return row["reason"] if row else None


def get_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    """Return a run row as a dict."""
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None
