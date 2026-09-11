"""SQLite logging for search queries, feeding the Monitoring tab."""

import json
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "logs" / "queries.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    retrieved_chunk_ids TEXT NOT NULL,
    top_score REAL,
    latency_ms REAL NOT NULL,
    n_tokens_in INTEGER,
    n_tokens_out INTEGER,
    model TEXT,
    refused INTEGER NOT NULL
);
"""


def _get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def _insert(
    db_path: Path,
    query: str,
    retrieved_chunk_ids: list[str],
    top_score: float | None,
    latency_ms: float,
    n_tokens_in: int | None,
    n_tokens_out: int | None,
    model: str | None,
    refused: bool,
) -> None:
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """INSERT INTO query_log
               (timestamp, query, retrieved_chunk_ids, top_score, latency_ms, n_tokens_in, n_tokens_out, model, refused)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                query,
                json.dumps(retrieved_chunk_ids),
                top_score,
                latency_ms,
                n_tokens_in,
                n_tokens_out,
                model,
                int(refused),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_query(
    query: str,
    retrieved_chunk_ids: list[str],
    top_score: float | None,
    latency_ms: float,
    n_tokens_in: int | None,
    n_tokens_out: int | None,
    model: str | None,
    refused: bool,
    db_path: Path = DB_PATH,
) -> None:
    """Log a query in a background thread so the write never delays what the
    user sees. SQLite inserts are fast either way, but doing it on a thread
    makes "in the background" literally true rather than "fast enough to not
    notice"."""
    thread = threading.Thread(
        target=_insert,
        args=(db_path, query, retrieved_chunk_ids, top_score, latency_ms, n_tokens_in, n_tokens_out, model, refused),
        daemon=True,
    )
    thread.start()


def fetch_all(db_path: Path = DB_PATH) -> list[dict]:
    if not db_path.exists():
        return []
    conn = _get_connection(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM query_log ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
