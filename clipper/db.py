"""SQLite state store. Each stage reads and writes rows here so runs can resume."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS source (
    id TEXT PRIMARY KEY,            -- '<extractor>-<video id>'
    campaign_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    channel_url TEXT,
    duration REAL,
    dir TEXT NOT NULL,
    status TEXT NOT NULL,           -- ingested | transcribed | ranked | rendered | failed
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS moment (
    id TEXT PRIMARY KEY,            -- '<source id>-m<rank>'
    source_id TEXT NOT NULL REFERENCES source(id),
    rank INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    score REAL,
    title TEXT,
    hooks TEXT,                     -- JSON list of hook texts, one per variant
    rationale TEXT,
    ranked_by TEXT,                 -- claude | heuristic | manual
    status TEXT NOT NULL DEFAULT 'picked'
);
CREATE TABLE IF NOT EXISTS clip (
    id TEXT PRIMARY KEY,            -- '<moment id>-v<n>'
    moment_id TEXT NOT NULL REFERENCES moment(id),
    variant INTEGER NOT NULL,
    path TEXT,
    layout TEXT,
    qa TEXT,                        -- JSON of automatic checks
    status TEXT NOT NULL,           -- rendered | qa_failed | failed
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(work_dir: Path) -> sqlite3.Connection:
    work_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(work_dir / "state.db")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(moment)")}
    if "subcaption" not in cols:  # added with the tastemaker
        con.execute("ALTER TABLE moment ADD COLUMN subcaption TEXT")
        con.commit()
    if "concept" not in cols:  # big-idea gate: the clip's one-line concept
        con.execute("ALTER TABLE moment ADD COLUMN concept TEXT")
        con.commit()
    return con


def upsert(con: sqlite3.Connection, table: str, row: dict) -> None:
    row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
    cols = ",".join(row)
    qs = ",".join("?" for _ in row)
    updates = ",".join(f"{k}=excluded.{k}" for k in row if k != "id")
    con.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({qs}) ON CONFLICT(id) DO UPDATE SET {updates}",
        list(row.values()),
    )
    con.commit()


def set_status(con: sqlite3.Connection, table: str, row_id: str, status: str) -> None:
    con.execute(f"UPDATE {table} SET status=? WHERE id=?", (status, row_id))
    con.commit()


def get_source(con: sqlite3.Connection, source_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM source WHERE id=?", (source_id,)).fetchone()
    if row is None:
        raise KeyError(f"No source '{source_id}'. Run `clipper ingest` first.")
    return row
