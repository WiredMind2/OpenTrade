"""
SQLite migrations for variant/run identity (params_hash, artifact FKs, backfill).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Set

from backend.utils.backtest_variants import compute_params_hash


def _table_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    cur = conn.cursor()
    return {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}


def ensure_variant_schema(conn: sqlite3.Connection) -> None:
    """Add variant columns and artifact FK columns; backfill where possible."""
    conn.execute("PRAGMA foreign_keys = ON;")
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='backtest_runs'")
    if cur.fetchone():
        br_cols = _table_columns(conn, "backtest_runs")
        for col, sql_type in (
            ("params_hash", "TEXT"),
            ("variant_label", "TEXT"),
            ("optimizer_mode", "TEXT"),
            ("experiment_id", "TEXT"),
            ("client_backtest_id", "TEXT"),
        ):
            if col not in br_cols:
                cur.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {sql_type}")

    for table in ("strategy_signals", "order_intents", "order_fills"):
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if not cur.fetchone():
            continue
        cols = _table_columns(conn, table)
        if "backtest_run_id" not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN backtest_run_id INTEGER REFERENCES backtest_runs(id)")

    conn.commit()

    # Backfill backtest_runs.params_hash and client_backtest_id from legacy JSON
    cur.execute("SELECT id, params, metrics FROM backtest_runs WHERE params IS NOT NULL OR metrics IS NOT NULL")
    rows = cur.fetchall()
    for row_id, params_raw, metrics_raw in rows:
        params_obj = None
        if params_raw:
            try:
                params_obj = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
            except (json.JSONDecodeError, TypeError):
                params_obj = None
        metrics_obj = None
        if metrics_raw:
            try:
                metrics_obj = json.loads(metrics_raw) if isinstance(metrics_raw, str) else metrics_raw
            except (json.JSONDecodeError, TypeError):
                metrics_obj = {}
        if not isinstance(metrics_obj, dict):
            metrics_obj = {}

        ph = None
        if isinstance(params_obj, dict):
            ph = compute_params_hash(params_obj)
        client_id = metrics_obj.get("backtest_id") if isinstance(metrics_obj, dict) else None

        cur.execute(
            """
            UPDATE backtest_runs
            SET params_hash = COALESCE(params_hash, ?),
                client_backtest_id = COALESCE(client_backtest_id, ?)
            WHERE id = ?
            """,
            (ph, client_id, row_id),
        )
    conn.commit()

    # Helpful indexes (safe after columns exist)
    for stmt in (
        "CREATE INDEX IF NOT EXISTS idx_backtest_runs_name_hash ON backtest_runs(name, params_hash)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_runs_client_id ON backtest_runs(client_backtest_id)",
        "CREATE INDEX IF NOT EXISTS idx_strategy_signals_run_id ON strategy_signals(backtest_run_id)",
        "CREATE INDEX IF NOT EXISTS idx_order_intents_run_id ON order_intents(backtest_run_id)",
        "CREATE INDEX IF NOT EXISTS idx_order_fills_run_id ON order_fills(backtest_run_id)",
    ):
        try:
            cur.execute(stmt)
        except sqlite3.OperationalError:
            pass
    conn.commit()

    # Link artifacts to runs via client_backtest_id
    for table in ("strategy_signals", "order_intents", "order_fills"):
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if not cur.fetchone():
            continue
        cur.execute(
            f"""
            UPDATE {table}
            SET backtest_run_id = (
                SELECT id FROM backtest_runs b
                WHERE b.client_backtest_id = {table}.backtest_id
                LIMIT 1
            )
            WHERE backtest_run_id IS NULL AND backtest_id IS NOT NULL
            """
        )
    conn.commit()
