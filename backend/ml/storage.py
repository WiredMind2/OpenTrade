"""
Storage helpers for ML metadata and operational logging.
"""

import sqlite3


def ensure_ml_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name IN ('ml_model_registry', 'ml_run_log', 'sentiment_predictions')
        """
    )
    found = {r[0] for r in cur.fetchall()}
    if missing := [t for t in ("sentiment_predictions", "ml_model_registry", "ml_run_log") if t not in found]:
        raise RuntimeError(
            "Database schema is missing required tables: "
            + ", ".join(missing)
            + ". Initialize the DB using db/schema.sql."
        )

    cur.execute("PRAGMA table_info('sentiment_predictions')")
    cols = [r[1] for r in cur.fetchall()]
    if "model_version" not in cols:
        cur.execute("ALTER TABLE sentiment_predictions ADD COLUMN model_version TEXT")
    if "feature_schema_version" not in cols:
        cur.execute("ALTER TABLE sentiment_predictions ADD COLUMN feature_schema_version TEXT")
    if "prediction_latency_ms" not in cols:
        cur.execute("ALTER TABLE sentiment_predictions ADD COLUMN prediction_latency_ms REAL")
    conn.commit()
