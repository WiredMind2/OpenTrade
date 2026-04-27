"""
Run scheduled ML retraining/evaluation cycle and write run log.
"""

import argparse
import sqlite3
from datetime import datetime

from backend.logging_config import get_component_logger
from backend.scripts.train_sentiment_model import train


logger = get_component_logger(__file__)


def log_run(conn: sqlite3.Connection, run_type: str, status: str, metadata: str) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ml_run_log(run_type, status, metadata, finished_at)
        VALUES (?, ?, ?, ?)
        """,
        (run_type, status, metadata, datetime.utcnow().isoformat()),
    )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="models")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        train(args.csv, outdir=args.outdir, db_path=args.db)
        log_run(conn, "retrain", "completed", f"csv={args.csv},outdir={args.outdir}")
        logger.info("ML cycle completed")
    except Exception as exc:
        log_run(conn, "retrain", "failed", str(exc))
        logger.error("ML cycle failed: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
