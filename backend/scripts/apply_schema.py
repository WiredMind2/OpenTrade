"""Apply SQLite schema to create the MVP database.

Usage:
  python apply_schema.py --db path/to/db.sqlite --schema db/schema.sql

This is a tiny utility that runs the SQL script in `db/schema.sql` against the target SQLite file.
"""
import argparse
import logging
import os
import sqlite3
import sys


def apply_schema(db_path: str, schema_path: str):
    with open(schema_path, 'r', encoding='utf-8') as f:
        sql = f.read()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql)
        conn.commit()
        logging.getLogger(__name__).info('Schema applied to %s', db_path)
    finally:
        conn.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=False, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')), help='SQLite DB path')
    parser.add_argument('--schema', required=False, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'db', 'schema.sql')), help='Path to SQL schema file')
    args = parser.parse_args()
    apply_schema(args.db, args.schema)


if __name__ == '__main__':
    main()
