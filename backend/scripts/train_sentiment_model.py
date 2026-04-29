import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# type: ignore[import, attr-defined, arg-type, union-attr]  # Complex file - ignoring type issues
"""
Train a simple embedding + LightGBM regression model to predict 1-day return from article text.

This script expects a CSV produced by `prepare_training_data.py`.

Usage:
  python scripts/train_sentiment_model.py --csv data/training_labels_1d_top10.csv --outdir models

Outputs:
  models/embedding_model_name.txt  (not saved, we load the transformer at runtime)
  models/lightgbm_1d_top10.joblib
"""
import argparse
import os
import sqlite3
import pandas as pd
from backend.scripts.script_logger import logger
from backend.ml.feature_pipeline import FeaturePipeline, FeatureInput
from backend.ml.training_pipeline import ensure_training_columns, train_horizon_model


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    # combine title + content for embedding
    df['text'] = (df['title'].fillna('') + '\n' + df['content'].fillna('')).str.strip()
    df = df[~df['text'].isna() & (df['text'].str.len() > 20)]
    return df


def _build_features_if_missing(df: pd.DataFrame, db_path: str | None = None) -> pd.DataFrame:
    feature_pipeline = FeaturePipeline()
    if all(name in df.columns for name in feature_pipeline.feature_names):
        return df
    if not db_path:
        logger.info("No db path provided; missing features will use defaults.")
        normalized, _ = ensure_training_columns(df)
        return normalized
    conn = sqlite3.connect(db_path)
    rows = []
    try:
        for _, row in df.iterrows():
            ticker = str(row.get("ticker", "")).upper()
            raw_date = row.get("date") or row.get("canonical_timestamp")
            if not ticker or not raw_date:
                rows.append({})
                continue
            as_of = pd.to_datetime(raw_date).to_pydatetime()
            vec = feature_pipeline.build_vector(conn, FeatureInput(ticker=ticker, as_of=as_of))
            rows.append(dict(zip(feature_pipeline.feature_names, vec.flatten().tolist())))
    finally:
        conn.close()
    for name in feature_pipeline.feature_names:
        df[name] = [r.get(name, 0.0) for r in rows]
    normalized, _ = ensure_training_columns(df)
    return normalized


def train(csv_path, outdir='models', model_name='all-MiniLM-L6-v2', db_path: str | None = None):
    os.makedirs(outdir, exist_ok=True)
    df = load_csv(csv_path)
    if df.empty:
        logger.warning('No training rows found in %s', csv_path)
        return
    df = _build_features_if_missing(df, db_path=db_path)
    logger.info('Rows: %d', len(df))

    for horizon in ("1d", "3d", "7d"):
        artifact = train_horizon_model(df, horizon=horizon, outdir=outdir)
        logger.info(
            "Saved %s model at %s (metrics=%s)",
            horizon,
            artifact.model_path,
            artifact.metrics,
        )


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--outdir', default='models')
    p.add_argument('--embedder', default='all-MiniLM-L6-v2')
    p.add_argument('--db-path', default=None)
    args = p.parse_args()
    train(args.csv, outdir=args.outdir, model_name=args.embedder, db_path=args.db_path)
