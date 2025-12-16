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
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
import joblib
from script_logger import logger


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    # combine title + content for embedding
    df['text'] = (df['title'].fillna('') + '\n' + df['content'].fillna('')).str.strip()
    df = df[~df['text'].isna() & (df['text'].str.len() > 20)]
    return df


def embed_texts(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True)
    return emb


def train(csv_path, outdir='models', model_name='all-MiniLM-L6-v2'):
    os.makedirs(outdir, exist_ok=True)
    df = load_csv(csv_path)
    if df.empty:
        logger.warning('No training rows found in %s', csv_path)
        return
    X_text = df['text'].tolist()
    y = df['label'].astype(float).values
    logger.info('Rows: %d', len(y))
    # embed
    emb = embed_texts(X_text, model_name=model_name)
    X_train, X_test, y_train, y_test = train_test_split(emb, y, test_size=0.2, random_state=42)

    # use sklearn API for LightGBM for simpler early-stopping handling
    logger.info('Training LightGBM (sklearn API)...')
    gbm = lgb.LGBMRegressor(n_estimators=200)
    # use callbacks for early stopping and disable logging
    callbacks = [lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)

    preds = gbm.predict(X_test)
    # Ensure predictions are numpy arrays for sklearn metrics
    if hasattr(preds, 'toarray'):
        preds = np.asarray(preds)  # type: ignore[arg-type]
    elif hasattr(preds, 'ravel'):
        preds = np.asarray(preds).ravel()  # type: ignore[arg-type]
    rmse = mean_squared_error(y_test, preds) ** 0.5  # type: ignore[arg-type]
    mae = mean_absolute_error(y_test, preds)  # type: ignore[arg-type]
    logger.info('RMSE: %.6f, MAE: %.6f', rmse, mae)

    # Derive model name from CSV filename
    import re
    match = re.search(r'training_labels_(\d+)d_top(\d+)\.csv', os.path.basename(csv_path))
    if match:
        horizon = match.group(1) + 'd'
        topn = f'top{match.group(2)}'
    else:
        # fallback
        horizon = '1d'
        topn = 'top10'
    out_model = os.path.join(outdir, f'lightgbm_{horizon}_{topn}.joblib')
    joblib.dump({'lgbm': gbm, 'embedder': model_name}, out_model)
    logger.info('Saved model to %s', out_model)


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--outdir', default='models')
    p.add_argument('--embedder', default='all-MiniLM-L6-v2')
    args = p.parse_args()
    train(args.csv, outdir=args.outdir, model_name=args.embedder)
