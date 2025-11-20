"""
Generate sentiment predictions using trained LightGBM models.

This script loads articles, embeds their text using the trained sentence transformer,
and predicts returns using the saved LightGBM models for different horizons.

Usage:
  python scripts/generate_sentiment_predictions.py --db data/backtest.db --horizon 1 --model models/lightgbm_1d_top10.joblib

Outputs:
  Rows in sentiment_predictions table with predicted_return values from the actual model.
"""
import argparse
import sqlite3
import pandas as pd
import os
import joblib
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer


def load_unlabeled_articles(conn, ticker):
    """Load articles that don't have sentiment predictions yet for this ticker."""
    cur = conn.cursor()
    cur.execute('''
        SELECT DISTINCT a.id, a.title, a.content, a.canonical_timestamp
        FROM articles a
        JOIN article_ticker at ON at.article_id = a.id
        WHERE at.ticker = ? AND a.title IS NOT NULL
        AND a.canonical_timestamp IS NOT NULL
        AND a.canonical_timestamp >= '2020-01-01'
        AND NOT EXISTS (
            SELECT 1 FROM sentiment_predictions sp 
            WHERE sp.article_id = a.id AND sp.ticker = at.ticker 
            AND sp.model LIKE 'lightgbm_%'
        )
        ORDER BY a.canonical_timestamp
    ''', (ticker,))
    rows = cur.fetchall()
    return rows


def embed_and_predict(model_path, articles):
    """Load model and embedder, predict returns for articles."""
    if not articles:
        return []
    
    # Load model bundle
    bundle = joblib.load(model_path)
    lgbm = bundle['lgbm']
    embedder_name = bundle['embedder']
    
    # Load embedder
    print(f'Loading embedder: {embedder_name}')
    embedder = SentenceTransformer(embedder_name)
    
    # Prepare texts
    texts = []
    for _, title, content, timestamp in articles:
        text = (str(title or '') + '\n' + str(content or '')).strip()
        if len(text) > 20:
            texts.append(text)
        else:
            texts.append('')
    
    # Embed texts
    print(f'Embedding {len(texts)} texts...')
    emb = embedder.encode(texts, show_progress_bar=True)
    
    # Predict
    print('Predicting...')
    preds = lgbm.predict(emb)
    
    return preds


def generate_predictions(conn, model_path, horizon):
    """Generate sentiment predictions for articles using the trained model."""
    cur = conn.cursor()
    
    # Get list of tickers we're modeling
    cur.execute('SELECT DISTINCT ticker FROM article_ticker ORDER BY ticker LIMIT 10')
    tickers = [r[0] for r in cur.fetchall()]
    print(f'Processing {len(tickers)} tickers: {tickers}')
    
    inserted_count = 0
    
    for ticker in tickers:
        print(f'\nProcessing ticker: {ticker}')
        
        # Get articles for this ticker
        articles = load_unlabeled_articles(conn, ticker)
        print(f'Found {len(articles)} unlabeled articles for {ticker}')
        
        if not articles:
            continue
        
        # Predict
        preds = embed_and_predict(model_path, articles)
        
        # Insert predictions
        for i, (aid, title, content, timestamp) in enumerate(articles):
            pred_return = float(preds[i]) if i < len(preds) else 0.0
            
            # Skip very small predictions
            if abs(pred_return) < 0.001:
                continue
            
            cur.execute('''
                INSERT INTO sentiment_predictions 
                (article_id, ticker, model, horizon, predicted_return, predicted_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                aid,
                ticker,
                f'lightgbm_{horizon}d',
                f'{horizon}d',
                pred_return,
                1.0  # Simple confidence
            ))
            
            if cur.rowcount > 0:
                inserted_count += 1
    
    conn.commit()
    print(f'\nInserted {inserted_count} sentiment predictions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    parser.add_argument('--horizon', type=int, default=1, choices=[1, 3, 7])
    parser.add_argument('--model', required=True, help='Path to trained model .joblib file')
    args = parser.parse_args()
    
    conn = sqlite3.connect(args.db)
    generate_predictions(conn, args.model, args.horizon)
    conn.close()
    print('Done')
