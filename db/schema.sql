-- SQLite schema for trading-backtesting MVP
-- All datetimes stored in UTC (ISO format)

PRAGMA foreign_keys = ON;

-- Tickers
CREATE TABLE IF NOT EXISTS tickers (
  ticker TEXT PRIMARY KEY,               -- e.g. 'AAPL'
  name TEXT,
  exchange TEXT,
  sector TEXT,
  added_at TEXT DEFAULT (datetime('now'))
);

-- Articles (raw + parsed)
CREATE TABLE IF NOT EXISTS articles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT,                           -- e.g. 'NewsAPI'
  url TEXT UNIQUE,
  canonical_timestamp TEXT,              -- UTC ISO string when article was published
  fetched_at TEXT DEFAULT (datetime('now')),
  title TEXT,
  author TEXT,
  raw_html TEXT,
  content TEXT,
  fingerprint TEXT,
  lang TEXT,
  metadata JSON
);
CREATE INDEX IF NOT EXISTS idx_articles_time ON articles(canonical_timestamp);

-- Many-to-many mapping between articles and tickers
CREATE TABLE IF NOT EXISTS article_ticker (
  article_id INTEGER,
  ticker TEXT,
  relevance_score REAL,
  PRIMARY KEY(article_id, ticker),
  FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE,
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_article_ticker_ticker ON article_ticker(ticker);

-- Daily price bars
CREATE TABLE IF NOT EXISTS price_daily (
  ticker TEXT,
  date TEXT,                             -- date in UTC YYYY-MM-DD
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  adjusted_close REAL,
  volume INTEGER,
  PRIMARY KEY (ticker, date),
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_price_daily_ticker_date ON price_daily(ticker, date);

-- Minute price bars (optional, may not exist for full 5 years for free sources)
CREATE TABLE IF NOT EXISTS price_minute (
  ticker TEXT,
  dt TEXT,                               -- UTC timestamp (YYYY-MM-DD HH:MM:SS)
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  volume INTEGER,
  PRIMARY KEY (ticker, dt),
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_price_minute_ticker_dt ON price_minute(ticker, dt);

-- Sentiment model predictions per article-ticker-horizon
CREATE TABLE IF NOT EXISTS sentiment_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id INTEGER,
  ticker TEXT,
  model TEXT,
  model_version TEXT,
  horizon TEXT,                          -- '1d','3d','7d'
  predicted_return REAL,                 -- percent (0.012 = +1.2%)
  predicted_confidence REAL,
  features_used TEXT,
  feature_schema_version TEXT,
  metadata JSON,
  prediction_latency_ms REAL,
  produced_at TEXT DEFAULT (datetime('now')),
  training_run_id TEXT,
  FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE SET NULL,
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_horizon ON sentiment_predictions(ticker, horizon);

-- Feature store (pickled/JSON-encoded vectors)
CREATE TABLE IF NOT EXISTS features (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT,
  dt TEXT,
  feature_blob BLOB,
  source TEXT,
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_features_ticker_dt ON features(ticker, dt);

-- Trading-model predictions / decisions (one row per ticker per decision timestamp)
CREATE TABLE IF NOT EXISTS trading_model_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT,
  dt TEXT,
  model TEXT,
  predicted_return REAL,
  enter_prob REAL,
  suggested_position_pct REAL,
  exit_prob REAL,
  produced_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trading_pred_ticker_dt ON trading_model_predictions(ticker, dt);

-- Backtest runs metadata
CREATE TABLE IF NOT EXISTS backtest_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  started_at TEXT DEFAULT (datetime('now')),
  completed_at TEXT,
  params JSON,
  initial_capital REAL,
  final_value REAL,
  total_return REAL,
  annualized_return REAL,
  sharpe_ratio REAL,
  max_drawdown REAL,
  win_rate REAL,
  total_trades INTEGER,
  avg_trade_return REAL,
  volatility REAL,
  equity_curve TEXT,
  metrics JSON
);

-- Trades recorded during backtests
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  backtest_run_id INTEGER,
  ticker TEXT,
  entry_dt TEXT,
  entry_price REAL,
  exit_dt TEXT,
  exit_price REAL,
  quantity INTEGER,
  position_pct REAL,
  fees REAL,
  slippage REAL,
  pnl REAL,
  FOREIGN KEY(backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE,
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trades_backtest ON trades(backtest_run_id);

-- Signal-level artifacts for signal-driven execution
CREATE TABLE IF NOT EXISTS strategy_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  backtest_id TEXT NOT NULL,
  signal_time TEXT NOT NULL,
  ticker TEXT NOT NULL,
  target_pct REAL NOT NULL,
  reason TEXT,
  confidence REAL,
  metadata JSON
);
CREATE INDEX IF NOT EXISTS idx_strategy_signals_backtest ON strategy_signals(backtest_id);
CREATE INDEX IF NOT EXISTS idx_strategy_signals_ticker_time ON strategy_signals(ticker, signal_time);

CREATE TABLE IF NOT EXISTS order_intents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  backtest_id TEXT NOT NULL,
  intent_time TEXT NOT NULL,
  ticker TEXT NOT NULL,
  side TEXT NOT NULL,
  notional_delta REAL NOT NULL,
  reason TEXT,
  metadata JSON
);
CREATE INDEX IF NOT EXISTS idx_order_intents_backtest ON order_intents(backtest_id);

CREATE TABLE IF NOT EXISTS order_fills (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  backtest_id TEXT NOT NULL,
  fill_time TEXT NOT NULL,
  ticker TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity INTEGER NOT NULL,
  fill_price REAL NOT NULL,
  fees REAL DEFAULT 0,
  slippage REAL DEFAULT 0,
  metadata JSON
);
CREATE INDEX IF NOT EXISTS idx_order_fills_backtest ON order_fills(backtest_id);

-- Portfolio snapshots for time-series of portfolio value
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
  backtest_run_id INTEGER,
  dt TEXT,
  cash REAL,
  market_value REAL,
  total_value REAL,
  exposure REAL,
  positions_json JSON,
  FOREIGN KEY(backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

-- Ingestion logs
CREATE TABLE IF NOT EXISTS ingestion_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT,
  success INTEGER,
  started_at TEXT,
  finished_at TEXT,
  details TEXT
);

-- Simple key/value config
CREATE TABLE IF NOT EXISTS config (
  key TEXT PRIMARY KEY,
  value TEXT,
  updated_at TEXT DEFAULT (datetime('now'))
);

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'viewer',  -- admin, trader, analyst, viewer
  avatar TEXT,
  is_active INTEGER DEFAULT 1,
  last_login TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

-- User sessions for JWT token management
CREATE TABLE IF NOT EXISTS user_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  token_hash TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token_hash ON user_sessions(token_hash);

-- Password reset tokens
CREATE TABLE IF NOT EXISTS password_reset_tokens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  token_hash TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  used INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_token_hash ON password_reset_tokens(token_hash);

-- User activity log
CREATE TABLE IF NOT EXISTS user_activity (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  action TEXT NOT NULL,
  details TEXT,
  ip_address TEXT,
  user_agent TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_user_activity_user_id ON user_activity(user_id);

-- Model retraining jobs
CREATE TABLE IF NOT EXISTS model_jobs (
  id TEXT PRIMARY KEY,
  model_name TEXT NOT NULL,
  status TEXT NOT NULL,  -- queued, running, completed, failed
  created_at DATETIME DEFAULT (datetime('now')),
  updated_at DATETIME DEFAULT (datetime('now')),
  config JSON,
  result JSON,
  error TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_jobs_model_name ON model_jobs(model_name);
CREATE INDEX IF NOT EXISTS idx_model_jobs_status ON model_jobs(status);

-- ML model registry (trained model artifacts + metrics)
CREATE TABLE IF NOT EXISTS ml_model_registry (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  horizon TEXT NOT NULL,
  feature_schema_version TEXT,
  metrics JSON,
  artifact_path TEXT,
  is_active INTEGER DEFAULT 0,
  trained_at TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ml_registry_horizon_active ON ml_model_registry(horizon, is_active);

-- ML operational run log (training/prediction runs, etc.)
CREATE TABLE IF NOT EXISTS ml_run_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_type TEXT NOT NULL,
  status TEXT NOT NULL,
  metadata JSON,
  started_at TEXT DEFAULT (datetime('now')),
  finished_at TEXT
);

-- End of schema
