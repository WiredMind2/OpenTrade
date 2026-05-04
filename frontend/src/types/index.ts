// API Response Types

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  uptime_seconds: number;
  services: Record<string, string>;
  database: string;
  models_loaded: number;
}

export interface PredictionRequest {
  ticker: string;
  horizon: string;
  context?: Record<string, any>;
  strategy_name?: string;
  strategy_params?: Record<string, any>;
  /** ISO datetime: use features only through this instant (walk-forward simulation). */
  as_of?: string;
  /** When false with as_of, skips writing to sentiment_predictions (default false if as_of set). */
  persist_prediction?: boolean;
  /** With as_of, request next-horizon realized daily closes in metadata for evaluation. */
  include_forward_actuals?: boolean;
}

export interface PredictionResponse {
  ticker: string;
  horizon: string;
  predicted_return: number;
  confidence: number;
  timestamp: string;
  strategy_name?: string | null;
  model_id?: string | null;
  features_used: string[];
  feature_schema_version?: string;
  interval_lower?: number | null;
  interval_upper?: number | null;
  metadata?: Record<string, any>;
}

export interface BacktestRequest {
  strategy_name: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  parameters?: Record<string, any>;
}

export interface BacktestResult {
  strategy_name: string;
  start_date: string;
  end_date: string;
  completed_at?: string | null;
  initial_capital: number;
  final_value: number;
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  avg_trade_return: number;
  volatility: number;
  timestamp: string;
  metrics: Record<string, any>;
  equity_curve: Array<Record<string, any>>;
  chart_data?: Array<Record<string, any>>;
  execution_engine?: string;
  signals_emitted?: number;
  order_intents?: number;
  order_fills?: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  horizon: string;
  accuracy?: number;
  last_trained: string;
  features: string[];
  status: string;
}

export interface PortfolioResponse {
  timestamp: string;
  total_value: number;
  cash: number;
  invested_value: number;
  exposure: number;
  positions: Array<Record<string, any>>;
  pnl: number;
  daily_return: number;
}

export interface SystemMetrics {
  timestamp: string;
  cpu_percent: number;
  memory_percent: number;
  disk_usage_percent: number;
  database_connections: number;
  active_models: number;
  recent_predictions: number;
  error_rate: number;
}

export interface PriceData {
  ticker: string;
  count: number;
  data: Array<{
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    adjusted_close: number;
    volume: number;
  }>;
}

// OHLC data structure for historical price data
export interface HistoricalDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
}

// OHLC data structure for chart display (extends historical with optional prediction flag)
export interface CandleData extends HistoricalDataPoint {
  isPrediction?: boolean;
}

// Prediction data point from API
export interface PredictionDataPoint {
  date: string;
  predicted_price: number | null;
  actual_price?: number | null;
  confidence: number;
  produced_at?: string;
}

// Prediction data point for chart display
export interface PredictionPoint {
  date: string;
  predicted: number | null;
  actual?: number;
  confidence: number;
}

// Aggregated prediction data point (when multiple predictions per date)
export interface AggregatedPredictionPoint extends PredictionPoint {
  count?: number;
}

// API response for chart data endpoint
export interface ChartDataResponse {
  ticker: string;
  historical_data: HistoricalDataPoint[];
  predictions: PredictionDataPoint[];
  raw_predictions?: PredictionDataPoint[];
}


export interface ScriptExecutionRequest {
  script_name: string;
  parameters?: Record<string, any>;
}

export interface ScriptExecutionResponse {
  script_name: string;
  status: string;
  execution_id: string;
  start_time: string;
  end_time?: string;
  output?: string;
  error?: string;
  duration_seconds?: number;
}

export interface PipelineStatus {
  execution_id: string;
  current_step?: string;
  completed_steps: string[];
  failed_steps: string[];
  status: string;
  start_time: string;
  estimated_completion?: string;
}

export interface ScriptExecution {
  execution_id: string;
  script_name: string;
  status: string;
  start_time: string;
  end_time?: string;
  duration_seconds?: number;
}

// WebSocket Message Types

export interface WebSocketMessage {
  type: string;
  data: any;
}

export interface ScriptStatusMessage extends WebSocketMessage {
  type: "script_status";
  data: ScriptExecutionResponse;
}

export interface PipelineStatusMessage extends WebSocketMessage {
  type: "pipeline_status";
  data: PipelineStatus;
}

export interface BacktestStatusMessage extends WebSocketMessage {
  type: "backtest_status";
  data: BacktestResult;
}

export interface ChartUpdateMessage extends WebSocketMessage {
  type: "chart_update";
  data: {
    symbol: string;
    resolution: string;
    bar: {
      time: number;
      open: number;
      high: number;
      low: number;
      close: number;
      volume?: number;
    };
  };
}

export interface TrainingProgressMessage extends WebSocketMessage {
  type: "training_progress";
  data: {
    job_id: string;
    status: string;
    progress?: number;
    logs?: string[];
    result?: any;
    error?: string;
  };
}

export interface StrategyMetadata {
  name: string;
  description: string;
  type: string;
  parameters_schema: Record<string, any>;
  can_train: boolean;
}

export interface StrategyForecastResponse {
  symbol: string;
  horizon_days: number;
  predicted_return: number;
  confidence: number;
  predicted_path: Array<{ time: string; price: number }>;
  metadata: Record<string, any>;
}

export interface StrategySignalPoint {
  ticker: string;
  target_pct: number;
  reason: string;
  confidence: number;
  timestamp: string;
  metadata: Record<string, any>;
}

export interface ProjectionPoint {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  predicted: boolean;
}

export interface ProjectionRequest {
  symbol: string;
  start_time: number;
  start_price: number;
  resolution: string;
  horizon_steps: number;
  params: Record<string, any>;
  mode: string;
}

export interface ProjectionResponse {
  strategy: string;
  points: ProjectionPoint[];
  precision: number;
  confidence: number;
  metadata: StrategyMetadata;
}

/** Overlay paths for optional multi-model prediction fan on the chart */
export interface PredictionProjectionPoint {
  time: number;
  price: number;
  confidence: number;
  upperBound?: number;
  lowerBound?: number;
}

export interface PredictionProjection {
  id: string;
  ticker: string;
  strategy_name: string;
  horizon: number;
  points: PredictionProjectionPoint[];
  confidence: number;
  color: string;
  createdAt: string;
  metadata?: Record<string, any>;
}

export interface StrategyAnalyticsFilters {
  strategies: string[];
  benchmarks: string[];
  available_presets: string[];
  available_granularities: Array<'daily' | 'weekly' | 'monthly'>;
  rolling_windows: number[];
  min_date?: string | null;
  max_date?: string | null;
}

export interface StrategyTimeseriesPoint {
  date: string;
  normalized_equity: number;
  drawdown: number;
  rolling_sharpe?: number | null;
  rolling_sortino?: number | null;
  rolling_volatility?: number | null;
  period_return?: number | null;
}

export interface DistributionBucket {
  bucket: string;
  count: number;
  value: number;
}

export interface StrategyDistributionResponse {
  strategy: string;
  returns_histogram: DistributionBucket[];
  trade_pnl_histogram: DistributionBucket[];
  holding_period_histogram: DistributionBucket[];
  pnl_by_symbol: DistributionBucket[];
}

export interface StrategyVariantRow {
  params_hash: string;
  variant_label?: string | null;
  strategy: string;
  representative_run_id: number;
  run_count: number;
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  volatility: number;
  params: Record<string, any>;
  last_completed_at?: string | null;
}

export interface StrategyVariantSummary {
  strategy: string;
  objective: string;
  top_n: number;
  variants: StrategyVariantRow[];
}

export interface VariantSeriesPayload {
  params_hash: string;
  variant_label?: string | null;
  representative_run_id: number;
  points: StrategyTimeseriesPoint[];
}

export interface StrategyVariantTimeseriesResponse {
  strategy: string;
  benchmark_ticker: string;
  granularity: 'daily' | 'weekly' | 'monthly';
  benchmark_points: StrategyTimeseriesPoint[];
  variant_series: VariantSeriesPayload[];
}

/** One strategy row within a ticker's leaderboard slice (matches backend TickerStrategyRow). */
export interface TickerStrategyRow {
  ticker: string;
  strategy: string;
  params_hash: string;
  variant_label?: string | null;
  representative_run_id: number;
  run_count: number;
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  volatility: number;
  params: Record<string, any>;
  last_completed_at?: string | null;
}

export interface TickerStrategyLeaderboard {
  ticker: string;
  strategies: TickerStrategyRow[];
}

export interface TickerStrategyLeaderboardResponse {
  objective: string;
  top_n: number;
  tickers: TickerStrategyLeaderboard[];
}