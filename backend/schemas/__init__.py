"""
Pydantic models for API requests and responses.

This module contains all the data models used by the Trading Backtester API.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = "healthy"
    timestamp: datetime
    version: str
    uptime_seconds: float
    services: Dict[str, str]
    database: str
    models_loaded: int


class PredictionRequest(BaseModel):
    """Request model for making predictions."""
    ticker: str = Field(..., description="Stock ticker symbol")
    horizon: str = Field(..., description="Prediction horizon: 1d, 3d, or 7d")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    strategy_name: Optional[str] = Field(default=None, description="Optional strategy to drive prediction output")
    strategy_params: Optional[Dict[str, Any]] = Field(default={}, description="Optional strategy parameter overrides")
    as_of: Optional[datetime] = Field(
        default=None,
        description="Simulate using data available through this timestamp (walk-forward). Omit for live 'now'.",
    )
    persist_prediction: Optional[bool] = Field(
        default=None,
        description="If true, write to sentiment_predictions. Defaults to true for live, false when as_of is set.",
    )
    include_forward_actuals: bool = Field(
        default=False,
        description="When as_of is set, add metadata.forward_actual_closes (next horizon daily closes) for evaluation.",
    )

    @field_validator('horizon')
    @classmethod
    def validate_horizon(cls, v):
        if v not in ['1d', '3d', '7d']:
            raise ValueError('horizon must be one of: 1d, 3d, 7d')
        return v

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        if not v or len(v) == 0:
            raise ValueError('ticker cannot be empty')
        return v.upper()


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    ticker: str
    horizon: str
    predicted_return: float
    confidence: float
    timestamp: datetime
    strategy_name: Optional[str] = None
    model_id: Optional[str] = None
    features_used: List[str]
    feature_schema_version: Optional[str] = None
    interval_lower: Optional[float] = None
    interval_upper: Optional[float] = None
    metadata: Dict[str, Any] = {}


class ModelInfo(BaseModel):
    """Information about available models."""
    name: str
    version: str
    horizon: str
    accuracy: Optional[float] = None
    last_trained: datetime
    features: List[str]
    status: str = "active"


class ModelSummary(BaseModel):
    """Summary information about available models."""
    name: str
    type: str
    version: str
    description: str
    capabilities: List[str]
    config_schema: Dict[str, Any]


class PortfolioResponse(BaseModel):
    """Portfolio information response."""
    timestamp: datetime
    total_value: float
    cash: float
    invested_value: float
    exposure: float
    positions: List[Dict[str, Any]]
    pnl: float
    daily_return: float


class SystemMetrics(BaseModel):
    """System metrics response."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    database_connections: int
    active_models: int
    recent_predictions: int
    error_rate: float


class ScriptExecutionRequest(BaseModel):
    """Request model for executing scripts."""
    script_name: str = Field(..., description="Name of the script to execute")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Script-specific parameters")

    @field_validator('script_name')
    @classmethod
    def validate_script_name(cls, v):
        valid_scripts = [
            'run_pipeline', 'train_sentiment_model', 'backtest_runner', 'train_all_strategies'
        ]
        if v not in valid_scripts:
            raise ValueError(f'script_name must be one of: {", ".join(valid_scripts)}')
        return v


class ScriptExecutionResponse(BaseModel):
    """Response model for script execution."""
    script_name: str
    status: str  # 'running', 'completed', 'failed'
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    output: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


class PipelineRequest(BaseModel):
    """Request model for running the data pipeline."""
    steps: Optional[List[str]] = Field(default=None, description="Pipeline steps to run. If not provided, runs all default steps.")


class PipelineStatus(BaseModel):
    """Status of pipeline execution."""
    execution_id: str
    current_step: Optional[str] = None
    completed_steps: List[str]
    failed_steps: List[str]
    status: str  # 'running', 'completed', 'failed'
    start_time: datetime
    estimated_completion: Optional[datetime] = None


class BatchStrategyTrainingRequest(BaseModel):
    """Batch signal-parameter training for every strategy that supports optimization training."""

    ticker: str = Field(..., description="Primary ticker symbol")
    start_date: str = Field(..., description="Training window start (YYYY-MM-DD)")
    end_date: str = Field(..., description="Training window end (YYYY-MM-DD)")
    initial_capital: float = Field(default=100000.0, gt=0)
    objective: str = Field(default="balanced", description="sharpe|return|drawdown|balanced")
    max_evals: int = Field(default=8, ge=1, le=50)
    optimizer_mode: str = Field(default="grid", description="grid|random")
    random_seed: Optional[int] = Field(default=None)
    pair_ticker: Optional[str] = Field(
        default=None,
        description="Second leg for pairs_trading (omit to skip pairs in the batch)",
    )
    universe_limit: int = Field(default=8, ge=2, le=15)
    stop_on_error: bool = Field(default=False, description="Stop on first strategy failure")

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        s = (v or "").strip().upper()
        if not s:
            raise ValueError("ticker cannot be empty")
        return s

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError as e:
            raise ValueError("Date must be in YYYY-MM-DD format") from e

    @field_validator("end_date")
    @classmethod
    def validate_end_after_start(cls, v: str, info) -> str:
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, v: str) -> str:
        o = (v or "balanced").lower()
        if o not in {"sharpe", "return", "drawdown", "balanced"}:
            raise ValueError("objective must be sharpe, return, drawdown, or balanced")
        return o

    @field_validator("optimizer_mode")
    @classmethod
    def validate_optimizer_mode(cls, v: str) -> str:
        m = (v or "grid").strip().lower()
        if m not in {"grid", "random"}:
            raise ValueError("optimizer_mode must be grid or random")
        return m


class HistoricalDataPoint(BaseModel):
    """Historical price data point."""
    date: str  # ISO format date string
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None


class PredictionDataPoint(BaseModel):
    """Prediction data point."""
    date: Optional[str] = None  # ISO format date string, None for invalid/malformed dates
    predicted_price: Optional[float] = None
    actual_price: Optional[float] = None
    confidence: float
    produced_at: Optional[str] = None  # ISO format timestamp
    count: Optional[int] = None  # Number of predictions aggregated into this point


class DataQualityMetadata(BaseModel):
    """Metadata about data quality and freshness."""
    data_freshness_score: float  # 0.0 to 1.0
    quality_level: str  # 'excellent', 'good', 'fair', 'poor', 'critical'
    last_updated: str  # ISO format timestamp
    data_age_hours: float
    validation_issues: int
    total_records: int
    data_source: str


class ChartDataResponse(BaseModel):
    """Response model for chart data endpoint."""
    ticker: str
    historical_data: List[HistoricalDataPoint]
    predictions: List[PredictionDataPoint]
    raw_predictions: Optional[List[PredictionDataPoint]] = None
    metadata: DataQualityMetadata

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('ticker cannot be empty')
        return v.upper().strip()


class ModelPredictRequest(BaseModel):
    """Request model for model predictions."""
    inputs: Dict[str, Any]
    config: Dict[str, Any]


class ModelPredictResponse(BaseModel):
    """Response model for model predictions."""
    predictions: List[Dict[str, Any]]
    meta: Dict[str, Any]


class StrategyForecastRequest(BaseModel):
    symbol: str
    as_of: Optional[datetime] = None
    current_price: Optional[float] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    horizon_days: int = Field(default=5, ge=1, le=365)


class StrategySignalRequest(BaseModel):
    symbols: List[str]
    as_of: Optional[datetime] = None
    current_prices: Dict[str, float] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)


class StrategySignalPoint(BaseModel):
    ticker: str
    target_pct: float
    reason: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategyFilterMetadataResponse(BaseModel):
    """Available filters for the strategy analytics dashboard."""
    strategies: List[str]
    benchmarks: List[str]
    available_presets: List[str]
    available_granularities: List[str]
    rolling_windows: List[int]
    min_date: Optional[str] = None
    max_date: Optional[str] = None


class StrategyTimeseriesPoint(BaseModel):
    """Timeseries point for per-strategy chart rendering."""
    date: str
    normalized_equity: float
    drawdown: float
    rolling_sharpe: Optional[float] = None
    rolling_sortino: Optional[float] = None
    rolling_volatility: Optional[float] = None
    period_return: Optional[float] = None


class DistributionBucket(BaseModel):
    """Histogram/distribution bucket."""
    bucket: str
    count: int
    value: float


class StrategyDistributionResponse(BaseModel):
    """Distribution payloads for trade and return analysis."""
    strategy: str
    returns_histogram: List[DistributionBucket]
    trade_pnl_histogram: List[DistributionBucket]
    holding_period_histogram: List[DistributionBucket]
    pnl_by_symbol: List[DistributionBucket]


class StrategyVariantRow(BaseModel):
    """One parameter-set variant (best run among same params_hash)."""
    params_hash: str
    variant_label: Optional[str] = None
    strategy: str
    representative_run_id: int
    run_count: int
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    volatility: float
    params: Dict[str, Any] = Field(default_factory=dict)
    last_completed_at: Optional[str] = None


class StrategyVariantSummaryResponse(BaseModel):
    """Top-N parameter variants for a single strategy."""
    strategy: str
    objective: str
    top_n: int
    variants: List[StrategyVariantRow]


class VariantSeriesPayload(BaseModel):
    """Equity series for one variant (for multi-line charts)."""
    params_hash: str
    variant_label: Optional[str] = None
    representative_run_id: int
    points: List[StrategyTimeseriesPoint]


class StrategyVariantTimeseriesResponse(BaseModel):
    """Benchmark + multiple variant curves for Performance tab."""
    strategy: str
    benchmark_ticker: str
    granularity: str
    benchmark_points: List[StrategyTimeseriesPoint]
    variant_series: List[VariantSeriesPayload]


class TickerStrategyRow(BaseModel):
    """One strategy row within a ticker's leaderboard slice."""

    ticker: str
    strategy: str
    params_hash: str
    variant_label: Optional[str] = None
    representative_run_id: int
    run_count: int
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    volatility: float
    params: Dict[str, Any] = Field(default_factory=dict)
    last_completed_at: Optional[str] = None


class TickerStrategyLeaderboard(BaseModel):
    """Per-ticker ranked strategies for the leaderboard."""

    ticker: str
    strategies: List[TickerStrategyRow]


class TickerStrategyLeaderboardResponse(BaseModel):
    """Ticker × strategy leaderboard for Performance tab."""

    objective: str
    top_n: int
    tickers: List[TickerStrategyLeaderboard]