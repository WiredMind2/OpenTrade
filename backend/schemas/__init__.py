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
    model_version: str
    features_used: List[str]
    feature_schema_version: Optional[str] = None
    interval_lower: Optional[float] = None
    interval_upper: Optional[float] = None
    metadata: Dict[str, Any] = {}


class BacktestRequest(BaseModel):
    """Request model for running backtests."""
    strategy_name: str = Field(..., description="Name of the trading strategy")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(default=100000.0, description="Initial capital amount")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Strategy parameters")

    @field_validator('initial_capital')
    @classmethod
    def validate_initial_capital(cls, v):
        if v <= 0:
            raise ValueError('initial_capital must be positive')
        return v

    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v, info):
        if 'start_date' in info.data and v <= info.data['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class BacktestResult(BaseModel):
    """Response model for backtest results."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    completed_at: Optional[datetime] = None
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    timestamp: datetime
    metrics: Dict[str, Any]
    equity_curve: List[Dict[str, Any]]


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
            'run_pipeline', 'train_sentiment_model', 'generate_sentiment_predictions',
            'generate_trading_predictions', 'backtest_runner'
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


class MAPredictionRequest(BaseModel):
    """Request model for generating MA predictions."""
    start_date: str = Field(..., description="Start date for prediction generation (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for prediction generation (YYYY-MM-DD)")
    short_ma_range: Optional[List[int]] = Field(default=[3, 5, 7], description="Range of short MA periods for optimization")
    medium_ma_range: Optional[List[int]] = Field(default=[15, 20, 25], description="Range of medium MA periods for optimization")
    long_ma_range: Optional[List[int]] = Field(default=[40, 50, 60], description="Range of long MA periods for optimization")
    skip_optimization: bool = Field(default=False, description="Skip optimization and use fixed MA periods")
    fixed_short: Optional[int] = Field(default=5, description="Fixed short MA period when skip_optimization is true")
    fixed_medium: Optional[int] = Field(default=20, description="Fixed medium MA period when skip_optimization is true")
    fixed_long: Optional[int] = Field(default=50, description="Fixed long MA period when skip_optimization is true")

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v, info):
        if 'start_date' in info.data and v <= info.data['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class MAPredictionResponse(BaseModel):
    """Response model for MA prediction generation."""
    status: str  # 'running', 'completed', 'failed'
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    output: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


class ModelPredictRequest(BaseModel):
    """Request model for model predictions."""
    inputs: Dict[str, Any]
    config: Dict[str, Any]


class ModelPredictResponse(BaseModel):
    """Response model for model predictions."""
    predictions: List[Dict[str, Any]]
    meta: Dict[str, Any]


class StrategyAnalyticsQuery(BaseModel):
    """Filter options for strategy analytics queries."""
    strategies: List[str] = Field(default_factory=list)
    benchmark_ticker: str = Field(default="SPY")
    preset: str = Field(default="MAX")
    granularity: str = Field(default="daily")
    rolling_window: int = Field(default=30)


class StrategyFilterMetadataResponse(BaseModel):
    """Available filters for the strategy analytics dashboard."""
    strategies: List[str]
    benchmarks: List[str]
    available_presets: List[str]
    available_granularities: List[str]
    rolling_windows: List[int]
    min_date: Optional[str] = None
    max_date: Optional[str] = None


class StrategyMetricPoint(BaseModel):
    """Summary metrics for a single strategy."""
    strategy: str
    run_count: int
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    information_ratio: float
    alpha: float
    beta: float
    volatility: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    total_trades: int


class StrategyComparisonSummaryResponse(BaseModel):
    """Response payload for strategy-level KPI comparison."""
    benchmark_ticker: str
    granularity: str
    rolling_window: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    metrics: List[StrategyMetricPoint]


class StrategyTimeseriesPoint(BaseModel):
    """Timeseries point for per-strategy chart rendering."""
    date: str
    normalized_equity: float
    drawdown: float
    rolling_sharpe: Optional[float] = None
    rolling_sortino: Optional[float] = None
    rolling_volatility: Optional[float] = None
    period_return: Optional[float] = None


class StrategyTimeseriesResponse(BaseModel):
    """Strategy and benchmark chart-ready timeseries."""
    strategy: str
    benchmark_ticker: str
    granularity: str
    points: List[StrategyTimeseriesPoint]
    benchmark_points: List[StrategyTimeseriesPoint]
    monthly_returns: Dict[str, Dict[str, float]]


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