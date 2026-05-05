"""
Model-related schemas for the Trading Backtester API.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field


class RetrainRequest(BaseModel):
    """Request model for retraining a model."""
    training_payload: Dict[str, Any]
    config: Dict[str, Any]
    options: Dict[str, Any]


class RetrainResponse(BaseModel):
    """Response model for retraining a model."""
    job_id: Optional[str] = None
    status: str
    model_meta: Optional[Dict[str, Any]] = None


DegradeStatus = Literal["healthy", "watch", "degraded"]
SignalAction = Literal["buy", "sell", "hold"]


class SavedModelCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    strategy_name: str = Field(..., min_length=1)
    ticker: str = Field(..., min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)
    objective: str = Field(default="balanced")
    is_active: bool = Field(default=True)


class SavedModelUpdateRequest(BaseModel):
    name: Optional[str] = None
    objective: Optional[str] = None
    is_active: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None


class SavedModelEvaluateRequest(BaseModel):
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    objective: Optional[str] = None
    drift_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "return_drop": 0.05,
            "sharpe_drop": 0.4,
            "drawdown_increase": 0.05,
        }
    )


class SavedModelSignalsBatchRequest(BaseModel):
    """Rank saved models (by last stored metrics), then evaluate each only at the latest daily bar."""

    ticker: str = Field(..., min_length=1)
    objective: str = "balanced"
    top_n: int = Field(default=5, ge=1, le=25)
    include_model_ids: List[int] = Field(default_factory=list)
    exclude_model_ids: List[int] = Field(default_factory=list)
    as_of_date: Optional[str] = Field(
        default=None,
        description="Optional YYYY-MM-DD; uses latest daily bar on or before this date.",
    )


class SavedModelSignalResponse(BaseModel):
    model_id: int
    name: Optional[str] = None
    strategy_name: str
    ticker: str
    params: Dict[str, Any] = Field(default_factory=dict)
    params_hash: str
    as_of: str
    last_price: float
    action: SignalAction = "hold"
    target_pct: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    degrade_status: DegradeStatus = "healthy"
    degrade_reason: Optional[str] = None
    error: Optional[str] = None


class SavedModelBatchEvaluateRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    objective: str = "balanced"
    top_n: int = Field(default=5, ge=1, le=25)
    rank_after_evaluation: bool = Field(
        default=False,
        description="Evaluate up to max_evaluate candidates (after filters), then rank by fresh metrics.",
    )
    max_evaluate: int = Field(default=25, ge=1, le=50)
    include_model_ids: List[int] = Field(default_factory=list)
    exclude_model_ids: List[int] = Field(default_factory=list)
    drift_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "return_drop": 0.05,
            "sharpe_drop": 0.4,
            "drawdown_increase": 0.05,
        }
    )


class SavedModelResponse(BaseModel):
    id: int
    name: str
    strategy_name: str
    ticker: str
    params: Dict[str, Any]
    params_hash: str
    objective: str
    baseline_metrics: Optional[Dict[str, Any]] = None
    latest_metrics: Optional[Dict[str, Any]] = None
    latest_equity_curve: Optional[List[Dict[str, Any]]] = None
    degrade_status: DegradeStatus = "healthy"
    degrade_reason: Optional[str] = None
    last_evaluated_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_active: bool = True


class SavedModelEvaluationResponse(BaseModel):
    model_id: int
    name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    status: str
    strategy_name: str
    ticker: str
    params_hash: str
    objective: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    degrade_status: DegradeStatus = "healthy"
    degrade_reason: Optional[str] = None
    evaluated_at: str
    error: Optional[str] = None