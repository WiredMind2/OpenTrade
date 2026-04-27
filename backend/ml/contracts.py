"""
Prediction contract models shared by training and inference.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


HORIZONS = ("1d", "3d", "7d")
HorizonType = Literal["1d", "3d", "7d"]


class FeatureContract(BaseModel):
    """Schema describing model feature layout."""

    schema_version: str = "ml_features_v1"
    feature_names: List[str]
    description: str = "Unified feature contract for multi-horizon prediction models."
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PredictionIntervals(BaseModel):
    """Optional prediction interval bounds."""

    lower: Optional[float] = None
    upper: Optional[float] = None


class ModelMetadata(BaseModel):
    """Metadata persisted with model bundles and predictions."""

    model_name: str
    model_version: str
    horizon: HorizonType
    feature_schema_version: str = "ml_features_v1"
    trained_at: Optional[datetime] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class PredictionResult(BaseModel):
    """Canonical output from prediction service."""

    ticker: str
    horizon: HorizonType
    predicted_return: float
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: ModelMetadata
    features_used: List[str]
    intervals: Optional[PredictionIntervals] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
