"""
Model-related schemas for the Trading Backtester API.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel


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