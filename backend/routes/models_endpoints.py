"""
Model information endpoints for the Trading Backtester API.
"""
from datetime import datetime
from typing import List
import sys
import os

from fastapi import APIRouter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from backend.schemas import ModelInfo


router = APIRouter()


@router.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List available models."""
    from backend.main import app_state  # Import here to avoid circular imports

    models = []

    for model_name, model_data in app_state["models_loaded"].items():
        if 'lightgbm' in model_name:
            horizon = model_name.split('_')[-1]
            models.append(ModelInfo(
                name=model_name,
                version="1.0.0",
                horizon=horizon,
                last_trained=datetime.utcnow(),
                features=["article_sentiment", "price_momentum", "volume"],
                status="active"
            ))

    return models