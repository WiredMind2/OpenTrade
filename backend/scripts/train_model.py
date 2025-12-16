"""
Model training and retraining utilities.

This module provides functions for training and retraining machine learning models.
"""

from typing import Dict, Any, Optional
from backend.logging_config import get_component_logger

logger = get_component_logger(__name__)


def retrain_model(model_name: str, start_date: Optional[str], end_date: Optional[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Retrain a model with new data.

    Args:
        model_name: Name of the model to retrain
        start_date: Start date for training data (required)
        end_date: End date for training data (required)
        config: Optional configuration for retraining

    Returns:
        Dictionary containing retraining results

    Raises:
        ValueError: If start_date or end_date are missing
        ValueError: If model is not found or doesn't support retraining
    """
    # Validate required parameters
    if not start_date or not end_date:
        raise ValueError("start_date and end_date required")

    # Get model registry from app state
    from backend.main import app_state
    registry = app_state.get("model_registry")

    if not registry:
        raise ValueError("Model registry not available")

    # Get the model
    model = registry.get(model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' not found")

    # Check if model supports retraining
    if "retrain" not in model.capabilities:
        raise ValueError(f"Model '{model_name}' does not support retraining")

    # Prepare training payload
    training_payload = {
        "start_date": start_date,
        "end_date": end_date
    }

    # Use default config if none provided
    if config is None:
        config = {}

    logger.info(f"Starting retraining for model {model_name} with date range {start_date} to {end_date}")

    # Call model's retrain method
    result = model.retrain(training_payload, config)

    logger.info(f"Retraining completed for model {model_name}")

    return result