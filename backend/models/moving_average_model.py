"""
Model adapter exposing the MovingAverageStrategy as a BaseModel-compatible object.

This wrapper lets the frontend and `/api/models` endpoint treat the
rule-based moving-average strategy as a model with a Pydantic config schema
and a `predict`/`project`-style interface.
"""
from typing import Dict, Any, List
from pydantic import BaseModel as PydanticBaseModel, create_model

from backend.models.base import BaseModel as ModelBase
from backend.strategies.moving_average import MovingAverageStrategy


class MovingAverageModel(ModelBase):
    """Expose the MovingAverageStrategy through the model registry."""

    def __init__(self):
        strategy = MovingAverageStrategy()
        name = getattr(strategy, "name", "moving_average")
        super().__init__(
            name=name,
            type="rule",
            version="1.0",
            description=getattr(strategy, "description", "Moving average crossover strategy"),
            capabilities=["project"]
        )

        self._strategy = strategy

    def get_config_schema(self) -> PydanticBaseModel:
        """Generate a Pydantic model from the strategy's parameter schema."""
        params = getattr(self._strategy, "parameters_schema", {}) or {}
        fields = {}

        for key, spec in params.items():
            t = spec.get("type", "str")
            default = spec.get("default")

            if t == "int":
                field_type = int
            elif t == "float":
                field_type = float
            elif t == "bool":
                field_type = bool
            else:
                field_type = str

            fields[key] = (field_type, default)

        Model = create_model("MovingAverageConfig", **fields)
        return Model

    def predict(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a projection/equity-curve based prediction using the strategy.project API.

        Returns a dict with keys `predictions` and `meta` to satisfy the models endpoint.
        """
        params = config or {}

        # Use inputs to populate ticker/date information if available
        ticker = None
        if isinstance(inputs, dict):
            # common input keys: 'tickers' (list), 'ticker' (str)
            if 'tickers' in inputs and isinstance(inputs['tickers'], (list, tuple)) and len(inputs['tickers']) > 0:
                ticker = inputs['tickers'][0]
            elif 'ticker' in inputs:
                ticker = inputs.get('ticker')

        # Fallback to a neutral name
        if not ticker:
            ticker = 'PORTFOLIO'

        projection = self._strategy.project(parameters=params)

        # projection is expected to contain summary fields like 'projected_return', 'projected_final_value', 'confidence'
        if not isinstance(projection, dict):
            projection = {"projected_return": None, "projected_final_value": None, "confidence": None}

        projected_return = projection.get('projected_return')
        projected_value = projection.get('projected_final_value')
        confidence = projection.get('confidence', 0.0)

        # Build a normalized prediction dict matching API expectations
        prediction = {
            'ticker': ticker,
            'date': projection.get('timestamp'),
            'predicted_return': float(projected_return) if projected_return is not None else None,
            'confidence': float(confidence) if confidence is not None else 0.0,
            'position_pct': params.get('max_position_pct') if isinstance(params, dict) else None,
            'model_version': getattr(self, 'version', '1.0'),
            'features_used': [],
            'metadata': projection
        }

        return {'predictions': [prediction], 'meta': {'model_name': getattr(self, 'name', 'moving_average'), 'model_version': getattr(self, 'version', '1.0'), 'projection': projection}}

    def retrain(self, training_payload: Dict[str, Any], config: Dict[str, Any], background: bool = False) -> Dict[str, Any]:
        raise NotImplementedError("Retraining is not supported for rule-based strategies")

    def save(self, path):
        # No-op for rule-based wrapper
        return None

    @classmethod
    def load(cls, path):
        # Loading not supported — return a fresh instance
        return cls()
