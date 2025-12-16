"""
Three Moving Averages Crossover Model Adapter.

This module provides an adapter for the three MA crossover trading model
that uses the generate_ma_predictions script.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

# Import with error handling for missing dependencies
try:
    import pydantic
except ImportError as e:
    raise ImportError(f"Failed to import pydantic: {e}. Install with: pip install pydantic")

try:
    from backend.logging_config import get_component_logger
except ImportError as e:
    raise ImportError(f"Failed to import get_component_logger from backend.logging_config: {e}")

try:
    from backend.models.adapters.script_adapter import ScriptModelAdapter
except ImportError as e:
    raise ImportError(f"Failed to import ScriptModelAdapter from backend.models.adapters.script_adapter: {e}")

try:
    from backend.scripts.predictions.generate_ma_predictions import optimize_ma_periods, generate_predictions
except ImportError as e:
    raise ImportError(f"Failed to import functions from backend.scripts.predictions.generate_ma_predictions: {e}")


class ThreeMAConfig(pydantic.BaseModel):
    """Configuration schema for three MA model."""
    short_range: Optional[List[int]] = [3, 5, 7]
    medium_range: Optional[List[int]] = [15, 20, 25]
    long_range: Optional[List[int]] = [40, 50, 60]


class ThreeMAAdapter(ScriptModelAdapter):
    """Adapter for three moving averages crossover model."""

    def __init__(self):
        try:
            super().__init__(
                name="three_ma_crossover_v1",
                model_type="script",
                version="1.0.0",
                description="Three Moving Averages Crossover Strategy",
                capabilities=["predict", "retrain"]
            )

            # Store optimized periods
            self._optimized_periods = None
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ThreeMAAdapter: {e}")

    def get_config_schema(self) -> type[pydantic.BaseModel]:
        """Return the configuration schema for this model."""
        return ThreeMAConfig

    def predict(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions using the three MA strategy."""
        try:
            start_date = inputs.get("start")
            end_date = inputs.get("end")
            tickers = inputs.get("tickers", [])
            skip_optimization = inputs.get("skip_optimization", True)
            fixed_short = inputs.get("fixed_short", 5)
            fixed_medium = inputs.get("fixed_medium", 20)
            fixed_long = inputs.get("fixed_long", 50)

            if not start_date or not end_date:
                raise ValueError("start and end dates are required")

            if not tickers:
                raise ValueError("tickers list cannot be empty")

            # Get database connection
            conn = self._get_db_connection()

            try:
                if skip_optimization:
                    # Use fixed periods
                    short_period = fixed_short
                    medium_period = fixed_medium
                    long_period = fixed_long
                    self.logger.info(f"Using fixed MA periods: {short_period}, {medium_period}, {long_period}")
                else:
                    # Optimize periods
                    short_range = config.get("short_range", [3, 5, 7])
                    medium_range = config.get("medium_range", [15, 20, 25])
                    long_range = config.get("long_range", [40, 50, 60])

                    short_period, medium_period, long_period, sharpe = optimize_ma_periods(
                        conn, start_date, end_date, short_range, medium_range, long_range, tickers
                    )
                    self._optimized_periods = (short_period, medium_period, long_period)
                    self.logger.info(f"Optimized MA periods: {short_period}, {medium_period}, {long_period} (Sharpe: {sharpe:.3f})")

                # Generate predictions
                generate_predictions(conn, start_date, end_date, short_period, medium_period, long_period)

                # Query predictions from database
                cur = conn.cursor()
                query = """
                    SELECT ticker, dt, predicted_return, enter_prob, suggested_position_pct, exit_prob
                    FROM trading_model_predictions
                    WHERE model = ? AND dt >= ? AND dt <= ?
                    ORDER BY ticker, dt
                """
                cur.execute(query, (self.name, start_date, end_date))
                rows = cur.fetchall()

                # Convert to unified format
                predictions = []
                for row in rows:
                    ticker, dt, predicted_return, enter_prob, suggested_position_pct, exit_prob = row
                    predictions.append({
                        "ticker": ticker,
                        "date": dt,
                        "predicted_return": predicted_return,
                        "confidence": enter_prob if suggested_position_pct > 0 else exit_prob,
                        "position_pct": suggested_position_pct,
                        "model_version": self.version,
                        "features_used": ["short_ma", "medium_ma", "long_ma"],
                        "metadata": {
                            "short_period": short_period,
                            "medium_period": medium_period,
                            "long_period": long_period,
                            "signal_type": "bullish" if suggested_position_pct > 0 else "bearish" if suggested_position_pct < 0 else "neutral"
                        }
                    })

                self.logger.info(f"Generated {len(predictions)} predictions for {len(tickers)} tickers")

                return {
                    "predictions": predictions,
                    "meta": {
                        "model_name": self.name,
                        "model_version": self.version,
                        "generated_at": datetime.now().isoformat(),
                        "date_range": {"start": start_date, "end": end_date},
                        "tickers": tickers,
                        "ma_periods": {
                            "short": short_period,
                            "medium": medium_period,
                            "long": long_period
                        },
                        "optimization_skipped": skip_optimization
                    }
                }

            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Prediction failed for model {self.name}: {str(e)}")
            raise

    def retrain(self, training_payload: Dict[str, Any], config: Dict[str, Any], background: bool = False) -> Dict[str, Any]:
        """Retrain the model by optimizing MA periods."""
        try:
            start_date = training_payload.get("start_date")
            end_date = training_payload.get("end_date")
            tickers = training_payload.get("tickers", [])

            if not start_date or not end_date:
                raise ValueError("start_date and end_date are required")

            # Get database connection
            conn = self._get_db_connection()

            try:
                short_range = config.get("short_range", [3, 5, 7])
                medium_range = config.get("medium_range", [15, 20, 25])
                long_range = config.get("long_range", [40, 50, 60])

                short_period, medium_period, long_period, sharpe = optimize_ma_periods(
                    conn, start_date, end_date, short_range, medium_range, long_range, tickers
                )

                # Store optimized periods
                self._optimized_periods = (short_period, medium_period, long_period)

                self.logger.info(f"Retrained model with optimized periods: {short_period}, {medium_period}, {long_period} (Sharpe: {sharpe:.3f})")

                return {
                    "status": "completed",
                    "optimized_periods": {
                        "short": short_period,
                        "medium": medium_period,
                        "long": long_period
                    },
                    "sharpe_ratio": sharpe,
                    "training_date_range": {"start": start_date, "end": end_date}
                }

            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Retraining failed for model {self.name}: {str(e)}")
            raise