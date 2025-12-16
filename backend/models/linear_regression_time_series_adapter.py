"""
Linear Regression Time Series Model Adapter.

This module provides an adapter for a linear regression model that predicts
percentage returns for 1-day, 3-day, and 7-day horizons based on lagged
daily returns as features.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import sqlite3

# Import with error handling for missing dependencies
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
except ImportError as e:
    raise ImportError(f"Failed to import scikit-learn: {e}. Install with: pip install scikit-learn")

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


class LinearRegressionConfig(pydantic.BaseModel):
    """Configuration schema for linear regression time series model."""
    lag_days: int = pydantic.Field(default=10, ge=1, le=100, description="Number of lagged returns to use as features")
    horizons: List[int] = pydantic.Field(default=[1, 3, 7], description="Supported prediction horizons in days")


class LinearRegressionTimeSeriesAdapter(ScriptModelAdapter):
    """Adapter for linear regression time series model."""

    def __init__(self):
        try:
            super().__init__(
                name="linear_regression_time_series_v1",
                model_type="linear_regression",
                version="1.0.0",
                description="Linear Regression Time Series Model for Return Prediction",
                capabilities=["predict", "retrain"]
            )

            # Store trained models for each horizon
            self._models: Dict[int, LinearRegression] = {}
            self._feature_columns: List[str] = []

        except Exception as e:
            raise RuntimeError(f"Failed to initialize LinearRegressionTimeSeriesAdapter: {e}")

    def get_config_schema(self) -> type[pydantic.BaseModel]:
        """Return the configuration schema for this model."""
        return LinearRegressionConfig

    def _predict_impl(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the linear regression model."""
        try:
            ticker = inputs.get("ticker")
            start_date = inputs.get("start_date")
            end_date = inputs.get("end_date")
            horizon = inputs.get("horizon")

            if not all([ticker, start_date, end_date, horizon]):
                raise ValueError("ticker, start_date, end_date, and horizon are required")

            horizon = int(horizon)
            if horizon not in [1, 3, 7]:
                raise ValueError("horizon must be 1, 3, or 7")

            if horizon not in self._models:
                raise ValueError(f"No trained model available for horizon {horizon}")

            # Fetch historical price data
            price_data = self._fetch_price_data(ticker, start_date, end_date)

            if len(price_data) < config.get("lag_days", 10) + horizon:
                raise ValueError(f"Insufficient data for prediction. Need at least {config.get('lag_days', 10) + horizon} days")

            # Compute lagged return features for the most recent date
            features = self._compute_features(price_data, config.get("lag_days", 10))

            if features.empty:
                raise ValueError("Could not compute features from available data")

            # Get the most recent feature vector
            latest_features = features.iloc[-1:].values

            # Make prediction
            model = self._models[horizon]
            predicted_return = float(model.predict(latest_features)[0])

            # Use fixed confidence for demo (could be improved with prediction intervals)
            confidence = 0.7

            # Get the prediction date (most recent date in data)
            prediction_date = price_data.index[-1].strftime('%Y-%m-%d')

            prediction = {
                "ticker": ticker,
                "date": prediction_date,
                "predicted_return": predicted_return,
                "confidence": confidence,
                "horizon": horizon,
                "model_version": self.version,
                "features_used": self._feature_columns
            }

            result = {
                "predictions": [prediction],
                "meta": {
                    "model_name": self.name,
                    "model_version": self.version,
                    "generated_at": prediction_date,
                    "date_range": {"start": start_date, "end": end_date},
                    "tickers": [ticker],
                    "horizon": horizon
                }
            }

            self.logger.info(f"LinearRegressionTimeSeriesAdapter prediction result: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Prediction failed for model {self.name}: {str(e)}")
            raise

    def retrain(self, training_payload: Dict[str, Any], config: Dict[str, Any], background: bool = False) -> Dict[str, Any]:
        """Retrain the model by fitting linear regression models for each horizon."""
        try:
            start_date = training_payload.get("start_date")
            end_date = training_payload.get("end_date")
            tickers = training_payload.get("tickers", [])

            if not all([start_date, end_date, tickers]):
                raise ValueError("start_date, end_date, and tickers are required")

            lag_days = config.get("lag_days", 10)
            horizons = config.get("horizons", [1, 3, 7])

            self.logger.info(f"Retraining model with lag_days={lag_days}, horizons={horizons}")

            # Get database connection
            conn = self._get_db_connection()

            try:
                all_features = []
                all_targets = {h: [] for h in horizons}

                # Collect training data from all tickers
                for ticker in tickers:
                    try:
                        price_data = self._fetch_price_data_db(conn, ticker, start_date, end_date)
                        if len(price_data) < lag_days + max(horizons):
                            self.logger.warning(f"Insufficient data for {ticker}, skipping")
                            continue

                        # Compute features
                        features = self._compute_features(price_data, lag_days)
                        if features.empty:
                            continue

                        # Compute targets for each horizon
                        for horizon in horizons:
                            targets = self._compute_targets(price_data, horizon)

                            # Align features and targets (features predict future returns)
                            # Features are computed from price_data, targets are shifted
                            aligned_features = features.iloc[:-horizon] if horizon > 0 else features
                            aligned_targets = targets.iloc[horizon:] if horizon > 0 else targets

                            # Ensure we have matching lengths and no NaN values
                            min_len = min(len(aligned_features), len(aligned_targets))
                            if min_len > 0:
                                aligned_features = aligned_features.iloc[:min_len]
                                aligned_targets = aligned_targets.iloc[:min_len]

                                # Drop any remaining NaN values
                                valid_idx = aligned_targets.notna()
                                if valid_idx.any():
                                    # Use loc with the same index to ensure alignment
                                    common_idx = aligned_features.index.intersection(aligned_targets.index)
                                    aligned_features = aligned_features.loc[common_idx]
                                    aligned_targets = aligned_targets.loc[common_idx]

                                    # Filter out NaN targets
                                    final_valid = aligned_targets.notna()
                                    if final_valid.any():
                                        aligned_features = aligned_features.loc[final_valid]
                                        aligned_targets = aligned_targets.loc[final_valid]

                                        if len(aligned_features) > 0 and len(aligned_targets) > 0:
                                            all_features.append(aligned_features)
                                            all_targets[horizon].append(aligned_targets)

                    except Exception as e:
                        self.logger.warning(f"Failed to process data for {ticker}: {e}")
                        continue

                if not all_features:
                    raise ValueError("No valid training data collected")

                # Combine all ticker data
                X = pd.concat(all_features, ignore_index=True)

                # Train model for each horizon
                trained_models = {}
                metrics = {}

                for horizon in horizons:
                    if not all_targets[horizon]:
                        self.logger.warning(f"No targets collected for horizon {horizon}")
                        continue

                    y = pd.concat(all_targets[horizon], ignore_index=True)

                    # Ensure X and y have same length
                    min_len = min(len(X), len(y))
                    X_h = X.iloc[:min_len]
                    y_h = y.iloc[:min_len]

                    # Train model
                    model = LinearRegression()
                    model.fit(X_h, y_h)

                    # Calculate training metrics
                    y_pred = model.predict(X_h)
                    mse = mean_squared_error(y_h, y_pred)
                    rmse = np.sqrt(mse)

                    trained_models[horizon] = model
                    metrics[horizon] = {
                        "mse": float(mse),
                        "rmse": float(rmse),
                        "samples": len(y_h)
                    }

                    self.logger.info(f"Trained model for horizon {horizon}: RMSE={rmse:.4f}, samples={len(y_h)}")

                # Store trained models
                self._models = trained_models
                self._feature_columns = [f"lag_{i}" for i in range(1, lag_days + 1)]

                # Mark as initialized
                self.is_initialized = True

                return {
                    "status": "completed",
                    "horizons_trained": list(trained_models.keys()),
                    "metrics": metrics,
                    "training_date_range": {"start": start_date, "end": end_date},
                    "tickers_used": len([t for t in tickers if any(all_targets[h] for h in horizons)])
                }

            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Retraining failed for model {self.name}: {str(e)}")
            raise

    def save(self, path: Path) -> None:
        """Save the trained models to disk."""
        try:
            save_data = {
                "models": self._models,
                "feature_columns": self._feature_columns,
                "meta": {
                    "name": self.name,
                    "type": self.type,
                    "version": self.version,
                    "description": self.description,
                    "capabilities": self.capabilities
                }
            }

            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(save_data, path)
            self.logger.info(f"Model saved to {path}")

        except Exception as e:
            self.logger.error(f"Failed to save model {self.name}: {str(e)}")
            raise

    @classmethod
    def load(cls, path: Path) -> 'LinearRegressionTimeSeriesAdapter':
        """Load a trained model from disk."""
        try:
            save_data = joblib.load(path)

            instance = cls()
            instance._models = save_data["models"]
            instance._feature_columns = save_data["feature_columns"]

            # Mark as initialized if models are loaded
            if instance._models:
                instance.is_initialized = True

            instance.logger.info(f"Model loaded from {path}")
            return instance

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}")

    def _fetch_price_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data for a ticker."""
        conn = self._get_db_connection()
        try:
            return self._fetch_price_data_db(conn, ticker, start_date, end_date)
        finally:
            conn.close()

    def _fetch_price_data_db(self, conn: sqlite3.Connection, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data from database."""
        query = """
            SELECT date, close
            FROM price_daily
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date
        """

        df = pd.read_sql_query(query, conn, params=[ticker.upper(), start_date, end_date])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        return df

    def _compute_features(self, price_data: pd.DataFrame, lag_days: int) -> pd.DataFrame:
        """Compute lagged return features from price data."""
        # Calculate daily returns
        returns = price_data['close'].pct_change()

        # Create lagged features
        features = pd.DataFrame(index=returns.index)
        for i in range(1, lag_days + 1):
            features[f'lag_{i}'] = returns.shift(i)

        # Drop rows with NaN values
        features = features.dropna()

        return features

    def _compute_targets(self, price_data: pd.DataFrame, horizon: int) -> pd.Series:
        """Compute target returns for a given horizon."""
        returns = price_data['close'].pct_change(horizon).shift(-horizon)
        return returns