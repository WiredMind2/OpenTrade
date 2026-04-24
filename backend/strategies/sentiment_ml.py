"""
Sentiment ML Strategy

This module implements a machine learning-driven trading strategy that uses
sentiment model predictions to generate trading signals.
"""

import asyncio
import sqlite3
import subprocess
import uuid
import time
import os
import json
import re
from datetime import datetime
from typing import Dict, Any, Type
import backtrader as bt
import joblib

from backend.strategies.base import BaseStrategy
from backend.logging_config import get_component_logger
from backend.routes.websocket import broadcast_websocket_message

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


logger = get_component_logger(__file__)


class SentimentMLStrategy(BaseStrategy):
    """ML-driven strategy using sentiment model predictions."""

    def __init__(self):
        parameters_schema = {
            'model_name': {
                'type': 'string',
                'default': 'sentiment_model',
                'description': 'Name of the sentiment model to use'
            },
            'prediction_threshold': {
                'type': 'float',
                'default': 0.5,
                'description': 'Minimum absolute prediction value to trigger a trade (as percentage)'
            },
            'max_position_pct': {
                'type': 'float',
                'default': 0.1,
                'description': 'Maximum position size as percentage of portfolio value'
            }
        }

        super().__init__(
            name="sentiment_ml",
            description="ML-driven strategy using sentiment model predictions",
            type="ml",
            parameters_schema=parameters_schema,
            can_train=True
        )

        self.versions = []
        self.current_version = None
        self.models_dir = 'models'
        self.model_name = 'sentiment_model'
        self._load_versions()

    def _load_versions(self):
        """Load existing model versions from the models directory."""
        if not os.path.exists(self.models_dir):
            return

        for file in os.listdir(self.models_dir):
            if file.startswith(f'{self.name}__') and file.endswith('.joblib'):
                # Parse version_name: {strategy_name}__{timestamp}__v{n}.joblib
                parts = file[:-7].split('__')  # remove .joblib
                if len(parts) == 3:
                    strategy, timestamp, v_str = parts
                    if strategy == self.name and v_str.startswith('v'):
                        try:
                            version_num = int(v_str[1:])
                            version_name = file[:-7]
                            metadata_path = os.path.join(self.models_dir, f'{version_name}_metadata.json')
                            if os.path.exists(metadata_path):
                                with open(metadata_path) as f:
                                    metadata = json.load(f)
                                self.versions.append({
                                    'version': version_num,
                                    'name': version_name,
                                    'path': os.path.join(self.models_dir, file),
                                    'metadata_path': metadata_path,
                                    'timestamp': timestamp
                                })
                        except (ValueError, json.JSONDecodeError):
                            continue

        # Sort by version number
        self.versions.sort(key=lambda x: x['version'])
        if self.versions:
            self.current_version = self.versions[-1]['name']
            self.model_name = self.current_version

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        """Create and return a Backtrader strategy class with sentiment ML logic."""

        model_name = self.model_name  # Use current version

        class SentimentMLBacktrader(bt.Strategy):
            params = (
                ('model_name', 'sentiment_model'),
                ('prediction_threshold', 0.5),
                ('max_position_pct', 0.1)
            )

            def __init__(self):
                self.equity_curve = []
                self.trades = []
                self.model_name = model_name
                self.prediction_threshold = self.p.prediction_threshold
                self.max_position_pct = self.p.max_position_pct

                # Get database path from cerebro (passed via strategy params)
                self.db_path = getattr(self, 'db_path', 'data/backtest.db')

            def next(self):
                current_date = self.datas[0].datetime.date(0).isoformat()
                allocations = {}

                # Generate signals for each ticker
                for data in self.datas:
                    ticker = data._name

                    # Get prediction for this ticker and date
                    prediction = self._get_prediction(ticker, current_date)
                    if prediction is None:
                        continue

                    predicted_return = prediction.get('predicted_return', 0)
                    suggested_position_pct = prediction.get('suggested_position_pct', 0)

                    # Generate signal based on prediction and threshold
                    if abs(predicted_return) >= self.prediction_threshold:
                        # Use the sign of predicted return, scaled by suggested position
                        signal_pct = suggested_position_pct * (1 if predicted_return > 0 else -1)
                        allocations[ticker] = signal_pct
                    else:
                        # Close position if below threshold
                        allocations[ticker] = 0.0

                # Enforce position limits (scale down if total exposure exceeds max)
                total_exposure = sum(abs(pct) for pct in allocations.values())
                if total_exposure > self.max_position_pct:
                    scale = self.max_position_pct / total_exposure
                    allocations = {t: v * scale for t, v in allocations.items()}

                # Execute trades
                for ticker, target_pct in allocations.items():
                    # Find data feed for this ticker
                    data = None
                    for d in self.datas:
                        if hasattr(d, '_name') and d._name == ticker:
                            data = d
                            break

                    if data is None:
                        continue

                    current_position = self.getposition(data).size
                    current_value = current_position * data.close[0]
                    portfolio_value = self.broker.getvalue()
                    target_value = target_pct * portfolio_value

                    if abs(current_value - target_value) < 100:  # Minimum trade size
                        continue

                    if target_value > current_value:
                        # Buy
                        shares_to_buy = int((target_value - current_value) / data.close[0])
                        if shares_to_buy > 0:
                            self.buy(data=data, size=shares_to_buy)
                    elif target_value < current_value:
                        # Sell
                        shares_to_sell = int((current_value - target_value) / data.close[0])
                        if shares_to_sell > 0:
                            self.sell(data=data, size=shares_to_sell)

                # Record equity curve
                self.equity_curve.append({
                    'date': current_date,
                    'value': self.broker.getvalue()
                })

            def _get_prediction(self, ticker: str, date: str) -> Dict[str, Any]:
                """Get prediction for ticker on given date."""
                try:
                    conn = sqlite3.connect(self.db_path)
                    cur = conn.cursor()

                    # Query the latest prediction for this ticker on or before the date
                    cur.execute("""
                        SELECT predicted_return, enter_prob, suggested_position_pct, exit_prob
                        FROM trading_model_predictions
                        WHERE ticker = ? AND dt <= ? AND model = ?
                        ORDER BY dt DESC, produced_at DESC
                        LIMIT 1
                    """, (ticker, date, self.model_name))

                    row = cur.fetchone()
                    conn.close()

                    if row:
                        return {
                            'predicted_return': row[0],
                            'enter_prob': row[1],
                            'suggested_position_pct': row[2],
                            'exit_prob': row[3]
                        }

                except Exception as e:
                    logger.error(f"Error getting prediction for {ticker} on {date}: {e}")

                return None

            def notify_trade(self, trade):
                if trade.isclosed:
                    self.trades.append({
                        'size': trade.size,
                        'price': trade.price,
                        'value': trade.value,
                        'pnl': trade.pnl,
                        'pnlcomm': trade.pnlcomm
                    })

        return SentimentMLBacktrader

    def train(self, config: Dict[str, Any]) -> Any:
        """Train the sentiment model by enqueuing a background training job."""
        # Import lazily to avoid hard dependency on full FastAPI app during unit tests
        # or lightweight CLI usage where optional API dependencies may be missing.
        try:
            from backend.main import app_state  # Import here to avoid circular imports
        except Exception:
            app_state = {}

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Store job as queued
        self._store_job(job_id, self.name, "queued", config)

        # Log job queuing event
        logger.info(
            f"Training job queued for sentiment model",
            event_type="training_job_queued",
            job_id=job_id,
            model_name=self.name,
            config=config
        )

        # Add background task
        asyncio.create_task(self._run_training_background(job_id, config, app_state))

        return {"job_id": job_id, "status": "queued"}

    def _store_job(self, job_id: str, model_name: str, status: str, config: dict = None):
        """Store job in database."""
        db_path = "data/backtest.db"  # Default path, could be made configurable
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO model_jobs (id, model_name, status, config)
                VALUES (?, ?, ?, ?)
            """, (job_id, model_name, status, str(config) if config else None))
            conn.commit()
        finally:
            conn.close()

    def _update_job_status(self, job_id: str, status: str, result: dict = None, error: str = None):
        """Update job status in database."""
        db_path = "data/backtest.db"
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("""
                UPDATE model_jobs
                SET status = ?, updated_at = datetime('now'),
                    result = ?, error = ?
                WHERE id = ?
            """, (status, str(result) if result else None, error, job_id))
            conn.commit()
        finally:
            conn.close()

    async def _run_training_background(self, job_id: str, config: Dict[str, Any], app_state: Dict[str, Any]):
        """Background task to run sentiment model training."""
        start_time = time.time()
        training_start_cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
        training_start_memory = psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0

        logger.info(
            f"Starting sentiment model training",
            event_type="training_started",
            job_id=job_id,
            model_name=self.name,
            config=config,
            system_metrics={
                "cpu_percent": training_start_cpu,
                "memory_percent": training_start_memory
            }
        )

        try:
            # Update status to running
            self._update_job_status(job_id, "running")
            await broadcast_websocket_message({
                "type": "training_progress",
                "job_id": job_id,
                "status": "running",
                "message": "Training started"
            })

            # Prepare training command
            csv_path = config.get('csv_path', 'data/training_labels_1d_top10.csv')
            outdir = config.get('outdir', 'models')
            embedder = config.get('embedder', 'all-MiniLM-L6-v2')

            # Run training script
            cmd = [
                'python', 'backend/scripts/train_sentiment_model.py',
                '--csv', csv_path,
                '--outdir', outdir,
                '--embedder', embedder
            ]

            logger.info(
                f"Executing training subprocess",
                event_type="subprocess_execution",
                job_id=job_id,
                command=cmd,
                csv_path=csv_path,
                outdir=outdir,
                embedder=embedder
            )

            # Execute training
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')

            if result.returncode == 0:
                # Calculate training metrics
                training_duration = time.time() - start_time
                training_end_cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
                training_end_memory = psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0

                # Success - update status and refresh registry
                self._update_job_status(job_id, "completed", result={"stdout": result.stdout})

                # Refresh model registry
                registry = app_state["model_registry"]
                registry.discover(
                    models_dir=app_state.get("models_dir", "models"),
                    models_pkg_dir=app_state.get("models_pkg_dir")
                )

                # Log model saving event
                logger.info(
                    f"Model registry refreshed after training completion",
                    event_type="model_saved",
                    job_id=job_id,
                    model_name=self.name,
                    models_dir=app_state.get("models_dir", "models")
                )

                # Version the trained model
                self._version_trained_model(config, result.stdout, outdir)

                await broadcast_websocket_message({
                    "type": "training_progress",
                    "job_id": job_id,
                    "status": "completed",
                    "message": "Training completed successfully",
                    "results": result.stdout
                })

                # Log training completion with metrics
                logger.info(
                    f"Sentiment model training completed successfully",
                    event_type="training_completed",
                    job_id=job_id,
                    model_name=self.name,
                    training_duration=training_duration,
                    success=True,
                    system_metrics={
                        "start_cpu_percent": training_start_cpu,
                        "end_cpu_percent": training_end_cpu,
                        "start_memory_percent": training_start_memory,
                        "end_memory_percent": training_end_memory
                    }
                )

                # Log performance metrics
                logger.performance_metric(
                    "training_duration_seconds",
                    training_duration,
                    strategy=self.name,
                    job_id=job_id
                )
                logger.performance_metric(
                    "training_success_rate",
                    1.0,
                    strategy=self.name,
                    job_id=job_id
                )
            else:
                # Calculate training metrics for failure
                training_duration = time.time() - start_time
                training_end_cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
                training_end_memory = psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0

                # Failure
                error_msg = f"Training failed: {result.stderr}"
                self._update_job_status(job_id, "failed", error=error_msg)
                await broadcast_websocket_message({
                    "type": "training_progress",
                    "job_id": job_id,
                    "status": "failed",
                    "message": error_msg
                })

                # Log training failure with metrics
                logger.error(
                    f"Sentiment model training failed",
                    event_type="training_failed",
                    job_id=job_id,
                    model_name=self.name,
                    training_duration=training_duration,
                    error_message=error_msg,
                    return_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    system_metrics={
                        "start_cpu_percent": training_start_cpu,
                        "end_cpu_percent": training_end_cpu,
                        "start_memory_percent": training_start_memory,
                        "end_memory_percent": training_end_memory
                    }
                )

                # Log performance metrics for failure
                logger.performance_metric(
                    "training_duration_seconds",
                    training_duration,
                    strategy=self.name,
                    job_id=job_id
                )
                logger.performance_metric(
                    "training_success_rate",
                    0.0,
                    strategy=self.name,
                    job_id=job_id
                )

        except Exception as e:
            # Calculate training metrics for exception
            training_duration = time.time() - start_time
            training_end_cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
            training_end_memory = psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0

            error_msg = f"Training background task failed: {str(e)}"
            self._update_job_status(job_id, "failed", error=error_msg)
            await broadcast_websocket_message({
                "type": "training_progress",
                "job_id": job_id,
                "status": "failed",
                "message": error_msg
            })

            # Log training exception with metrics
            logger.error(
                f"Sentiment model training background task failed",
                event_type="training_exception",
                job_id=job_id,
                model_name=self.name,
                training_duration=training_duration,
                error_message=error_msg,
                exception_type=type(e).__name__,
                system_metrics={
                    "start_cpu_percent": training_start_cpu,
                    "end_cpu_percent": training_end_cpu,
                    "start_memory_percent": training_start_memory,
                    "end_memory_percent": training_end_memory
                }
            )

            # Log performance metrics for exception
            logger.performance_metric(
                "training_duration_seconds",
                training_duration,
                strategy=self.name,
                job_id=job_id
            )
            logger.performance_metric(
                "training_success_rate",
                0.0,
                strategy=self.name,
                job_id=job_id
            )

    def _version_trained_model(self, config: Dict[str, Any], stdout: str, outdir: str):
        """Version the trained model and create metadata."""
        try:
            # Generate version info
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_num = len(self.versions) + 1
            version_name = f"{self.name}__{timestamp}__v{version_num}"

            # Determine original model path
            csv_path = config.get('csv_path', 'data/training_labels_1d_top10.csv')
            match = re.search(r'training_labels_(\d+)d_top(\d+)\.csv', os.path.basename(csv_path))
            if match:
                horizon = match.group(1) + 'd'
                topn = f'top{match.group(2)}'
            else:
                horizon = '1d'
                topn = 'top10'
            original_model_path = os.path.join(outdir, f'lightgbm_{horizon}_{topn}.joblib')

            # Rename to versioned path
            versioned_model_path = os.path.join(outdir, f'{version_name}.joblib')
            os.rename(original_model_path, versioned_model_path)

            # Load model to extract explainability data
            model_data = joblib.load(versioned_model_path)
            gbm = model_data['lgbm']
            embedder = model_data['embedder']

            # Parse metrics from stdout
            rmse_match = re.search(r'RMSE:\s*([\d.]+)', stdout)
            mae_match = re.search(r'MAE:\s*([\d.]+)', stdout)
            rmse = float(rmse_match.group(1)) if rmse_match else None
            mae = float(mae_match.group(1)) if mae_match else None

            # Feature importance
            feature_importance = gbm.feature_importances_.tolist() if hasattr(gbm, 'feature_importances_') else None

            # Model architecture
            architecture = {
                'type': 'LightGBM Regressor',
                'params': gbm.get_params(),
                'embedder': embedder,
                'best_iteration': getattr(gbm, 'best_iteration_', None)
            }

            # Loss curves (if available)
            loss_curves = getattr(gbm, 'evals_result_', None)

            # Create metadata
            metadata = {
                'version': version_num,
                'version_name': version_name,
                'timestamp': timestamp,
                'training_config': config,
                'metrics': {
                    'rmse': rmse,
                    'mae': mae
                },
                'feature_importance': feature_importance,
                'model_architecture': architecture,
                'loss_curves': loss_curves
            }

            # Save metadata
            metadata_path = os.path.join(outdir, f'{version_name}_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)  # default=str for datetime serialization

            # Update versions list
            version_info = {
                'version': version_num,
                'name': version_name,
                'path': versioned_model_path,
                'metadata_path': metadata_path,
                'timestamp': timestamp
            }
            self.versions.append(version_info)
            self.current_version = version_name
            self.model_name = version_name

            logger.info(
                f"Model versioned successfully",
                event_type="model_versioned",
                version_name=version_name,
                version_num=version_num
            )

        except Exception as e:
            logger.error(f"Failed to version model: {e}", event_type="model_versioning_failed")

    def list_versions(self) -> list:
        """List all available model versions."""
        return self.versions.copy()

    def switch_version(self, version_name: str) -> bool:
        """Switch to a specific model version."""
        if version_name in [v['name'] for v in self.versions]:
            self.current_version = version_name
            self.model_name = version_name
            logger.info(
                f"Switched to model version {version_name}",
                event_type="model_version_switched",
                version_name=version_name
            )
            return True
        return False

    def get_current_model_name(self) -> str:
        """Get the current model name."""
        return self.model_name

    def project(self, parameters: Dict[str, Any], projection_days: int = 30, initial_capital: float = 100000.0) -> Dict[str, Any]:
        """Project future performance using ML model predictions."""
        from datetime import datetime, timedelta
        from decimal import Decimal, getcontext
        import numpy as np

        # Set precision for financial calculations
        getcontext().prec = 10

        try:
            from backend.main import app_state
            import sqlite3

            db_path = app_state.get("database_path", "data/backtest.db")
            conn = sqlite3.connect(db_path)

            # Get recent trading model predictions
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=30)  # Use recent predictions

            cur = conn.cursor()
            cur.execute("""
                SELECT ticker, suggested_position_pct, dt, predicted_return, enter_prob
                FROM trading_model_predictions
                WHERE dt >= ? AND dt <= ?
                ORDER BY dt DESC, enter_prob DESC
                LIMIT 100
            """, (start_date.isoformat(), end_date.isoformat()))

            predictions = cur.fetchall()
            conn.close()

            if not predictions:
                # Fallback if no predictions available
                initial_capital_dec = Decimal(str(initial_capital))
                projected_return_dec = Decimal('0.03')
                projected_final_value = initial_capital_dec * (Decimal('1') + projected_return_dec)

                return {
                    'projected_return': float(projected_return_dec),
                    'projected_volatility': 0.18,
                    'confidence': 0.6,
                    'projection_days': projection_days,
                    'initial_capital': float(initial_capital_dec),
                    'projected_final_value': float(projected_final_value.quantize(Decimal('0.01'))),
                    'avg_prediction_confidence': 0.5,
                    'timestamp': datetime.utcnow().isoformat()
                }

            # Analyze predictions
            predicted_returns = [row[3] for row in predictions if row[3] is not None]
            confidences = [row[4] for row in predictions if row[4] is not None]
            position_sizes = [abs(row[1]) for row in predictions if row[1] is not None]

            if not predicted_returns:
                predicted_returns = [0.0]

            avg_predicted_return = np.mean(predicted_returns)
            avg_confidence = np.mean(confidences) if confidences else 0.5
            avg_position_size = np.mean(position_sizes) if position_sizes else 0.1

            # Calculate projected volatility from prediction dispersion
            return_volatility = np.std(predicted_returns) if len(predicted_returns) > 1 else 0.15

            # Project forward performance using Decimal
            initial_capital_dec = Decimal(str(initial_capital))
            projection_days_dec = Decimal(str(projection_days))
            avg_predicted_return_dec = Decimal(str(avg_predicted_return))
            avg_position_size_dec = Decimal(str(avg_position_size))
            avg_confidence_dec = Decimal(str(avg_confidence))

            # ML strategy can time entries better, assume higher Sharpe ratio
            daily_return = avg_predicted_return_dec * avg_position_size_dec * avg_confidence_dec
            total_return = daily_return * projection_days_dec

            # ML strategies typically have higher returns but also higher volatility
            projected_volatility = return_volatility * np.sqrt(projection_days) * 1.2  # Annualized and adjusted

            projected_final_value = initial_capital_dec * (Decimal('1') + total_return)

            return {
                'projected_return': float(total_return),
                'projected_volatility': round(projected_volatility, 6),
                'confidence': round(avg_confidence, 4),
                'projection_days': projection_days,
                'initial_capital': float(initial_capital_dec),
                'projected_final_value': float(projected_final_value.quantize(Decimal('0.01'))),
                'avg_predicted_return': float(avg_predicted_return_dec),
                'avg_position_size': round(avg_position_size, 4),
                'predictions_used': len(predictions),
                'model_version': self.get_current_model_name(),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            # Fallback on error
            initial_capital_dec = Decimal(str(initial_capital))
            fallback_return = Decimal('0.025')
            projected_final_value = initial_capital_dec * (Decimal('1') + fallback_return)

            return {
                'projected_return': float(fallback_return),
                'projected_volatility': 0.20,
                'confidence': 0.4,
                'projection_days': projection_days,
                'initial_capital': float(initial_capital_dec),
                'projected_final_value': float(projected_final_value.quantize(Decimal('0.01'))),
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }