"""
Integration tests for model API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
import os

from backend.main import app, app_state
from backend.models.registry import ModelRegistry


class TestModelAPI:
    """Test the model API endpoints."""

    def setup_method(self):
        """Set up test client and mock registry."""
        self.client = TestClient(app)

        # Create a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()

        # Mock the database path
        app_state["database_path"] = self.temp_db.name

        # Create and populate registry with mock models
        registry = ModelRegistry()

        # Create a mock model
        mock_model = MagicMock()
        mock_model.name = "test_model"
        mock_model.type = "test"
        mock_model.version = "1.0.0"
        mock_model.description = "Test model"
        mock_model.capabilities = ["predict"]
        mock_model.get_config_schema.return_value.model_json_schema.return_value = {"type": "object"}

        registry.register(mock_model)
        app_state["model_registry"] = registry

    def teardown_method(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_db.name)
        except Exception:
            pass

    def test_get_models_list(self):
        """Test GET /api/models returns list of models."""
        response = self.client.get("/api/models")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) >= 1  # Should have at least our mock model

        # Find our test model
        test_model = next((m for m in data if m["name"] == "test_model"), None)
        assert test_model is not None
        assert test_model["type"] == "test"
        assert test_model["version"] == "1.0.0"
        assert test_model["description"] == "Test model"
        assert "predict" in test_model["capabilities"]
        assert "config_schema" in test_model

    @patch('backend.models.three_ma_adapter.ThreeMAAdapter')
    def test_predict_with_three_ma_model(self, mock_adapter_class):
        """Test POST /api/models/{name}/predict with three MA model."""
        # Mock the adapter instance
        mock_adapter = MagicMock()
        mock_adapter.name = "three_ma_crossover_v1"
        mock_adapter.predict.return_value = {
            "predictions": [
                {
                    "ticker": "AAPL",
                    "date": "2023-01-01",
                    "predicted_return": 0.05,
                    "confidence": 0.8,
                    "position_pct": 0.05,
                    "model_version": "1.0.0",
                    "features_used": ["short_ma", "medium_ma", "long_ma"],
                    "metadata": {
                        "short_period": 5,
                        "medium_period": 20,
                        "long_period": 50,
                        "signal_type": "bullish"
                    }
                }
            ],
            "meta": {
                "model_name": "three_ma_crossover_v1",
                "model_version": "1.0.0",
                "generated_at": "2023-01-01T12:00:00",
                "date_range": {"start": "2023-01-01", "end": "2023-01-02"},
                "tickers": ["AAPL"],
                "ma_periods": {"short": 5, "medium": 20, "long": 50},
                "optimization_skipped": True
            }
        }
        mock_adapter_class.return_value = mock_adapter

        # Register the three MA model
        registry = app_state["model_registry"]
        registry.register(mock_adapter)

        # Test prediction request
        request_data = {
            "inputs": {
                "start": "2023-01-01",
                "end": "2023-01-02",
                "tickers": ["AAPL"],
                "skip_optimization": True,
                "fixed_short": 5,
                "fixed_medium": 20,
                "fixed_long": 50
            },
            "config": {}
        }

        response = self.client.post("/api/models/three_ma_crossover_v1/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert "meta" in data
        assert len(data["predictions"]) == 1

        prediction = data["predictions"][0]
        assert prediction["ticker"] == "AAPL"
        assert prediction["predicted_return"] == 0.05
        assert prediction["confidence"] == 0.8

        assert data["meta"]["model_name"] == "three_ma_crossover_v1"
        assert data["meta"]["optimization_skipped"] is True

        # Verify the adapter was called correctly
        mock_adapter.predict.assert_called_once_with(request_data["inputs"], request_data["config"])

    def test_predict_model_not_found(self):
        """Test POST /api/models/{name}/predict with non-existent model."""
        request_data = {
            "inputs": {"test": "data"},
            "config": {}
        }

        response = self.client.post("/api/models/non_existent_model/predict", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert "Model 'non_existent_model' not found" in data["detail"]

    def test_predict_invalid_request(self):
        """Test POST /api/models/{name}/predict with invalid request data."""
        # Register a mock model that will raise an exception
        registry = app_state["model_registry"]
        mock_model = MagicMock()
        mock_model.name = "failing_model"
        mock_model.predict.side_effect = ValueError("Test error")
        registry.register(mock_model)

        request_data = {
            "inputs": {"invalid": "data"},
            "config": {}
        }

        response = self.client.post("/api/models/failing_model/predict", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "Prediction failed: Test error" in data["detail"]

    def test_predict_malformed_request(self):
        """Test POST /api/models/{name}/predict with malformed request."""
        # Register a valid model
        registry = app_state["model_registry"]
        mock_model = MagicMock()
        mock_model.name = "valid_model"
        mock_model.predict.return_value = {"predictions": [], "meta": {}}
        registry.register(mock_model)

        # Send malformed JSON
        response = self.client.post(
            "/api/models/valid_model/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422  # Validation error

    def test_retrain_model_background(self):
        """Test POST /api/models/{name}/retrain with background execution."""
        # Register a mock model that supports retraining
        registry = app_state["model_registry"]
        mock_model = MagicMock()
        mock_model.name = "retrainable_model"
        mock_model.capabilities = ["predict", "retrain"]
        registry.register(mock_model)

        request_data = {
            "training_payload": {"start_date": "2023-01-01", "end_date": "2023-12-31", "tickers": ["AAPL"]},
            "config": {"short_range": [3, 5, 7]},
            "options": {"background": True}
        }

        response = self.client.post("/api/models/retrainable_model/retrain", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "job_id" in data
        assert data["status"] in ["queued", "running", "completed"]
        assert data["model_meta"] is None

    def test_retrain_model_immediate(self):
        """Test POST /api/models/{name}/retrain with immediate execution."""
        # Register a mock model that supports retraining
        registry = app_state["model_registry"]
        mock_model = MagicMock()
        mock_model.name = "retrainable_model"
        mock_model.capabilities = ["predict", "retrain"]
        mock_model.retrain.return_value = {"status": "completed", "optimized_periods": {"short": 5, "medium": 20, "long": 50}}
        registry.register(mock_model)

        request_data = {
            "training_payload": {"start_date": "2023-01-01", "end_date": "2023-12-31", "tickers": ["AAPL"]},
            "config": {"short_range": [3, 5, 7]},
            "options": {"background": False}
        }

        response = self.client.post("/api/models/retrainable_model/retrain", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["job_id"] is None
        assert data["status"] == "completed"
        assert "optimized_periods" in data["model_meta"]

        # Verify retrain was called
        mock_model.retrain.assert_called_once_with(request_data["training_payload"], request_data["config"], background=False)

    def test_retrain_model_not_found(self):
        """Test POST /api/models/{name}/retrain with non-existent model."""
        request_data = {
            "training_payload": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            "config": {},
            "options": {}
        }

        response = self.client.post("/api/models/non_existent_model/retrain", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert "Model 'non_existent_model' not found" in data["detail"]

    def test_retrain_model_not_supported(self):
        """Test POST /api/models/{name}/retrain with model that doesn't support retraining."""
        # Register a mock model that doesn't support retraining
        registry = app_state["model_registry"]
        mock_model = MagicMock()
        mock_model.name = "non_retrainable_model"
        mock_model.capabilities = ["predict"]  # No "retrain"
        registry.register(mock_model)

        request_data = {
            "training_payload": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            "config": {},
            "options": {}
        }

        response = self.client.post("/api/models/non_retrainable_model/retrain", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "does not support retraining" in data["detail"]

    def test_retrain_model_joblib_not_supported(self):
        """Test POST /api/models/{name}/retrain with joblib model (should fail)."""
        # Register a mock joblib model
        registry = app_state["model_registry"]
        mock_model = MagicMock()
        mock_model.name = "joblib_model"
        mock_model.capabilities = ["predict", "retrain"]
        mock_model.retrain.side_effect = NotImplementedError("Retraining not supported for legacy joblib models")
        registry.register(mock_model)

        request_data = {
            "training_payload": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            "config": {},
            "options": {"background": False}
        }

        response = self.client.post("/api/models/joblib_model/retrain", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "Retraining not supported for legacy joblib models" in data["detail"]

    def test_get_job_status(self):
        """Test GET /api/jobs/{job_id} to get job status."""
        # First create a job by calling retrain in background mode
        registry = app_state["model_registry"]
        mock_model = MagicMock()
        mock_model.name = "test_job_model"
        mock_model.capabilities = ["predict", "retrain"]
        registry.register(mock_model)

        request_data = {
            "training_payload": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            "config": {},
            "options": {"background": True}
        }

        response = self.client.post("/api/models/test_job_model/retrain", json=request_data)
        assert response.status_code == 200
        job_data = response.json()
        job_id = job_data["job_id"]

        # Now get the job status
        response = self.client.get(f"/api/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == job_id
        assert data["model_name"] == "test_job_model"
        assert data["status"] in ["queued", "running", "completed"]

    def test_get_job_status_not_found(self):
        """Test GET /api/jobs/{job_id} with non-existent job."""
        response = self.client.get("/api/jobs/non-existent-job-id")

        assert response.status_code == 404
        data = response.json()
        assert "Job 'non-existent-job-id' not found" in data["detail"]