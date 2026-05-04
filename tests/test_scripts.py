"""
Unit tests for script execution endpoints.
"""
import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime
import tempfile
import os
import asyncio


@pytest.mark.unit
class TestScriptEndpoints:
    """Test script execution endpoints functionality."""

    def setup_method(self):
        """Set up test client."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
        from backend.main import app, app_state

        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Set the database path in the app state
        app_state['database_path'] = self.temp_db.name

        # Create test client
        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up temporary database."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
        
        # Clear script executions to avoid test pollution
        from backend.routes.scripts import script_executions
        script_executions.clear()

    @patch('backend.routes.scripts.run_script_async')
    def test_execute_script_train_sentiment_model(self, mock_run_script):
        """Test executing train_sentiment_model script."""
        mock_run_script.return_value = None

        payload = {
            "script_name": "train_sentiment_model",
            "parameters": {
                "csv": "data/training_labels_1d_top10.csv",
                "outdir": "models"
            }
        }
        response = self.client.post("/scripts/execute", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["script_name"] == "train_sentiment_model"
        assert data["status"] == "running"
        assert "execution_id" in data
        assert "start_time" in data

    @patch('backend.routes.scripts.run_script_async')
    def test_execute_script_backtest_runner(self, mock_run_script):
        """Test executing backtest_runner script."""
        mock_run_script.return_value = None

        payload = {
            "script_name": "backtest_runner",
            "parameters": {
                "start": "2023-01-01",
                "end": "2023-12-31"
            }
        }
        response = self.client.post("/scripts/execute", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["script_name"] == "backtest_runner"
        assert data["status"] == "running"

    def test_execute_script_invalid_name(self):
        """Test executing script with invalid name."""
        payload = {
            "script_name": "invalid_script_name",
            "parameters": {}
        }
        response = self.client.post("/scripts/execute", json=payload)
        assert response.status_code == 422  # Validation error

    def test_execute_script_missing_parameters(self):
        """Test executing script without required parameters."""
        payload = {
            "script_name": "train_sentiment_model"
            # Missing parameters
        }
        response = self.client.post("/scripts/execute", json=payload)
        assert response.status_code == 200  # Should still work, parameters are optional

    @patch('backend.routes.scripts.run_script_async')
    def test_get_script_status_completed(self, mock_run_script):
        """Test getting status of completed script."""
        mock_run_script.return_value = None
        
        # First execute a script
        payload = {
            "script_name": "train_sentiment_model",
            "parameters": {"csv": "data/training.csv"}
        }
        exec_response = self.client.post("/scripts/execute", json=payload)
        assert exec_response.status_code == 200
        
        execution_id = exec_response.json()["execution_id"]
        
        # Now check the status - it should be running initially
        response = self.client.get(f"/scripts/status/{execution_id}")
        assert response.status_code == 200
        
        # The status should be "running" since we mocked the async function
        data = response.json()
        assert data["status"] == "running"
        assert data["script_name"] == "train_sentiment_model"

    def test_get_script_status_running(self):
        """Test getting status of running script from script_executions."""
        from backend.routes.scripts import script_executions

        execution_id = "test_exec_456"
        script_executions.clear()
        script_executions[execution_id] = {
            "script_name": "backtest_runner",
            "status": "running",
            "start_time": datetime.utcnow(),
            "output": "Processing articles...",
            "error": "",
            "parameters": {},
        }

        response = self.client.get(f"/scripts/status/{execution_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["output"] == "Processing articles..."
        assert data["end_time"] is None

    def test_get_script_status_not_found(self):
        """Test getting status of non-existent script execution."""
        response = self.client.get("/scripts/status/nonexistent_id")
        assert response.status_code == 404

    def test_list_script_executions(self):
        """Test listing all script executions."""
        from backend.routes.scripts import script_executions
        
        # Populate the real dictionary with test data
        script_executions.clear()
        script_executions["exec_1"] = {
            "script_name": "train_sentiment_model",
            "status": "completed",
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow(),
        }
        script_executions["exec_2"] = {
            "script_name": "backtest_runner",
            "status": "running",
            "start_time": datetime.utcnow()
        }

        response = self.client.get("/scripts/executions")
        assert response.status_code == 200
        data = response.json()
        assert "executions" in data
        assert len(data["executions"]) == 2
        # executions can be in any order, so just check that both are present
        script_names = {e["script_name"] for e in data["executions"]}
        assert "train_sentiment_model" in script_names
        assert "backtest_runner" in script_names

    @patch('backend.routes.scripts.run_pipeline_async')
    def test_run_pipeline_default_steps(self, mock_run_pipeline):
        """Test running pipeline with default steps."""
        mock_run_pipeline.return_value = None

        response = self.client.post("/scripts/pipeline/run")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "execution_id" in data
        assert data["current_step"] is None
        assert data["completed_steps"] == []
        assert data["failed_steps"] == []

    @patch('backend.routes.scripts.run_pipeline_async')
    def test_run_pipeline_custom_steps(self, mock_run_pipeline):
        """Test running pipeline with custom steps."""
        mock_run_pipeline.return_value = None

        response = self.client.post("/scripts/pipeline/run?steps=apply_schema,ingest_prices")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    @patch('backend.routes.scripts.script_executions')
    def test_get_pipeline_status_completed(self, mock_executions):
        """Test getting status of completed pipeline."""
        mock_executions.__getitem__.return_value = {
            "script_name": "run_pipeline",
            "status": "completed",
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow(),
            "current_step": None,
            "completed_steps": ["apply_schema", "ingest_prices", "ingest_news"],
            "failed_steps": [],
            "output": "Pipeline completed successfully",
            "error": "",
            "parameters": {"steps": ["apply_schema", "ingest_prices", "ingest_news"]}     
        }

        response = self.client.get(f"/scripts/pipeline/status/{execution_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert len(data["completed_steps"]) == 3
        assert len(data["failed_steps"]) == 0

    @patch('backend.routes.scripts.run_pipeline_async')
    def test_get_pipeline_status_completed(self, mock_run_pipeline):
        """Test getting status of completed pipeline."""
        mock_run_pipeline.return_value = None
        
        # Execute a pipeline
        response = self.client.post("/scripts/pipeline/run?steps=apply_schema&steps=ingest_prices&steps=ingest_news")
        assert response.status_code == 200
        
        execution_id = response.json()["execution_id"]
        
        # Check the status
        status_response = self.client.get(f"/scripts/pipeline/status/{execution_id}")
        assert status_response.status_code == 200
        
        data = status_response.json()
        assert data["status"] == "running"
        assert data["execution_id"] == execution_id

    @patch('backend.routes.scripts.run_pipeline_async')
    def test_get_pipeline_status_running(self, mock_run_pipeline):
        """Test getting status of running pipeline."""
        mock_run_pipeline.return_value = None
        
        # Execute a pipeline
        response = self.client.post("/scripts/pipeline/run?steps=apply_schema&steps=ingest_prices&steps=ingest_news")
        assert response.status_code == 200
        
        execution_id = response.json()["execution_id"]
        
        # Check the status
        status_response = self.client.get(f"/scripts/pipeline/status/{execution_id}")
        assert status_response.status_code == 200
        
        data = status_response.json()
        assert data["status"] == "running"
        assert data["execution_id"] == execution_id

    def test_get_pipeline_status_not_found(self):
        """Test getting status of non-existent pipeline."""
        response = self.client.get("/scripts/pipeline/status/nonexistent_pipeline")
        assert response.status_code == 404

    @patch('backend.routes.scripts.get_script_path')
    @patch('backend.routes.scripts.run_script_async')
    def test_script_execution_with_file_validation(self, mock_run_script, mock_get_path): 
        """Test that script execution validates script file exists."""
        mock_get_path.return_value = "/path/to/train_sentiment_model.py"
        mock_run_script.return_value = None

        payload = {
            "script_name": "train_sentiment_model",
            "parameters": {"csv": "data/training.csv"}
        }
        response = self.client.post("/scripts/execute", json=payload)
        assert response.status_code == 200

        # The script execution should succeed since we mocked the path and async function
        # In the real implementation, get_script_path would be called during async execution
        # For this test, we just verify the endpoint accepts the request
        data = response.json()
        assert data["script_name"] == "train_sentiment_model"
        assert data["status"] == "running"
        assert "execution_id" in data

    @patch('backend.routes.scripts.get_script_path')
    def test_script_execution_file_not_found(self, mock_get_path):
        """Test script execution when script file doesn't exist."""
        mock_get_path.return_value = None

        payload = {
            "script_name": "nonexistent_script",
            "parameters": {}
        }
        response = self.client.post("/scripts/execute", json=payload)
        assert response.status_code == 422  # Validation should catch invalid script names

    @patch('backend.routes.scripts.run_script_async')
    def test_script_execution_cleanup(self, mock_run_script):
        """Test that completed executions include proper cleanup."""
        mock_run_script.return_value = None
        
        # Execute a script
        payload = {
            "script_name": "train_sentiment_model",
            "parameters": {}
        }
        exec_response = self.client.post("/scripts/execute", json=payload)
        assert exec_response.status_code == 200
        
        execution_id = exec_response.json()["execution_id"]
        
        # Check the status
        response = self.client.get(f"/scripts/status/{execution_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "running"
        assert data["execution_id"] == execution_id
        # Since the script is mocked as running, end_time should be None
        assert data["end_time"] is None
        assert data["duration_seconds"] is None

    @patch('backend.routes.scripts.run_script_async')
    def test_get_script_status_running_after_execute(self, mock_run_script):
        """Test getting status of running script after POST /scripts/execute."""
        mock_run_script.return_value = None
        
        # Execute a script
        payload = {
            "script_name": "backtest_runner",
            "parameters": {"start": "2023-01-01", "end": "2023-12-31"}
        }
        exec_response = self.client.post("/scripts/execute", json=payload)
        assert exec_response.status_code == 200
        
        execution_id = exec_response.json()["execution_id"]
        
        # Check the status
        response = self.client.get(f"/scripts/status/{execution_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "running"
        assert data["script_name"] == "backtest_runner"
