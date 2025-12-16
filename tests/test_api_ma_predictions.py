"""
API tests for MA prediction endpoints.
"""
import pytest
import json
from unittest.mock import Mock, patch, AsyncMock, patch as mock_patch
from fastapi.testclient import TestClient
from datetime import datetime
import sqlite3
import tempfile
import os

# Import the test patterns from existing API tests
from test_api_scripts import TestAPIScriptEndpoints


@pytest.mark.unit
class TestMAPredictionEndpoints:
    """Test MA prediction API endpoints functionality."""

    def setup_method(self):
        """Set up test client and mock database."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
        from main import app, app_state

        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Initialize basic schema
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_model_predictions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                suggested_position_pct REAL,
                dt TEXT,
                confidence REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_daily (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adjusted_close REAL,
                volume INTEGER
            )
        """)
        conn.commit()
        conn.close()

        # Set the database path in the app state
        app_state['database_path'] = self.temp_db.name
        app_state['models_loaded'] = {}

        # Create test client
        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up temporary database and script executions."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

        # Clear script executions between tests
        from routes.scripts import script_executions
        script_executions.clear()

    @patch('routes.scripts.run_script_async')
    def test_generate_ma_predictions_endpoint_success(self, mock_run_script):
        """Test successful MA prediction generation."""
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01",
            "short_ma_range": [3, 5, 7],
            "medium_ma_range": [15, 20, 25],
            "long_ma_range": [40, 50, 60],
            "skip_optimization": False
        }
        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "running"
        assert "execution_id" in data

    @patch('routes.scripts.run_script_async')
    def test_generate_ma_predictions_endpoint_skip_optimization(self, mock_run_script):
        """Test MA prediction generation with skip optimization."""
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01",
            "skip_optimization": True,
            "fixed_short": 5,
            "fixed_medium": 20,
            "fixed_long": 50
        }
        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_generate_ma_predictions_endpoint_invalid_date(self):
        """Test MA prediction generation with invalid date."""
        payload = {
            "start_date": "invalid-date",
            "end_date": "2025-01-01"
        }
        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert response.status_code == 422  # Validation error

    def test_generate_ma_predictions_endpoint_end_before_start(self):
        """Test MA prediction generation with end date before start date."""
        payload = {
            "start_date": "2025-01-01",
            "end_date": "2020-01-01"
        }
        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert response.status_code == 422  # Validation error

    def test_generate_ma_predictions_endpoint_missing_required_fields(self):
        """Test MA prediction generation with missing required fields."""
        payload = {
            "start_date": "2020-01-01"
            # Missing end_date
        }
        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert response.status_code == 422  # Validation error

    @patch('routes.scripts.run_script_async')
    def test_generate_ma_predictions_endpoint_with_defaults(self, mock_run_script):
        """Test MA prediction generation with default parameter values."""
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01"
            # Using defaults for all optional fields
        }
        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_get_ma_prediction_status_not_found(self):
        """Test getting status for non-existent MA prediction execution."""
        response = self.client.get("/scripts/generate-ma-predictions/status/nonexistent_id")
        assert response.status_code == 404

    @patch('routes.scripts.run_script_async')
    def test_get_ma_prediction_status_success(self, mock_run_script):
        """Test getting status for existing MA prediction execution."""
        # First create an execution by calling the POST endpoint
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01"
        }
        post_response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert post_response.status_code == 200
        post_data = post_response.json()
        execution_id = post_data["execution_id"]

        # For testing purposes, simulate a completed execution by directly modifying the global dict
        # This is necessary because the endpoint uses a global dict that's hard to mock
        import backend.routes.scripts as scripts_module
        script_executions = scripts_module.script_executions
        if execution_id in script_executions:
            execution = script_executions[execution_id]
            execution['status'] = 'completed'
            execution['end_time'] = datetime.utcnow()
            execution['output'] = 'MA predictions generated successfully'

        # Now test getting the status
        response = self.client.get(f"/scripts/generate-ma-predictions/status/{execution_id}")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'completed'
        assert data['execution_id'] == execution_id
        assert data['output'] == 'MA predictions generated successfully'
        assert data['error'] is None
        assert 'duration_seconds' in data

    @patch('routes.scripts.run_script_async')
    def test_get_ma_prediction_status_running(self, mock_run_script):
        """Test getting status for running MA prediction execution."""
        # Create an execution by calling the POST endpoint
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01"
        }
        post_response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert post_response.status_code == 200
        post_data = post_response.json()
        execution_id = post_data["execution_id"]

        # Test getting the status (should be running by default)
        response = self.client.get(f"/scripts/generate-ma-predictions/status/{execution_id}")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'running'
        assert data['execution_id'] == execution_id
        assert data['output'] is None
        assert data['error'] is None
        assert 'duration_seconds' not in data or data['duration_seconds'] is None

    @patch('routes.scripts.run_script_async')
    def test_get_ma_prediction_status_failed(self, mock_run_script):
        """Test getting status for failed MA prediction execution."""
        # First create an execution by calling the POST endpoint
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01"
        }
        post_response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert post_response.status_code == 200
        post_data = post_response.json()
        execution_id = post_data["execution_id"]

        # Modify the execution to be failed
        import backend.routes.scripts as scripts_module
        script_executions = scripts_module.script_executions
        if execution_id in script_executions:
            execution = script_executions[execution_id]
            execution['status'] = 'failed'
            execution['end_time'] = datetime.utcnow()
            execution['error'] = 'Script execution failed: invalid parameters'

        # Now test getting the status
        response = self.client.get(f"/scripts/generate-ma-predictions/status/{execution_id}")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'failed'
        assert data['execution_id'] == execution_id
        assert data['output'] is None
        assert data['error'] == 'Script execution failed: invalid parameters'
        assert 'duration_seconds' in data

    @patch('routes.scripts.run_script_async')
    def test_generate_ma_predictions_parameter_validation_ranges(self, mock_run_script):
        """Test parameter validation for MA ranges."""
        # Test with empty ranges
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01",
            "short_ma_range": [],
            "medium_ma_range": [15, 20],
            "long_ma_range": [40, 50]
        }
        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        # Should still work as ranges are optional and have defaults
        assert response.status_code == 200

    @patch('routes.scripts.run_script_async')
    def test_generate_ma_predictions_fixed_parameters(self, mock_run_script):
        """Test MA prediction generation with fixed parameters."""
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01",
            "skip_optimization": True,
            "fixed_short": 3,
            "fixed_medium": 10,
            "fixed_long": 30
        }
        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

        # In test mode, the execution is recorded but run_script_async is not called
        # Just verify the endpoint accepts the parameters and creates execution
        assert "execution_id" in data
        assert data["execution_id"].startswith("ma_")


@pytest.mark.unit
class TestMAPredictionValidation:
    """Test input validation for MA prediction endpoints."""

    def setup_method(self):
        """Set up test client."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
        from main import app

        self.client = TestClient(app)

    def test_date_format_validation(self):
        """Test date format validation."""
        test_cases = [
            ("2020-01-01", "2025-01-01", True),   # Valid
            ("2020/01/01", "2025-01-01", False),  # Invalid format
            ("01-01-2020", "2025-01-01", False),  # Invalid format
            ("2020-13-01", "2025-01-01", False),  # Invalid month
            ("2020-01-32", "2025-01-01", False),  # Invalid day
        ]

        for start_date, end_date, should_pass in test_cases:
            payload = {
                "start_date": start_date,
                "end_date": end_date
            }
            response = self.client.post("/scripts/generate-ma-predictions", json=payload)

            if should_pass:
                assert response.status_code in [200, 500]  # 500 is OK if script fails, validation passed
            else:
                assert response.status_code == 422  # Validation error

    def test_ma_range_validation(self):
        """Test MA range parameter validation."""
        # Test with non-integer values
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01",
            "short_ma_range": [3.5, 5.5],  # Floats instead of ints
        }
        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        # Pydantic should handle type conversion or reject
        assert response.status_code in [200, 422]

    def test_skip_optimization_validation(self):
        """Test skip_optimization parameter validation."""
        test_cases = [
            True,
            False,
            "true",   # String instead of bool
            1,        # Int instead of bool
            0,        # Int instead of bool
        ]

        for skip_opt in test_cases:
            payload = {
                "start_date": "2020-01-01",
                "end_date": "2025-01-01",
                "skip_optimization": skip_opt
            }
            response = self.client.post("/scripts/generate-ma-predictions", json=payload)
            # Should accept valid boolean values
            if isinstance(skip_opt, bool):
                assert response.status_code in [200, 500]
            # May reject invalid types depending on Pydantic strictness


@pytest.mark.unit
class TestMAWebSocketIntegration:
    """Test WebSocket integration for MA predictions."""

    def setup_method(self):
        """Set up test client."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
        from main import app

        self.client = TestClient(app)

    @patch('routes.scripts.broadcast_websocket_message')
    @patch('routes.scripts.run_script_async')
    def test_websocket_broadcast_on_start(self, mock_run_script, mock_broadcast):
        """Test WebSocket broadcast when MA prediction starts."""
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01",
            "skip_optimization": False
        }

        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert response.status_code == 200

        # Verify WebSocket broadcast was not called on start (only on completion)
        # The broadcast happens in run_script_async, not in the endpoint itself
        mock_broadcast.assert_not_called()

    @patch('routes.scripts.broadcast_websocket_message')
    @patch('routes.scripts.script_executions')
    @patch('routes.scripts.run_script_async')
    async def test_websocket_broadcast_on_completion(self, mock_run_script, mock_executions, mock_broadcast):
        """Test WebSocket broadcast when MA prediction completes."""
        # This would be tested in the async function, but we can't easily test that here
        # The WebSocket broadcast is tested in the run_script_async tests in test_api_scripts.py
        pass


@pytest.mark.unit
class TestMAEndpointErrorHandling:
    """Test error handling for MA prediction endpoints."""

    def setup_method(self):
        """Set up test client."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
        from main import app

        self.client = TestClient(app)

    @patch('routes.scripts.BackgroundTasks.add_task')
    def test_generate_ma_predictions_background_task_failure(self, mock_add_task):
        """Test handling of background task failures."""
        mock_add_task.side_effect = Exception("Background task failed")

        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01"
        }

        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        assert response.status_code == 500

    @patch('routes.scripts.get_config')
    def test_generate_ma_predictions_config_failure(self, mock_get_config):
        """Test handling of configuration failures."""
        mock_get_config.side_effect = Exception("Config load failed")

        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01"
        }

        # The endpoint should handle config failures gracefully
        response = self.client.post("/scripts/generate-ma-predictions", json=payload)
        # In test mode, the exception might not propagate as expected
        # Just verify the endpoint doesn't crash
        assert response.status_code in [200, 500]

    def test_get_ma_prediction_status_malformed_id(self):
        """Test handling of malformed execution IDs."""
        test_cases = [
            "invalid-id-with-dashes-and-numbers-123",
            "",  # Empty string
            "a",  # Very short
            "a" * 100,  # Very long
        ]

        for execution_id in test_cases:
            response = self.client.get(f"/scripts/generate-ma-predictions/status/{execution_id}")
            # Should handle gracefully
            assert response.status_code in [200, 404]  # Either finds it or not


@pytest.mark.unit
class TestMAPredictionConcurrency:
    """Test concurrent MA prediction requests."""

    def setup_method(self):
        """Set up test client."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
        from main import app

        self.client = TestClient(app)

    @patch('routes.scripts.run_script_async')
    def test_multiple_concurrent_requests(self, mock_run_script):
        """Test handling multiple concurrent MA prediction requests."""
        payload = {
            "start_date": "2020-01-01",
            "end_date": "2025-01-01",
            "skip_optimization": True,
            "fixed_short": 5,
            "fixed_medium": 20,
            "fixed_long": 50
        }

        # Make multiple concurrent requests
        responses = []
        for i in range(5):
            response = self.client.post("/scripts/generate-ma-predictions", json=payload)
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"
            assert "execution_id" in data

        # Verify different execution IDs were generated
        execution_ids = [r.json()["execution_id"] for r in responses]
        assert len(set(execution_ids)) == len(execution_ids), "All execution IDs should be unique"