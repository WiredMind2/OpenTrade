#!/usr/bin/env python3
"""
Test script for LinearRegressionTimeSeriesAdapter integration.

This script tests:
1. Model discovery by registry
2. Retraining operation
3. Prediction calls
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Also add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from backend.logging_config import get_component_logger
from backend.models.registry import ModelRegistry
from backend.config import get_config

# Mock app_state for testing
import backend.main
backend.main.app_state = {
    "database_path": "data/backtest.db",
    "model_registry": None  # Will be set later
}

logger = get_component_logger(__name__)


def test_model_discovery():
    """Test that the LinearRegressionTimeSeriesAdapter is discovered by the registry."""
    print("=== Testing Model Discovery ===")

    # Initialize registry
    registry = ModelRegistry()

    # Set up paths
    config = get_config()
    models_dir = Path(config.model.model_dir)
    models_pkg_dir = backend_dir / "models"

    print(f"Models directory: {models_dir}")
    print(f"Models package directory: {models_pkg_dir}")

    # Discover models
    registry.discover(models_dir, models_pkg_dir)

    # List discovered models
    models = registry.list()
    print(f"Discovered {len(models)} models:")
    for model in models:
        print(f"  - {model.name} ({model.type}) - {model.description}")

    # Check if our model is there
    linear_model = registry.get("linear_regression_time_series_v1")
    if linear_model:
        print("PASS: LinearRegressionTimeSeriesAdapter found in registry")
        return True
    else:
        print("FAIL: LinearRegressionTimeSeriesAdapter NOT found in registry")
        return False


def test_retraining(registry):
    """Test retraining the model."""
    print("\n=== Testing Retraining ===")

    # Get the model
    model = registry.get("linear_regression_time_series_v1")
    if not model:
        print("FAIL: Model not found for retraining")
        return False

    # Prepare training data - use a date range that definitely has data
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    training_payload = {
        "start_date": start_date,
        "end_date": end_date,
        "tickers": ["AAPL"]  # AAPL has the most complete data
    }

    config = {
        "lag_days": 10,
        "horizons": [1, 3, 7]
    }

    try:
        print(f"Retraining model from {start_date} to {end_date}")
        result = model.retrain(training_payload, config)
        print("PASS: Retraining completed successfully")
        print(f"  Horizons trained: {result.get('horizons_trained', [])}")
        print(f"  Tickers used: {result.get('tickers_used', 0)}")
        return True
    except Exception as e:
        print(f"FAIL: Retraining failed: {e}")
        return False


def test_prediction(registry):
    """Test making predictions."""
    print("\n=== Testing Prediction ===")

    # Get the model
    model = registry.get("linear_regression_time_series_v1")
    if not model:
        print("FAIL: Model not found for prediction")
        return False

    # Check if model is trained
    if not hasattr(model, '_models') or not model._models:
        print("FAIL: Model not trained")
        return False

    # Prepare prediction inputs - use recent data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

    prediction_inputs = {
        "ticker": "AAPL",
        "start_date": start_date,
        "end_date": end_date,
        "horizon": 1
    }

    config = {
        "lag_days": 10
    }

    try:
        print(f"Making prediction for {prediction_inputs['ticker']} with horizon {prediction_inputs['horizon']}")
        result = model.predict(prediction_inputs, config)
        print("PASS: Prediction completed successfully")
        print(f"  Predicted return: {result.get('predicted_return', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Date: {result.get('date', 'N/A')}")
        return True
    except Exception as e:
        print(f"FAIL: Prediction failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing LinearRegressionTimeSeriesAdapter Integration")
    print("=" * 60)

    results = []

    # Initialize registry once for all tests
    registry = ModelRegistry()
    config = get_config()
    models_dir = Path(config.model.model_dir)
    models_pkg_dir = backend_dir / "models"
    registry.discover(models_dir, models_pkg_dir)

    # Test 1: Model Discovery
    results.append(test_model_discovery())

    # Test 2: Retraining
    results.append(test_retraining(registry))

    # Test 3: Prediction
    results.append(test_prediction(registry))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    tests = ["Model Discovery", "Retraining", "Prediction"]
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "PASS" if result else "FAIL"
        print(f"Test {i}: {test} - {status}")

    all_passed = all(results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())