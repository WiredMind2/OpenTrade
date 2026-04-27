import pytest
from datetime import datetime
from backend.routes.predictions import generate_prediction
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.main import app


def test_prediction_with_dates():
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    tickers = ['AAPL']

    result = generate_prediction(start_date, end_date, tickers)
    assert result is not None


def test_prediction_missing_dates():
    tickers = ['AAPL']

    with pytest.raises(ValueError, match="start and end dates required"):
        generate_prediction(None, None, tickers)


def test_prediction_with_tickers():
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    tickers = ['AAPL', 'GOOGL']

    result = generate_prediction(start_date, end_date, tickers)
    assert result is not None


def test_prediction_empty_tickers():
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    tickers = []

    with pytest.raises(ValueError, match="tickers list cannot be empty"):
        generate_prediction(start_date, end_date, tickers)


def test_projection_overlays_endpoint_returns_series():
    client = TestClient(app)

    mock_strategy = MagicMock()
    mock_strategy.project_series.return_value = [
        {
            "time": "2026-04-27T00:00:00+00:00",
            "price": 100.0,
            "confidence": 0.8,
            "upperBound": 102.0,
            "lowerBound": 98.0,
        },
        {
            "time": "2026-04-28T00:00:00+00:00",
            "price": 101.0,
            "confidence": 0.78,
            "upperBound": 103.0,
            "lowerBound": 99.0,
        },
    ]

    mock_registry = MagicMock()
    mock_registry.list.return_value = [
        {
            "name": "moving_average",
            "type": "rule",
        }
    ]
    mock_registry.get.return_value = mock_strategy

    with patch("backend.main.app_state", {"strategy_registry": mock_registry}):
        response = client.post(
            "/api/predictions/projections",
            json={
                "symbol": "aapl",
                "anchor_time": "2026-04-27T00:00:00Z",
                "anchor_price": 100.0,
                "horizon_days": 2,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["ticker"] == "AAPL"
    assert body[0]["modelName"] == "moving_average"
    assert len(body[0]["points"]) == 2
    assert body[0]["points"][0]["time"] == 1777248000


def test_projection_overlays_endpoint_rejects_unknown_strategy():
    client = TestClient(app)
    mock_registry = MagicMock()
    mock_registry.list.return_value = [{"name": "moving_average", "type": "rule"}]

    with patch("backend.main.app_state", {"strategy_registry": mock_registry}):
        response = client.post(
            "/api/predictions/projections",
            json={
                "symbol": "AAPL",
                "anchor_time": "2026-04-27T00:00:00Z",
                "anchor_price": 100.0,
                "horizon_days": 2,
                "strategy_names": ["unknown"],
            },
        )

    assert response.status_code == 400
    assert "Unknown strategies requested" in response.json()["detail"]


def test_projection_overlays_endpoint_passes_strategy_parameters():
    client = TestClient(app)
    mock_strategy = MagicMock()
    mock_strategy.project_series.return_value = [
        {"time": "2026-04-27T00:00:00Z", "price": 100.0, "confidence": 0.9},
    ]

    mock_registry = MagicMock()
    mock_registry.list.return_value = [{"name": "moving_average", "type": "rule"}]
    mock_registry.get.return_value = mock_strategy

    request_payload = {
        "symbol": "AAPL",
        "anchor_time": "2026-04-27T00:00:00Z",
        "anchor_price": 100.0,
        "horizon_days": 5,
        "strategy_names": ["moving_average"],
        "params_by_strategy": {"moving_average": {"short_window": 5, "long_window": 20}},
    }

    with patch("backend.main.app_state", {"strategy_registry": mock_registry}):
        response = client.post("/api/predictions/projections", json=request_payload)

    assert response.status_code == 200
    mock_strategy.project_series.assert_called_once()
    kwargs = mock_strategy.project_series.call_args.kwargs
    assert kwargs["projection_days"] == 5
    assert kwargs["anchor_price"] == 100.0
    assert kwargs["parameters"]["symbol"] == "AAPL"
    assert kwargs["parameters"]["short_window"] == 5
    assert kwargs["parameters"]["long_window"] == 20


def test_projection_overlays_endpoint_skips_strategy_failures_and_returns_valid_series():
    client = TestClient(app)

    failing_strategy = MagicMock()
    failing_strategy.project_series.side_effect = RuntimeError("boom")

    good_strategy = MagicMock()
    good_strategy.project_series.return_value = [
        {
            "time": 1777248000000,  # milliseconds input should be normalized
            "price": 102.5,
            "confidence": 0.77,
            "upperBound": 104.0,
            "lowerBound": 101.0,
        }
    ]

    mock_registry = MagicMock()
    mock_registry.list.return_value = [
        {"name": "moving_average", "type": "rule"},
        {"name": "sentiment_ml", "type": "ml"},
    ]
    mock_registry.get.side_effect = lambda name: failing_strategy if name == "moving_average" else good_strategy

    with patch("backend.main.app_state", {"strategy_registry": mock_registry}):
        response = client.post(
            "/api/predictions/projections",
            json={
                "symbol": "AAPL",
                "anchor_time": "2026-04-27T00:00:00Z",
                "anchor_price": 100.0,
                "horizon_days": 2,
                "strategy_names": ["moving_average", "sentiment_ml"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["modelName"] == "sentiment_ml"
    assert body[0]["points"][0]["time"] == 1777248000
    assert body[0]["points"][0]["price"] == 102.5


def test_predict_endpoint_returns_interval_metadata():
    client = TestClient(app)
    mock_service = MagicMock()
    class _Model:
        model_version = "lightgbm_1d_v2"
        model_name = "lightgbm_1d_v2"
        feature_schema_version = "ml_features_v1"

    class _Intervals:
        lower = -0.01
        upper = 0.02

    class _Result:
        ticker = "AAPL"
        horizon = "1d"
        predicted_return = 0.012
        confidence = 0.77
        timestamp = datetime.utcnow()
        model = _Model()
        features_used = ["avg_sentiment"]
        intervals = _Intervals()
        metadata = {"request_id": "r1"}

    fake_result = _Result()
    mock_service.predict.return_value = fake_result

    with patch("backend.routes.predictions.PredictionService", return_value=mock_service):
        with patch("backend.main.app_state", {"database_path": "x", "models_loaded": {"lightgbm_1d": {}}}):
            response = client.post("/predict", json={"ticker": "AAPL", "horizon": "1d"})
    assert response.status_code == 200
    body = response.json()
    assert body["feature_schema_version"] == "ml_features_v1"
    assert body["interval_lower"] == -0.01
    assert body["interval_upper"] == 0.02