import pytest
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