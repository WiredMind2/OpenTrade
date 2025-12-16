import pytest
from backend.routes.predictions import generate_prediction


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