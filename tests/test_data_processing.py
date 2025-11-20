"""
Unit tests for data processing functions used in chart data flow.
"""
import pytest
import pandas as pd
from datetime import datetime, date
from backend.data_processing import (
    aggregate_predictions,
    validate_confidence,
    calculate_predicted_price,
    find_base_price,
    process_prediction_record
)


class TestAggregatePredictions:
    """Test prediction aggregation logic."""

    def test_no_aggregation_when_none(self):
        """Test that predictions are returned as-is when aggregate is None."""
        raw_predictions = [
            {"date": "2024-01-01", "predicted_price": 100.0, "confidence": 0.8},
            {"date": "2024-01-02", "predicted_price": 105.0, "confidence": 0.7}
        ]
        result = aggregate_predictions(raw_predictions, None)
        assert result == raw_predictions

    def test_no_aggregation_when_invalid(self):
        """Test that predictions are returned as-is when aggregate is invalid."""
        raw_predictions = [
            {"date": "2024-01-01", "predicted_price": 100.0, "confidence": 0.8}
        ]
        result = aggregate_predictions(raw_predictions, "invalid")
        assert result == raw_predictions

    def test_aggregate_avg_mode(self):
        """Test average aggregation mode."""
        raw_predictions = [
            {"date": "2024-01-01", "predicted_price": 100.0, "confidence": 0.8},
            {"date": "2024-01-01", "predicted_price": 110.0, "confidence": 0.6},
            {"date": "2024-01-02", "predicted_price": 105.0, "confidence": 0.7}
        ]
        result = aggregate_predictions(raw_predictions, "avg")

        # Should have 2 aggregated predictions
        assert len(result) == 2

        # Check aggregated values for date 2024-01-01
        agg1 = next(p for p in result if p["date"] == "2024-01-01")
        assert agg1["predicted_price"] == 105.0  # (100 + 110) / 2
        assert agg1["confidence"] == 0.7  # (0.8 + 0.6) / 2
        assert agg1["count"] == 2

        # Check single prediction for date 2024-01-02
        agg2 = next(p for p in result if p["date"] == "2024-01-02")
        assert agg2["predicted_price"] == 105.0
        assert agg2["confidence"] == 0.7
        assert agg2["count"] == 1

    def test_aggregate_avg_with_none_prices(self):
        """Test average aggregation handles None prices correctly."""
        raw_predictions = [
            {"date": "2024-01-01", "predicted_price": 100.0, "confidence": 0.8},
            {"date": "2024-01-01", "predicted_price": None, "confidence": 0.6},
            {"date": "2024-01-01", "predicted_price": 110.0, "confidence": 0.9}
        ]
        result = aggregate_predictions(raw_predictions, "avg")

        agg = next(p for p in result if p["date"] == "2024-01-01")
        assert agg["predicted_price"] == 105.0  # (100 + 110) / 2, ignoring None
        assert agg["confidence"] == pytest.approx(0.7667, abs=0.01)  # (0.8 + 0.6 + 0.9) / 3
        assert agg["count"] == 3

    def test_aggregate_max_conf_mode(self):
        """Test max confidence aggregation mode."""
        raw_predictions = [
            {"date": "2024-01-01", "predicted_price": 100.0, "confidence": 0.6},
            {"date": "2024-01-01", "predicted_price": 110.0, "confidence": 0.9},
            {"date": "2024-01-01", "predicted_price": 105.0, "confidence": 0.7}
        ]
        result = aggregate_predictions(raw_predictions, "max_conf")

        agg = next(p for p in result if p["date"] == "2024-01-01")
        assert agg["predicted_price"] == 110.0  # Highest confidence prediction
        assert agg["confidence"] == 0.9
        assert agg["count"] == 3

    def test_aggregate_latest_mode(self):
        """Test latest aggregation mode."""
        raw_predictions = [
            {"date": "2024-01-01", "predicted_price": 100.0, "confidence": 0.8, "produced_at": "2024-01-01T10:00:00"},
            {"date": "2024-01-01", "predicted_price": 110.0, "confidence": 0.6, "produced_at": "2024-01-01T15:00:00"},
            {"date": "2024-01-01", "predicted_price": 105.0, "confidence": 0.7, "produced_at": "2024-01-01T12:00:00"}
        ]
        result = aggregate_predictions(raw_predictions, "latest")

        agg = next(p for p in result if p["date"] == "2024-01-01")
        assert agg["predicted_price"] == 110.0  # Latest produced_at
        assert agg["confidence"] == 0.6
        assert agg["count"] == 3

    def test_aggregate_null_dates_preserved(self):
        """Test that predictions with null dates are preserved without aggregation."""
        raw_predictions = [
            {"date": None, "predicted_price": 100.0, "confidence": 0.8},
            {"date": "2024-01-01", "predicted_price": 110.0, "confidence": 0.6}
        ]
        result = aggregate_predictions(raw_predictions, "avg")

        # Null date prediction should be preserved as-is
        null_preds = [p for p in result if p["date"] is None]
        assert len(null_preds) == 1
        assert null_preds[0]["predicted_price"] == 100.0

        # Regular date should be aggregated (single item)
        regular_preds = [p for p in result if p["date"] == "2024-01-01"]
        assert len(regular_preds) == 1
        assert regular_preds[0]["predicted_price"] == 110.0


class TestValidateConfidence:
    """Test confidence validation function."""

    def test_valid_confidence_values(self):
        """Test that valid confidence values are returned unchanged."""
        assert validate_confidence(0.0) == 0.0
        assert validate_confidence(0.5) == 0.5
        assert validate_confidence(1.0) == 1.0
        assert validate_confidence(0.75) == 0.75

    def test_clamp_high_values(self):
        """Test that values above 1.0 are clamped to 1.0."""
        assert validate_confidence(1.5) == 1.0
        assert validate_confidence(2.0) == 1.0
        assert validate_confidence(100) == 1.0

    def test_clamp_low_values(self):
        """Test that values below 0.0 are clamped to 0.0."""
        assert validate_confidence(-0.5) == 0.0
        assert validate_confidence(-1.0) == 0.0
        assert validate_confidence(-100) == 0.0

    def test_invalid_values_default_to_0_5(self):
        """Test that invalid values default to 0.5."""
        assert validate_confidence("invalid") == 0.5
        assert validate_confidence(None) == 0.5
        assert validate_confidence(float('nan')) == 0.5

    def test_string_numbers_converted(self):
        """Test that string numbers are properly converted."""
        assert validate_confidence("0.8") == 0.8
        assert validate_confidence("1.2") == 1.0  # Clamped
        assert validate_confidence("-0.1") == 0.0  # Clamped


class TestCalculatePredictedPrice:
    """Test predicted price calculation."""

    def test_normal_calculation(self):
        """Test normal predicted price calculation."""
        result = calculate_predicted_price(100.0, 0.05)
        assert result == 105.0

        result = calculate_predicted_price(200.0, -0.1)
        assert result == 180.0

    def test_extreme_returns_allowed(self):
        """Test that extreme returns are still calculated (warnings handled elsewhere)."""
        result = calculate_predicted_price(100.0, 0.6)  # 60% return
        assert result == 160.0

        result = calculate_predicted_price(100.0, -0.7)  # -70% return
        assert result == pytest.approx(30.0)

    def test_none_base_price(self):
        """Test that None base price returns None."""
        result = calculate_predicted_price(None, 0.05)
        assert result is None

    def test_none_predicted_return(self):
        """Test that None predicted return returns None."""
        result = calculate_predicted_price(100.0, None)
        assert result is None

    def test_invalid_predicted_return(self):
        """Test that invalid predicted return returns None."""
        result = calculate_predicted_price(100.0, "invalid")
        assert result is None

    def test_zero_base_price(self):
        """Test calculation with zero base price."""
        result = calculate_predicted_price(0.0, 0.05)
        assert result == 0.0


class TestFindBasePrice:
    """Test base price finding logic."""

    def test_exact_date_match(self):
        """Test finding price for exact date match."""
        price_by_date = {
            "2024-01-01": 100.0,
            "2024-01-02": 105.0,
            "2024-01-03": 110.0
        }
        produced_date = date(2024, 1, 2)
        result = find_base_price(price_by_date, produced_date)
        assert result == 105.0

    def test_nearest_prior_date(self):
        """Test finding nearest prior date when exact match not found."""
        price_by_date = {
            "2024-01-01": 100.0,
            "2024-01-03": 110.0,
            "2024-01-05": 120.0
        }
        produced_date = date(2024, 1, 4)  # Between 2024-01-03 and 2024-01-05
        result = find_base_price(price_by_date, produced_date)
        assert result == 110.0  # Nearest prior date

    def test_no_prior_dates(self):
        """Test that None is returned when no prior dates exist."""
        price_by_date = {
            "2024-01-03": 110.0,
            "2024-01-05": 120.0
        }
        produced_date = date(2024, 1, 1)  # Before all dates
        result = find_base_price(price_by_date, produced_date)
        assert result is None

    def test_none_produced_date(self):
        """Test that None produced date returns None."""
        price_by_date = {"2024-01-01": 100.0}
        result = find_base_price(price_by_date, None)
        assert result is None

    def test_empty_price_data(self):
        """Test with empty price data."""
        price_by_date = {}
        produced_date = date(2024, 1, 1)
        result = find_base_price(price_by_date, produced_date)
        assert result is None


class TestProcessPredictionRecord:
    """Test prediction record processing."""

    def test_complete_prediction_record(self):
        """Test processing a complete prediction record."""
        prow = pd.Series({
            'produced_at': '2024-01-01T10:00:00',
            'predicted_return': 0.05,
            'predicted_confidence': 0.8
        })
        price_by_date = {
            '2024-01-01': 100.0,
            '2024-01-02': 105.0
        }

        result = process_prediction_record(prow, price_by_date, '1d')

        assert result['date'] == '2024-01-02'  # 1 day later
        assert result['predicted_price'] == 105.0  # 100 * (1 + 0.05)
        assert result['actual_price'] == 105.0  # Actual price on target date
        assert result['confidence'] == 0.8
        assert result['produced_at'] == '2024-01-01T10:00:00'

    def test_missing_actual_price(self):
        """Test when actual price is not available."""
        prow = pd.Series({
            'produced_at': '2024-01-01T10:00:00',
            'predicted_return': 0.05,
            'predicted_confidence': 0.8
        })
        price_by_date = {
            '2024-01-01': 100.0
            # No price for target date 2024-01-02
        }

        result = process_prediction_record(prow, price_by_date, '1d')

        assert result['actual_price'] is None

    def test_missing_base_price(self):
        """Test when base price is not available."""
        prow = pd.Series({
            'produced_at': '2023-12-31T10:00:00',  # Before available prices
            'predicted_return': 0.05,
            'predicted_confidence': 0.8
        })
        price_by_date = {
            '2024-01-01': 100.0
        }

        result = process_prediction_record(prow, price_by_date, '1d')

        assert result['predicted_price'] is None

    def test_invalid_confidence_defaults(self):
        """Test that invalid confidence defaults to 0.5."""
        prow = pd.Series({
            'produced_at': '2024-01-01T10:00:00',
            'predicted_return': 0.05,
            'predicted_confidence': 'invalid'
        })
        price_by_date = {'2024-01-01': 100.0}

        result = process_prediction_record(prow, price_by_date, '1d')
        assert result['confidence'] == 0.5

    def test_extreme_predicted_return(self):
        """Test handling of extreme predicted returns."""
        prow = pd.Series({
            'produced_at': '2024-01-01T10:00:00',
            'predicted_return': 0.6,  # 60% return
            'predicted_confidence': 0.8
        })
        price_by_date = {'2024-01-01': 100.0}

        result = process_prediction_record(prow, price_by_date, '1d')
        assert result['predicted_price'] == 160.0  # Still calculated

    def test_different_horizons(self):
        """Test different prediction horizons."""
        price_by_date = {'2024-01-01': 100.0}

        # 1d horizon
        prow = pd.Series({
            'produced_at': '2024-01-01T10:00:00',
            'predicted_return': 0.05,
            'predicted_confidence': 0.8
        })
        result = process_prediction_record(prow, price_by_date, '1d')
        assert result['date'] == '2024-01-02'

        # 3d horizon
        result = process_prediction_record(prow, price_by_date, '3d')
        assert result['date'] == '2024-01-04'

        # 7d horizon
        result = process_prediction_record(prow, price_by_date, '7d')
        assert result['date'] == '2024-01-08'

    def test_missing_produced_at(self):
        """Test handling of missing produced_at."""
        prow = pd.Series({
            'predicted_return': 0.05,
            'predicted_confidence': 0.8
        })
        price_by_date = {'2024-01-01': 100.0}

        result = process_prediction_record(prow, price_by_date, '1d')
        assert result['date'] is None
        assert result['predicted_price'] is None
        assert result['produced_at'] is None