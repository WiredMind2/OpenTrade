"""
Unit tests for data validation functions.
"""
import pytest
from datetime import datetime
from backend.data_validation import validate_date_range


def test_valid_date_range():
    """Test validation passes for valid date range."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    assert validate_date_range(start_date, end_date) is True


def test_invalid_date_range():
    """Test validation fails for invalid date range."""
    start_date = datetime(2023, 12, 31)
    end_date = datetime(2023, 1, 1)

    with pytest.raises(ValueError, match="Start date must be before end date"):
        validate_date_range(start_date, end_date)


def test_equal_dates():
    """Test validation fails when start and end dates are equal."""
    date = datetime(2023, 6, 15)

    with pytest.raises(ValueError, match="Start date must be before end date"):
        validate_date_range(date, date)