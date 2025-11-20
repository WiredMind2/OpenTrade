"""
Integration tests for the complete chart data flow from API to response.
"""
import sqlite3
import pytest
from fastapi.testclient import TestClient
from backend.main import app, app_state


@pytest.fixture
def populated_test_db_with_predictions(tmp_path):
    """Create a test database with historical data and predictions."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)

    # Create price_minute table
    conn.execute("""
        CREATE TABLE price_minute (
            ticker TEXT,
            dt TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
    """)

    # Insert historical price data for AAPL
    price_data = [
        ('AAPL', '2024-01-01', 150.0, 155.0, 149.0, 152.0, 1000000),
        ('AAPL', '2024-01-02', 152.0, 158.0, 151.0, 155.0, 1200000),
        ('AAPL', '2024-01-03', 155.0, 160.0, 154.0, 158.0, 1100000),
        ('AAPL', '2024-01-04', 158.0, 162.0, 157.0, 160.0, 1300000),
        ('AAPL', '2024-01-05', 160.0, 165.0, 159.0, 163.0, 1400000),
    ]

    conn.executemany("""
        INSERT INTO price_minute (ticker, dt, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, price_data)

    # Create sentiment_predictions table
    conn.execute("""
        CREATE TABLE sentiment_predictions (
            article_id INTEGER,
            ticker TEXT,
            model TEXT,
            horizon TEXT,
            predicted_return REAL,
            predicted_confidence REAL,
            produced_at TEXT
        )
    """)

    # Insert prediction data
    prediction_data = [
        (1, 'AAPL', 'lightgbm_1d', '1d', 0.02, 0.8, '2024-01-01T10:00:00'),  # Target: 2024-01-02
        (2, 'AAPL', 'lightgbm_1d', '1d', 0.03, 0.9, '2024-01-02T10:00:00'),  # Target: 2024-01-03
        (3, 'AAPL', 'lightgbm_1d', '1d', 0.025, 0.7, '2024-01-02T15:00:00'), # Target: 2024-01-03 (duplicate)
        (4, 'AAPL', 'lightgbm_3d', '3d', 0.05, 0.85, '2024-01-01T10:00:00'), # Target: 2024-01-04
    ]

    conn.executemany("""
        INSERT INTO sentiment_predictions
        (article_id, ticker, model, horizon, predicted_return, predicted_confidence, produced_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, prediction_data)

    conn.commit()
    conn.close()

    return str(db_path)


class TestChartDataIntegration:
    """Integration tests for chart data API flow."""

    def test_complete_chart_data_flow_no_aggregation(self, populated_test_db_with_predictions):
        """Test complete chart data flow without aggregation."""
        app_state['database_path'] = populated_test_db_with_predictions
        client = TestClient(app)

        response = client.get('/predictions/chart-data/AAPL?start_date=2024-01-01&end_date=2024-01-05&horizon=1d')

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert 'ticker' in data
        assert 'historical_data' in data
        assert 'predictions' in data
        assert 'metadata' in data

        assert data['ticker'] == 'AAPL'

        # Check historical data
        historical = data['historical_data']
        assert len(historical) == 5

        # Verify historical data structure and values (dates come back as ISO with time)
        expected_historical = [
            {'date': '2024-01-01T00:00:00', 'open': 150.0, 'high': 155.0, 'low': 149.0, 'close': 152.0, 'volume': 1000000},
            {'date': '2024-01-02T00:00:00', 'open': 152.0, 'high': 158.0, 'low': 151.0, 'close': 155.0, 'volume': 1200000},
            {'date': '2024-01-03T00:00:00', 'open': 155.0, 'high': 160.0, 'low': 154.0, 'close': 158.0, 'volume': 1100000},
            {'date': '2024-01-04T00:00:00', 'open': 158.0, 'high': 162.0, 'low': 157.0, 'close': 160.0, 'volume': 1300000},
            {'date': '2024-01-05T00:00:00', 'open': 160.0, 'high': 165.0, 'low': 159.0, 'close': 163.0, 'volume': 1400000},
        ]

        for expected, actual in zip(expected_historical, historical):
            assert actual['date'] == expected['date']
            assert actual['open'] == expected['open']
            assert actual['close'] == expected['close']

        # Check predictions (no aggregation)
        predictions = data['predictions']
        assert len(predictions) == 3  # 3 predictions for 1d horizon

        # Verify prediction calculations - should have predictions for 2024-01-02 and 2024-01-03
        pred_2024_01_02 = next(p for p in predictions if p['date'] == '2024-01-02')
        assert pred_2024_01_02['predicted_price'] == 152.0 * 1.02  # 152 * (1 + 0.02)
        assert pred_2024_01_02['actual_price'] == 155.0
        assert pred_2024_01_02['confidence'] == 0.8

        # Should have two predictions for 2024-01-03 (no aggregation)
        preds_2024_01_03 = [p for p in predictions if p['date'] == '2024-01-03']
        assert len(preds_2024_01_03) == 2

        # Check metadata
        metadata = data['metadata']
        assert 'data_freshness_score' in metadata
        assert 'quality_level' in metadata
        assert 'last_updated' in metadata
        assert metadata['total_records'] == 8  # 5 historical + 3 predictions

    def test_chart_data_with_aggregation_avg(self, populated_test_db_with_predictions):
        """Test chart data flow with average aggregation."""
        app_state['database_path'] = populated_test_db_with_predictions
        client = TestClient(app)

        response = client.get('/predictions/chart-data/AAPL?horizon=1d&aggregate=avg')

        assert response.status_code == 200
        data = response.json()

        predictions = data['predictions']

        # Should have aggregated predictions for dates with multiple predictions
        pred_2024_01_03 = next(p for p in predictions if p['date'] == '2024-01-03')

        # Two predictions for 2024-01-03: 0.03 and 0.025 returns
        # Base price: 155.0 (close on 2024-01-02)
        expected_price = 155.0 * (1 + (0.03 + 0.025) / 2)  # Average return
        assert pred_2024_01_03['predicted_price'] == pytest.approx(expected_price)
        assert pred_2024_01_03['confidence'] == (0.9 + 0.7) / 2  # Average confidence


    def test_chart_data_different_horizons(self, populated_test_db_with_predictions):
        """Test chart data flow with different prediction horizons."""
        app_state['database_path'] = populated_test_db_with_predictions
        client = TestClient(app)

        # Test 3d horizon
        response = client.get('/predictions/chart-data/AAPL?horizon=3d')
        assert response.status_code == 200
        data = response.json()

        predictions = data['predictions']
        # Should have prediction for 2024-01-04 (3 days after 2024-01-01)
        pred_2024_01_04 = next((p for p in predictions if p['date'] == '2024-01-04'), None)
        assert pred_2024_01_04 is not None
        expected_price = 152.0 * (1 + 0.05)  # Base on 2024-01-01 close
        assert pred_2024_01_04['predicted_price'] == pytest.approx(expected_price)

    def test_chart_data_confidence_validation(self, populated_test_db_with_predictions):
        """Test that confidence values are properly validated and clamped."""
        # Insert prediction with invalid confidence
        conn = sqlite3.connect(populated_test_db_with_predictions)
        conn.execute("""
            INSERT INTO sentiment_predictions
            (article_id, ticker, model, horizon, predicted_return, predicted_confidence, produced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (5, 'AAPL', 'lightgbm_1d', '1d', 0.01, 1.5, '2024-01-03T10:00:00'))  # Confidence > 1.0
        conn.commit()
        conn.close()

        app_state['database_path'] = populated_test_db_with_predictions
        client = TestClient(app)

        response = client.get('/predictions/chart-data/AAPL?horizon=1d')
        assert response.status_code == 200
        data = response.json()

        predictions = data['predictions']
        pred_2024_01_04 = next(p for p in predictions if p['date'] == '2024-01-04')

        # Confidence should be clamped to 1.0
        assert pred_2024_01_04['confidence'] == 1.0

    def test_chart_data_data_quality_metadata(self, populated_test_db_with_predictions):
        """Test that data quality metadata is properly calculated."""
        app_state['database_path'] = populated_test_db_with_predictions
        client = TestClient(app)

        response = client.get('/predictions/chart-data/AAPL?horizon=1d')
        assert response.status_code == 200
        data = response.json()

        metadata = data['metadata']
        assert 'data_freshness_score' in metadata
        assert 'quality_level' in metadata
        assert 'validation_issues' in metadata
        assert 'total_records' in metadata

        # Should have excellent quality for this test data
        assert metadata['quality_level'] in ['excellent', 'good', 'fair', 'poor', 'critical']
        assert isinstance(metadata['data_freshness_score'], float)
        assert 0.0 <= metadata['data_freshness_score'] <= 1.0

    def test_chart_data_empty_results(self, tmp_path):
        """Test chart data API with empty database."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(db_path)
        # Create empty tables
        conn.execute("CREATE TABLE price_minute (ticker TEXT, dt TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)")
        conn.execute("CREATE TABLE sentiment_predictions (article_id INTEGER, ticker TEXT, model TEXT, horizon TEXT, predicted_return REAL, predicted_confidence REAL, produced_at TEXT)")
        conn.commit()
        conn.close()

        app_state['database_path'] = str(db_path)
        client = TestClient(app)

        # Use a different ticker to avoid cache hits from other tests
        response = client.get('/predictions/chart-data/EMPTY?horizon=1d')
        assert response.status_code == 200
        data = response.json()

        assert data['historical_data'] == []
        assert data['predictions'] == []
        assert data['ticker'] == 'EMPTY'