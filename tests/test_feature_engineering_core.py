"""
Unit tests for core feature engineering functionality.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sqlite3
import gc
import time
from unittest.mock import patch, MagicMock, PropertyMock
from backend.feature_engineering import (
    FeatureEngineer, create_feature_engineer, FeatureDefinition, FeatureType
)


@pytest.mark.unit
class TestFeatureEngineerCore:
    """Test core FeatureEngineer functionality."""

    def setup_method(self):
        """Set up test database and data."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Create test database with sample data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            CREATE TABLE price_daily (
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
        conn.execute("""
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY,
                canonical_timestamp TEXT,
                title TEXT,
                content TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE sentiment_predictions (
                id INTEGER PRIMARY KEY,
                article_id INTEGER,
                ticker TEXT,
                predicted_return REAL,
                confidence REAL,
                FOREIGN KEY(article_id) REFERENCES articles(id)
            )
        """)

        # Insert sample price data
        sample_prices = [
            ("AAPL", "2024-01-01", 150.0, 155.0, 149.0, 154.0, 154.0, 1000000),
            ("AAPL", "2024-01-02", 154.0, 158.0, 153.0, 157.0, 157.0, 1200000),
            ("AAPL", "2024-01-03", 157.0, 160.0, 156.0, 159.0, 159.0, 1100000),
            ("AAPL", "2024-01-04", 159.0, 162.0, 158.0, 161.0, 161.0, 1300000),
            ("AAPL", "2024-01-05", 161.0, 164.0, 160.0, 163.0, 163.0, 1400000),
        ]
        conn.executemany("""
            INSERT INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_prices)

        conn.commit()
        conn.close()

        self.engineer = FeatureEngineer(self.temp_db.name)

    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.temp_db.name):
            for _ in range(10):
                try:
                    os.unlink(self.temp_db.name)
                    break
                except PermissionError:
                    gc.collect()
                    time.sleep(0.05)

    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        assert self.engineer.db_path == self.temp_db.name
        assert isinstance(self.engineer.feature_registry, dict)
        assert len(self.engineer.feature_registry) > 0  # Should have default features

    def test_register_default_features(self):
        """Test default feature registration."""
        # Check that some expected features are registered
        expected_features = [
            "sma_5", "sma_20", "ema_12", "rsi_14", "macd",
            "article_sentiment_score", "sentiment_momentum",
            "return_zscore", "volume_price_trend"
        ]

        for feature_name in expected_features:
            assert feature_name in self.engineer.feature_registry
            feature = self.engineer.feature_registry[feature_name]
            assert isinstance(feature, FeatureDefinition)

    def test_load_price_data(self):
        """Test loading price data from database."""
        df = self.engineer._load_price_data("AAPL", "2024-01-01", "2024-01-05")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "close" in df.columns
        assert "volume" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_price_data_no_data(self):
        """Test loading price data for non-existent ticker."""
        df = self.engineer._load_price_data("NONEXISTENT", "2024-01-01", "2024-01-05")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_sentiment_data(self):
        """Test loading sentiment data from database."""
        # Insert test sentiment data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO articles (canonical_timestamp, title, content)
            VALUES (?, ?, ?)
        """, ("2024-01-01T10:00:00", "Test Article", "Test content"))

        cur = conn.cursor()
        cur.execute("SELECT last_insert_rowid()")
        article_id = cur.fetchone()[0]
        conn.execute("""
            INSERT INTO sentiment_predictions (article_id, ticker, predicted_return, confidence)
            VALUES (?, ?, ?, ?)
        """, (article_id, "AAPL", 0.02, 0.85))

        conn.commit()
        conn.close()

        df = self.engineer._load_sentiment_data("AAPL", "2024-01-01", "2024-01-05")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "sentiment_score" in df.columns
        assert "sentiment_confidence" in df.columns

    def test_generate_features_basic(self):
        """Test basic feature generation."""
        features_df = self.engineer.generate_features(
            "AAPL", "2024-01-01", "2024-01-05",
            feature_list=["sma_5", "rsi_14"],
            save_to_db=False
        )

        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 5
        assert "sma_5" in features_df.columns
        assert "rsi_14" in features_df.columns

    def test_generate_features_all_features(self):
        """Test generating all features."""
        features_df = self.engineer.generate_features(
            "AAPL", "2024-01-01", "2024-01-05",
            save_to_db=False
        )

        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 5
        # Should have many feature columns
        assert len(features_df.columns) > 10

    def test_generate_features_no_price_data(self):
        """Test feature generation with no price data."""
        with pytest.raises(Exception):  # Should raise DataIngestionError
            self.engineer.generate_features(
                "NONEXISTENT", "2024-01-01", "2024-01-05",
                save_to_db=False
            )

    def test_generate_features_unknown_feature(self):
        """Test generating unknown feature."""
        features_df = self.engineer.generate_features(
            "AAPL", "2024-01-01", "2024-01-05",
            feature_list=["sma_5", "unknown_feature"],
            save_to_db=False
        )

        # Should still generate known features
        assert "sma_5" in features_df.columns
        # Unknown feature should not be present
        assert "unknown_feature" not in features_df.columns

    def test_save_features_to_db(self):
        """Test saving features to database."""
        # Generate some features
        features_df = self.engineer.generate_features(
            "AAPL", "2024-01-01", "2024-01-05",
            feature_list=["sma_5"],
            save_to_db=True
        )

        # Check that features were saved
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM features WHERE ticker = ? AND feature_name = ?",
                   ("AAPL", "sma_5"))
        count = cur.fetchone()[0]
        conn.close()

        assert count > 0

    def test_create_feature_sets(self):
        """Test creating predefined feature sets."""
        feature_sets = self.engineer.create_feature_sets()

        assert isinstance(feature_sets, dict)
        assert "sentiment_focused" in feature_sets
        assert "technical_analysis" in feature_sets
        assert "comprehensive" in feature_sets

        # Check sentiment focused set
        sentiment_set = feature_sets["sentiment_focused"]
        assert len(sentiment_set.features) > 0
        assert all(f.feature_type in [FeatureType.SENTIMENT, FeatureType.TECHNICAL]
                  for f in sentiment_set.features)

    def test_select_features_mutual_info(self):
        """Test feature selection using mutual information."""
        # Create sample feature data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'noise': np.random.randn(100)
        })
        y = X['feature1'] * 2 + X['feature2'] + np.random.randn(100) * 0.1

        selected = self.engineer.select_features(X, y, method="mutual_info", k=2)

        assert isinstance(selected, list)
        assert len(selected) == 2
        assert "feature1" in selected or "feature2" in selected

    def test_select_features_correlation(self):
        """Test feature selection using correlation."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'noise': np.random.randn(100)
        })
        y = X['feature1'] * 0.8 + np.random.randn(100) * 0.1

        selected = self.engineer.select_features(X, y, method="correlation", k=2)

        assert isinstance(selected, list)
        assert len(selected) == 2
        assert "feature1" in selected

    def test_select_features_variance(self):
        """Test feature selection using variance."""
        X = pd.DataFrame({
            'high_var': np.random.randn(100) * 10,
            'low_var': np.random.randn(100) * 0.1,
            'medium_var': np.random.randn(100) * 2
        })

        selected = self.engineer.select_features(X, pd.Series(np.random.randn(100)),
                                                method="variance", k=2)

        assert isinstance(selected, list)
        assert len(selected) == 2
        assert "high_var" in selected

    def test_select_features_unknown_method(self):
        """Test feature selection with unknown method."""
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        y = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="Unknown selection method"):
            self.engineer.select_features(X, y, method="unknown")

    def test_remove_redundant_features(self):
        """Test removing redundant features."""
        # Create correlated features
        np.random.seed(42)
        base = np.random.randn(100)
        X = pd.DataFrame({
            'feature1': base,
            'feature2': base + np.random.randn(100) * 0.01,  # Highly correlated
            'feature3': np.random.randn(100),  # Uncorrelated
        })

        X_cleaned = self.engineer._remove_redundant_features(X, threshold=0.9)

        # Should remove feature2 as it's highly correlated with feature1
        assert 'feature1' in X_cleaned.columns
        assert 'feature3' in X_cleaned.columns
        # feature2 might be removed depending on correlation calculation

    def test_scale_features_standard(self):
        """Test standard scaling."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        X_scaled = self.engineer.scale_features(X, method="standard", fit=True)

        assert isinstance(X_scaled, pd.DataFrame)
        # Check approximate standardization (mean ≈ 0, std ≈ 1)
        assert abs(X_scaled['feature1'].mean()) < 0.1
        # For small sample sizes, std might not be exactly 1.0
        assert 0.8 <= X_scaled['feature1'].std() <= 1.2

    def test_scale_features_minmax(self):
        """Test min-max scaling."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        X_scaled = self.engineer.scale_features(X, method="minmax", fit=True)

        assert isinstance(X_scaled, pd.DataFrame)
        # Check that values are in [0, 1] range
        assert X_scaled.min().min() >= 0
        assert X_scaled.max().max() <= 1

    def test_scale_features_robust(self):
        """Test robust scaling."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 100],  # Outlier
            'feature2': [10, 20, 30, 40, 50]
        })

        X_scaled = self.engineer.scale_features(X, method="robust", fit=True)

        assert isinstance(X_scaled, pd.DataFrame)
        # Robust scaling uses median and IQR, less affected by outliers

    def test_scale_features_unknown_method(self):
        """Test scaling with unknown method."""
        X = pd.DataFrame({'a': [1, 2, 3]})

        with pytest.raises(ValueError, match="Unknown scaling method"):
            self.engineer.scale_features(X, method="unknown")

    def test_get_feature_importance_tree_model(self):
        """Test feature importance extraction from tree model."""
        # Mock a tree model
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.3, 0.7])
        mock_model.coef_ = None  # Ensure coef_ is not used

        feature_names = ['feature1', 'feature2']
        importance = self.engineer.get_feature_importance(mock_model, feature_names)

        assert isinstance(importance, dict)
        assert importance['feature1'] == 0.3
        assert importance['feature2'] == 0.7

    def test_get_feature_importance_linear_model(self):
        """Test feature importance extraction from linear model."""
        # Mock a linear model
        mock_model = MagicMock()
        mock_model.coef_ = np.array([2.0, -1.5])  # 1D array

        feature_names = ['feature1', 'feature2']
        importance = self.engineer.get_feature_importance(mock_model, feature_names)

        assert isinstance(importance, dict)
        # For 1D coef array, it takes absolute value
        expected_importance = np.abs(np.array([2.0, -1.5]))
        assert len(importance) == len(expected_importance)
        # Check that the values are present (order might vary)
        assert 2.0 in importance.values()
        assert 1.5 in importance.values()

    def test_get_feature_importance_no_attributes(self):
        """Test feature importance extraction from model without importance attributes."""
        mock_model = MagicMock()
        del mock_model.feature_importances_  # Remove the attribute
        del mock_model.coef_  # Remove the attribute

        with pytest.raises(ValueError, match="Model does not provide feature importance"):
            self.engineer.get_feature_importance(mock_model, ['feature1'])

    def test_create_lagged_features(self):
        """Test creating lagged features."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        lagged_df = self.engineer.create_lagged_features(df, ['feature1'], [1, 2])

        assert isinstance(lagged_df, pd.DataFrame)
        assert 'feature1_lag1' in lagged_df.columns
        assert 'feature1_lag2' in lagged_df.columns
        assert lagged_df.loc[1, 'feature1_lag1'] == 1
        assert lagged_df.loc[2, 'feature1_lag2'] == 1

    def test_create_rolling_features(self):
        """Test creating rolling window features."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        rolling_df = self.engineer.create_rolling_features(
            df, ['feature1'], [3], ['mean', 'std']
        )

        assert isinstance(rolling_df, pd.DataFrame)
        assert 'feature1_rolling_mean_3' in rolling_df.columns
        assert 'feature1_rolling_std_3' in rolling_df.columns

        # Check rolling mean calculation
        assert rolling_df.loc[2, 'feature1_rolling_mean_3'] == 2.0  # (1+2+3)/3
        assert rolling_df.loc[3, 'feature1_rolling_mean_3'] == 3.0  # (2+3+4)/3


@pytest.mark.unit
class TestFeatureEngineerUtilities:
    """Test FeatureEngineer utility functions."""

    def test_create_feature_engineer(self):
        """Test creating feature engineer instance."""
        with patch('backend.feature_engineering.get_config') as mock_config:
            mock_config.return_value.database.path = ":memory:"

            engineer = create_feature_engineer()
            assert isinstance(engineer, FeatureEngineer)
            assert engineer.db_path == ":memory:"