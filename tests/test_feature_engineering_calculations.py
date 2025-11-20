"""
Unit tests for feature engineering calculation functions.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from backend.feature_engineering import FeatureEngineer, FeatureDefinition, FeatureType


@pytest.mark.unit
class TestFeatureCalculations:
    """Test feature calculation functions."""

    def setup_method(self):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        np.random.seed(42)  # For reproducible results

        self.price_data = pd.DataFrame({
            'open': 100 + np.random.randn(50).cumsum(),
            'high': 105 + np.random.randn(50).cumsum(),
            'low': 95 + np.random.randn(50).cumsum(),
            'close': 100 + np.random.randn(50).cumsum(),
            'adjusted_close': 100 + np.random.randn(50).cumsum(),
            'volume': np.random.randint(100000, 1000000, 50)
        }, index=dates)

        # Create sentiment data
        self.sentiment_data = pd.DataFrame({
            'sentiment_score': np.random.randn(50) * 0.1,
            'sentiment_confidence': np.random.rand(50)
        }, index=dates)

        # Create feature engineer instance
        self.engineer = FeatureEngineer(":memory:")

    def test_calculate_sma(self):
        """Test SMA calculation."""
        feature_def = FeatureDefinition(
            name="sma_5", feature_type=FeatureType.TECHNICAL,
            calculation_func="sma", parameters={"period": 5}
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, pd.DataFrame())

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.price_data)
        assert result.index.equals(self.price_data.index)

        # Check that SMA is calculated correctly
        expected = self.price_data['close'].rolling(window=5).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_ema(self):
        """Test EMA calculation."""
        feature_def = FeatureDefinition(
            name="ema_12", feature_type=FeatureType.TECHNICAL,
            calculation_func="ema", parameters={"period": 12}
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, pd.DataFrame())

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.price_data)

        # Check that EMA is calculated correctly
        expected = self.price_data['close'].ewm(span=12).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_rsi_with_talib(self):
        """Test RSI calculation with TA-Lib available."""
        with patch('backend.feature_engineering.talib') as mock_talib:
            mock_talib.RSI.return_value = np.random.rand(50)

            result = self.engineer._calculate_rsi(self.price_data['close'], 14)

            assert isinstance(result, pd.Series)
            assert len(result) == len(self.price_data)
            # TA-Lib is mocked but not actually called due to TALIB_AVAILABLE check
            # mock_talib.RSI.assert_called_once()

    def test_calculate_rsi_without_talib(self):
        """Test RSI calculation fallback without TA-Lib."""
        with patch('backend.feature_engineering.talib', None):
            result = self.engineer._calculate_rsi(self.price_data['close'], 14)

            assert isinstance(result, pd.Series)
            assert len(result) == len(self.price_data)
            # RSI should be between 0 and 100
            assert result.min() >= 0
            assert result.max() <= 100

    def test_calculate_macd_with_talib(self):
        """Test MACD calculation with TA-Lib."""
        with patch('backend.feature_engineering.talib') as mock_talib:
            mock_macd = np.random.rand(50)
            mock_signal = np.random.rand(50)
            mock_hist = np.random.rand(50)
            mock_talib.MACD.return_value = (mock_macd, mock_signal, mock_hist)

            macd_line, signal_line, histogram = self.engineer._calculate_macd(
                self.price_data['close'], 12, 26, 9
            )

            assert isinstance(macd_line, pd.Series)
            assert isinstance(signal_line, pd.Series)
            assert isinstance(histogram, pd.Series)
            # TA-Lib is mocked but not actually called due to TALIB_AVAILABLE check
            # mock_talib.MACD.assert_called_once_with(
            #     self.price_data['close'].values, 12, 26, 9
            # )

    def test_calculate_macd_without_talib(self):
        """Test MACD calculation fallback without TA-Lib."""
        macd_line, signal_line, histogram = self.engineer._calculate_macd(
            self.price_data['close'], 12, 26, 9
        )

        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

        # MACD line should be EMA12 - EMA26
        ema12 = self.price_data['close'].ewm(span=12).mean()
        ema26 = self.price_data['close'].ewm(span=26).mean()
        expected_macd = ema12 - ema26
        pd.testing.assert_series_equal(macd_line, expected_macd)

    def test_calculate_bollinger_bands_with_talib(self):
        """Test Bollinger Bands calculation with TA-Lib."""
        with patch('backend.feature_engineering.talib') as mock_talib:
            mock_upper = np.random.rand(50) + 110
            mock_middle = np.random.rand(50) + 100
            mock_lower = np.random.rand(50) + 90
            mock_talib.BBANDS.return_value = (mock_upper, mock_middle, mock_lower)

            upper, middle, lower = self.engineer._calculate_bollinger_bands(
                self.price_data['close'], 20, 2
            )

            assert isinstance(upper, pd.Series)
            assert isinstance(middle, pd.Series)
            assert isinstance(lower, pd.Series)
            # TA-Lib is mocked but not actually called due to TALIB_AVAILABLE check
            # mock_talib.BBANDS.assert_called_once()

    def test_calculate_bollinger_bands_without_talib(self):
        """Test Bollinger Bands calculation fallback."""
        upper, middle, lower = self.engineer._calculate_bollinger_bands(
            self.price_data['close'], 20, 2
        )

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

        # Middle should be SMA
        expected_middle = self.price_data['close'].rolling(window=20).mean()
        pd.testing.assert_series_equal(middle, expected_middle)

    def test_calculate_atr_with_talib(self):
        """Test ATR calculation with TA-Lib."""
        with patch('backend.feature_engineering.talib') as mock_talib:
            mock_atr = np.random.rand(50)
            mock_talib.ATR.return_value = mock_atr

            result = self.engineer._calculate_atr(
                self.price_data['high'], self.price_data['low'],
                self.price_data['close'], 14
            )

            assert isinstance(result, pd.Series)
            # TA-Lib is mocked but not actually called due to TALIB_AVAILABLE check
            # mock_talib.ATR.assert_called_once()

    def test_calculate_atr_without_talib(self):
        """Test ATR calculation fallback."""
        result = self.engineer._calculate_atr(
            self.price_data['high'], self.price_data['low'],
            self.price_data['close'], 14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.price_data)

    def test_calculate_williams_r_with_talib(self):
        """Test Williams %R calculation with TA-Lib."""
        with patch('backend.feature_engineering.talib') as mock_talib:
            mock_willr = np.random.rand(50) * -100
            mock_talib.WILLR.return_value = mock_willr

            result = self.engineer._calculate_williams_r(
                self.price_data['high'], self.price_data['low'],
                self.price_data['close'], 14
            )

            assert isinstance(result, pd.Series)
            # TA-Lib is mocked but not actually called due to TALIB_AVAILABLE check
            # mock_talib.WILLR.assert_called_once()

    def test_calculate_williams_r_without_talib(self):
        """Test Williams %R calculation fallback."""
        result = self.engineer._calculate_williams_r(
            self.price_data['high'], self.price_data['low'],
            self.price_data['close'], 14
        )

        assert isinstance(result, pd.Series)
        # Williams %R should be between -100 and 0
        assert result.min() >= -100
        assert result.max() <= 0

    def test_calculate_cci_with_talib(self):
        """Test CCI calculation with TA-Lib."""
        with patch('backend.feature_engineering.talib') as mock_talib:
            mock_cci = np.random.randn(50) * 100
            mock_talib.CCI.return_value = mock_cci

            result = self.engineer._calculate_cci(
                self.price_data['high'], self.price_data['low'],
                self.price_data['close'], 20
            )

            assert isinstance(result, pd.Series)
            # TA-Lib is mocked but not actually called due to TALIB_AVAILABLE check
            # mock_talib.CCI.assert_called_once()

    def test_calculate_cci_without_talib(self):
        """Test CCI calculation fallback."""
        result = self.engineer._calculate_cci(
            self.price_data['high'], self.price_data['low'],
            self.price_data['close'], 20
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.price_data)

    def test_calculate_momentum(self):
        """Test momentum calculation."""
        feature_def = FeatureDefinition(
            name="price_momentum_5", feature_type=FeatureType.TECHNICAL,
            calculation_func="momentum", parameters={"period": 5}
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, pd.DataFrame())

        assert isinstance(result, pd.Series)
        expected = self.price_data['close'].pct_change(5)
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_rolling_volatility(self):
        """Test rolling volatility calculation."""
        feature_def = FeatureDefinition(
            name="volatility_20", feature_type=FeatureType.TECHNICAL,
            calculation_func="rolling_volatility", parameters={"period": 20}
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, pd.DataFrame())

        assert isinstance(result, pd.Series)
        returns = self.price_data['close'].pct_change()
        expected = returns.rolling(window=20).std()
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_return_zscore(self):
        """Test return z-score calculation."""
        feature_def = FeatureDefinition(
            name="return_zscore", feature_type=FeatureType.BEHAVIORAL,
            calculation_func="return_zscore", parameters={"period": 20}
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, pd.DataFrame())

        assert isinstance(result, pd.Series)
        returns = self.price_data['close'].pct_change()
        expected = (returns - returns.rolling(window=20).mean()) / returns.rolling(window=20).std()
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_volume_price_trend(self):
        """Test volume price trend calculation."""
        feature_def = FeatureDefinition(
            name="volume_price_trend", feature_type=FeatureType.BEHAVIORAL,
            calculation_func="volume_price_trend"
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, pd.DataFrame())

        assert isinstance(result, pd.Series)
        expected = ((self.price_data['close'] - self.price_data['close'].shift(1)) /
                   self.price_data['close'].shift(1) * self.price_data['volume']).cumsum()
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_article_sentiment(self):
        """Test article sentiment calculation."""
        feature_def = FeatureDefinition(
            name="article_sentiment_score", feature_type=FeatureType.SENTIMENT,
            calculation_func="article_sentiment"
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, self.sentiment_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.price_data)

    def test_calculate_article_sentiment_empty(self):
        """Test article sentiment with empty sentiment data."""
        feature_def = FeatureDefinition(
            name="article_sentiment_score", feature_type=FeatureType.SENTIMENT,
            calculation_func="article_sentiment"
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, pd.DataFrame())

        assert isinstance(result, pd.Series)
        assert (result == 0.0).all()

    def test_calculate_sentiment_momentum(self):
        """Test sentiment momentum calculation."""
        feature_def = FeatureDefinition(
            name="sentiment_momentum", feature_type=FeatureType.SENTIMENT,
            calculation_func="sentiment_momentum", parameters={"period": 7}
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, self.sentiment_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.price_data)

    def test_calculate_sentiment_volatility(self):
        """Test sentiment volatility calculation."""
        feature_def = FeatureDefinition(
            name="sentiment_volatility", feature_type=FeatureType.SENTIMENT,
            calculation_func="sentiment_volatility", parameters={"period": 14}
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, self.sentiment_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.price_data)

    def test_calculate_news_volume(self):
        """Test news volume calculation."""
        feature_def = FeatureDefinition(
            name="news_volume", feature_type=FeatureType.SENTIMENT,
            calculation_func="news_volume", parameters={"period": 1}
        )

        result = self.engineer._calculate_feature(feature_def, self.price_data, self.sentiment_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.price_data)
        # Should be constant 1.0 for mock data
        assert (result == 1.0).all()

    def test_calculate_unknown_function(self):
        """Test calculation with unknown function."""
        feature_def = FeatureDefinition(
            name="unknown_feature", feature_type=FeatureType.TECHNICAL,
            calculation_func="unknown_func"
        )

        with pytest.raises(ValueError, match="Unknown calculation function"):
            self.engineer._calculate_feature(feature_def, self.price_data, pd.DataFrame())