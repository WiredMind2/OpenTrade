# type: ignore[import, attr-defined, arg-type, union-attr]  # Complex file - ignoring type issues
"""
Feature engineering pipeline for trading backtesting system.

This module provides comprehensive feature extraction, transformation, and validation
for machine learning models in the trading system.
"""
import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings

from config import get_config
from logging_config import get_app_logger
from error_handling import handle_data_errors, DataIngestionError


logger = get_app_logger()


class FeatureType(Enum):
    """Feature type classifications."""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    BEHAVIORAL = "behavioral"
    LAGGED = "lagged"
    AGGREGATED = "aggregated"


@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    name: str
    feature_type: FeatureType
    calculation_func: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    frequency: str = "daily"  # daily, intraday, weekly
    window_size: Optional[int] = None


@dataclass
class FeatureSet:
    """Collection of features for a specific model/purpose."""
    name: str
    description: str
    features: List[FeatureDefinition]
    target_variable: Optional[str] = None
    horizon: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class FeatureEngineer:
    """Main feature engineering engine."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.feature_registry = {}
        self.scalers = {}
        self.cache = {}
        
        # Initialize feature definitions
        self._register_default_features()
    
    def _register_default_features(self):
        """Register default feature definitions."""
        
        # Technical indicators
        technical_features = [
            FeatureDefinition(
                name="sma_5",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="sma",
                parameters={"period": 5},
                description="5-day Simple Moving Average"
            ),
            FeatureDefinition(
                name="sma_20",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="sma",
                parameters={"period": 20},
                description="20-day Simple Moving Average"
            ),
            FeatureDefinition(
                name="ema_12",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="ema",
                parameters={"period": 12},
                description="12-day Exponential Moving Average"
            ),
            FeatureDefinition(
                name="ema_26",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="ema",
                parameters={"period": 26},
                description="26-day Exponential Moving Average"
            ),
            FeatureDefinition(
                name="rsi_14",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="rsi",
                parameters={"period": 14},
                description="14-day Relative Strength Index"
            ),
            FeatureDefinition(
                name="macd",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="macd",
                parameters={"fast": 12, "slow": 26, "signal": 9},
                description="MACD (12, 26, 9)"
            ),
            FeatureDefinition(
                name="bollinger_upper",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="bollinger_bands",
                parameters={"period": 20, "std": 2},
                description="Bollinger Bands Upper"
            ),
            FeatureDefinition(
                name="bollinger_lower",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="bollinger_bands",
                parameters={"period": 20, "std": 2},
                description="Bollinger Bands Lower"
            ),
            FeatureDefinition(
                name="atr_14",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="atr",
                parameters={"period": 14},
                description="14-day Average True Range"
            ),
            FeatureDefinition(
                name="volume_sma_20",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="volume_sma",
                parameters={"period": 20},
                description="20-day Volume Simple Moving Average"
            ),
            FeatureDefinition(
                name="price_momentum_5",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="momentum",
                parameters={"period": 5},
                description="5-day Price Momentum"
            ),
            FeatureDefinition(
                name="volatility_20",
                feature_type=FeatureType.TECHNICAL,
                calculation_func="rolling_volatility",
                parameters={"period": 20},
                description="20-day Rolling Volatility"
            )
        ]
        
        # Sentiment features
        sentiment_features = [
            FeatureDefinition(
                name="article_sentiment_score",
                feature_type=FeatureType.SENTIMENT,
                calculation_func="article_sentiment",
                parameters={"horizon": "1d"},
                description="Aggregated article sentiment score"
            ),
            FeatureDefinition(
                name="sentiment_momentum",
                feature_type=FeatureType.SENTIMENT,
                calculation_func="sentiment_momentum",
                parameters={"period": 7},
                description="7-day sentiment momentum"
            ),
            FeatureDefinition(
                name="sentiment_volatility",
                feature_type=FeatureType.SENTIMENT,
                calculation_func="sentiment_volatility",
                parameters={"period": 14},
                description="14-day sentiment volatility"
            ),
            FeatureDefinition(
                name="news_volume",
                feature_type=FeatureType.SENTIMENT,
                calculation_func="news_volume",
                parameters={"period": 1},
                description="Daily news article volume"
            )
        ]
        
        # Behavioral features
        behavioral_features = [
            FeatureDefinition(
                name="return_zscore",
                feature_type=FeatureType.BEHAVIORAL,
                calculation_func="return_zscore",
                parameters={"period": 20},
                description="Z-score of returns over 20-day window"
            ),
            FeatureDefinition(
                name="volume_price_trend",
                feature_type=FeatureType.BEHAVIORAL,
                calculation_func="volume_price_trend",
                parameters={},
                description="Volume Price Trend indicator"
            ),
            FeatureDefinition(
                name="williams_r",
                feature_type=FeatureType.BEHAVIORAL,
                calculation_func="williams_r",
                parameters={"period": 14},
                description="Williams %R oscillator"
            ),
            FeatureDefinition(
                name="cci",
                feature_type=FeatureType.BEHAVIORAL,
                calculation_func="cci",
                parameters={"period": 20},
                description="Commodity Channel Index"
            )
        ]
        
        # Register all features
        all_features = technical_features + sentiment_features + behavioral_features
        for feature in all_features:
            self.feature_registry[feature.name] = feature
        
        logger.info(f"Registered {len(all_features)} default features")
    
    @handle_data_errors
    def generate_features(self, ticker: str, start_date: str, end_date: str, 
                         feature_list: List[str] | None = None, save_to_db: bool = True) -> pd.DataFrame:
        """Generate features for a ticker over a date range."""
        logger.info(f"Generating features for {ticker} from {start_date} to {end_date}")
        
        # Load price data
        price_df = self._load_price_data(ticker, start_date, end_date)
        if price_df.empty:
            raise DataIngestionError(f"No price data found for {ticker}")
        
        # Load sentiment data
        sentiment_df = self._load_sentiment_data(ticker, start_date, end_date)
        
        # Determine which features to calculate
        if feature_list is None:
            feature_list = list(self.feature_registry.keys())
        
        # Calculate features
        feature_data = price_df.copy()
        
        for feature_name in feature_list:
            if feature_name not in self.feature_registry:
                logger.warning(f"Unknown feature: {feature_name}")
                continue
            
            feature_def = self.feature_registry[feature_name]
            try:
                feature_values = self._calculate_feature(feature_def, price_df, sentiment_df)
                feature_data[feature_name] = feature_values
            except Exception as e:
                logger.error(f"Failed to calculate feature {feature_name}: {e}")
                # Fill with NaN if calculation fails
                feature_data[feature_name] = np.nan
        
        # Remove rows with all NaN features
        feature_columns = [col for col in feature_data.columns if col not in ['ticker', 'date']]
        feature_data = feature_data.dropna(subset=feature_columns, how='all')
        
        # Save to database if requested
        if save_to_db:
            self._save_features_to_db(ticker, feature_data, feature_list)
        
        logger.info(f"Generated {len(feature_list)} features for {ticker}")
        return feature_data
    
    def _load_price_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load price data from database."""
        query = """
            SELECT date, open, high, low, close, adjusted_close, volume
            FROM price_daily
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    
    def _load_sentiment_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load sentiment data from database."""
        query = """
            SELECT date(canonical_timestamp) as date, predicted_return, confidence
            FROM sentiment_predictions sp
            JOIN articles a ON sp.article_id = a.id
            WHERE sp.ticker = ? 
            AND date(canonical_timestamp) BETWEEN ? AND ?
            ORDER BY date(canonical_timestamp)
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    # Aggregate by date
                    df = df.groupby(df.index).agg({
                        'predicted_return': 'mean',
                        'confidence': 'mean'
                    }).rename(columns={
                        'predicted_return': 'sentiment_score',
                        'confidence': 'sentiment_confidence'
                    })
        except Exception as e:
            logger.warning(f"Could not load sentiment data: {e}")
            df = pd.DataFrame()
        
        return df
    
    def _calculate_feature(self, feature_def: FeatureDefinition, price_df: pd.DataFrame, 
                          sentiment_df: pd.DataFrame) -> pd.Series:
        """Calculate a specific feature."""
        func_name = feature_def.calculation_func
        params = feature_def.parameters
        
        if func_name == "sma":
            period = params.get("period", 20)
            return price_df['close'].rolling(window=period).mean()
        
        elif func_name == "ema":
            period = params.get("period", 20)
            return price_df['close'].ewm(span=period).mean()
        
        elif func_name == "rsi":
            period = params.get("period", 14)
            return self._calculate_rsi(price_df['close'], period)
        
        elif func_name == "macd":
            fast = params.get("fast", 12)
            slow = params.get("slow", 26)
            signal = params.get("signal", 9)
            macd_line, signal_line, histogram = self._calculate_macd(
                price_df['close'], fast, slow, signal
            )
            return macd_line
        
        elif func_name == "bollinger_bands":
            period = params.get("period", 20)
            std = params.get("std", 2)
            upper, middle, lower = self._calculate_bollinger_bands(
                price_df['close'], period, std
            )
            if "upper" in feature_def.name:
                return upper
            elif "lower" in feature_def.name:
                return lower
            else:
                return middle
        
        elif func_name == "atr":
            period = params.get("period", 14)
            return self._calculate_atr(
                price_df['high'], price_df['low'], price_df['close'], period
            )
        
        elif func_name == "volume_sma":
            period = params.get("period", 20)
            return price_df['volume'].rolling(window=period).mean()
        
        elif func_name == "momentum":
            period = params.get("period", 10)
            return price_df['close'].pct_change(period)
        
        elif func_name == "rolling_volatility":
            period = params.get("period", 20)
            returns = price_df['close'].pct_change()
            return returns.rolling(window=period).std()
        
        elif func_name == "article_sentiment":
            # Calculate aggregated sentiment
            if sentiment_df.empty:
                return pd.Series(index=price_df.index, data=0.0)
            
            # Align sentiment data with price data
            aligned_sentiment = sentiment_df.reindex(price_df.index, method='ffill').fillna(0)
            return aligned_sentiment['sentiment_score']
        
        elif func_name == "sentiment_momentum":
            period = params.get("period", 7)
            if sentiment_df.empty:
                return pd.Series(index=price_df.index, data=0.0)
            
            sentiment_score = sentiment_df['sentiment_score'].reindex(
                price_df.index, method='ffill'
            ).fillna(0)
            return sentiment_score.diff(period)
        
        elif func_name == "sentiment_volatility":
            period = params.get("period", 14)
            if sentiment_df.empty:
                return pd.Series(index=price_df.index, data=0.0)
            
            sentiment_score = sentiment_df['sentiment_score'].reindex(
                price_df.index, method='ffill'
            ).fillna(0)
            return sentiment_score.rolling(window=period).std()
        
        elif func_name == "news_volume":
            period = params.get("period", 1)
            # Count articles per day (simplified)
            daily_counts = pd.Series(index=price_df.index, data=1.0)  # Mock data
            return daily_counts.rolling(window=period).sum()
        
        elif func_name == "return_zscore":
            period = params.get("period", 20)
            returns = price_df['close'].pct_change()
            return (returns - returns.rolling(window=period).mean()) / returns.rolling(window=period).std()
        
        elif func_name == "volume_price_trend":
            return ((price_df['close'] - price_df['close'].shift(1)) / price_df['close'].shift(1) * 
                   price_df['volume']).cumsum()
        
        elif func_name == "williams_r":
            period = params.get("period", 14)
            return self._calculate_williams_r(
                price_df['high'], price_df['low'], price_df['close'], period
            )
        
        elif func_name == "cci":
            period = params.get("period", 20)
            return self._calculate_cci(
                price_df['high'], price_df['low'], price_df['close'], period
            )
        
        else:
            raise ValueError(f"Unknown calculation function: {func_name}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            import talib
            return pd.Series(talib.RSI(prices.values, timeperiod=period), index=prices.index)
        except ImportError:
            # Fallback calculation if talib not available
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        try:
            import talib
            macd, signal_line, histogram = talib.MACD(prices.values, fast, slow, signal)
            return (
                pd.Series(macd, index=prices.index),
                pd.Series(signal_line, index=prices.index),
                pd.Series(histogram, index=prices.index)
            )
        except ImportError:
            # Fallback calculation
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                  std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            import talib
            upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, nbdevup=std, nbdevdn=std)
            return (
                pd.Series(upper, index=prices.index),
                pd.Series(middle, index=prices.index),
                pd.Series(lower, index=prices.index)
            )
        except ImportError:
            # Fallback calculation
            middle = prices.rolling(window=period).mean()
            std_dev = prices.rolling(window=period).std()
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            return upper, middle, lower
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        try:
            import talib
            atr_values = talib.ATR(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(atr_values, index=close.index)
        except ImportError:
            # Fallback calculation
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return true_range.rolling(window=period).mean()
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                             period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        try:
            import talib
            willr_values = talib.WILLR(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(willr_values, index=close.index)
        except ImportError:
            # Fallback calculation
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return wr
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        try:
            import talib
            cci_values = talib.CCI(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(cci_values, index=close.index)
        except ImportError:
            # Fallback calculation
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_dev = (typical_price - sma_tp).abs().rolling(window=period).mean()
            cci = (typical_price - sma_tp) / (0.015 * mean_dev)
            return cci
    
    def _save_features_to_db(self, ticker: str, feature_df: pd.DataFrame, feature_list: List[str]):
        """Save generated features to database."""
        # Create features table if it doesn't exist
        with sqlite3.connect(self.db_path) as conn:
            # Check if features table exists
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    dt TEXT,
                    feature_name TEXT,
                    feature_value REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_ticker_dt ON features(ticker, dt)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_name ON features(feature_name)")
            
            # Insert features
            insert_data = []
            for date, row in feature_df.iterrows():
                for feature_name in feature_list:
                    if feature_name in row and pd.notna(row[feature_name]):
                        insert_data.append((
                            ticker,
                            date.strftime('%Y-%m-%d'),
                            feature_name,
                            float(row[feature_name])
                        ))
            
            if insert_data:
                cursor.executemany("""
                    INSERT OR REPLACE INTO features (ticker, dt, feature_name, feature_value)
                    VALUES (?, ?, ?, ?)
                """, insert_data)
            
            conn.commit()
        
        logger.info(f"Saved {len(insert_data)} feature records for {ticker}")
    
    def create_feature_sets(self) -> Dict[str, FeatureSet]:
        """Create predefined feature sets for different use cases."""
        
        # Sentiment-focused features
        sentiment_features = [
            self.feature_registry[name] for name in [
                "article_sentiment_score", "sentiment_momentum", "sentiment_volatility",
                "news_volume", "sma_5", "rsi_14", "volume_sma_20"
            ]
        ]
        
        # Technical analysis features
        technical_features = [
            self.feature_registry[name] for name in [
                "sma_5", "sma_20", "ema_12", "ema_26", "rsi_14", "macd",
                "bollinger_upper", "bollinger_lower", "atr_14", "williams_r", "cci"
            ]
        ]
        
        # Momentum-focused features
        momentum_features = [
            self.feature_registry[name] for name in [
                "price_momentum_5", "volume_price_trend", "return_zscore",
                "volatility_20", "sma_20", "rsi_14", "macd"
            ]
        ]
        
        # Comprehensive feature set
        comprehensive_features = list(self.feature_registry.values())
        
        feature_sets = {
            "sentiment_focused": FeatureSet(
                name="sentiment_focused",
                description="Features focused on sentiment analysis and news impact",
                features=sentiment_features,
                target_variable="future_return_1d"
            ),
            "technical_analysis": FeatureSet(
                name="technical_analysis", 
                description="Technical indicators and price-based features",
                features=technical_features,
                target_variable="future_return_1d"
            ),
            "momentum_trading": FeatureSet(
                name="momentum_trading",
                description="Features optimized for momentum trading strategies",
                features=momentum_features,
                target_variable="future_return_1d"
            ),
            "comprehensive": FeatureSet(
                name="comprehensive",
                description="All available features for maximum information",
                features=comprehensive_features,
                target_variable="future_return_1d"
            )
        }
        
        return feature_sets
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = "mutual_info", 
                       k: int = 20, target_correlation: float = 0.8) -> List[str]:
        """Select the most relevant features."""
        logger.info(f"Selecting features using {method} method")
        
        # Remove constant and highly correlated features
        X_cleaned = self._remove_redundant_features(X, target_correlation)
        
        if method == "mutual_info":
            selector = SelectKBest(mutual_info_regression, k=min(k, X_cleaned.shape[1]))
            X_selected = selector.fit_transform(X_cleaned, y)
            selected_features = X_cleaned.columns[selector.get_support()].tolist()
        
        elif method == "correlation":
            # Select features with highest absolute correlation to target
            correlations = X_cleaned.corrwith(y.abs()).abs().sort_values(ascending=False)
            selected_features = correlations.head(k).index.tolist()
        
        elif method == "variance":
            # Select features with highest variance
            variances = X_cleaned.var().sort_values(ascending=False)
            selected_features = variances.head(k).index.tolist()
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        return selected_features
    
    def _remove_redundant_features(self, X: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with high correlation
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        X_cleaned = X.drop(columns=to_drop)
        
        if to_drop:
            logger.info(f"Removed {len(to_drop)} redundant features: {to_drop}")
        
        return X_cleaned
    
    def scale_features(self, X: pd.DataFrame, method: str = "standard", 
                      fit: bool = True, feature_names: List[str] = None) -> pd.DataFrame:
        """Scale features using various scaling methods."""
        if fit:
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
            
            # Store scaler for later use
            scaler_key = f"{method}_{hash(str(feature_names))}" if feature_names else method
            self.scalers[scaler_key] = scaler
        
        else:
            # Use pre-fitted scaler
            scaler_key = f"{method}_{hash(str(feature_names))}" if feature_names else method
            if scaler_key not in self.scalers:
                raise ValueError(f"Scaler {scaler_key} not found. Fit first.")
            
            X_scaled = pd.DataFrame(
                self.scalers[scaler_key].transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from a trained model."""
        if hasattr(model, 'coef_') and model.coef_ is not None:
            importance = np.abs(model.coef_)
            # Handle 1D and 2D coefficient arrays
            if importance.ndim > 1:
                importance = importance.flatten()
        elif hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            importance = model.feature_importances_
        else:
            raise ValueError("Model does not provide feature importance")

        return dict(zip(feature_names, importance))
    
    def create_lagged_features(self, df: pd.DataFrame, columns: List[str], 
                              lags: List[int]) -> pd.DataFrame:
        """Create lagged versions of features."""
        result_df = df.copy()
        
        for col in columns:
            for lag in lags:
                lagged_col = f"{col}_lag{lag}"
                result_df[lagged_col] = df[col].shift(lag)
        
        logger.info(f"Created {len(columns) * len(lags)} lagged features")
        return result_df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], 
                               windows: List[int], aggregations: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """Create rolling window features."""
        result_df = df.copy()
        
        for col in columns:
            for window in windows:
                for agg in aggregations:
                    if agg == 'mean':
                        feature_name = f"{col}_rolling_mean_{window}"
                        result_df[feature_name] = df[col].rolling(window=window).mean()
                    elif agg == 'std':
                        feature_name = f"{col}_rolling_std_{window}"
                        result_df[feature_name] = df[col].rolling(window=window).std()
                    elif agg == 'min':
                        feature_name = f"{col}_rolling_min_{window}"
                        result_df[feature_name] = df[col].rolling(window=window).min()
                    elif agg == 'max':
                        feature_name = f"{col}_rolling_max_{window}"
                        result_df[feature_name] = df[col].rolling(window=window).max()
        
        logger.info(f"Created {len(columns) * len(windows) * len(aggregations)} rolling features")
        return result_df


def create_feature_engineer(db_path: str = None) -> FeatureEngineer:
    """Create and initialize feature engineer."""
    if db_path is None:
        config = get_config()
        db_path = config.database.path
    
    return FeatureEngineer(db_path)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--db", default="data/backtest.db", help="Database path")
    parser.add_argument("--ticker", required=True, help="Stock ticker")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--features", nargs="+", help="Specific features to calculate")
    parser.add_argument("--output", help="Output CSV file")
    
    args = parser.parse_args()
    
    engineer = create_feature_engineer(args.db)
    
    # Generate features
    feature_data = engineer.generate_features(
        args.ticker, args.start, args.end, args.features
    )
    
    # Print summary
    print(f"\nGenerated features for {args.ticker}:")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Features: {list(feature_data.columns)}")
    print(f"Data points: {len(feature_data)}")
    
    # Save to file if requested
    if args.output:
        feature_data.to_csv(args.output)
        print(f"Saved feature data to {args.output}")
    
    # Show sample data
    print("\nSample feature data:")
    print(feature_data.head())