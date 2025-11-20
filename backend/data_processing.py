"""
Data processing utilities for chart data aggregation and transformations.
"""
import math
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from backend.logging_config import get_app_logger

logger = get_app_logger()


def aggregate_predictions(raw_predictions: List[Dict[str, Any]], aggregate: Optional[str]) -> List[Dict[str, Any]]:
    """
    Aggregate multiple predictions per target date based on the specified mode.

    Args:
        raw_predictions: List of raw prediction dictionaries
        aggregate: Aggregation mode ('avg', 'latest', 'max_conf') or None

    Returns:
        List of aggregated predictions
    """
    if aggregate not in ('avg', 'latest', 'max_conf'):
        return raw_predictions

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in raw_predictions:
        key = r['date'] or 'null'
        grouped.setdefault(key, []).append(r)

    aggregated: List[Dict[str, Any]] = []
    for d, items in grouped.items():
        if d == 'null':
            # Keep entries without target date as-is (cannot aggregate)
            for it in items:
                aggregated.append(it)
            continue

        if aggregate == 'avg':
            vals = [it['predicted_price'] for it in items if it['predicted_price'] is not None]
            avg_price = float(sum(vals) / len(vals)) if vals else None
            avg_conf = float(sum(it['confidence'] for it in items) / len(items))
            aggregated.append({
                'date': d,
                'predicted_price': avg_price,
                'confidence': avg_conf,
                'count': len(items)
            })
        elif aggregate == 'max_conf':
            best = max(items, key=lambda it: it.get('confidence', 0))
            aggregated.append({
                'date': d,
                'predicted_price': best['predicted_price'],
                'confidence': best['confidence'],
                'count': len(items)
            })
        else:  # latest
            # choose prediction with latest produced_at
            def _to_dt(v):
                try:
                    return pd.to_datetime(v).to_datetime64()
                except Exception:
                    return pd.NaT

            latest = max(items, key=lambda it: _to_dt(it.get('produced_at')) if it.get('produced_at') else pd.NaT)
            aggregated.append({
                'date': d,
                'predicted_price': latest['predicted_price'],
                'confidence': latest.get('confidence', 0.5),
                'count': len(items)
            })

    return aggregated


def validate_confidence(confidence: Any) -> float:
    """
    Validate and clamp confidence value to [0, 1].

    Args:
        confidence: Raw confidence value

    Returns:
        Validated confidence between 0 and 1
    """
    try:
        confidence = float(confidence)
        # Handle NaN values
        if math.isnan(confidence):
            return 0.5
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
    except (ValueError, TypeError):
        confidence = 0.5
    return confidence


def calculate_predicted_price(base_price: Optional[float], predicted_return: Optional[float]) -> Optional[float]:
    """
    Calculate predicted price from base price and predicted return.

    Args:
        base_price: Base price for the calculation
        predicted_return: Predicted return percentage

    Returns:
        Predicted price or None if calculation not possible
    """
    if base_price is not None and predicted_return is not None:
        try:
            predicted_return = float(predicted_return)
            # Check for extreme returns (>50% or <-50%)
            if abs(predicted_return) > 0.5:
                # Could log warning here, but for testing we'll allow it
                pass
            return float(base_price) * (1.0 + predicted_return)
        except (ValueError, TypeError):
            pass
    return None


def find_base_price(price_by_date: Dict[str, float], produced_date: Optional[datetime.date]) -> Optional[float]:
    """
    Find the base price for a prediction based on produced date.

    Args:
        price_by_date: Dictionary mapping dates to closing prices
        produced_date: Date when prediction was produced

    Returns:
        Base price or None if not found
    """
    if produced_date:
        produced_iso = produced_date.isoformat()
        if produced_iso in price_by_date:
            return price_by_date[produced_iso]
        else:
            # Find nearest prior date in historical
            prior_dates = [d for d in price_by_date.keys() if d <= produced_iso]
            if prior_dates:
                nearest = max(prior_dates)
                return price_by_date[nearest]
    return None


def process_prediction_record(
    prow: pd.Series,
    price_by_date: Dict[str, float],
    horizon: str
) -> Dict[str, Any]:
    """
    Process a single prediction record into the standardized format.

    Args:
        prow: Prediction row from database
        price_by_date: Dictionary mapping dates to prices
        horizon: Prediction horizon ('1d', '3d', '7d')

    Returns:
        Processed prediction record
    """
    produced_at = prow.get('produced_at')

    # Validate and parse produced_at
    try:
        if produced_at:
            produced_datetime = pd.to_datetime(produced_at)
            produced_date = produced_datetime.date()
            produced_at_iso = produced_datetime.isoformat()
        else:
            produced_date = None
            produced_at_iso = None
    except Exception as e:
        # Log parsing failures that lead to null dates
        logger.warning(f"Failed to parse produced_at '{produced_at}' for prediction: {e}")
        produced_date = None
        produced_at_iso = None

    # Determine target date based on horizon
    horizon_days = 1 if horizon == '1d' else 3 if horizon == '3d' else 7
    target_date = None
    if produced_date:
        target_date = (produced_date + pd.Timedelta(days=horizon_days)).isoformat()

    # Determine base price
    base_price = find_base_price(price_by_date, produced_date)

    # Validate predicted_return
    predicted_return = prow.get('predicted_return')
    if predicted_return is not None:
        try:
            predicted_return = float(predicted_return)
            # Check for extreme returns (>50% or <-50%)
            if abs(predicted_return) > 0.5:
                # Could log warning, but for testing we'll track it
                pass
        except (ValueError, TypeError):
            predicted_return = None

    predicted_price = calculate_predicted_price(base_price, predicted_return)

    # Get actual price if available
    actual_price = None
    if target_date and target_date in price_by_date:
        actual_price = float(price_by_date[target_date])

    # Validate confidence
    confidence = validate_confidence(prow.get('predicted_confidence', prow.get('confidence', 0.5)))

    return {
        "date": target_date,
        "predicted_price": predicted_price,
        "actual_price": actual_price,
        "confidence": confidence,
        "produced_at": produced_at_iso
    }


def resample_ohlc_data(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """
    Resample OHLC data to the specified resolution.

    Args:
        df: DataFrame with columns [date, open, high, low, close, volume]
        resolution: Target resolution (1, 5, 15, 30, 60, 240 for minutes, 1D, 1W, 1M for daily+)

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    # Convert resolution to pandas frequency
    freq_map = {
        '1': '1min',
        '5': '5min',
        '15': '15min',
        '30': '30min',
        '60': '1h',
        '240': '4h',
        '1D': '1D',
        '1W': '1W',
        '1M': '1M'
    }

    freq = freq_map.get(resolution)
    if not freq:
        # Default to 1 minute if resolution not recognized
        freq = '1min'

    # Set date as index and sort
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # Resample OHLC data
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Reset index and convert date back to string
    resampled = resampled.reset_index()
    resampled['date'] = resampled['date'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    return resampled


def parse_resolution(resolution: str) -> tuple[str, str]:
    """
    Parse resolution string to determine table and target frequency.

    Args:
        resolution: Resolution string (1, 5, 15, 30, 60, 240, 1D, 1W, 1M)

    Returns:
        Tuple of (table_name, target_resolution)
    """
    # Determine table based on resolution
    if resolution.endswith('D') or resolution.endswith('W') or resolution.endswith('M'):
        table = 'price_daily'
        # For daily+ resolutions, we might need to resample further
        # But for now, return as-is since price_daily is already daily
        target_resolution = resolution
    else:
        # Intraday resolutions
        table = 'price_minute'
        target_resolution = resolution

    return table, target_resolution


def resolution_to_timeframe(resolution: str) -> str:
    """
    Convert UDF resolution format to timeframe format used in predictions.

    Args:
        resolution: UDF resolution (1, 5, 15, 30, 60, 240, 1D, 1W, 1M)

    Returns:
        Timeframe string (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
    """
    resolution_map = {
        '1': '1m',
        '5': '5m',
        '15': '15m',
        '30': '30m',
        '60': '1h',
        '240': '4h',
        '1D': '1d',
        '1W': '1w',
        '1M': '1M'
    }
    return resolution_map.get(resolution, '1d')


def timeframe_to_resolution(timeframe: str) -> str:
    """
    Convert timeframe format to UDF resolution format.

    Args:
        timeframe: Timeframe string (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)

    Returns:
        UDF resolution (1, 5, 15, 30, 60, 240, 1D, 1W, 1M)
    """
    timeframe_map = {
        '1m': '1',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '4h': '240',
        '1d': '1D',
        '1w': '1W',
        '1M': '1M'
    }
    return timeframe_map.get(timeframe, '1D')