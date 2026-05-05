"""
Pydantic models for UDF (Universal Data Feed) API responses.

These models correspond to the TradingView charting library's DatafeedConfiguration interface.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class Exchange(BaseModel):
    """Exchange descriptor for datafeed configuration."""
    value: str
    name: str
    desc: str


class DatafeedSymbolType(BaseModel):
    """Symbol type descriptor for datafeed configuration."""
    name: str
    value: str


class Unit(BaseModel):
    """Unit descriptor for unit conversion."""
    id: str
    name: str
    description: str


class CurrencyItem(BaseModel):
    """Currency descriptor for currency conversion."""
    id: str
    code: str
    logoUrl: Optional[str] = None
    description: Optional[str] = None


class DatafeedConfiguration(BaseModel):
    """
    Datafeed configuration data for TradingView charting library.

    This corresponds to the DatafeedConfiguration interface in the charting library.
    """
    # List of exchange descriptors
    exchanges: Optional[List[Exchange]] = None

    # List of supported resolutions
    supported_resolutions: Optional[List[str]] = None

    # Supported unit groups
    units: Optional[Dict[str, List[Unit]]] = None

    # Supported currencies for currency conversion
    currency_codes: Optional[List[str]] = None

    # Does the datafeed supports marks on bars
    supports_marks: Optional[bool] = None

    # Set this one to true if your datafeed provides server time
    supports_time: Optional[bool] = None

    # Does the datafeed supports marks on the timescale
    supports_timescale_marks: Optional[bool] = None

    # List of filter descriptors for symbol types
    symbols_types: Optional[List[DatafeedSymbolType]] = None

    # Symbol grouping configuration
    symbols_grouping: Optional[Dict[str, str]] = None

    # Does the datafeed supports search
    supports_search: Optional[bool] = None

    # Does the datafeed supports group request
    supports_group_request: Optional[bool] = None


class SymbolInfo(BaseModel):
    """Symbol information for TradingView charting library."""
    name: str
    ticker: Optional[str] = None
    description: str
    type: str
    session: str
    timezone: str
    exchange: str
    listed_exchange: str
    minmov: int = 1
    pricescale: int = 100
    has_intraday: bool = True
    supported_resolutions: List[str]
    has_daily: bool = True
    has_weekly_and_monthly: bool = True
    data_status: str = "streaming"
    currency_code: Optional[str] = None
    original_currency_code: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


class HistoricalDataResponse(BaseModel):
    """Historical bars data response."""
    s: str  # "ok" or "error"
    t: Optional[List[int]] = None  # timestamps
    o: Optional[List[float]] = None  # opens
    h: Optional[List[float]] = None  # highs
    l: Optional[List[float]] = None  # lows
    c: Optional[List[float]] = None  # closes
    v: Optional[List[int]] = None  # volumes
    errmsg: Optional[str] = None


class QuoteData(BaseModel):
    """Quote data for a symbol."""
    s: str  # "ok" or "error"
    n: str  # symbol name
    v: Dict[str, Any]  # quote values
