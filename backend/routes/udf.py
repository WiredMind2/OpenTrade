"""
UDF (Universal Data Feed) endpoints for TradingView charting integration.

This module provides endpoints that conform to TradingView's UDF API specification
for supplying chart data to the TradingView charting library.
"""
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import sys
import os
from urllib.parse import quote_plus
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException, Query

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

from backend.logging_config import get_component_logger
from backend.config import get_config
from backend.schemas.udf import DatafeedConfiguration, Exchange, DatafeedSymbolType, SymbolInfo
from backend.data_processing import (
    resample_ohlc_data,
    parse_resolution,
    resolution_to_timeframe,
    resolution_to_pandas_freq,
)
from backend.data_validation import DataValidator
from backend.cache import udf_history_cache


logger = get_component_logger(__file__)
router = APIRouter()

# Yahoo Finance rejects 1m requests spanning more than ~8 calendar days ("Only 8 days worth
# of 1m granularity data are allowed to be fetched per request."). A timedelta of 7 days spans
# eight calendar dates inclusive, so chunk by six calendar days to stay under the cap.
YAHOO_1M_MAX_CHUNK_DAYS = 6

# Yahoo only serves 1-minute bars for a rolling recent window (~30 days from "now").
YAHOO_1M_RETENTION_DAYS = 29

YAHOO_QUOTE_TYPE_TO_UDF = {
    "EQUITY": "stock",
    "ETF": "fund",
    "MUTUALFUND": "fund",
    "FUTURE": "futures",
    "CURRENCY": "forex",
    "CRYPTOCURRENCY": "crypto",
    "INDEX": "index",
}


def _normalize_symbol(symbol: str) -> str:
    normalized = str(symbol or "").upper().strip()
    if ":" in normalized:
        normalized = normalized.split(":")[-1]
    if normalized.endswith(".US"):
        normalized = normalized[:-3]
    return normalized


def _infer_currency(symbol: str, quote_type: str = "") -> str:
    upper = symbol.upper()
    qtype = quote_type.upper()
    if upper.endswith("=X"):
        pair = upper[:-2]
        return pair[-3:] if len(pair) >= 6 else "USD"
    if upper.endswith("-USD") or qtype == "CRYPTOCURRENCY":
        return "USD"
    return "USD"


def _infer_pricescale(symbol: str, quote_type: str = "") -> int:
    upper = symbol.upper()
    qtype = quote_type.upper()
    if qtype in {"CURRENCY", "CRYPTOCURRENCY"} or upper.endswith("=X") or upper.endswith("-USD"):
        return 100000
    return 100


def invalidate_symbol_cache(symbol: str) -> None:
    """Invalidate cache entries for a specific symbol.

    This should be called whenever new data is ingested for a symbol
    to ensure TradingView gets fresh data.
    """
    try:
        udf_history_cache.invalidate_symbol(symbol)
        logger.info(f"Invalidated UDF cache for symbol {symbol}")
    except Exception as e:
        logger.error(f"Failed to invalidate cache for symbol {symbol}: {e}")



def fetch_external_data(symbol: str, table: str, from_date: datetime, to_date: datetime) -> bool:
    """
    Fetch data from external sources and insert into database.

    Args:
        symbol: Stock ticker symbol
        table: Target table ('price_daily' or 'price_minute')
        from_date: Start date for data fetch
        to_date: End date for data fetch

    Returns:
        True if data was successfully fetched and inserted, False otherwise
    """
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance not available for data fetching")
        return False

    try:
        # Performance consideration: Limit the date range to avoid excessive data fetching
        # For minute data, limit to 30 days max to avoid huge datasets
        # For daily data, limit to 2 years max
        current_time = datetime.now()
        max_days = 30 if table == 'price_minute' else 730  # 30 days for minute, 2 years for daily

        # Adjust from_date if the range is too large
        days_requested = (to_date - from_date).days
        if days_requested > max_days:
            from_date = to_date - timedelta(days=max_days)
            logger.info(f"Limited data fetch range for {symbol} to {max_days} days: {from_date.date()} to {to_date.date()}")

        # Don't fetch data from the future
        if to_date > current_time:
            to_date = current_time

        # Don't fetch data that's too old (more than 5 years ago for performance)
        min_date = current_time - timedelta(days=5*365)
        if from_date < min_date:
            from_date = min_date
            logger.info(f"Limited data fetch start date for {symbol} to 5 years ago: {from_date.date()}")

        # Guarantee valid chronological order after all clamps.
        if from_date >= to_date:
            fallback_days = 30 if table == 'price_minute' else 365
            to_date = current_time
            from_date = to_date - timedelta(days=fallback_days)
            logger.warning(
                f"Adjusted invalid fetch window for {symbol} to recent range: "
                f"{from_date.date()} to {to_date.date()}"
            )

        # Minute data: Yahoo only allows ~30 days of 1m history ending at "now", regardless of
        # the chart's requested `to_date` (which may be months in the past when panning).
        if table == "price_minute":
            yahoo_1m_earliest = current_time - timedelta(days=YAHOO_1M_RETENTION_DAYS)
            if to_date < yahoo_1m_earliest:
                logger.info(
                    "Skipping Yahoo 1m fetch for %s: requested window ends before Yahoo retention "
                    "(%s < %s)",
                    symbol,
                    to_date.isoformat(),
                    yahoo_1m_earliest.isoformat(),
                )
                return False
            if from_date < yahoo_1m_earliest:
                logger.info(
                    "Clamping Yahoo 1m fetch start for %s from %s to %s (Yahoo ~%sd retention)",
                    symbol,
                    from_date.isoformat(),
                    yahoo_1m_earliest.isoformat(),
                    YAHOO_1M_RETENTION_DAYS,
                )
                from_date = yahoo_1m_earliest
            if from_date >= to_date:
                logger.info(
                    "No Yahoo 1m window left for %s after retention clamp (%s >= %s)",
                    symbol,
                    from_date.isoformat(),
                    to_date.isoformat(),
                )
                return False

        logger.info(f"Fetching external data for {symbol} from {from_date.date()} to {to_date.date()}")

        # Determine interval based on table
        if table == 'price_daily':
            interval = '1d'
        else:  # price_minute
            # For minute data, fetch 1-minute intervals
            interval = '1m'

        # Fetch data from Yahoo Finance
        stock = yf.Ticker(symbol)
        if table == 'price_daily':
            df = stock.history(start=from_date, end=to_date, interval=interval)
        else:
            # Minute: Yahoo caps 1m range per request — walk the window in chunks.
            # prepost=True: without it, windows that fall only in extended hours (common when
            # naive UTC bounds line up with post-market) return empty and look "delisted".
            yf_kw: Dict[str, Any] = {"interval": interval, "prepost": True}
            parts: List[pd.DataFrame] = []
            chunk_start = from_date
            while chunk_start < to_date:
                chunk_end = min(chunk_start + timedelta(days=YAHOO_1M_MAX_CHUNK_DAYS), to_date)
                part = stock.history(start=chunk_start, end=chunk_end, **yf_kw)
                if not part.empty:
                    parts.append(part)
                if chunk_end >= to_date:
                    break
                chunk_start = chunk_end

            if not parts:
                df = pd.DataFrame()
            else:
                df = pd.concat(parts)
                if isinstance(df.index, pd.DatetimeIndex) and df.index.has_duplicates:
                    df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()

        if df.empty:
            logger.warning(f"No data found for {symbol} from Yahoo Finance")
            return False

        # Reset index to get datetime as column
        df = df.reset_index()

        # Rename columns to match our schema
        df = df.rename(columns={
            'Datetime': 'dt',
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adjusted_close',
            'Volume': 'volume'
        })

        # Ensure date format matches table schema
        if table == 'price_daily':
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        else:  # price_minute
            if 'dt' in df.columns:
                df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d %H:%M:%S')

        df['ticker'] = symbol.upper()

        # Insert data into database
        from backend.main import app_state
        config = get_config()
        db_path = app_state.get('database_path') or config.database.path

        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            table_columns = {
                row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()
            }

            # Ensure ticker exists in tickers table
            metadata = _search_yahoo_symbols(symbol.upper(), "", 1)
            meta = metadata[0] if metadata else {}
            cur.execute(
                "INSERT OR IGNORE INTO tickers (ticker, name, exchange) VALUES (?, ?, ?)",
                (
                    symbol.upper(),
                    meta.get("description") if meta else None,
                    meta.get("exchange") if meta else None,
                ),
            )

            inserted = 0
            date_col = "date" if table == "price_daily" else "dt"
            has_adjusted_close = "adjusted_close" in table_columns

            for _, row in df.iterrows():
                try:
                    if table == "price_daily":
                        if has_adjusted_close:
                            cur.execute(
                                f"""
                                INSERT OR REPLACE INTO {table}
                                (ticker, date, open, high, low, close, adjusted_close, volume)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    row["ticker"],
                                    row["date"],
                                    float(row["open"]) if pd.notna(row["open"]) else None,
                                    float(row["high"]) if pd.notna(row["high"]) else None,
                                    float(row["low"]) if pd.notna(row["low"]) else None,
                                    float(row["close"]) if pd.notna(row["close"]) else None,
                                    (
                                        float(row["adjusted_close"])
                                        if ("adjusted_close" in row and pd.notna(row["adjusted_close"]))
                                        else (float(row["close"]) if pd.notna(row["close"]) else None)
                                    ),
                                    int(row["volume"]) if pd.notna(row["volume"]) else None,
                                ),
                            )
                        else:
                            cur.execute(
                                f"""
                                INSERT OR REPLACE INTO {table}
                                (ticker, date, open, high, low, close, volume)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    row["ticker"],
                                    row["date"],
                                    float(row["open"]) if pd.notna(row["open"]) else None,
                                    float(row["high"]) if pd.notna(row["high"]) else None,
                                    float(row["low"]) if pd.notna(row["low"]) else None,
                                    float(row["close"]) if pd.notna(row["close"]) else None,
                                    int(row["volume"]) if pd.notna(row["volume"]) else None,
                                ),
                            )
                    else:  # price_minute
                        cur.execute(
                            f"""
                            INSERT OR REPLACE INTO {table}
                            (ticker, dt, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                row["ticker"],
                                row["dt"],
                                float(row["open"]) if pd.notna(row["open"]) else None,
                                float(row["high"]) if pd.notna(row["high"]) else None,
                                float(row["low"]) if pd.notna(row["low"]) else None,
                                float(row["close"]) if pd.notna(row["close"]) else None,
                                int(row["volume"]) if pd.notna(row["volume"]) else None,
                            ),
                        )
                    inserted += 1
                except Exception as e:
                    logger.error(f"Error inserting row for {row['ticker']} at {row[date_col]}: {e}")

            conn.commit()

        # Invalidate cache for this symbol
        invalidate_symbol_cache(symbol.upper())

        logger.info(f"Successfully inserted {inserted} rows for {symbol} into {table}")
        return inserted > 0

    except Exception as e:
        logger.error(f"Failed to fetch external data for {symbol}: {str(e)}")
        return False


def _get_latest_dt_for_symbol(conn: sqlite3.Connection, table: str, symbol: str) -> Optional[datetime]:
    """Return latest timestamp (UTC) we have for a symbol in a given table."""
    date_col = "date" if table == "price_daily" else "dt"
    cur = conn.cursor()
    cur.execute(
        f"SELECT MAX({date_col}) FROM {table} WHERE ticker = ?",
        (symbol.upper(),),
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    try:
        # Normalize to a naive UTC datetime to avoid tz-aware/naive arithmetic issues.
        dt = pd.to_datetime(row[0], utc=True, errors="coerce").to_pydatetime()
        return dt.replace(tzinfo=None) if dt else None
    except Exception:
        return None


def _maybe_refresh_latest_data(symbol: str, table: str, to_date: datetime, from_date_hint: datetime) -> bool:
    """Fetch and persist missing recent bars if DB data is stale.

    This prevents the chart from getting stuck on old historical data when the DB already
    contains *some* rows for the symbol but hasn't been updated recently.
    """
    try:
        from backend.main import app_state
        config = get_config()
        db_path = app_state.get('database_path') or config.database.path

        with sqlite3.connect(db_path) as conn:
            latest = _get_latest_dt_for_symbol(conn, table, symbol)

        if latest is None:
            return False

        # Choose minimal "freshness" threshold to avoid refetching on every request.
        # Daily: if we're behind by >= 1 day. Minute: behind by >= 5 minutes.
        if table == "price_daily":
            latest_floor = latest.date()
            if to_date.date() <= latest_floor:
                return False
            fetch_from = max(from_date_hint, datetime(latest_floor.year, latest_floor.month, latest_floor.day) + timedelta(days=1))
        else:
            # Minute data timestamps can be naive/UTC-ish in yfinance; treat as UTC strings in DB.
            # Fetch a small overlap (1 minute) to guarantee we don't miss a boundary bar.
            if (to_date - latest) <= timedelta(minutes=5):
                return False
            fetch_from = max(from_date_hint, latest - timedelta(minutes=1))

        if fetch_from >= to_date:
            return False

        logger.info(
            "UDF history: refreshing latest %s data for %s from %s to %s",
            table,
            symbol.upper(),
            fetch_from,
            to_date,
        )
        return fetch_external_data(symbol.upper(), table, fetch_from, to_date)
    except Exception as e:
        logger.warning("UDF history: latest refresh check failed for %s (%s): %s", symbol, table, e)
        return False


def _ensure_symbol_available(db_path: str, symbol: str) -> bool:
    """Ensure a symbol exists in DB, bootstrapping via external fetch when needed."""
    normalized_symbol = _normalize_symbol(symbol)
    validator = DataValidator(db_path)
    if validator.validate_symbol_exists(normalized_symbol):
        return True

    logger.warning(
        "UDF symbols: %s missing in DB, attempting bootstrap fetch",
        normalized_symbol,
    )
    # Fetch a reasonable recent daily window; this also inserts into tickers table.
    fetch_to = datetime.now()
    fetch_from = fetch_to - timedelta(days=365 * 2)
    if not fetch_external_data(normalized_symbol, "price_daily", fetch_from, fetch_to):
        return False

    validator = DataValidator(db_path)
    return validator.validate_symbol_exists(normalized_symbol)


def _search_yahoo_symbols(q: str, exchange: str, limit: int) -> List[Dict[str, Any]]:
    """Search Yahoo Finance symbols to support symbols not yet stored locally."""
    cleaned = q.strip()
    if not cleaned:
        return []

    payload: Dict[str, Any] = {}
    try:
        max_results = max(1, min(limit, 50))
        search_url = (
            "https://query1.finance.yahoo.com/v1/finance/search"
            f"?q={quote_plus(cleaned)}&quotesCount={max_results}&newsCount=0"
        )
        with urlopen(search_url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        logger.warning("UDF yahoo search failed for %s: %s", cleaned, e)

    wanted_exchange = exchange.upper().strip() if exchange else ""
    exchange_aliases = {
        "NASDAQ": {"NASDAQ", "NMS", "NGM", "NCM"},
        "NYSE": {"NYSE", "NYQ"},
        "AMEX": {"AMEX", "ASE"},
        "OTC": {"OTC", "PNK"},
    }
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload.get("quotes", []) or []:
        symbol = str(item.get("symbol") or "").upper().strip()
        if not symbol or symbol in seen:
            continue

        quote_type = str(item.get("quoteType") or "").upper()
        if quote_type and quote_type not in YAHOO_QUOTE_TYPE_TO_UDF:
            continue

        exchange_code = str(item.get("exchange") or "").upper().strip()
        accepted_codes = exchange_aliases.get(wanted_exchange, {wanted_exchange}) if wanted_exchange else set()
        if wanted_exchange and exchange_code and exchange_code not in accepted_codes:
            continue

        short_name = str(item.get("shortname") or item.get("longname") or "").strip()
        if wanted_exchange:
            display_exchange = wanted_exchange
        else:
            display_exchange = exchange_code or "UNKNOWN"
        results.append({
            "symbol": symbol,
            "full_name": f"{display_exchange}:{symbol}",
            "description": short_name or f"{symbol} Stock",
            "exchange": display_exchange,
            "ticker": symbol,
            "type": YAHOO_QUOTE_TYPE_TO_UDF.get(quote_type, "stock"),
            "currency_code": _infer_currency(symbol, quote_type),
            "pricescale": _infer_pricescale(symbol, quote_type),
        })
        seen.add(symbol)
        if len(results) >= max_results:
            break

    if results:
        return results

    # Last-resort fallback: allow exact ticker-like queries even when upstream search is throttled.
    ticker_like = all(ch.isalnum() or ch in ".-=^/" for ch in cleaned) and len(cleaned) <= 16
    if ticker_like:
        fallback_symbol = cleaned.upper()
        display_exchange = wanted_exchange or "UNKNOWN"
        return [{
            "symbol": fallback_symbol,
            "full_name": f"{display_exchange}:{fallback_symbol}",
            "description": fallback_symbol,
            "exchange": display_exchange,
            "ticker": fallback_symbol,
            "type": "stock",
            "currency_code": _infer_currency(fallback_symbol),
            "pricescale": _infer_pricescale(fallback_symbol),
        }]

    return []


@router.get("/config", tags=["UDF"])
async def get_config_endpoint():
    """Return datafeed configuration for TradingView charting library."""
    try:
        from backend.main import app_state

        config = get_config()
        db_path = app_state.get('database_path') or config.database.path

        # Query database for available exchanges
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get distinct exchanges from tickers table
        cursor.execute("SELECT DISTINCT exchange FROM tickers WHERE exchange IS NOT NULL ORDER BY exchange")
        exchange_rows = cursor.fetchall()
        conn.close()

        # Create exchange objects
        exchanges = []
        for (exchange_name,) in exchange_rows:
            # Map common exchange codes to full names and descriptions
            exchange_mapping = {
                "NASDAQ": ("NASDAQ", "NASDAQ Stock Exchange"),
                "NYSE": ("NYSE", "New York Stock Exchange"),
                "AMEX": ("AMEX", "American Stock Exchange"),
                "OTC": ("OTC", "Over-the-Counter"),
            }

            name, desc = exchange_mapping.get(exchange_name, (exchange_name, f"{exchange_name} Exchange"))
            exchanges.append(Exchange(value=exchange_name, name=name, desc=desc))

        # If no exchanges found in database, provide defaults
        if not exchanges:
            exchanges = [
                Exchange(value="NASDAQ", name="NASDAQ", desc="NASDAQ Stock Exchange"),
                Exchange(value="NYSE", name="NYSE", desc="New York Stock Exchange")
            ]

        # Create symbol types
        symbols_types = [
            DatafeedSymbolType(name="Stock", value="stock")
        ]

        # Supported resolutions based on available data
        # We have price_minute (intraday) and price_daily tables
        supported_resolutions = [
            "1", "3", "5", "10", "15", "30", "45", "60", "120", "180", "240",
            "1D", "2D", "3D", "1W", "2W", "1M", "3M"
        ]

        # Create DatafeedConfiguration object
        udf_config = DatafeedConfiguration(
            exchanges=exchanges,
            symbols_types=symbols_types,
            supported_resolutions=supported_resolutions,
            supports_search=True,
            supports_group_request=True,
            supports_marks=False,
            supports_timescale_marks=False,
            supports_time=True
        )

        logger.info(f"UDF config requested with {len(exchanges)} exchanges")
        return udf_config

    except Exception as e:
        logger.error(f"Failed to get UDF config: {str(e)}")
        # Return UDF error format for consistency
        return {
            "s": "error",
            "errmsg": f"Internal server error: {str(e)}"
        }


@router.get("/search", tags=["UDF"])
async def search_symbols(
    q: str = Query(..., description="Search query"),
    type: str = Query("", description="Symbol type filter"),
    exchange: str = Query("", description="Exchange filter"),
    limit: int = Query(50, description="Maximum number of results")
):
    """Search for symbols with fuzzy matching.

    Performs fuzzy search across symbol names, tickers, and descriptions.
    Supports filtering by type and exchange.
    """
    try:
        from backend.main import app_state

        config = get_config()
        db_path = app_state.get('database_path') or config.database.path

        # Build search query with fuzzy matching
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Base query for fuzzy search
        query = """
            SELECT ticker, name, exchange, sector
            FROM tickers
            WHERE 1=1
        """
        params = []

        # Add search filter with fuzzy matching
        if q:
            # Fuzzy search on ticker, name, and sector
            search_conditions = []
            search_term = f"%{q.upper()}%"

            # Exact match on ticker gets highest priority
            if len(q) <= 10:  # Reasonable ticker length
                search_conditions.append("ticker LIKE ?")
                params.append(q.upper())

            # Fuzzy match on name and ticker
            search_conditions.append("ticker LIKE ?")
            params.append(search_term)
            search_conditions.append("name LIKE ?")
            params.append(search_term)

            query += f" AND ({' OR '.join(search_conditions)})"

        # Add exchange filter
        if exchange:
            query += " AND exchange = ?"
            params.append(exchange.upper())

        # Local DB does not store asset class yet. Keep local matches for any
        # requested type; Yahoo fallback supplies futures/forex/crypto metadata.

        # Add ordering for relevance (exact ticker matches first, then fuzzy matches)
        if q:
            query += """
                ORDER BY
                    CASE WHEN ticker = ? THEN 1
                         WHEN ticker LIKE ? THEN 2
                         ELSE 3 END,
                    LENGTH(ticker), ticker
            """
            params.extend([q.upper(), f"{q.upper()}%"])
        else:
            query += " ORDER BY ticker"

        # Add limit
        query += " LIMIT ?"
        params.append(min(limit, 100))  # Cap at 100 for performance

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # Format results for UDF
        results = []
        for ticker, name, exchange_val, sector in rows:
            # Enhanced description with sector information
            if name and sector:
                description = f"{name} ({sector})"
            elif name:
                description = name
            else:
                description = f"{ticker} Stock"

            result = {
                "symbol": ticker,
                "full_name": f"{exchange_val}:{ticker}",
                "description": description,
                "exchange": exchange_val or "UNKNOWN",
                "ticker": ticker,
                "type": "stock"
            }
            results.append(result)

        # Fallback to Yahoo search so empty DBs can discover new symbols.
        if q and len(results) < min(limit, 100):
            yahoo_results = _search_yahoo_symbols(q, exchange, min(limit, 100))
            existing = {item["ticker"] for item in results}
            for item in yahoo_results:
                if item["ticker"] not in existing:
                    results.append(item)
                    existing.add(item["ticker"])
                if len(results) >= min(limit, 100):
                    break

        logger.info(f"UDF search: query='{q}', type='{type}', exchange='{exchange}', limit={limit}, returned {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Failed to search symbols: {str(e)}")
        # Return empty results on error (UDF standard)
        return []


@router.get("/symbols", tags=["UDF"])
async def get_symbol_info(symbol: str = Query(..., description="Symbol name")):
    """Return symbol information for TradingView charting library.

    Performs symbol validation, normalization, and resolution to ensure
    only valid, known symbols are returned with accurate information.
    """
    try:
        from backend.main import app_state

        config = get_config()
        db_path = app_state.get('database_path') or config.database.path

        # Initialize data validator
        validator = DataValidator(db_path)

        # Normalize symbol (uppercase, remove common suffixes/prefixes if needed)
        normalized_symbol = _normalize_symbol(symbol)

        # Validate symbol exists (bootstrap fetch for empty DBs).
        if not _ensure_symbol_available(db_path, normalized_symbol):
            logger.warning(f"Symbol {normalized_symbol} not found in database after bootstrap attempt")
            return {
                "s": "error",
                "errmsg": f"Symbol '{symbol}' not found"
            }

        # Query database for symbol information
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ticker, name, exchange, sector
            FROM tickers
            WHERE ticker = ?
        """, (normalized_symbol,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            logger.error(f"Symbol validation passed but query failed for {normalized_symbol}")
            return {
                "s": "error",
                "errmsg": f"Symbol '{symbol}' not found"
            }

        ticker, name, exchange, sector = row
        # Treat placeholder exchanges as missing so we can rehydrate metadata from Yahoo.
        # Previously `exchange == "UNKNOWN"` was considered "present", so we skipped Yahoo
        # enrichment and kept returning UNKNOWN forever.
        exchange_norm = str(exchange or "").strip().upper()
        needs_remote_meta = (
            not name
            or not exchange_norm
            or exchange_norm == "UNKNOWN"
            or any(ch in str(ticker) for ch in ("=", "-", "^"))
        )
        yahoo_meta = _search_yahoo_symbols(ticker, "", 1) if needs_remote_meta else []
        exact_meta = next((item for item in yahoo_meta if item.get("ticker") == ticker), None)
        symbol_type = str((exact_meta or {}).get("type") or "stock")
        currency_code = str((exact_meta or {}).get("currency_code") or _infer_currency(ticker))
        pricescale = int((exact_meta or {}).get("pricescale") or _infer_pricescale(ticker))
        if (not name or name == f"{ticker} Stock") and exact_meta and exact_meta.get("description"):
            name = str(exact_meta["description"])
        if (not exchange_norm or exchange_norm == "UNKNOWN") and exact_meta and exact_meta.get("exchange"):
            exchange = str(exact_meta["exchange"])

        # Enhanced description with sector information
        if name and sector:
            description = f"{name} ({sector})"
        elif name:
            description = name
        else:
            description = f"{ticker} Stock"

        # Keep symbol-level resolution contract backward compatible for existing clients/tests.
        supported_resolutions = ["1", "5", "15", "30", "60", "240", "1D", "1W", "1M"]

        # Determine market hours based on exchange
        market_hours = {
            "NASDAQ": "0930-1600",
            "NYSE": "0930-1600",
            "AMEX": "0930-1600",
            "OTC": "0930-1600"
        }
        session = market_hours.get(exchange, "0930-1600") if exchange else "0930-1600"

        # Create SymbolInfo object with enhanced information
        symbol_info = SymbolInfo(
            name=ticker,
            ticker=ticker,
            description=description,
            type=symbol_type,
            session=session,
            timezone="America/New_York",
            exchange=exchange or "UNKNOWN",
            listed_exchange=exchange or "UNKNOWN",
            minmov=1,
            pricescale=pricescale,
            has_intraday=True,
            supported_resolutions=supported_resolutions,
            has_daily=True,
            has_weekly_and_monthly=True,
            data_status="streaming",
            currency_code=currency_code,
            original_currency_code=currency_code,
            sector=sector,
        )

        logger.info(f"UDF symbol info resolved for {symbol} -> {normalized_symbol}")
        return symbol_info

    except Exception as e:
        logger.error(f"Failed to get symbol info for {symbol}: {str(e)}")
        # Return UDF error format for consistency
        return {
            "s": "error",
            "errmsg": f"Internal server error: {str(e)}"
        }


@router.get("/symbol_info", tags=["UDF"])
async def get_symbol_info_group(
    group: str = Query(..., description="Symbol group request")
):
    """Return symbol information for group requests in table format.

    This endpoint handles group requests for multiple symbols,
    returning data in a tabular format as required by UDF v2.
    """
    try:
        from backend.main import app_state

        config = get_config()
        db_path = app_state.get('database_path') or config.database.path

        # Parse group request - typically format is "exchange:symbol1,symbol2,..."
        # For now, we'll handle simple group requests
        if not group or ':' not in group:
            return {
                "s": "error",
                "errmsg": "Invalid group format"
            }

        exchange, symbols_str = group.split(':', 1)
        symbol_list = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]

        if not symbol_list:
            return {
                "s": "error",
                "errmsg": "No symbols specified"
            }

        # Initialize data validator
        validator = DataValidator(db_path)

        # Query database for symbol information
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Build query for multiple symbols
        placeholders = ','.join('?' * len(symbol_list))
        query = f"""
            SELECT ticker, name, exchange, sector
            FROM tickers
            WHERE ticker IN ({placeholders})
            AND (exchange = ? OR ? = '')
        """

        params = symbol_list + [exchange.upper(), exchange.upper()]
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {
                "s": "error",
                "errmsg": f"No symbols found for group {group}"
            }

        # Format as table response for UDF group requests
        # UDF v2 expects table format with columns and data
        symbols_data = []
        for ticker, name, exchange_val, sector in rows:
            # Enhanced description with sector information
            if name and sector:
                description = f"{name} ({sector})"
            elif name:
                description = name
            else:
                description = f"{ticker} Stock"

            # Supported resolutions (same as config)
            supported_resolutions = [
                "1", "3", "5", "10", "15", "30", "45", "60", "120", "180", "240",
                "1D", "2D", "3D", "1W", "2W", "1M", "3M"
            ]

            # Determine market hours based on exchange
            market_hours = {
                "NASDAQ": "0930-1600",
                "NYSE": "0930-1600",
                "AMEX": "0930-1600",
                "OTC": "0930-1600"
            }
            session = market_hours.get(exchange_val, "0930-1600") if exchange_val else "0930-1600"

            symbol_data = {
                "symbol": ticker,
                "ticker": ticker,
                "name": ticker,
                "description": description,
                "type": "stock",
                "session": session,
                "timezone": "America/New_York",
                "exchange": exchange_val or "UNKNOWN",
                "listed_exchange": exchange_val or "UNKNOWN",
                "minmov": 1,
                "pricescale": 100,
                "has_intraday": True,
                "supported_resolutions": supported_resolutions,
                "has_daily": True,
                "has_weekly_and_monthly": True,
                "data_status": "streaming"
            }
            symbols_data.append(symbol_data)

        # Return table format response
        response = {
            "s": "ok",
            "symbols": symbols_data
        }

        logger.info(f"UDF symbol_info group request: {group}, returned {len(symbols_data)} symbols")
        return response

    except Exception as e:
        logger.error(f"Failed to get symbol info for group {group}: {str(e)}")
        return {
            "s": "error",
            "errmsg": f"Internal server error: {str(e)}"
        }


@router.get("/history", tags=["UDF"])
async def get_historical_data(
    symbol: str = Query(..., description="Symbol name"),
    resolution: str = Query(..., description="Resolution (1, 5, 15, etc.)"),
    from_ts: int = Query(..., description="From timestamp (Unix)"),
    to_ts: int = Query(..., description="To timestamp (Unix)"),
    countback: Optional[int] = Query(None, description="Number of bars to return")
):
    """Return historical bars data for TradingView charting library.

    This endpoint serves PURE historical OHLC data only. No predictions, forecasts,
    or future data are included. All data is validated to ensure it represents
    actual historical market data.
    """
    logger.info(f"UDF history request: symbol={symbol}, resolution={resolution}, from_ts={from_ts}, to_ts={to_ts}, countback={countback}")
    try:
        from backend.main import app_state

        config = get_config()
        db_path = app_state.get('database_path') or config.database.path
        logger.info(f"UDF history: using database path {db_path}")
        symbol = _normalize_symbol(symbol)

        # Initialize data validator
        validator = DataValidator(db_path)

        # Parse resolution early so timestamp normalization can use granularity.
        table, target_resolution = parse_resolution(resolution)
        date_col = 'date' if table == 'price_daily' else 'dt'

        # Convert timestamps to datetime (Windows can reject negative Unix timestamps).
        def _safe_from_timestamp(ts: int) -> datetime:
            try:
                return datetime.fromtimestamp(ts)
            except (OverflowError, OSError, ValueError):
                return datetime(1970, 1, 1) + timedelta(seconds=ts)

        from_date = _safe_from_timestamp(from_ts)
        to_date = _safe_from_timestamp(to_ts)

        # Ensure we never return future data - cap to_date at current time
        current_time = datetime.now()
        if to_date > current_time:
            to_date = current_time
            logger.info(f"UDF history: capped to_date to current time {to_date}")

        # TradingView can occasionally send malformed mixed-sign epoch ranges
        # (e.g. negative "from" with tiny positive "to"), but it also legitimately
        # requests very old/negative ranges while backfilling history. Only
        # normalize the malformed cases, otherwise we risk infinite backfill loops.
        stale_cutoff = current_time - timedelta(days=365 * 2)
        has_invalid_order = from_date >= to_date
        has_malformed_mixed_sign_epoch = (
            from_ts < 0 < to_ts and
            to_date < stale_cutoff
        )
        if has_invalid_order:
            logger.warning(
                "UDF history: invalid date range for %s (from_ts=%s, to_ts=%s)",
                symbol,
                from_ts,
                to_ts,
            )
            return {
                "s": "error",
                "errmsg": (
                    "Invalid date range: 'from' timestamp must be earlier than 'to' timestamp."
                ),
            }

        should_normalize_to_recent = has_malformed_mixed_sign_epoch

        if countback and countback > 0 and should_normalize_to_recent:
            timeframe = resolution_to_timeframe(resolution)
            lookback_bars = max(countback, 1)

            if timeframe.endswith('m'):
                bar_minutes = int(timeframe[:-1])
                lookback_delta = timedelta(minutes=bar_minutes * lookback_bars * 3)
            elif timeframe.endswith('h'):
                bar_hours = int(timeframe[:-1])
                lookback_delta = timedelta(hours=bar_hours * lookback_bars * 3)
            elif timeframe == '1d':
                lookback_delta = timedelta(days=lookback_bars * 2)
            elif timeframe == '1w':
                lookback_delta = timedelta(weeks=lookback_bars * 2)
            elif timeframe == '1M':
                lookback_delta = timedelta(days=lookback_bars * 62)
            else:
                lookback_delta = timedelta(days=max(lookback_bars * 2, 30))

            to_date = current_time
            from_date = current_time - lookback_delta
            logger.warning(
                f"UDF history: normalized stale/invalid range using countback={countback}: "
                f"{from_date} to {to_date}"
            )
        elif should_normalize_to_recent:
            # Some clients send invalid/stale windows without countback on initial load.
            # Use a safe recent fallback so bootstrap fetches can populate empty DBs.
            if table == "price_daily":
                lookback_delta = timedelta(days=365 * 2)
            else:
                lookback_delta = timedelta(days=30)

            to_date = current_time
            from_date = current_time - lookback_delta
            logger.warning(
                "UDF history: normalized stale/invalid range without countback: "
                f"{from_date} to {to_date}"
            )

        logger.info(f"UDF history: date range {from_date} to {to_date}")

        logger.info(f"UDF history: using table {table}, date_col {date_col}, target_resolution {target_resolution}")

        # If the colleague's DB is empty, TradingView will otherwise show "No data here".
        # Attempt external fetch for unknown symbols before failing fast.
        if not validator.validate_symbol_exists(symbol.upper()):
            logger.warning(
                f"UDF history: symbol {symbol} not found in database; attempting external fetch "
                f"for {from_date.date()} to {to_date.date()} (table={table})"
            )

            fetch_success = fetch_external_data(symbol.upper(), table, from_date, to_date)
            if not fetch_success:
                logger.error(f"UDF history: external fetch failed for unknown symbol {symbol}")
                return {
                    "s": "error",
                    "errmsg": f"Symbol '{symbol}' not found and external fetch failed"
                }

            # Rebuild validator after potential DB updates.
            validator = DataValidator(db_path)
            if not validator.validate_symbol_exists(symbol.upper()):
                logger.error(f"UDF history: external fetch reported success but symbol still missing: {symbol}")
                return {
                    "s": "error",
                    "errmsg": f"Failed to register symbol '{symbol}' after external fetch"
                }

        # If we already have data but it's stale (e.g. DB only contains historical Kaggle dump),
        # pull the missing recent bars and persist them before serving the query.
        _maybe_refresh_latest_data(symbol.upper(), table, to_date, from_date)

        # Let TradingView stop requesting older chunks naturally.
        if to_ts < 0 and from_ts < 0:
            logger.info(
                f"UDF history: old negative backfill window detected "
                f"({from_ts} -> {to_ts}), returning no_data"
            )
            return {
                "s": "no_data",
                "errmsg": "No older historical data available"
            }

        # Check cache first with improved key that includes current date to prevent stale data
        cache_key = f"{symbol.upper()}:{resolution}:{from_ts}:{to_ts}:{countback}:{current_time.date().isoformat()}"
        cached_response = udf_history_cache.get(cache_key)
        if cached_response:
            logger.info(f"Cache hit for UDF history: {symbol}, {resolution}")
            return cached_response

        conn = sqlite3.connect(db_path)

        # Format dates according to table type for proper string comparison
        if table == 'price_daily':
            # Daily table uses YYYY-MM-DD format
            from_date_str = from_date.date().isoformat()
            to_date_str = to_date.date().isoformat()
            current_date_str = current_time.date().isoformat()
        else:
            # Minute table stores dt as "YYYY-MM-DD HH:MM:SS" (space, not "T").
            # Keep query bounds in the same format so SQLite text comparisons match.
            from_date_str = from_date.strftime('%Y-%m-%d %H:%M:%S')
            to_date_str = to_date.strftime('%Y-%m-%d %H:%M:%S')
            current_date_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # For countback requests, prioritize bars up to "to" over strict "from" bounds.
        # TradingView often sends sparse/holiday windows while still expecting previous bars.
        is_countback_request = bool(countback and countback > 0)
        raw_fetch_limit: Optional[int] = None
        if is_countback_request:
            if table == "price_minute":
                target_minutes = 1
                normalized_target = str(target_resolution or "").upper()
                if normalized_target.isdigit():
                    target_minutes = max(int(normalized_target), 1)
                elif normalized_target.endswith("H"):
                    hour_part = normalized_target[:-1]
                    target_minutes = max(int(hour_part), 1) * 60 if hour_part.isdigit() else 60
                raw_fetch_limit = max(countback * target_minutes * 3, countback)
            else:
                # Keep additional daily bars so 2W/3M style resolutions can resample accurately.
                raw_fetch_limit = max(countback * 6, countback)

        if raw_fetch_limit is not None:
            query = f"""
                SELECT {date_col} as date, open, high, low, close, volume
                FROM {table}
                WHERE ticker = ?
                AND {date_col} <= ?
                AND {date_col} <= ?
                ORDER BY {date_col} DESC
                LIMIT ?
            """
            params = [symbol.upper(), to_date_str, current_date_str, raw_fetch_limit]
        else:
            # Build query - get raw historical data only
            # Explicitly filter to ensure no future data and only historical tables
            query = f"""
                SELECT {date_col} as date, open, high, low, close, volume
                FROM {table}
                WHERE ticker = ?
                AND {date_col} >= ?
                AND {date_col} <= ?
                AND {date_col} <= ?
                ORDER BY {date_col} ASC
            """
            # Add current time as additional filter to ensure no future data
            params = [symbol.upper(), from_date_str, to_date_str, current_date_str]

        logger.info(f"UDF history: executing query for {symbol.upper()} in table {table} (historical data only)")

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if not df.empty and raw_fetch_limit is not None:
            df = df.sort_values("date")

        logger.info(f"UDF history: query returned {len(df)} rows for {symbol}")
        if df.empty:
            # Avoid noisy provider calls for tiny daily windows (weekends/holidays/backfill gaps).
            # These commonly return empty data and produce misleading "possibly delisted" errors.
            if table == "price_daily" and (to_date.date() - from_date.date()).days <= 2:
                logger.info(
                    "UDF history: no local rows for narrow daily window %s -> %s; returning no_data without external fetch",
                    from_date.date(),
                    to_date.date(),
                )
                return {
                    "s": "no_data",
                    "errmsg": "No data available for requested range",
                }

            logger.warning(f"UDF history: no data found for {symbol} in table {table}, attempting external fetch")

            # Attempt to fetch data from external sources
            fetch_success = fetch_external_data(symbol.upper(), table, from_date, to_date)

            if fetch_success:
                logger.info(f"UDF history: external fetch successful for {symbol}, re-querying data")

                # Re-query the database after fetching external data
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(query, conn, params=params)
                conn.close()

                logger.info(f"UDF history: after external fetch, query returned {len(df)} rows for {symbol}")

                if df.empty:
                    logger.error(f"UDF history: still no data after external fetch for {symbol}")
                    return {
                        "s": "no_data",
                        "errmsg": "No data available"
                    }
            else:
                logger.error(f"UDF history: external fetch failed for {symbol}")
                # Yahoo does not serve 1m for arbitrary history; return no_data so TradingView
                # can back off instead of surfacing a hard datafeed error on every pan.
                if table == "price_minute":
                    return {
                        "s": "no_data",
                        "errmsg": (
                            "No intraday data for this range (Yahoo ~30d of 1m). "
                            "Zoom to recent dates, use a coarser resolution, or load minute data locally."
                        ),
                    }
                return {
                    "s": "error",
                    "errmsg": "Failed to fetch data from external sources",
                }

        # Convert resolution to timeframe for consistent resampling
        timeframe = resolution_to_timeframe(resolution)

        # Optimized resampling for UDF with better performance
        import time
        udf_resample_start = time.time()

        if not df.empty and timeframe not in ['1m', '1d']:
            logger.info(f"UDF: Starting optimized resampling for {symbol}: {len(df)} rows from {timeframe}")

            try:
                # Prepare data for resampling
                df_resample = df.copy()
                df_resample['date'] = pd.to_datetime(df_resample['date'], errors='coerce')
                df_resample = df_resample.dropna(subset=['date']).set_index('date').sort_index()

                # Convert requested resolution to pandas frequency dynamically.
                freq = resolution_to_pandas_freq(resolution)

                # Perform efficient OHLC resampling
                ohlc_df = df_resample[['open', 'high', 'low', 'close']].resample(freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last'
                })

                # Resample volume
                volume_df = df_resample[['volume']].resample(freq).sum()

                # Combine and clean
                df = pd.concat([ohlc_df, volume_df], axis=1).dropna()

                # Reset index and format dates
                df = df.reset_index()
                df['date'] = df['date'].dt.strftime('%Y-%m-%dT%H:%M:%S')

                udf_resample_time = time.time() - udf_resample_start
                logger.info(f"UDF: Resampling completed for {symbol}: {len(df)} rows in {udf_resample_time:.3f}s")

            except Exception as e:
                logger.warning(f"UDF: Resampling failed for {symbol}, using original data: {e}")
                # Keep original data if resampling fails
                udf_resample_time = time.time() - udf_resample_start
                logger.info(f"UDF: Resampling failed, keeping original data: {udf_resample_time:.3f}s")
        else:
            udf_resample_time = time.time() - udf_resample_start
            logger.info(f"UDF: No resampling needed for {symbol}, timeframe {timeframe}: {udf_resample_time:.3f}s")

        # Apply countback limit after resampling
        if countback and len(df) > countback:
            df = df.tail(countback)

        # Validate historical data quality with enhanced checks
        udf_validation_start = time.time()
        if not df.empty:
            logger.info(f"UDF: Starting data validation for {symbol}: {len(df)} rows")
            df = df.sort_values('date')

            # Enhanced price validation
            price_columns = ['open', 'high', 'low', 'close']
            validation_issues = 0

            for col in price_columns:
                if col in df.columns:
                    # Check for negative prices
                    negative_prices = (df[col] < 0).sum()
                    if negative_prices > 0:
                        logger.warning(f"Found {negative_prices} negative {col} prices for {symbol}")
                        validation_issues += negative_prices

                    # Check for extremely high prices (> $100,000)
                    extreme_prices = (df[col] > 100000).sum()
                    if extreme_prices > 0:
                        logger.warning(f"Found {extreme_prices} extreme {col} prices for {symbol}")
                        validation_issues += extreme_prices

            # Validate OHLC relationships
            invalid_ohlc = ((df['high'] < df['low']) |
                           (df['open'] < df['low']) |
                           (df['open'] > df['high']) |
                           (df['close'] < df['low']) |
                           (df['close'] > df['high'])).sum()
            if invalid_ohlc > 0:
                logger.warning(f"Found {invalid_ohlc} invalid OHLC relationships for {symbol}")
                validation_issues += invalid_ohlc

            # Check for zero volume (might indicate missing data)
            if 'volume' in df.columns:
                zero_volume = (df['volume'] == 0).sum()
                if zero_volume > len(df) * 0.5:  # More than 50% zero volume
                    logger.warning(f"Found {zero_volume} zero volume entries for {symbol} ({zero_volume/len(df):.1%})")

            # Check for data gaps (missing dates)
            if len(df) > 1:
                df_temp = df.copy()
                df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
                date_diffs = df_temp['date'].diff().dt.days
                gaps = (date_diffs > 1).sum()  # Assuming daily data, gaps > 1 day
                if gaps > 0:
                    logger.info(f"Found {gaps} data gaps in {symbol} historical data")

            if validation_issues > 0:
                logger.warning(f"Data validation found {validation_issues} issues for {symbol}")

            udf_validation_time = time.time() - udf_validation_start
            logger.info(f"UDF: Data validation completed for {symbol}: {udf_validation_time:.3f}s")
        else:
            udf_validation_time = time.time() - udf_validation_start
            logger.info(f"UDF: No data validation needed for {symbol}: {udf_validation_time:.3f}s")

        # Vectorized conversion to UDF format with millisecond timestamps
        if not df.empty:
            # Ensure dates are datetime objects
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce', utc=True)
            df_copy = df_copy.dropna(subset=['date'])

            # Normalize epoch scale robustly (s/us/ns -> ms) to satisfy TradingView.
            epoch_raw = df_copy['date'].astype('int64')
            abs_max = int(epoch_raw.abs().max()) if not epoch_raw.empty else 0
            if abs_max < 10**11:  # seconds
                timestamps = (epoch_raw * 1000).astype('int64').tolist()
            elif abs_max < 10**14:  # milliseconds
                timestamps = epoch_raw.astype('int64').tolist()
            elif abs_max < 10**17:  # microseconds
                timestamps = (epoch_raw // 1000).astype('int64').tolist()
            else:  # nanoseconds
                timestamps = (epoch_raw // 10**6).astype('int64').tolist()

            opens = df_copy['open'].astype(float).tolist()
            highs = df_copy['high'].astype(float).tolist()
            lows = df_copy['low'].astype(float).tolist()
            closes = df_copy['close'].astype(float).tolist()
            volumes = df_copy['volume'].fillna(0).astype(int).tolist()
        else:
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

        response = {
            "s": "ok",
            "t": timestamps,
            "o": opens,
            "h": highs,
            "l": lows,
            "c": closes,
            "v": volumes
        }

        return response

    except Exception as e:
        logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
        return {
            "s": "error",
            "errmsg": f"Internal server error: {str(e)}"
        }


@router.get("/quotes", tags=["UDF"])
async def get_quotes(symbols: str = Query(..., description="Comma-separated list of symbols")):
    """Return quote data for TradingView charting library."""
    try:
        from backend.main import app_state

        config = get_config()
        db_path = app_state.get('database_path') or config.database.path

        symbol_list = [_normalize_symbol(s) for s in symbols.split(',') if _normalize_symbol(s)]

        conn = sqlite3.connect(db_path)

        quotes = []
        for symbol in symbol_list:
            # Try to get data from price_minute first (most recent), fall back to price_daily
            tables_to_try = [
                ('price_minute', 'dt', 'dt'),
                ('price_daily', 'date', 'date')
            ]

            current_data = None
            previous_close = None

            for table, date_col, order_col in tables_to_try:
                # Get the 2 most recent rows to calculate change
                query = f"""
                    SELECT open, high, low, close, volume, {date_col} as date
                    FROM {table}
                    WHERE ticker = ?
                    ORDER BY {order_col} DESC
                    LIMIT 2
                """

                df = pd.read_sql_query(query, conn, params=[symbol])

                if not df.empty:
                    current_data = df.iloc[0]  # Most recent
                    if len(df) > 1:
                        previous_close = float(df.iloc[1]['close'])
                    break

            if current_data is not None:
                current_close = float(current_data['close'])
                current_open = float(current_data['open'])
                current_high = float(current_data['high'])
                current_low = float(current_data['low'])
                current_volume = int(current_data['volume']) if not pd.isna(current_data['volume']) else 0

                # Calculate change and change percentage
                if previous_close is not None and previous_close != 0:
                    change = current_close - previous_close
                    change_pct = (change / previous_close) * 100
                else:
                    change = 0.0
                    change_pct = 0.0

                quote = {
                    "s": "ok",
                    "n": symbol,
                    "v": {
                        "lp": current_close,  # Last price
                        "open_price": current_open,  # Today's opening price
                        "high_price": current_high,  # Today's high price
                        "low_price": current_low,  # Today's low price
                        "volume": current_volume,  # Today's trading volume
                        "ch": change,  # Price change
                        "chp": change_pct,  # Price change percentage
                        "prev_close_price": previous_close if previous_close is not None else current_close  # Previous close
                    }
                }
            else:
                quote = {
                    "s": "error",
                    "n": symbol,
                    "v": {}
                }

            quotes.append(quote)

        conn.close()

        logger.info(f"UDF quotes requested for {len(symbol_list)} symbols")
        return quotes

    except Exception as e:
        logger.error(f"Failed to get quotes: {str(e)}")
        # Return error for all requested symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        return [
            {
                "s": "error",
                "n": symbol,
                "v": {}
            }
            for symbol in symbol_list
        ]
