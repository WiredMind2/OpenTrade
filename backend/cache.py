"""
In-memory caching for frequently requested data to improve performance.
"""
import time
import hashlib
from typing import Any, Dict, Optional
from threading import Lock
import json

from backend.logging_config import get_app_logger

logger = get_app_logger()


class TTLCache:
    """Thread-safe TTL (Time-To-Live) cache with automatic expiration."""

    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.lock = Lock()

    def _make_key(self, *args, **kwargs) -> str:
        """Create a consistent cache key from arguments."""
        # Sort kwargs for consistent key generation
        key_parts = list(args) + [f"{k}:{v}" for k, v in sorted(kwargs.items())]
        key_string = json.dumps(key_parts, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < entry['ttl']:
                    return entry['data']
                else:
                    # Expired, remove it
                    del self.cache[key]
        return None

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        with self.lock:
            self.cache[key] = {
                'data': data,
                'timestamp': time.time(),
                'ttl': ttl or self.default_ttl
            }

    def invalidate(self, key: str) -> None:
        """Remove specific key from cache."""
        with self.lock:
            self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            # Clean expired entries
            current_time = time.time()
            expired_keys = [k for k, v in self.cache.items()
                          if current_time - v['timestamp'] >= v['ttl']]
            for k in expired_keys:
                del self.cache[k]
            return len(self.cache)


class UDFHistoryCache(TTLCache):
    """Specialized cache for UDF history data with symbol-aware invalidation."""

    def __init__(self, default_ttl: int = 600):
        super().__init__(default_ttl)
        self.symbol_last_update = {}  # Track last update time per symbol

    def _make_key(self, symbol: str, resolution: str, from_ts: int, to_ts: int,
                  countback: Optional[int], current_date: str) -> str:
        """Create a cache key that includes symbol and date awareness."""
        return super()._make_key(symbol, resolution, from_ts, to_ts, countback, current_date)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired and symbol data is still current."""
        entry = super().get(key)
        if entry:
            # Extract symbol from key (first part before first separator)
            try:
                symbol = key.split(':', 1)[0].upper()
                # Check if we have a last update time for this symbol
                if symbol in self.symbol_last_update:
                    last_update = self.symbol_last_update[symbol]
                    # If cache entry is older than symbol's last update, invalidate
                    if entry['timestamp'] < last_update:
                        self.invalidate(key)
                        return None
            except (IndexError, AttributeError):
                pass  # If we can't parse symbol, proceed with normal cache check
        return entry['data'] if entry else None

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache and update symbol last update time."""
        super().set(key, data, ttl)
        # Extract symbol from key and update last update time
        try:
            symbol = key.split(':', 1)[0].upper()
            self.symbol_last_update[symbol] = time.time()
        except (IndexError, AttributeError):
            pass  # If we can't parse symbol, skip update

    def invalidate_symbol(self, symbol: str) -> None:
        """Invalidate all cache entries for a specific symbol."""
        symbol = symbol.upper()
        keys_to_remove = []

        with self.lock:
            for key in self.cache.keys():
                try:
                    if key.split(':', 1)[0].upper() == symbol:
                        keys_to_remove.append(key)
                except (IndexError, AttributeError):
                    continue

            for key in keys_to_remove:
                self.cache.pop(key, None)

        # Update last update time to force cache miss for future requests
        self.symbol_last_update[symbol] = time.time()

        logger.info(f"Invalidated cache for symbol {symbol}, removed {len(keys_to_remove)} entries")

    def invalidate_all(self) -> None:
        """Invalidate all cache entries and reset symbol update times."""
        super().clear()
        self.symbol_last_update.clear()
        logger.info("Invalidated all UDF history cache entries")


# Global cache instances
chart_data_cache = TTLCache(default_ttl=300)  # 5 minutes for chart data
udf_history_cache = UDFHistoryCache(default_ttl=600)  # 10 minutes for UDF history with symbol-aware invalidation