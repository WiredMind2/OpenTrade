"""
Helpers for parameter-set (variant) identity and fingerprinting.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

# Keys that affect execution environment but not the strategy's parameter "variant"
# for comparison purposes. Keep in sync with API/backtest payloads.
_EXECUTION_META_KEYS = frozenset(
    {
        "execution_mode",
        "ticker",
        "commission_per_share",
        "slippage_bps",
        "min_trade_notional",
        "max_gross_exposure",
        "rebalance_frequency",
        "optimizer_mode",
        "experiment_id",
        "variant_label",
    }
)


def strip_execution_meta(parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of parameters without execution/meta keys for fingerprinting."""
    if not parameters:
        return {}
    return {k: v for k, v in parameters.items() if k not in _EXECUTION_META_KEYS}


def compute_params_hash(parameters: Optional[Dict[str, Any]]) -> str:
    """Stable SHA-256 hex digest of canonical strategy params (execution keys stripped)."""
    payload = strip_execution_meta(parameters)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def variant_label_from_params(parameters: Optional[Dict[str, Any]], max_len: int = 120) -> str:
    """Short human-readable label for UI (truncated JSON of strategy params)."""
    s = json.dumps(strip_execution_meta(parameters), sort_keys=True, default=str)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
