"""
Small helpers to keep strategy modules focused on trading logic (schemas + capability dicts).
"""

from __future__ import annotations

from typing import Any, Dict


def param_int(default: int, description: str, **extra: Any) -> Dict[str, Any]:
    return {"type": "int", "default": default, "description": description, **extra}


def param_float(default: float, description: str, **extra: Any) -> Dict[str, Any]:
    return {"type": "float", "default": default, "description": description, **extra}


def param_str(default: str, description: str, **extra: Any) -> Dict[str, Any]:
    return {"type": "str", "default": default, "description": description, **extra}


def param_bool(default: bool, description: str, **extra: Any) -> Dict[str, Any]:
    return {"type": "bool", "default": default, "description": description, **extra}


def default_capability_profile() -> Dict[str, Any]:
    """Baseline used by preflight / optimizer; override fields per strategy."""
    return {
        "requires_predictions": False,
        "required_prediction_horizons": [],
        "supports_signal_execution": True,
        "supports_backtrader_execution": True,
        "min_history_bars": 30,
        "supported_objectives": ["balanced", "sharpe", "return", "drawdown"],
    }


def capability_profile(**overrides: Any) -> Dict[str, Any]:
    """Merge overrides onto defaults for ``get_capability_profile``."""
    merged = default_capability_profile()
    merged.update(overrides)
    return merged
