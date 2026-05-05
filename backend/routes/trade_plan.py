"""Actionable trade-plan endpoint."""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import get_config


router = APIRouter()

TraderStyle = Literal["auto", "short", "swing", "long"]
Direction = Literal["long", "short", "wait", "exit"]


class TradePlanRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=16)
    style: TraderStyle = "auto"
    account_size: float = Field(default=10000.0, gt=0)
    risk_percent: float = Field(default=1.0, gt=0, le=10)
    as_of_date: Optional[datetime] = None
    signal_action: Optional[str] = None
    signal_confidence: Optional[float] = None
    signal_reason: Optional[str] = None


class TradePlanResponse(BaseModel):
    ticker: str
    style: Literal["short", "swing", "long"]
    trader_type: str
    direction: Direction
    confidence: float
    entry: Optional[float]
    stop_loss: Optional[float]
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    trailing_stop: Optional[float]
    invalidation: str
    time_exit: str
    risk_reward: Optional[float]
    risk_amount: float
    position_size: int
    latest_close: float
    price_date: str
    strategy: str
    reasons: List[str]
    warnings: List[str]
    indicators: Dict[str, Optional[float]]
    style_scores: Dict[str, float]


def _db_path() -> str:
    for module_name in ("backend.main", "main"):
        try:
            module = __import__(module_name, fromlist=["app_state"])
            app_state = getattr(module, "app_state", None)
            if isinstance(app_state, dict) and app_state.get("database_path"):
                return str(app_state["database_path"])
        except Exception:
            continue
    return get_config().database.path


def _load_prices(ticker: str, as_of: Optional[datetime], limit: int = 280) -> pd.DataFrame:
    conn = sqlite3.connect(_db_path())
    try:
        where_as_of = "AND date <= ?" if as_of else ""
        params: List[Any] = [ticker.upper()]
        if as_of:
            params.append(as_of.date().isoformat())
        params.append(limit)
        df = pd.read_sql_query(
            f"""
            SELECT date, open, high, low, close, volume
            FROM price_daily
            WHERE ticker = ? {where_as_of}
            ORDER BY date DESC
            LIMIT ?
            """,
            conn,
            params=params,
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def _num(value: Any) -> Optional[float]:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _round_price(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value >= 100:
        return round(value, 2)
    if value >= 10:
        return round(value, 3)
    return round(value, 4)


def _style_scores(df: pd.DataFrame) -> Dict[str, float]:
    close = df["close"]
    returns = close.pct_change()
    latest = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50
    vol20 = float(returns.tail(20).std() or 0.0)
    vol60 = float(returns.tail(60).std() or 0.0)
    return {
        "short": round(min(1.0, max(0.0, 0.40 + vol20 * 18.0 + abs(latest / ma20 - 1.0) * 4.0)), 3),
        "swing": round(min(1.0, max(0.0, 0.45 + abs(latest / ma50 - 1.0) * 3.0 + vol60 * 8.0)), 3),
        "long": round(min(1.0, max(0.0, 0.45 + max(latest / ma200 - 1.0, 0.0) * 4.0 - vol60 * 4.0)), 3),
    }


@router.post("/trade-plan", response_model=TradePlanResponse, tags=["Trade Plan"])
async def create_trade_plan(req: TradePlanRequest) -> TradePlanResponse:
    ticker = req.ticker.strip().upper()
    df = _load_prices(ticker, req.as_of_date)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No price history found for {ticker}")
    if len(df) < 60:
        raise HTTPException(status_code=400, detail="Need at least 60 daily bars to build a trade plan")

    df = df.copy()
    df["atr14"] = _atr(df)
    df["rsi14"] = _rsi(df["close"])
    for w in (5, 20, 50, 200):
        df[f"ma{w}"] = df["close"].rolling(w).mean()

    scores = _style_scores(df)
    style = req.style if req.style != "auto" else max(scores.items(), key=lambda item: item[1])[0]
    cfg = {
        "short": ("Short-term trader", 0.10, 1.00, 1.25, 2.00, "Exit after 1-5 trading days unless momentum expands.", 5, 20),
        "swing": ("Swing trader", 0.15, 1.50, 2.00, 3.00, "Exit after 2-6 weeks or when the swing setup invalidates.", 20, 50),
        "long": ("Long-term trader", 0.25, 2.50, 4.00, 6.00, "Review monthly; exit on trend break or thesis invalidation.", 50, 200),
    }[style]
    label, entry_buf, stop_atr, target1_atr, target2_atr, time_exit, fast_w, slow_w = cfg

    latest = float(df["close"].iloc[-1])
    price_date = pd.to_datetime(df["date"].iloc[-1]).date().isoformat()
    atr = _num(df["atr14"].iloc[-1]) or latest * 0.02
    rsi = _num(df["rsi14"].iloc[-1])
    ma_fast = _num(df[f"ma{fast_w}"].iloc[-1])
    ma_slow = _num(df[f"ma{slow_w}"].iloc[-1])
    ma20 = _num(df["ma20"].iloc[-1])
    ma50 = _num(df["ma50"].iloc[-1])
    ma200 = _num(df["ma200"].iloc[-1])
    high20 = float(df["high"].tail(20).max())
    low20 = float(df["low"].tail(20).min())

    reasons: List[str] = []
    warnings: List[str] = []
    trend_up = ma_fast is not None and ma_slow is not None and latest > ma_fast > ma_slow
    trend_down = ma_fast is not None and ma_slow is not None and latest < ma_fast < ma_slow
    overbought = rsi is not None and rsi >= 70
    oversold = rsi is not None and rsi <= 30

    direction: Direction = "wait"
    strategy = f"{style} wait"
    if req.signal_action in {"buy", "sell"} and (req.signal_confidence or 0) >= 0.55:
        direction = "long" if req.signal_action == "buy" else "short"
        strategy = f"saved-model {req.signal_action} signal"
        reasons.append(req.signal_reason or "Saved model signal supports this direction.")
    elif trend_up and not overbought:
        direction, strategy = "long", f"{style} trend continuation"
        reasons.append("Price is above the active trend averages.")
    elif trend_down and not oversold:
        direction, strategy = "short", f"{style} trend continuation short"
        reasons.append("Price is below the active trend averages.")
    else:
        reasons.append("No clean directional edge is active yet.")

    if overbought:
        warnings.append("RSI is overbought; avoid chasing without confirmation.")
    if oversold:
        warnings.append("RSI is oversold; short entries have elevated squeeze risk.")

    if direction == "long":
        entry = latest + entry_buf * atr
        stop = entry - stop_atr * atr
        target1 = entry + target1_atr * atr
        target2 = entry + target2_atr * atr
        invalidation = f"Exit if price closes below {_round_price(stop)} or loses the active trend average."
    elif direction == "short":
        entry = latest - entry_buf * atr
        stop = entry + stop_atr * atr
        target1 = entry - target1_atr * atr
        target2 = entry - target2_atr * atr
        invalidation = f"Exit if price closes above {_round_price(stop)} or reclaims the active trend average."
    else:
        entry = high20 + entry_buf * atr
        stop = entry - stop_atr * atr
        target1 = entry + target1_atr * atr
        target2 = entry + target2_atr * atr
        invalidation = f"Only enter on a break above {_round_price(entry)}; cancel below {_round_price(stop)}."

    risk_amount = float(req.account_size) * (float(req.risk_percent) / 100.0)
    per_share_risk = abs(entry - stop) if entry is not None and stop is not None else 0.0
    position_size = int(risk_amount / per_share_risk) if per_share_risk > 0 else 0
    rr = abs(target1 - entry) / per_share_risk if target1 is not None and per_share_risk > 0 else None
    confidence = min(0.95, scores[style] + (0.08 if direction != "wait" else 0.0))

    return TradePlanResponse(
        ticker=ticker,
        style=style,  # type: ignore[arg-type]
        trader_type=label,
        direction=direction,
        confidence=round(float(confidence), 3),
        entry=_round_price(entry),
        stop_loss=_round_price(stop),
        take_profit_1=_round_price(target1),
        take_profit_2=_round_price(target2),
        trailing_stop=_round_price(stop_atr * atr),
        invalidation=invalidation,
        time_exit=time_exit,
        risk_reward=round(rr, 2) if rr is not None else None,
        risk_amount=round(risk_amount, 2),
        position_size=position_size,
        latest_close=_round_price(latest) or latest,
        price_date=price_date,
        strategy=strategy,
        reasons=reasons,
        warnings=warnings,
        indicators={
            "atr14": _round_price(atr),
            "rsi14": round(rsi, 2) if rsi is not None else None,
            "ma20": _round_price(ma20),
            "ma50": _round_price(ma50),
            "ma200": _round_price(ma200),
            "high20": _round_price(high20),
            "low20": _round_price(low20),
        },
        style_scores=scores,
    )
