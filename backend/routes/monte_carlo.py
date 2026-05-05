"""
Monte Carlo simulation endpoints for the Trading Backtester API.
"""
import uuid
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.logging_config import get_component_logger
from backend.monte_carlo import MonteCarloGenerator, aggregate_monte_carlo_results, calculate_value_at_risk, calculate_expected_shortfall
from backend.services.strategy_framework import StrategyPreflightService


router = APIRouter()
logger = get_component_logger(__file__)


class MonteCarloRequest(BaseModel):
    """Request model for Monte Carlo simulation."""
    strategy_name: str = Field(..., description="Name of the strategy to simulate")
    ticker: str = Field(..., description="Ticker symbol for the simulation")
    start_date: str = Field(..., description="Start date for simulation (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for simulation (YYYY-MM-DD)")
    initial_capital: float = Field(default=100000.0, gt=0, description="Initial capital for simulation")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    num_simulations: int = Field(default=1000, ge=100, le=10000, description="Number of Monte Carlo simulations")
    time_horizon_days: int = Field(default=252, ge=30, le=2520, description="Time horizon in trading days")


class MonteCarloResult(BaseModel):
    """Response model for Monte Carlo simulation results."""
    simulation_id: str
    strategy_name: str
    ticker: str
    num_simulations: int
    time_horizon_days: int
    aggregated_results: Dict[str, Any]
    risk_metrics: Dict[str, float]
    created_at: str


@router.post("/simulate", response_model=MonteCarloResult, tags=["Monte Carlo"])
async def run_monte_carlo_simulation(request: MonteCarloRequest):
    """Run Monte Carlo simulation for a strategy to assess risk and potential outcomes."""
    from backend.config import get_config
    from backend.main import app_state

    try:
        # Get strategy registry
        registry = app_state.get("strategy_registry")
        if not registry:
            raise HTTPException(status_code=500, detail="Strategy registry not available")

        # Validate strategy exists
        strategy = registry.get(request.strategy_name)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy '{request.strategy_name}' not found")

        # Preflight check - validate ticker and dates
        preflight_service = StrategyPreflightService()
        preflight_result = await preflight_service.check_strategy_readiness(
            strategy_name=request.strategy_name,
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date
        )

        if not preflight_result.ready:
            issues = [issue.message for issue in preflight_result.issues]
            raise HTTPException(status_code=400, detail=f"Preflight failed: {', '.join(issues)}")

        # Get historical data to estimate drift and volatility
        config = get_config()
        db_path = app_state.get("database_path") or config.database.path

        # Fetch historical prices for parameter estimation
        historical_returns = get_historical_returns(db_path, request.ticker, request.start_date, request.end_date)
        if len(historical_returns) < 30:
            raise HTTPException(status_code=400, detail="Insufficient historical data for Monte Carlo simulation")

        # Estimate drift and volatility from historical data
        drift = np.mean(historical_returns)
        volatility = np.std(historical_returns)

        # Run Monte Carlo simulations
        generator = MonteCarloGenerator()
        simulation_results = []

        # Get initial price
        initial_price = get_initial_price(db_path, request.ticker, request.start_date)

        for i in range(request.num_simulations):
            # Generate price path
            price_path = generator.generate_price_path(
                initial_price=initial_price,
                drift=drift,
                volatility=volatility,
                time_horizon=request.time_horizon_days / 252,  # Convert to years
                num_steps=request.time_horizon_days
            )

            # Calculate returns from price path
            final_value = price_path[-1]
            total_return = (final_value - initial_price) / initial_price

            # For now, simulate strategy performance as a simplified model
            # In a full implementation, this would run the actual backtest on simulated data
            strategy_return = simulate_strategy_performance(total_return, request.strategy_params)

            simulation_results.append({
                'final_value': request.initial_capital * (1 + strategy_return),
                'total_return': strategy_return,
                'simulation_id': i
            })

        # Aggregate results
        aggregated = aggregate_monte_carlo_results(simulation_results)

        # Calculate additional risk metrics
        returns_list = [r['total_return'] for r in simulation_results]
        var_95 = calculate_value_at_risk(returns_list, 0.95)
        es_95 = calculate_expected_shortfall(returns_list, 0.95)

        risk_metrics = {
            'value_at_risk_95': var_95,
            'expected_shortfall_95': es_95,
            'volatility': np.std(returns_list),
            'probability_positive_return': np.mean([1 if r > 0 else 0 for r in returns_list])
        }

        # Generate unique simulation ID
        simulation_id = f"mc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        result = MonteCarloResult(
            simulation_id=simulation_id,
            strategy_name=request.strategy_name,
            ticker=request.ticker,
            num_simulations=request.num_simulations,
            time_horizon_days=request.time_horizon_days,
            aggregated_results=aggregated,
            risk_metrics=risk_metrics,
            created_at=datetime.utcnow().isoformat()
        )

        logger.info(f"Completed Monte Carlo simulation: {simulation_id} for {request.strategy_name}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


def get_historical_returns(db_path: str, ticker: str, start_date: str, end_date: str) -> List[float]:
    """Fetch historical daily returns for drift/volatility estimation."""
    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT close
            FROM price_daily
            WHERE ticker = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """
        cursor = conn.execute(query, [ticker, start_date, end_date])
        rows = cursor.fetchall()

        if not rows:
            return []

        # Calculate daily returns
        prices = [row[0] for row in rows]
        returns = np.diff(prices) / prices[:-1]

        return returns.tolist()
    finally:
        conn.close()


def get_initial_price(db_path: str, ticker: str, start_date: str) -> float:
    """Get the closing price on the start date."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT close
            FROM price_daily
            WHERE ticker = ?
              AND date >= ?
            ORDER BY date ASC
            LIMIT 1
        """
        cursor = conn.execute(query, [ticker, start_date])
        row = cursor.fetchone()
        return row[0] if row else 100.0  # Default fallback
    finally:
        conn.close()


def simulate_strategy_performance(market_return: float, strategy_params: Dict[str, Any]) -> float:
    """
    Simplified strategy performance simulation.
    In a full implementation, this would run the actual strategy logic on simulated price data.
    """
    # For now, apply a simple risk-adjusted return model
    # Strategy might have different beta, alpha, etc.
    beta = strategy_params.get('beta', 1.0)  # Market sensitivity
    alpha = strategy_params.get('alpha', 0.0)  # Excess return

    # Simulate strategy return with some noise
    strategy_return = alpha + beta * market_return + np.random.normal(0, 0.02)

    return strategy_return