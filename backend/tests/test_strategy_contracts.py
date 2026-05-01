from datetime import datetime

from backend.strategies.moving_average import MovingAverageStrategy


def test_forecast_contract_enforces_bounds():
    strategy = MovingAverageStrategy()
    output = strategy.forecast(
        parameters={},
        symbol="AAPL",
        as_of=datetime(2026, 5, 1),
        current_price=100.0,
        horizon_days=5,
    )
    assert output.symbol == "AAPL"
    assert output.horizon_days == 5
    assert 0.0 <= output.confidence <= 1.0


def test_generate_target_allocations_contract_bounds():
    strategy = MovingAverageStrategy()
    allocations = strategy.generate_target_allocations(
        parameters={"max_position_pct": 0.1, "prediction_threshold": 0.001},
        symbols=["AAPL"],
        as_of=datetime(2026, 5, 1),
        current_prices={"AAPL": 100.0},
    )
    for allocation in allocations:
        assert -0.1 <= allocation.target_pct <= 0.1
        assert 0.0 <= allocation.confidence <= 1.0
