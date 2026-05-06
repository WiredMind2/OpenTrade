from types import SimpleNamespace

from backend.services.backtrader_engine import extract_run_metrics


class _Analyzer:
    def __init__(self, analysis):
        self._analysis = analysis

    def get_analysis(self):
        return self._analysis


def _strategy_result_with_trade_analysis(analysis: dict):
    return SimpleNamespace(
        analyzers=SimpleNamespace(
            sharpe=_Analyzer({"sharperatio": 0.0}),
            drawdown=_Analyzer({"max": {"drawdown": 0.0}}),
            returns=_Analyzer({"rnorm100": 0.0}),
            trades=_Analyzer(analysis),
        ),
        trades=[],
        equity_curve=[],
        decision_markers=[],
    )


def test_extract_run_metrics_uses_closed_trades_for_win_rate():
    strategy_result = _strategy_result_with_trade_analysis(
        {
            "total": {"total": 10, "open": 7, "closed": 3},
            "won": {"total": 1},
            "lost": {"total": 2},
        }
    )
    metrics = extract_run_metrics(strategy_result, initial_capital=100_000.0, final_value=100_000.0)
    assert metrics["total_trades"] == 3
    assert metrics["win_rate"] == (1 / 3)


def test_extract_run_metrics_counts_no_closed_trades_as_zero_total_trades():
    strategy_result = _strategy_result_with_trade_analysis(
        {
            "total": {"total": 5, "open": 5, "closed": 0},
            "won": {"total": 0},
            "lost": {"total": 0},
        }
    )
    metrics = extract_run_metrics(strategy_result, initial_capital=100_000.0, final_value=100_000.0)
    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] == 0.0


def test_extract_run_metrics_does_not_count_open_trades_as_closed_when_closed_missing():
    # Backtrader can omit "closed" entirely and only report total/open. If open == total,
    # there were no closed trades and total_trades must remain 0.
    strategy_result = _strategy_result_with_trade_analysis(
        {
            "total": {"total": 5, "open": 5},
        }
    )
    metrics = extract_run_metrics(strategy_result, initial_capital=100_000.0, final_value=100_000.0)
    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] == 0.0


def test_extract_run_metrics_counts_virtual_close_trades_for_open_positions():
    class _CloseLine:
        def __init__(self, v: float):
            self._v = v

        def __getitem__(self, idx: int) -> float:
            assert idx == 0
            return self._v

    class _Data:
        def __init__(self, name: str, last_close: float):
            self._name = name
            self.close = _CloseLine(last_close)

    class _Pos:
        def __init__(self, size: float, price: float):
            self.size = size
            self.price = price

    # No closed trades in analyzer; one open position that is profitable at last close.
    s = _strategy_result_with_trade_analysis({"total": {"total": 1, "open": 1}})
    s.datas = [_Data("AAPL", 110.0)]

    def _getposition(d):
        assert getattr(d, "_name", "") == "AAPL"
        return _Pos(size=10.0, price=100.0)

    s.getposition = _getposition

    metrics = extract_run_metrics(s, initial_capital=100_000.0, final_value=100_000.0)
    assert metrics["total_trades"] == 1
    assert metrics["win_rate"] == 1.0
    assert len(metrics["trades"]) == 1
    assert metrics["trades"][0]["ticker"] == "AAPL"
    assert metrics["trades"][0]["side"] == "virtual_close"

