"""
回测主引擎 + 绩效评估 集成测试（合成数据，零网络）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _bt(**kw):
    from src.engine import Backtester
    cfg = {"backtest": {"initial_capital": 1_000_000}}
    return Backtester(config=cfg, **kw)


def test_full_pipeline_metrics(sample_data):
    from src.engine import BacktestResult, Trade
    from src.strategy.examples.ma_rsi import MaRsiStrategy

    bench = pd.Series(
        np.linspace(3000, 3300, len(sample_data["000001.SZ"])),
        index=sample_data["000001.SZ"]["date"],
    )
    res = _bt(position_size=0.4).run(
        MaRsiStrategy(fast_period=5, slow_period=20),
        "2024-01-01", "2024-12-31", data=sample_data, benchmark=bench,
    )
    assert isinstance(res, BacktestResult)
    assert len(res.equity_curve) > 2
    assert abs(res.equity_curve.iloc[0] - 1.0) < 1e-9          # 初始净值 = 1.0
    assert abs(res.total_return - (res.equity_curve.iloc[-1] - 1.0)) < 1e-9
    assert -1.0 <= res.max_drawdown <= 0.0
    assert all(isinstance(t, Trade) for t in res.trades)
    assert res.benchmark_return != 0.0                          # 提供基准


def test_metric_consistency(sample_data):
    from src.strategy.examples.ma_rsi import MaRsiStrategy

    res = _bt(position_size=0.4).run(
        MaRsiStrategy(fast_period=5, slow_period=20), "2024-01-01", "2024-12-31", data=sample_data
    )
    if res.total_trades > 0:
        wins = sum(1 for t in res.trades if t.pnl > 0)
        assert abs(res.win_rate - wins / res.total_trades) < 1e-9
    if res.max_drawdown < 0:
        assert abs(res.calmar_ratio - res.annual_return / abs(res.max_drawdown)) < 1e-6
    for t in res.trades:
        assert t.shares > 0 and t.holding_days >= 0
        assert t.entry_date is not None and t.exit_date is not None
    assert res.benchmark_return == 0.0                          # 未提供基准


def test_hold_strategy_no_trades(sample_data):
    from src.strategy.base import BaseStrategy

    class Flat(BaseStrategy):
        strategy_name = "flat"

        def generate_signals(self, d):
            return self.empty_signals(next(iter(d.values()))["date"], list(d.keys()))

    res = _bt().run(Flat(), "2024-01-01", "2024-12-31", data=sample_data)
    assert res.total_trades == 0
    assert abs(res.total_return) < 1e-9                          # 从未建仓 → 净值持平


def test_empty_data_safe():
    from src.strategy.base import BaseStrategy

    class Flat(BaseStrategy):
        strategy_name = "flat"

        def generate_signals(self, d):
            return self.empty_signals([], [])

    res = _bt().run(Flat(), "2024-01-01", "2024-02-01", data={})
    assert res.total_trades == 0 and len(res.equity_curve) == 0


def test_no_future_function_first_day_flat(sample_data):
    """首日不应有成交（无上一日信号），净值等于初始资金。"""
    from src.strategy.examples.ma_rsi import MaRsiStrategy

    res = _bt(position_size=0.4).run(
        MaRsiStrategy(fast_period=5, slow_period=20), "2024-01-01", "2024-12-31", data=sample_data
    )
    assert abs(res.equity_curve.iloc[0] * 1_000_000 - 1_000_000) < 1e-6
