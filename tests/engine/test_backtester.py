"""
回测主引擎 + 绩效评估 集成测试（合成数据，零网络）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def test_risk_manager_position_cap_wired(sample_data):
    """注入 RiskManager 时，单票上限应裁减建仓市值（验证风控已接入回测主循环）。"""
    from src.engine import Backtester
    from src.risk import RiskManager
    from src.strategy.examples.ma_rsi import MaRsiStrategy

    cfg = {"backtest": {"initial_capital": 1_000_000}}
    strat = lambda: MaRsiStrategy(fast_period=5, slow_period=20)  # noqa: E731
    cap_value = 1_000_000 * 0.10 * 1.2  # 10% 上限 + 20% 容差（建仓时点 total 波动）

    # 单票上限 10%，关闭止损/熔断以隔离仓位裁减
    rm = RiskManager(max_single_position=0.10, daily_stop_loss=1.0, total_drawdown_stop=1.0)
    res = Backtester(config=cfg, risk_manager=rm, position_size=0.5).run(
        strat(), "2024-01-01", "2024-12-31", data=sample_data
    )
    assert res.trades, "应至少有若干笔交易用于验证"
    for t in res.trades:
        assert t.entry_price * t.shares <= cap_value, "风控未生效：建仓市值超出单票上限"

    # 对照：不接风控时 position_size=0.5 会建出超过 10% 的仓位
    res2 = Backtester(config=cfg, position_size=0.5).run(
        strat(), "2024-01-01", "2024-12-31", data=sample_data
    )
    assert any(t.entry_price * t.shares > cap_value for t in res2.trades), \
        "对照组应出现超过 10% 的建仓（否则测试无区分力）"
