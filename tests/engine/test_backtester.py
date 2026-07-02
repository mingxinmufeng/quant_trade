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


def test_delisted_position_force_liquidated():
    """P0-3：某股行情中途终止（退市）且被持有 → 数据结束后按最后市价强制清仓。

    验证不再"持仓幻值永久计入净值"：退市股应产生一笔完成交易、回测末日不再持有它，
    而正常持有到末日的股票仍保留仓位（不被误清）。
    """
    from src.engine import Backtester
    from src.strategy.base import BaseStrategy, Signal

    dates_full = pd.date_range("2024-01-02", periods=40, freq="B")

    def _mk(dates, base):
        close = pd.Series(np.linspace(base, base + len(dates) * 0.1, len(dates)))
        return pd.DataFrame({
            "date": dates, "open": close, "high": close + 0.2, "low": close - 0.2,
            "close": close, "volume": 1e6, "is_suspended": False,
        })

    data = {
        "000001.SZ": _mk(dates_full, 10.0),       # 全程有行情
        "000002.SZ": _mk(dates_full[:20], 20.0),  # 第 20 根后"退市"（行情终止）
    }

    class BuyHold(BaseStrategy):
        strategy_name = "buyhold"

        def generate_signals(self, d):
            idx = pd.to_datetime(d["000001.SZ"]["date"])
            sig = self.empty_signals(idx, list(d.keys()))
            sig.iloc[0] = int(Signal.BUY)   # 次日撮合，两只都建仓后持有
            return self.validate_signals(sig)

    cfg = {"backtest": {"initial_capital": 1_000_000}}
    res = Backtester(config=cfg, position_size=0.4).run(
        BuyHold(), "2024-01-02", dates_full[-1].strftime("%Y-%m-%d"), data=data
    )

    assert "000002.SZ" in {t.code for t in res.trades}, "退市股应被强制清仓并入账为完成交易"
    last_pos = res.daily_positions.iloc[-1]
    assert last_pos.get("000002.SZ", 0) == 0, "退市股回测末日不应仍被持有（幻值）"
    assert last_pos.get("000001.SZ", 0) > 0, "正常持有到末日的股票不应被误清"


def test_none_adjust_corporate_action_uses_cum_factor():
    """不复权 raw 价格跨除权日时，应按 cum_factor 调整股数，避免把除权跳空算亏损。"""
    from src.engine import Backtester
    from src.strategy.base import BaseStrategy, Signal

    dates = pd.bdate_range("2024-01-02", periods=4)
    data = {
        "000001.SZ": pd.DataFrame(
            {
                "date": dates,
                "open": [10.0, 10.0, 5.0, 5.0],
                "high": [10.0, 10.0, 5.0, 5.0],
                "low": [10.0, 10.0, 5.0, 5.0],
                "close": [10.0, 10.0, 5.0, 5.0],
                "volume": 1e6,
                "is_suspended": False,
                "adj_factor": 1.0,                    # none 口径下价格乘数恒为 1
                "cum_factor": [1.0, 1.0, 2.0, 2.0],   # 除权真实触发器
            }
        )
    }

    class BuyHold(BaseStrategy):
        strategy_name = "buyhold"

        def generate_signals(self, d):
            sig = self.empty_signals(d["000001.SZ"]["date"], ["000001.SZ"])
            sig.iloc[0, 0] = int(Signal.BUY)
            return self.validate_signals(sig)

    res = Backtester(
        config={"backtest": {"initial_capital": 1_000_000}},
        position_size=0.5,
        apply_corporate_actions=True,
    ).run(BuyHold(), dates[0].date(), dates[-1].date(), data=data)

    pos = res.daily_positions["000001.SZ"]
    assert pos.iloc[1] == 50_000
    assert pos.iloc[2] == 100_000
    assert res.equity_curve.iloc[2] > 0.99


def test_apply_corporate_actions_rejects_adjusted_data_without_cum_factor():
    """apply_corporate_actions=True 不应接受疑似 hfq/qfq 数据，避免除权双重计提。"""
    from src.engine import Backtester
    from src.strategy.base import BaseStrategy

    dates = pd.bdate_range("2024-01-02", periods=3)
    data = {
        "000001.SZ": pd.DataFrame(
            {
                "date": dates,
                "open": [10.0, 10.0, 10.0],
                "high": [10.0, 10.0, 10.0],
                "low": [10.0, 10.0, 10.0],
                "close": [10.0, 10.0, 10.0],
                "volume": 1e6,
                "is_suspended": False,
                "adj_factor": [1.0, 2.0, 2.0],
            }
        )
    }

    class Flat(BaseStrategy):
        strategy_name = "flat"

        def generate_signals(self, d):
            return self.empty_signals(d["000001.SZ"]["date"], ["000001.SZ"])

    with pytest.raises(ValueError, match="疑似 hfq/qfq 数据"):
        Backtester(
            config={"backtest": {"initial_capital": 1_000_000}},
            apply_corporate_actions=True,
        ).run(Flat(), dates[0].date(), dates[-1].date(), data=data)


def test_run_uses_signal_data_for_strategy_and_trade_data_for_execution():
    """双数据口径：策略看复权价，成交股数和成交价使用 raw 价。"""
    from src.engine import Backtester
    from src.strategy.base import BaseStrategy, Signal

    dates = pd.bdate_range("2024-01-02", periods=3)
    signal_data = {
        "000001.SZ": pd.DataFrame(
            {
                "date": dates,
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0],
                "volume": 1e6,
                "is_suspended": False,
                "adj_factor": 10.0,
            }
        )
    }
    trade_data = {
        "000001.SZ": pd.DataFrame(
            {
                "date": dates,
                "open": [10.0, 10.0, 10.0],
                "high": [10.0, 10.0, 10.0],
                "low": [10.0, 10.0, 10.0],
                "close": [10.0, 10.0, 10.0],
                "volume": 1e6,
                "is_suspended": False,
                "adj_factor": 1.0,
                "cum_factor": 1.0,
            }
        )
    }

    class BuyIfAdjusted(BaseStrategy):
        strategy_name = "buy_if_adjusted"

        def generate_signals(self, d):
            assert float(d["000001.SZ"]["close"].iloc[0]) == 100.0
            sig = self.empty_signals(d["000001.SZ"]["date"], ["000001.SZ"])
            sig.iloc[0, 0] = int(Signal.BUY)
            return self.validate_signals(sig)

    res = Backtester(
        config={"backtest": {"initial_capital": 1_000_000}},
        position_size=0.5,
        apply_corporate_actions=True,
    ).run(
        BuyIfAdjusted(),
        dates[0].date(),
        dates[-1].date(),
        data=signal_data,
        trade_data=trade_data,
    )

    assert res.daily_positions["000001.SZ"].iloc[1] == 50_000
    assert abs(res.equity_curve.iloc[1] - 1.0) < 0.001


def test_point_in_time_signal_adjust_anchors_to_signal_day():
    """历史时点前复权：每个信号日用当日 cum_factor 作锚点，不看未来因子。"""
    from src.engine import Backtester
    from src.strategy.base import BaseStrategy

    dates = pd.bdate_range("2024-01-02", periods=3)
    data = {
        "000001.SZ": pd.DataFrame(
            {
                "date": dates,
                "open": [10.0, 5.0, 5.0],
                "high": [10.0, 5.0, 5.0],
                "low": [10.0, 5.0, 5.0],
                "close": [10.0, 5.0, 5.0],
                "volume": 1e6,
                "is_suspended": False,
                "adj_factor": 1.0,
                "cum_factor": [1.0, 2.0, 4.0],
            }
        )
    }
    seen: list[tuple[float, float]] = []

    class Inspect(BaseStrategy):
        strategy_name = "inspect"

        def generate_signals(self, d):
            close = d["000001.SZ"]["close"].tolist()
            seen.append((close[0], close[-1]))
            return self.empty_signals(d["000001.SZ"]["date"], ["000001.SZ"])

    Backtester(config={"backtest": {"initial_capital": 1_000_000}}).run(
        Inspect(),
        dates[0].date(),
        dates[-1].date(),
        data=data,
        trade_data=data,
        point_in_time_signal_adjust=True,
    )

    assert seen[0] == (10.0, 10.0)  # D0 锚点=1，不能被未来 cum_factor=4 缩放
    assert seen[1] == (5.0, 5.0)    # D1 锚点=2，历史 D0 前复权到 5
    assert seen[2] == (2.5, 5.0)    # D2 锚点=4，历史 D0 前复权到 2.5


def test_point_in_time_signal_adjust_batches_by_corporate_action_segment():
    """锚点不变的连续交易日应合并成一次策略调用，且不泄露分段之外的未来行。"""
    from src.engine import Backtester
    from src.strategy.base import BaseStrategy

    dates = pd.bdate_range("2024-01-02", periods=5)
    data = {
        "000001.SZ": pd.DataFrame(
            {
                "date": dates,
                "open": [10.0, 10.0, 10.0, 5.0, 5.0],
                "high": [10.0, 10.0, 10.0, 5.0, 5.0],
                "low": [10.0, 10.0, 10.0, 5.0, 5.0],
                "close": [10.0, 10.0, 10.0, 5.0, 5.0],
                "volume": 1e6,
                "is_suspended": False,
                "adj_factor": 1.0,
                # 前 3 天锚点不变 (1.0)，第 4/5 天除权后锚点变为 2.0：应合并为 2 段/2 次调用。
                "cum_factor": [1.0, 1.0, 1.0, 2.0, 2.0],
            }
        )
    }
    calls: list[int] = []  # 每次调用时看到的行数（用于验证未泄露分段外的未来行）

    class Inspect(BaseStrategy):
        strategy_name = "inspect"

        def generate_signals(self, d):
            calls.append(len(d["000001.SZ"]))
            return self.empty_signals(d["000001.SZ"]["date"], ["000001.SZ"])

    Backtester(config={"backtest": {"initial_capital": 1_000_000}}).run(
        Inspect(),
        dates[0].date(),
        dates[-1].date(),
        data=data,
        trade_data=data,
        point_in_time_signal_adjust=True,
    )

    assert calls == [3, 5], "应合并成 2 次调用（前 3 天一段、后 2 天一段），而非逐日 5 次"


def test_benchmark_beta_date_aligned_partial_coverage():
    """P1-1：基准不覆盖回测起点时，beta 须按日期对齐而非按位置。

    构造 strat 与 benchmark 在重叠日（D3/D4/D5）日收益完全相同 → 正确对齐时 beta==1；
    旧的按位置拼接会把 strat 的 D1/D2/D3 与 bench 的 D3/D4/D5 错配，得到 beta≠1。
    """
    from src.engine import Backtester

    dates = pd.bdate_range("2024-01-01", periods=6)
    strat_rets = [0.05, -0.02, 0.10, -0.03, 0.04]   # D1..D5
    eq = [1.0]
    for r in strat_rets:
        eq.append(eq[-1] * (1 + r))
    equity_curve = pd.Series(eq, index=dates)
    strat_returns = equity_curve.pct_change().dropna()

    # benchmark 仅覆盖后 4 日（缺 D0/D1）；重叠日 D3/D4/D5 收益 = strat 的 [0.10,-0.03,0.04]
    bench = pd.Series(
        [100.0, 110.0, 110.0 * 0.97, 110.0 * 0.97 * 1.04], index=dates[2:6]
    )
    bt = Backtester(config={"backtest": {"initial_capital": 1_000_000}})
    _, _, beta = bt._benchmark_metrics(equity_curve, strat_returns, 0.0, bench)
    assert abs(beta - 1.0) < 1e-6, f"beta 应≈1（日期对齐），实得 {beta}"


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
