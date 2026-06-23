"""分钟级回测引擎 MinuteBacktester 集成测试（合成数据，零网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import BaseStrategy, Signal


def _minute_data(code="000001.SZ"):
    """两个交易日、每日 4 根 1 分钟 bar 的合成 hfq 数据。"""
    frames = []
    for day, base in [("2024-01-02", 10.0), ("2024-01-03", 11.0)]:
        ts = pd.to_datetime([f"{day} 09:31", f"{day} 09:32", f"{day} 14:59", f"{day} 15:00"])
        px = base + np.arange(4) * 0.1
        frames.append(pd.DataFrame({
            "datetime": ts, "code": code,
            "open": px, "high": px + 0.05, "low": px - 0.05, "close": px,
            "volume": 1e8, "limit_up": base + 5, "limit_down": base - 5,
            "is_suspended": False, "adj_factor": 1.0,
        }))
    return pd.concat(frames, ignore_index=True)


class _BuyThenSell(BaseStrategy):
    """bar0 买入（次日 T+1 冻结）、倒数第 2 根 bar 卖出（次日已解冻）→ 产生完成交易。"""

    strategy_name = "buy_then_sell"

    def generate_signals(self, data):
        d = next(iter(data.values()))
        sig = self.empty_signals(pd.to_datetime(d["datetime"]), list(data.keys()))
        sig.iloc[0] = int(Signal.BUY)
        sig.iloc[-2] = int(Signal.SELL)
        return self.validate_signals(sig)


def test_minute_risk_position_cap_wired():
    """P1-2：注入 RiskManager 时，分钟引擎应对建仓施加单票上限（验证风控已接入 bar 循环）。"""
    from src.engine import MinuteBacktester
    from src.risk import RiskManager

    data = {"000001.SZ": _minute_data()}
    cap_value = 1_000_000 * 0.10 * 1.2  # 10% 上限 + 容差

    # 单票上限 10%，关闭止损/熔断以隔离仓位裁减
    rm = RiskManager(max_single_position=0.10, daily_stop_loss=1.0, total_drawdown_stop=1.0)
    res = MinuteBacktester(position_size=0.5, risk_manager=rm).run(
        _BuyThenSell(), "2024-01-02", "2024-01-03", data=data
    )
    assert res.trades, "应至少有一笔完成交易用于验证"
    for t in res.trades:
        assert t.entry_price * t.shares <= cap_value, "分钟风控未生效：建仓市值超出单票上限"

    # 对照：不接风控时 position_size=0.5 会建出远超 10% 的仓位
    res2 = MinuteBacktester(position_size=0.5).run(
        _BuyThenSell(), "2024-01-02", "2024-01-03", data=data
    )
    assert any(t.entry_price * t.shares > cap_value for t in res2.trades), \
        "对照组应出现超过 10% 的建仓（否则测试无区分力）"


def test_minute_risk_drawdown_halt_blocks_buy():
    """注入低熔断线 + 强制起点回撤 → 新开仓被拦截（check_order 已接入）。"""
    from src.engine import MinuteBacktester
    from src.risk import RiskManager

    data = {"000001.SZ": _minute_data()}
    # 熔断线 0%：reset 后任何非正回撤即触发，buy 应全被拦截 → 无任何交易
    rm = RiskManager(total_drawdown_stop=0.0)
    res = MinuteBacktester(position_size=0.5, risk_manager=rm).run(
        _BuyThenSell(), "2024-01-02", "2024-01-03", data=data
    )
    assert res.total_trades == 0, "熔断后不应有任何新开仓成交"


def test_minute_apply_corporate_actions_raises():
    """P1-2 / C2：apply_corporate_actions=True 应明确报错（旧版静默忽略产出错误结果）。"""
    from src.engine import MinuteBacktester

    with pytest.raises(NotImplementedError):
        MinuteBacktester(apply_corporate_actions=True)
