"""Portfolio 资金 / T+1 口径单元测试（零网络）。"""

from __future__ import annotations

from datetime import date


def _buy_then_sell(pf):
    """T 日买入 1000@10，次日解冻后卖出 1000@11。返回卖出后的 pf。"""
    pf.buy("000001.SZ", 1000, 10.0, 5.0, date(2024, 1, 2))
    pf.settle_new_day()                                  # 解冻持仓 T+1
    pf.sell("000001.SZ", 1000, 11.0, 5.0, date(2024, 1, 3))
    return pf


def test_sell_proceeds_available_same_day_default():
    """P0-2 默认(t1_cash_freeze=False)：卖出回款当日即计入可用资金，可继续买入（A 股真实规则）。"""
    from src.engine import Portfolio

    pf = _buy_then_sell(Portfolio(100_000))             # 默认 False
    assert pf.frozen_cash == 0.0
    assert pf.available_cash == pf.cash                 # 回款全部可用
    # 回款已可用于当日再买入（不抛资金不足）
    pf.buy("600519.SH", 100, 50.0, 5.0, date(2024, 1, 3))
    assert pf.has_position("600519.SH")


def test_sell_proceeds_frozen_when_enabled():
    """保守口径(t1_cash_freeze=True)：卖出回款当日冻结，不计入可用资金。"""
    from src.engine import Portfolio

    pf = _buy_then_sell(Portfolio(100_000, t1_cash_freeze=True))
    assert pf.frozen_cash > 0.0
    assert pf.available_cash < pf.cash                  # 回款被冻结，可用减少


def test_share_t1_freeze_independent_of_cash_switch():
    """无论资金开关如何，'卖出当日买入的股票'始终受 T+1 约束（Position.frozen）。"""
    from src.engine import Portfolio

    for freeze in (False, True):
        pf = Portfolio(100_000, t1_cash_freeze=freeze)
        pf.buy("000001.SZ", 1000, 10.0, 5.0, date(2024, 1, 2))
        pos = pf.get_position("000001.SZ")
        assert pos.available == 0                       # 当日买入不可卖
