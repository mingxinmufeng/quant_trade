"""Portfolio 资金 / T+1 口径单元测试（零网络）。"""

from __future__ import annotations

from datetime import date

import pytest


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


def test_buy_insufficient_cash_raises():
    """买入超过可用资金 → ValueError（撮合层应在调用前裁减，这里是账本最后一道防线）。"""
    from src.engine import Portfolio

    pf = Portfolio(10_000)
    with pytest.raises(ValueError):
        pf.buy("000001.SZ", 10_000, 10.0, 5.0, date(2024, 1, 2))   # 需 ~10万 > 1万


def test_sell_exceeds_available_raises():
    """卖出超过可卖股数（T+1 冻结未解冻）→ ValueError。"""
    from src.engine import Portfolio

    pf = Portfolio(100_000)
    pf.buy("000001.SZ", 1000, 10.0, 5.0, date(2024, 1, 2))         # 当日冻结
    with pytest.raises(ValueError):
        pf.sell("000001.SZ", 1000, 11.0, 5.0, date(2024, 1, 2))    # 未 settle，available=0


def test_apply_split_keeps_zero_lot_and_total_cost():
    """送转默认保留零股、总成本不变、avg_cost 等比下调（[[portfolio 除权记账口径]]）。"""
    from src.engine import Portfolio

    pf = Portfolio(100_000)
    pf.buy("000001.SZ", 1000, 10.0, 0.0, date(2024, 1, 2))         # 成本基 10000
    pos = pf.get_position("000001.SZ")
    cost_before = pos.cost_value
    pf.apply_split("000001.SZ", 1.5)                               # 10 送 5 → ×1.5
    assert pos.shares == 1500                                      # 保留零股（非整手取整）
    assert abs(pos.cost_value - cost_before) < 1e-6                # 总成本不变
    assert abs(pos.avg_cost - 10000 / 1500) < 1e-9                # 等比下调


def test_cash_dividend_counts_as_realized_not_cost_offset():
    """现金分红计入当期已实现收益、不冲减成本基（避免超额分红虚高卖出 pnl）。"""
    from src.engine import Portfolio

    pf = Portfolio(100_000)
    pf.buy("000001.SZ", 1000, 10.0, 0.0, date(2024, 1, 2))
    pos = pf.get_position("000001.SZ")
    cash_before, avg_before = pf.cash, pos.avg_cost
    amt = pf.add_cash_dividend("000001.SZ", 0.5)                   # 每股税后 0.5
    assert abs(amt - 500.0) < 1e-9
    assert abs(pf.cash - (cash_before + 500.0)) < 1e-9
    assert abs(pf.realized_pnl - 500.0) < 1e-9                     # 计入已实现
    assert pos.avg_cost == avg_before                             # 成本基不变


def test_taxable_cash_dividend_withheld_on_sell_by_holding_days():
    """gbbq 税前分红先全额入账，卖出时按持股天数扣红利税。"""
    from src.engine import Portfolio

    pf = Portfolio(100_000)
    pf.buy("000001.SZ", 1000, 10.0, 0.0, date(2024, 1, 2))
    pf.settle_new_day()
    pf.add_taxable_cash_dividend("000001.SZ", 0.5, date(2024, 1, 3))
    cash_after_dividend = pf.cash

    pnl = pf.sell("000001.SZ", 1000, 11.0, 0.0, date(2024, 1, 20))

    assert cash_after_dividend == 90_500.0
    assert pnl == 900.0  # 卖出价差 1000 - 红利税 500*20%
    assert pf.cash == 101_400.0
    assert pf.realized_pnl == 1_400.0  # 税前分红 500 + 卖出 pnl 900
