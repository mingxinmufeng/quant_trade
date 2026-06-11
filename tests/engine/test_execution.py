"""
撮合引擎单元测试：滑点/tick、涨跌停一字板、成交量上限、T+1、费用、除权探测。
"""

from __future__ import annotations

from datetime import date

import pytest


def _bar(**kw):
    base = {"date": date(2024, 1, 3), "open": 10.0, "high": 10.6, "low": 9.8,
            "close": 10.4, "volume": 1_000_000, "limit_up": 11.0, "limit_down": 9.0,
            "is_suspended": False}
    base.update(kw)
    return base


def _eng(**kw):
    from src.engine import ExecutionEngine
    return ExecutionEngine(**kw)


def _order(direction, shares, code="000001.SZ"):
    from src.engine import Order
    return Order(code=code, direction=direction, shares=shares)


def _signal():
    from src.strategy.base import Signal
    return int(Signal.BUY), int(Signal.SELL)


def test_slippage_percent_tick():
    eng = _eng(slippage_type="percent", percent_rate=0.001, tick_size=0.01, min_ticks=1)
    assert abs(eng.execution_price(_bar(), True) - 10.01) < 1e-9
    assert abs(eng.execution_price(_bar(), False) - 9.99) < 1e-9


def test_slippage_min_ticks_floor():
    eng = _eng(slippage_type="percent", percent_rate=1e-6, tick_size=0.01, min_ticks=1)
    assert abs(eng.execution_price(_bar(), True) - 10.01) < 1e-9


def test_slippage_fixed_and_open_gap():
    f = _eng(slippage_type="fixed", fixed_amount=0.05, tick_size=0.01)
    assert abs(f.execution_price(_bar(), True) - 10.05) < 1e-9
    g = _eng(slippage_type="open_gap")
    assert abs(g.execution_price(_bar(), True) - 10.0) < 1e-9


def test_buy_t1_freeze_then_sell_next_day():
    from src.engine import OrderStatus, Portfolio
    BUY, SELL = _signal()
    eng = _eng(slippage_type="open_gap", min_order_amount=0)
    pf = Portfolio(1_000_000)
    ob = _order(BUY, 5000)
    eng.match(ob, _bar(), pf)
    assert ob.status == OrderStatus.FILLED and pf.get_position("000001.SZ").frozen == 5000
    # same day cannot sell
    os = _order(SELL, 5000)
    eng.match(os, _bar(), pf)
    assert os.status == OrderStatus.FAILED
    # next day ok
    pf.settle_new_day()
    os2 = _order(SELL, 5000)
    eng.match(os2, _bar(open=11.0, date=date(2024, 1, 4)), pf)
    assert os2.status == OrderStatus.FILLED and not pf.has_position("000001.SZ")


def test_volume_cap_partial():
    from src.engine import OrderStatus, Portfolio
    BUY, _ = _signal()
    eng = _eng(slippage_type="open_gap", volume_pct_limit=0.0001)  # cap=100
    o = _order(BUY, 5000)
    eng.match(o, _bar(), Portfolio(1_000_000))
    assert o.status == OrderStatus.PARTIAL and o.filled_shares == 100


def test_one_word_boards():
    from src.engine import OrderStatus, Portfolio
    BUY, SELL = _signal()
    eng = _eng(slippage_type="open_gap")
    ob = _order(BUY, 1000)
    eng.match(ob, _bar(open=11.0, high=11.0, low=11.0, close=11.0), Portfolio(1_000_000))
    assert ob.status == OrderStatus.FAILED and "涨停" in ob.reason

    pf = Portfolio(1_000_000)
    eng.match(_order(BUY, 1000, "Y"), _bar(code=None), pf)  # buy Y
    pf.settle_new_day()
    osd = _order(SELL, 1000, "Y")
    eng.match(osd, _bar(open=9.0, high=9.0, low=9.0, close=9.0, date=date(2024, 1, 4)), pf)
    assert osd.status == OrderStatus.FAILED and "跌停" in osd.reason


def test_suspended_fails():
    from src.engine import OrderStatus, Portfolio
    BUY, _ = _signal()
    o = _order(BUY, 1000)
    _eng().match(o, _bar(is_suspended=True), Portfolio(1_000_000))
    assert o.status == OrderStatus.FAILED


def test_min_order_amount_ignored():
    from src.engine import OrderStatus, Portfolio
    BUY, _ = _signal()
    eng = _eng(slippage_type="open_gap", min_order_amount=1e9)
    o = _order(BUY, 1000)
    eng.match(o, _bar(), Portfolio(1_000_000))
    assert o.status == OrderStatus.IGNORED


def test_commission_internal():
    eng = _eng()
    assert abs(eng.commission(100_000, True) - 100_000 * 0.00025) < 1e-9
    assert abs(eng.commission(100_000, False) - 100_000 * (0.00025 + 0.0005)) < 1e-9
    assert abs(eng.commission(1000, True) - 5.0) < 1e-9


def test_detect_ex_factor_ratio():
    from src.engine import ExecutionEngine
    assert abs(ExecutionEngine.detect_ex_factor_ratio(1.0, 2.0) - 2.0) < 1e-9
    assert ExecutionEngine.detect_ex_factor_ratio(1.0, 1.0005) == 1.0


def test_from_config():
    from src.engine import ExecutionEngine
    cfg = {"execution": {"volume_pct_limit": 0.1, "min_order_amount": 1000,
                         "slippage": {"type": "percent", "percent_rate": 0.002, "tick_size": 0.01, "min_ticks": 1}},
           "risk": {"commission_rate": 0.0003, "stamp_duty": 0.001, "min_commission": 5.0}}
    ec = ExecutionEngine.from_config(cfg)
    assert ec.percent_rate == 0.002 and ec.commission_rate == 0.0003
