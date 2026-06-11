"""
交易日历单元测试 —— 离线（从预写缓存加载，零网络）。
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def test_is_trading_day(offline_calendar):
    cal = offline_calendar
    assert cal.is_trading_day(date(2024, 1, 2)) is True       # 周二
    assert cal.is_trading_day(date(2024, 1, 6)) is False      # 周六（合成集不含）


def test_next_prev_trading_day(offline_calendar):
    cal = offline_calendar
    # 2024-01-05 周五 → 下一交易日 2024-01-08 周一
    assert cal.next_trading_day(date(2024, 1, 5)) == date(2024, 1, 8)
    assert cal.previous_trading_day(date(2024, 1, 8)) == date(2024, 1, 5)
    # 跨多日
    assert cal.next_trading_day(date(2024, 1, 1), n=1) == date(2024, 1, 2)


def test_get_trading_days_range(offline_calendar):
    cal = offline_calendar
    days = cal.get_trading_days(date(2024, 1, 1), date(2024, 1, 7))
    # 周一~周日中工作日：1(周一?) 2024-01-01 是周一 → 含 1,2,3,4,5
    assert all(isinstance(d, date) for d in days)
    assert date(2024, 1, 6) not in days and date(2024, 1, 7) not in days
    assert days == sorted(days)


def test_get_trading_days_invalid(offline_calendar):
    import pytest

    with pytest.raises(ValueError):
        offline_calendar.get_trading_days(date(2024, 2, 1), date(2024, 1, 1))


def test_trading_days_property_sorted_unique(offline_calendar):
    td = offline_calendar.trading_days
    assert isinstance(td, pd.DatetimeIndex)
    assert td.is_monotonic_increasing and td.is_unique
