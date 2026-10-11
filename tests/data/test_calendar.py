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


# ------------------------------------------------------------
# 回源双源容灾（tushare 主 → akshare 兜底），mock 两源、零网络
# ------------------------------------------------------------

def _mk_cal(tmp_path):
    from src.data.trading_calendar import TradingCalendar

    return TradingCalendar(store_path=tmp_path, auto_load=False)


def test_fetch_remote_prefers_tushare(tmp_path, monkeypatch):
    cal = _mk_cal(tmp_path)
    called = []
    monkeypatch.setattr(cal, "_fetch_tushare",
                        lambda: called.append("ts") or pd.Series(["20240102", "20240103"]))
    monkeypatch.setattr(cal, "_fetch_akshare",
                        lambda: called.append("ak") or pd.Series(["20240102"]))
    s = cal._fetch_remote()
    assert called == ["ts"]  # 主源成功即返回，不触碰兜底源
    assert len(s) == 2


def test_fetch_remote_falls_back_to_akshare(tmp_path, monkeypatch):
    cal = _mk_cal(tmp_path)
    monkeypatch.setattr(cal, "_fetch_tushare",
                        lambda: (_ for _ in ()).throw(RuntimeError("未配置 TUSHARE_TOKEN")))
    monkeypatch.setattr(cal, "_fetch_akshare", lambda: pd.Series(["20240102", "20240103"]))
    s = cal._fetch_remote()
    assert list(s) == ["20240102", "20240103"]


def test_fetch_remote_all_fail_raises(tmp_path, monkeypatch):
    import pytest

    cal = _mk_cal(tmp_path)
    boom = lambda: (_ for _ in ()).throw(RuntimeError("x"))  # noqa: E731
    monkeypatch.setattr(cal, "_fetch_tushare", boom)
    monkeypatch.setattr(cal, "_fetch_akshare", boom)
    from a_share_quant_pro.utils import helpers

    monkeypatch.setattr(helpers.time, "sleep", lambda s: None)  # retry 退避不真等
    # 空返回也算失败
    with pytest.raises(RuntimeError, match="全部日历源失败"):
        cal._fetch_remote()
