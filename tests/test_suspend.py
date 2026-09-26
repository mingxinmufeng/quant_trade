"""
停牌名单 Provider + fetcher 停牌判定单元测试（无网络）
======================================================

覆盖：
  - ``src.data.suspend.SuspendProvider``：代码归一、磁盘/内存缓存、lookback 限频、
    东财主源 → tushare 兜底的容灾切换。
  - ``DataFetcher``：原始层只落成交行（``_normalize_daily_raw``）；加工层按停牌名单区分
    suspend / missing 并延伸确认停牌尾部（``_build_daily_meta``）；增量游标跳过确认停牌日。

所有网络调用均通过 monkeypatch 替换，测试零网络。

运行：
    python -m pytest tests/test_suspend.py -q
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.suspend import SuspendProvider, _norm_code  # noqa: E402

# ============================================================
# _norm_code
# ============================================================

def test_norm_code_variants():
    assert _norm_code("600766") == "600766.SH"
    assert _norm_code("000001") == "000001.SZ"
    assert _norm_code("000001.SZ") == "000001.SZ"
    assert _norm_code("600766.SH") == "600766.SH"
    assert _norm_code("830799") == "830799.BJ"
    # 数字补零（个别接口丢前导 0）
    assert _norm_code(2) == "000002.SZ"
    # 非法/空
    assert _norm_code(None) is None
    assert _norm_code("") is None
    assert _norm_code("nan") is None


# ============================================================
# 缓存：内存 + 磁盘
# ============================================================

def test_disk_and_memory_cache(tmp_path, monkeypatch):
    calls = {"n": 0}
    today_key = date.today().strftime("%Y%m%d")

    def fake_em(self, d, key):
        calls["n"] += 1
        return frozenset({"600000.SH", "000001.SZ"})

    monkeypatch.setattr(SuspendProvider, "_fetch_eastmoney", fake_em)

    p = SuspendProvider(store_path=tmp_path, sources=("eastmoney",))
    s1 = p.get_suspended_set(date.today())
    assert s1 == frozenset({"600000.SH", "000001.SZ"})
    assert calls["n"] == 1
    # 第二次命中内存缓存，不再调用
    s2 = p.get_suspended_set(date.today())
    assert s2 == s1
    assert calls["n"] == 1
    # 磁盘缓存已落盘
    assert (tmp_path / "suspend" / f"{today_key}.parquet").exists()

    # 新建 provider（清空内存），应命中磁盘缓存而不联网
    p2 = SuspendProvider(store_path=tmp_path, sources=("eastmoney",))
    s3 = p2.get_suspended_set(date.today())
    assert s3 == s1
    assert calls["n"] == 1  # fake_em 未被新实例调用（calls 共享，仍为 1）


def test_lookback_cap_skips_network(tmp_path, monkeypatch):
    called = {"n": 0}

    def fake_em(self, d, key):
        called["n"] += 1
        return frozenset({"600000.SH"})

    monkeypatch.setattr(SuspendProvider, "_fetch_eastmoney", fake_em)

    p = SuspendProvider(store_path=tmp_path, sources=("eastmoney",), lookback_days=30)
    old_day = date.today() - timedelta(days=400)
    s = p.get_suspended_set(old_day)
    assert s == frozenset()
    assert called["n"] == 0  # 超出 lookback，未联网


def test_disabled_returns_empty(tmp_path, monkeypatch):
    def fake_em(self, d, key):  # 不应被调用
        raise AssertionError("disabled 时不应取数")

    monkeypatch.setattr(SuspendProvider, "_fetch_eastmoney", fake_em)
    p = SuspendProvider(store_path=tmp_path, enabled=False)
    assert p.get_suspended_set(date.today()) == frozenset()
    assert not p.enabled


# ============================================================
# 源容灾：东财失败 → tushare 兜底
# ============================================================

def test_eastmoney_fail_fallback_tushare(tmp_path, monkeypatch):
    def fake_em(self, d, key):
        raise RuntimeError("东财风控")

    def fake_ts(self, d, key):
        return frozenset({"000002.SZ"})

    monkeypatch.setattr(SuspendProvider, "_fetch_eastmoney", fake_em)
    monkeypatch.setattr(SuspendProvider, "_fetch_tushare", fake_ts)

    p = SuspendProvider(store_path=tmp_path, sources=("eastmoney", "tushare"))
    s = p.get_suspended_set(date.today())
    assert s == frozenset({"000002.SZ"})


def test_all_sources_fail_returns_empty_no_cache(tmp_path, monkeypatch):
    def boom(self, d, key):
        raise RuntimeError("挂了")

    monkeypatch.setattr(SuspendProvider, "_fetch_eastmoney", boom)
    monkeypatch.setattr(SuspendProvider, "_fetch_tushare", boom)

    p = SuspendProvider(store_path=tmp_path, sources=("eastmoney", "tushare"))
    today_key = date.today().strftime("%Y%m%d")
    s = p.get_suspended_set(date.today())
    assert s == frozenset()
    # 失败不落盘，下次可重试
    assert not (tmp_path / "suspend" / f"{today_key}.parquet").exists()


def test_empty_authoritative_set_is_cached(tmp_path, monkeypatch):
    """当日确无停牌（源返回空）也应缓存，避免重复联网。"""
    calls = {"n": 0}

    def fake_em(self, d, key):
        calls["n"] += 1
        return frozenset()

    monkeypatch.setattr(SuspendProvider, "_fetch_eastmoney", fake_em)
    p = SuspendProvider(store_path=tmp_path, sources=("eastmoney",))
    today_key = date.today().strftime("%Y%m%d")
    assert p.get_suspended_set(date.today()) == frozenset()
    assert (tmp_path / "suspend" / f"{today_key}.parquet").exists()
    # 新实例读磁盘
    p2 = SuspendProvider(store_path=tmp_path, sources=("eastmoney",))
    assert p2.get_suspended_set(date.today()) == frozenset()


# ============================================================
# fetcher：原始层只落成交行，停牌由加工层按名单标注；增量游标跳过确认停牌日
# ============================================================

from src.data.fetcher import DataFetcher  # noqa: E402
from src.data.trading_calendar import TradingCalendar  # noqa: E402


class _FakeSuspend:
    """注入用的停牌名单桩：指定日期集合视为停牌。"""

    enabled = True
    _lookback_days = 0

    def __init__(self, confirmed_by_date):
        self._m = confirmed_by_date  # {date: set(codes)}

    def get_suspended_set(self, d):
        key = pd.Timestamp(d).date()
        return frozenset(self._m.get(key, set()))


def _fetcher_with_calendar(tmp_path, trading_days, suspend_stub):
    cal = TradingCalendar(store_path=tmp_path, auto_load=False)
    cal._trading_days = pd.DatetimeIndex(pd.to_datetime(trading_days))
    f = DataFetcher(store_path=tmp_path, calendar=cal, suspend_enabled=False, tdx_path="__nonexistent__")
    f._suspend = suspend_stub
    f._stock_name_cache["000001.SZ"] = "平安银行"
    return f


def _raw(dates, closes, volumes=None):
    n = len(dates)
    closes = [float(c) for c in closes]
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": volumes if volumes is not None else [1000.0] * n,
        "amount": [1e4] * n,
    })


def test_normalize_daily_raw_drops_placeholders(tmp_path):
    """原始层：丢弃无价量空行与数据源自带停牌标记行，零成交有价行保留。"""
    f = _fetcher_with_calendar(tmp_path, ["2024-01-02"], _FakeSuspend({}))
    df = _raw(["2024-01-02", "2024-01-03", "2024-01-04"], [10.0, 10.0, 10.2], [1000.0, 0.0, 0.0])
    df.loc[1, "_suspended"] = True           # 源标记的停牌前收占位行
    df = pd.concat([df, pd.DataFrame({"date": pd.to_datetime(["2024-01-05"])})], ignore_index=True)
    out = f._normalize_daily_raw("000001.SZ", df, source="pytdx")
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-04"]
    assert list(out.columns) == ["date", "code", "open", "high", "low", "close", "volume", "amount", "source"]
    assert (out["source"] == "pytdx").all()


def test_build_meta_status_and_confirmed_tail(tmp_path):
    """时间轴 = 首末原始行间全部交易日 + 其后连续确认停牌日；缺口按名单分 suspend / missing。"""
    days = ["2022-01-04", "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-10", "2022-01-11"]
    stub = _FakeSuspend({
        date(2022, 1, 5): {"000001.SZ"},     # 中间缺口：确认停牌
        date(2022, 1, 10): {"000001.SZ"},     # 尾部：确认停牌
    })
    f = _fetcher_with_calendar(tmp_path, days, stub)
    raw = _raw(["2022-01-04", "2022-01-07"], [10.0, 10.5])   # 01-05 停牌、01-06 缺数据
    meta = f._build_daily_meta("000001.SZ", raw, until=date(2022, 1, 11))
    got = meta.set_index(meta["date"].dt.strftime("%Y-%m-%d"))
    assert list(got.index) == days[:5]                        # 01-11 未确认 → 不进时间轴
    assert list(got["status"]) == ["trade", "suspend", "missing", "trade", "suspend"]
    assert list(got["is_suspended"]) == [False, True, True, False, True]
    assert np.isclose(got.loc["2022-01-07", "limit_up"], 11.0)   # 复牌日取停牌前收 10.0（核准制上市，无新股不设限窗口）


def test_build_meta_delisted_right_after_last_trade(tmp_path):
    """退市日紧跟末个成交日（中间只隔周末）：停牌尾部区间为空，不应报错。"""
    days = ["2022-06-23", "2022-06-24", "2022-06-27"]
    f = _fetcher_with_calendar(tmp_path, days, _FakeSuspend({}))
    f._delist_date_cache["000001.SZ"] = date(2022, 6, 27)
    meta = f._build_daily_meta("000001.SZ", _raw(days[:2], [10.0, 10.1]), until=date(2022, 6, 27))
    assert list(meta["status"]) == ["trade", "trade"]


def test_build_meta_zero_volume_needs_confirmation(tmp_path):
    """零成交但有价：名单确认才算停牌；未确认按真实（低流动性）交易日。"""
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    stub = _FakeSuspend({date(2024, 1, 4): {"000001.SZ"}})
    f = _fetcher_with_calendar(tmp_path, days, stub)
    raw = _raw(days, [10.0, 10.1, 10.1], [1000.0, 0.0, 0.0])
    meta = f._build_daily_meta("000001.SZ", raw, until=date(2024, 1, 4))
    assert list(meta["status"]) == ["trade", "trade", "suspend"]


def test_first_unconfirmed_day_skips_suspension(tmp_path):
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    stub = _FakeSuspend({date(2024, 1, 2): {"000001.SZ"}, date(2024, 1, 3): {"000001.SZ"}})
    f = _fetcher_with_calendar(tmp_path, days, stub)
    assert f._first_unconfirmed_day("000001.SZ", date(2024, 1, 2), date(2024, 1, 4)) == date(2024, 1, 4)
    assert f._first_unconfirmed_day("000001.SZ", date(2024, 1, 2), date(2024, 1, 3)) is None


def test_update_suspended_tail_extends_meta_without_fetch(tmp_path, monkeypatch):
    """原始行之后整段确认停牌：不取数、不改原始文件，加工层时间轴延伸到最后确认停牌日。"""
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    stub = _FakeSuspend({date(2024, 1, 3): {"000001.SZ"}, date(2024, 1, 4): {"000001.SZ"}})
    f = _fetcher_with_calendar(tmp_path, days, stub)
    raw = f._normalize_daily_raw("000001.SZ", _raw(["2024-01-02"], [10.0]), source="pytdx")
    f._store.write_raw("000001.SZ", "daily", raw)
    f._refresh_daily_meta("000001.SZ", raw, until=date(2024, 1, 2))

    def boom(*a, **k):
        raise AssertionError("整段停牌不应取数")

    monkeypatch.setattr(f, "_fetch_with_fallback", boom)
    monkeypatch.setattr("src.data.fetcher.date", type("D", (date,), {"today": staticmethod(lambda: date(2024, 1, 5))}))
    f._update_one("000001.SZ", "daily")

    assert len(f._store.read_raw("000001.SZ", "daily")) == 1
    view = f._store.read_view("000001.SZ", "daily")
    assert list(view["date"].dt.strftime("%Y-%m-%d")) == days
    assert list(view["status"]) == ["trade", "suspend", "suspend"]
    assert view["close"].isna().tolist() == [False, True, True]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
