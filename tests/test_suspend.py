"""
停牌名单 Provider + fetcher 停牌判定单元测试（无网络）
======================================================

覆盖：
  - ``src.data.suspend.SuspendProvider``：代码归一、磁盘/内存缓存、lookback 限频、
    东财主源 → tushare 兜底的容灾切换。
  - ``src.data.fetcher.DataFetcher._trim_unconfirmed_trailing``：尾部未确认缺口裁剪。
  - ``DataFetcher._post_process_daily``：权威停牌名单区分 :suspend / :gap，空区间补行。

所有网络调用均通过 monkeypatch 替换，测试零网络。

运行：
    python -m pytest tests/test_suspend.py -q
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

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
# fetcher 尾部裁剪
# ============================================================

from src.data.fetcher import DataFetcher  # noqa: E402


def _src_df(dates, sources):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "close": [float("nan")] * len(dates),
        "source": pd.array(sources, dtype="string"),
    })


def test_trim_trailing_gap():
    df = _src_df(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        ["sina", "sina:gap", "sina:gap"],
    )
    out = DataFetcher._trim_unconfirmed_trailing(df)
    assert len(out) == 1
    assert out["source"].iloc[0] == "sina"


def test_trim_keeps_suspend_tail():
    df = _src_df(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        ["sina", "sina:gap", "sina:suspend"],
    )
    out = DataFetcher._trim_unconfirmed_trailing(df)
    # 末行是 :suspend（已确认停牌），不裁；中间 :gap 保留
    assert len(out) == 3


def test_trim_interior_gap_kept():
    df = _src_df(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        ["sina:gap", "sina", "sina"],
    )
    out = DataFetcher._trim_unconfirmed_trailing(df)
    assert len(out) == 3  # 中间/开头 gap 不裁，只裁尾部


def test_trim_all_gap_becomes_empty():
    df = _src_df(
        ["2024-01-02", "2024-01-03"],
        ["sina:gap", "sina:gap"],
    )
    out = DataFetcher._trim_unconfirmed_trailing(df)
    assert out.empty


# ============================================================
# _post_process_daily 端到端：权威停牌名单区分 :suspend / :gap
# ============================================================

from src.data.trading_calendar import TradingCalendar  # noqa: E402


class _FakeSuspend:
    """注入用的停牌名单桩：指定日期集合视为停牌。"""

    enabled = True

    def __init__(self, confirmed_by_date):
        self._m = confirmed_by_date  # {date: set(codes)}

    def get_suspended_set(self, d):
        key = pd.Timestamp(d).date()
        return frozenset(self._m.get(key, set()))


def _fetcher_with_calendar(tmp_path, trading_days, suspend_stub):
    cal = TradingCalendar(store_path=tmp_path, auto_load=False)
    cal._trading_days = pd.DatetimeIndex(pd.to_datetime(trading_days))
    f = DataFetcher(store_path=tmp_path, calendar=cal, suspend_enabled=False)
    f._suspend = suspend_stub
    DataFetcher._STOCK_NAME_CACHE["000001.SZ"] = "平安银行"
    return f


def test_post_process_empty_range_confirmed_suspend(tmp_path):
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    stub = _FakeSuspend({date(2024, 1, 4): {"000001.SZ"}})
    f = _fetcher_with_calendar(tmp_path, days, stub)

    # 整段空区间：reindex 补全 3 个交易日全 NaN 行
    out = f._post_process_daily("000001.SZ", pd.DataFrame(), date(2024, 1, 2), date(2024, 1, 4), source="pytdx")

    assert list(out["date"].dt.strftime("%Y-%m-%d")) == days
    # 全部缺口 → 全 is_suspended
    assert out["is_suspended"].all()
    src = out.set_index(out["date"].dt.strftime("%Y-%m-%d"))["source"]
    assert src["2024-01-02"] == "pytdx:gap"
    assert src["2024-01-03"] == "pytdx:gap"
    assert src["2024-01-04"] == "pytdx:suspend"  # 仅 01-04 被名单确认

    # 经尾部裁剪：末行是 :suspend，保留全部
    trimmed = DataFetcher._trim_unconfirmed_trailing(out)
    assert len(trimmed) == 3


def test_post_process_empty_range_all_unconfirmed_trimmed(tmp_path):
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    stub = _FakeSuspend({})  # 无任何确认停牌
    f = _fetcher_with_calendar(tmp_path, days, stub)

    out = f._post_process_daily("000001.SZ", pd.DataFrame(), date(2024, 1, 2), date(2024, 1, 4), source="pytdx")
    assert (out["source"] == "pytdx:gap").all()
    # 尾部裁剪后全空 → 不写入、不推进游标
    assert DataFetcher._trim_unconfirmed_trailing(out).empty


def test_post_process_with_data_then_trailing_suspend(tmp_path):
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    stub = _FakeSuspend({date(2024, 1, 4): {"000001.SZ"}})
    f = _fetcher_with_calendar(tmp_path, days, stub)

    # 仅 01-02、01-03 有真实数据，01-04 缺（停牌）
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [10.0, 10.5], "high": [10.2, 10.6], "low": [9.9, 10.4],
        "close": [10.1, 10.5], "volume": [1000.0, 1200.0], "amount": [1e4, 1.2e4],
        "raw_close": [10.1, 10.5],
    })
    out = f._post_process_daily("000001.SZ", raw, date(2024, 1, 2), date(2024, 1, 4), source="pytdx")
    src = out.set_index(out["date"].dt.strftime("%Y-%m-%d"))["source"]
    assert src["2024-01-02"] == "pytdx"
    assert src["2024-01-03"] == "pytdx"
    assert src["2024-01-04"] == "pytdx:suspend"
    susp = out.set_index(out["date"].dt.strftime("%Y-%m-%d"))["is_suspended"]
    assert not bool(susp["2024-01-02"])
    assert bool(susp["2024-01-04"])
    # 末行确认停牌，裁剪保留全部 3 行（推进游标到 01-04）
    assert len(DataFetcher._trim_unconfirmed_trailing(out)) == 3


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
