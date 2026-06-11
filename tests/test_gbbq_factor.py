"""gbbq 自算复权因子（FactorCalculator）与 GbbqStore 查询逻辑测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.data.adjust import apply_adjust
from src.data.factors import FactorCalculator
from src.data.gbbq import GbbqStore
from src.data.storage import DataStore


def _raw(dates, closes):
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": closes})


# ============================================================
# FactorCalculator._compute 公式
# ============================================================


def test_compute_song_only():
    """10 送 10（每股送转 1.0）：除权价减半，因子翻倍。"""
    raw = _raw(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
               [10.0, 10.0, 5.5, 5.5])
    events = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03"]),
        "fenhong": [0.0], "peijia": [0.0], "song": [10.0], "pei": [0.0],
    })
    out = FactorCalculator._compute(raw, events)
    assert list(out["date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-03"]))
    # ex = (10 - 0 + 0) / (1 + 1) = 5 ; cum = 10/5 = 2.0
    assert out["cum_factor"].iloc[0] == pytest.approx(1.0)
    assert out["cum_factor"].iloc[1] == pytest.approx(2.0)


def test_compute_dividend_only():
    """每 10 股派现 10 元（每股 1.0）：因子 = 前收/(前收-派现)。"""
    raw = _raw(["2024-01-01", "2024-01-02", "2024-01-03"], [10.0, 10.0, 9.0])
    events = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03"]),
        "fenhong": [10.0], "peijia": [0.0], "song": [0.0], "pei": [0.0],
    })
    out = FactorCalculator._compute(raw, events)
    # ex = (10 - 1) / 1 = 9 ; cum = 10/9
    assert out["cum_factor"].iloc[-1] == pytest.approx(10.0 / 9.0)


def test_compute_rights_issue():
    """配股：每 10 股配 5 股（每股 0.5），配股价 5 元，前收 10。"""
    raw = _raw(["2024-01-01", "2024-01-02", "2024-01-03"], [10.0, 10.0, 8.5])
    events = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03"]),
        "fenhong": [0.0], "peijia": [5.0], "song": [0.0], "pei": [5.0],
    })
    out = FactorCalculator._compute(raw, events)
    # ex = (10 - 0 + 0.5*5) / (1 + 0.5) = 12.5/1.5 ; cum = 10 / (12.5/1.5)
    expected = 10.0 / (12.5 / 1.5)
    assert out["cum_factor"].iloc[-1] == pytest.approx(expected)


def test_compute_no_events_is_flat():
    """无事件：仅基准行，因子恒 1.0。"""
    raw = _raw(["2024-01-01", "2024-01-02"], [10.0, 11.0])
    out = FactorCalculator._compute(raw, pd.DataFrame(columns=["date", "fenhong", "peijia", "song", "pei"]))
    assert len(out) == 1
    assert out["cum_factor"].iloc[0] == pytest.approx(1.0)


def test_compute_event_before_first_close_skipped():
    """除权日早于本地最早行情（无前收）→ 跳过该事件。"""
    raw = _raw(["2024-01-02", "2024-01-03"], [10.0, 10.0])
    events = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),  # 早于最早行情
        "fenhong": [0.0], "peijia": [0.0], "song": [10.0], "pei": [0.0],
    })
    out = FactorCalculator._compute(raw, events)
    assert len(out) == 1  # 仅基准行


def test_compute_empty_raw():
    out = FactorCalculator._compute(None, pd.DataFrame())
    assert out.empty


# ============================================================
# 与 apply_adjust 集成：后复权连续性
# ============================================================


def test_hfq_continuity_with_calculator():
    """自算因子喂给 apply_adjust(hfq)：除权前后后复权价连续。"""
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "open": [10.0, 10.0, 5.0, 5.0],
        "high": [10.0, 10.0, 5.0, 5.0],
        "low": [10.0, 10.0, 5.0, 5.0],
        "close": [10.0, 10.0, 5.0, 5.0],  # 01-03 起 10送10，价格减半
    })
    events = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03"]),
        "fenhong": [0.0], "peijia": [0.0], "song": [10.0], "pei": [0.0],
    })
    factor = FactorCalculator._compute(raw, events)
    hfq = apply_adjust(raw, factor, mode="hfq", time_col="date")
    # 除权前 hfq=10；除权后 raw 5 * 因子 2 = 10，连续
    assert hfq["close"].tolist() == pytest.approx([10.0, 10.0, 10.0, 10.0])


# ============================================================
# GbbqStore 查询 / 过滤逻辑（绕过文件，直接注入解析结果）
# ============================================================


def _make_store_with_df(df: pd.DataFrame) -> GbbqStore:
    gs = GbbqStore(tdx_path="__nonexistent__")
    gs._loaded = True  # 跳过文件解析
    gs._df = df
    return gs


def _gbbq_df(rows):
    """rows: list of (market, code, date, category, f1, f2, f3, f4)。"""
    cols = ["market", "code", "date", "category",
            "hongli_panqianliutong", "peigujia_qianzongguben",
            "songgu_qianzongguben", "peigu_houzongguben"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_store_filters_by_market_and_category():
    df = _gbbq_df([
        (0, "000001", "2024-01-03", 1, 10.0, 0.0, 0.0, 0.0),   # SZ 除权除息
        (1, "000001", "2024-02-03", 1, 5.0, 0.0, 0.0, 0.0),    # SH 同号码不同市场
        (0, "000001", "2024-03-03", 2, 0.0, 0.0, 0.0, 0.0),    # SZ 但非除权除息类别
    ])
    gs = _make_store_with_df(df)
    ev = gs.events("000001.SZ")
    assert len(ev) == 1
    assert ev["date"].iloc[0] == pd.Timestamp("2024-01-03")
    assert ev["fenhong"].iloc[0] == pytest.approx(10.0)


def test_store_last_event_date():
    df = _gbbq_df([
        (1, "600519", "2023-06-30", 1, 30.0, 0.0, 0.0, 0.0),
        (1, "600519", "2024-06-28", 1, 35.0, 0.0, 0.0, 0.0),
    ])
    gs = _make_store_with_df(df)
    assert gs.last_event_date("600519.SH") == pd.Timestamp("2024-06-28")
    assert gs.last_event_date("000002.SZ") is None  # 无事件


def test_store_bj_query_by_current_code():
    """北交所按当前（新）代码直查命中；实测表明新代码已含完整除权记录（老代码为子集）。"""
    df = _gbbq_df([
        (2, "920799", "2024-01-03", 1, 1.0, 0.0, 0.0, 0.0),  # 当前代码
        (2, "830799", "2023-05-06", 1, 2.0, 0.0, 0.0, 0.0),  # 老代码（子集，不再单独查）
    ])
    gs = _make_store_with_df(df)
    ev = gs.events("920799.BJ")
    assert len(ev) == 1
    assert ev["date"].iloc[0] == pd.Timestamp("2024-01-03")
    # 老代码查询只返回老代码自身事件，不再自动并入新代码（已无映射）
    ev_old = gs.events("830799.BJ")
    assert len(ev_old) == 1
    assert ev_old["date"].iloc[0] == pd.Timestamp("2023-05-06")


# ============================================================
# 事件快照落盘 / 回读
# ============================================================


def test_snapshot_save_and_roundtrip(tmp_path):
    df = _gbbq_df([
        (0, "000001", "2024-01-03", 1, 10.0, 0.0, 0.0, 0.0),
        (1, "600519", "2024-06-28", 1, 35.0, 0.0, 0.0, 0.0),
    ])
    snap = tmp_path / "gbbq_events.parquet"
    gs = GbbqStore(tdx_path="__nonexistent__", snapshot_path=snap)
    gs._loaded = True
    gs._df = df

    p = gs.save_snapshot(force=True)
    assert p == snap and snap.exists()

    back = gs._read_snapshot()
    assert {"market", "code", "date", "category"}.issubset(back.columns)

    # 从快照重建一个 store，events() 应等价
    gs2 = GbbqStore(tdx_path="__nonexistent__", snapshot_path=snap)
    gs2._loaded = True
    gs2._df = back
    ev = gs2.events("000001.SZ")
    assert len(ev) == 1
    assert ev["fenhong"].iloc[0] == pytest.approx(10.0)


def test_snapshot_noop_without_path():
    gs = GbbqStore(tdx_path="__nonexistent__", snapshot_path=None)
    gs._loaded = True
    gs._df = pd.DataFrame()
    assert gs.save_snapshot() is None


def test_snapshot_diff_counts_new_events():
    old = _gbbq_df([
        (0, "000001", "2024-01-03", 1, 10.0, 0.0, 0.0, 0.0),
        (1, "600519", "2024-06-28", 1, 35.0, 0.0, 0.0, 0.0),
    ])
    new = _gbbq_df([
        (0, "000001", "2024-01-03", 1, 10.0, 0.0, 0.0, 0.0),
        (1, "600519", "2024-06-28", 1, 35.0, 0.0, 0.0, 0.0),
        (1, "600519", "2025-06-20", 1, 38.0, 0.0, 0.0, 0.0),   # 新增 1 条
    ])
    added, delta = GbbqStore._snapshot_diff(old, new)
    assert added == 1
    assert delta == 1


def test_snapshot_rewrite_reports_delta(tmp_path):
    snap = tmp_path / "gbbq_events.parquet"
    gs = GbbqStore(tdx_path="__nonexistent__", snapshot_path=snap)
    gs._loaded = True
    gs._df = _gbbq_df([(0, "000001", "2024-01-03", 1, 10.0, 0.0, 0.0, 0.0)])
    gs.save_snapshot(force=True)  # 首次生成

    # 新增一条后重写：diff 应识别 +1
    gs._df = _gbbq_df([
        (0, "000001", "2024-01-03", 1, 10.0, 0.0, 0.0, 0.0),
        (0, "000001", "2025-10-15", 1, 2.36, 0.0, 0.0, 0.0),
    ])
    old = gs._read_snapshot()
    added, delta = GbbqStore._snapshot_diff(old, gs._df)
    assert added == 1 and delta == 1
    assert gs.save_snapshot(force=True) == snap


# ============================================================
# DataStore：factors_gbbq/ 与 factors/ 并存不互相覆盖
# ============================================================


def test_store_factor_gbbq_separate_dir(tmp_path):
    ds = DataStore(tmp_path)
    gbbq_factor = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "cum_factor": [1.5]})
    ds.write_factor("000001.SZ", gbbq_factor, gbbq=True)

    # factors_gbbq/ 有，factors/ 仍空
    assert ds.read_factor("000001.SZ", gbbq=True)["cum_factor"].iloc[0] == pytest.approx(1.5)
    assert ds.read_factor("000001.SZ", gbbq=False).empty
    assert (Path(tmp_path) / "factors_gbbq" / "000001.SZ.parquet").exists()

    # 写 factors/ 不影响 factors_gbbq/
    ext_factor = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "cum_factor": [2.0]})
    ds.write_factor("000001.SZ", ext_factor, gbbq=False)
    assert ds.read_factor("000001.SZ", gbbq=False)["cum_factor"].iloc[0] == pytest.approx(2.0)
    assert ds.read_factor("000001.SZ", gbbq=True)["cum_factor"].iloc[0] == pytest.approx(1.5)


# ============================================================
# load(use_gbbq=...)：因子口径切换、缓存隔离、缺失回退
# ============================================================


def _seed_store(tmp_path, code="000001.SZ", active=2.0, gbbq=None):
    ds = DataStore(tmp_path)
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "open": [10.0, 10.0], "high": [10.0, 10.0], "low": [10.0, 10.0],
        "close": [10.0, 10.0], "volume": [1.0, 1.0], "amount": [1.0, 1.0],
    })
    ds.write_raw(code, "daily", raw)
    ds.write_factor(code, pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01"]), "cum_factor": [active]}), gbbq=False)
    if gbbq is not None:
        ds.write_factor(code, pd.DataFrame(
            {"date": pd.to_datetime(["2024-01-01"]), "cum_factor": [gbbq]}), gbbq=True)
    return ds


def test_load_switch_and_cache_isolation(tmp_path):
    ds = _seed_store(tmp_path, active=2.0, gbbq=3.0)
    a = ds.load("000001.SZ", "daily", "hfq", use_gbbq=False)
    g = ds.load("000001.SZ", "daily", "hfq", use_gbbq=True)
    assert a["close"].iloc[0] == pytest.approx(20.0)   # 10 * 2.0（生效因子）
    assert g["close"].iloc[0] == pytest.approx(30.0)   # 10 * 3.0（gbbq 因子）
    # 两套 hfq 缓存隔离落盘
    assert (Path(tmp_path) / "adjusted" / "daily" / "000001.SZ.parquet").exists()
    assert (Path(tmp_path) / "adjusted_gbbq" / "daily" / "000001.SZ.parquet").exists()


def test_load_gbbq_fallback_when_missing(tmp_path):
    # 无 factors_gbbq → use_gbbq 应回退到生效因子
    ds = _seed_store(tmp_path, active=2.0, gbbq=None)
    g = ds.load("000001.SZ", "daily", "hfq", use_gbbq=True)
    assert g["close"].iloc[0] == pytest.approx(20.0)
