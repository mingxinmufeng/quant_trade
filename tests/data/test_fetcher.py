"""
数据层（存储 / 复权 / 重采样 / 涨跌停回补）单元测试 —— 零网络。

聚焦 ``src.data`` 中可离线验证的纯逻辑：``DataStore`` 读写 + 按需复权、``resample_*``、
``apply_adjust``、``fetcher._backfill_limit_df``。网络拉取路径不在本测试范围。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _raw_daily(code="000001.SZ", n=10):
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(np.arange(10, 10 + n), dtype="float64")
    return pd.DataFrame(
        {
            "date": dates,
            "code": code,
            "open": close.astype("float64"),
            "high": (close + 0.5).astype("float64"),
            "low": (close - 0.5).astype("float64"),
            "close": close.astype("float64"),
            "volume": pd.Series(np.full(n, 1e5)).astype("float64"),
            "amount": pd.Series(np.full(n, 1e6)).astype("float64"),
            "is_suspended": False,
            "limit_up": (close * 1.1).astype("float64"),
            "limit_down": (close * 0.9).astype("float64"),
            "name": "平安银行",
            "source": "test",
        }
    )


def test_datastore_roundtrip_and_none_adjust(tmp_path):
    from src.data import DataStore

    store = DataStore(tmp_path)
    df = _raw_daily()
    store.write_raw("000001.SZ", "daily", df)
    got = store.read_raw("000001.SZ", "daily")
    assert got is not None and len(got) == len(df)
    # none 复权：原样价 + adj_factor=1
    none_df = store.load("000001.SZ", "daily", adjust="none")
    assert np.allclose(none_df["close"], df["close"])
    assert np.allclose(none_df["adj_factor"], 1.0)


def test_datastore_hfq_adjust(tmp_path):
    from src.data import DataStore

    store = DataStore(tmp_path)
    df = _raw_daily(n=10)
    store.write_raw("000001.SZ", "daily", df)
    # 因子：前 5 日 1.0，后 5 日 2.0（除权日翻倍）
    factor = pd.DataFrame(
        {"date": df["date"], "cum_factor": [1.0] * 5 + [2.0] * 5}
    )
    store.write_factor("000001.SZ", factor)
    hfq = store.load("000001.SZ", "daily", adjust="hfq")
    # 后 5 日收盘价应为原始价 ×2
    assert np.allclose(hfq["close"].to_numpy()[5:], df["close"].to_numpy()[5:] * 2.0)
    assert np.allclose(hfq["close"].to_numpy()[:5], df["close"].to_numpy()[:5] * 1.0)


def test_apply_adjust_modes():
    from src.data import apply_adjust

    df = _raw_daily(n=6)
    factor = pd.DataFrame({"date": df["date"], "cum_factor": [1, 1, 1, 2, 2, 2]})
    hfq = apply_adjust(df, factor, mode="hfq", time_col="date")
    assert np.allclose(hfq["close"].to_numpy()[3:], df["close"].to_numpy()[3:] * 2)
    # qfq 锚最后一日：最后一日因子 2 → 早期价 /2 的相对刻度
    qfq = apply_adjust(df, factor, mode="qfq", time_col="date")
    assert qfq["close"].iloc[-1] == df["close"].iloc[-1]  # 锚点日不变


def test_resample_daily_weekly():
    from src.data import resample_daily

    df = _raw_daily(n=10)
    wk = resample_daily(df, "weekly")
    assert not wk.empty
    assert {"open", "high", "low", "close", "volume"}.issubset(wk.columns)
    # 周线高点 = 周内最高
    assert wk["high"].iloc[0] <= df["high"].max() + 1e-9


def test_backfill_limit_df():
    from src.data.fetcher import _backfill_limit_df

    df = _raw_daily(n=4)
    df.loc[0, "limit_up"] = np.nan
    df.loc[0, "limit_down"] = np.nan
    df.loc[2, "limit_up"] = np.nan
    out = _backfill_limit_df(df.copy(), limit_pct=0.10)
    # 首行无前收 → 仍 NaN；第 3 行用前一行 close 反推
    assert pd.isna(out.loc[0, "limit_up"])
    assert not pd.isna(out.loc[2, "limit_up"])
    assert np.isclose(out.loc[2, "limit_up"], round(df.loc[1, "close"] * 1.1, 2))


def test_minute_resample():
    from src.data import resample_minute

    n = 20
    dt = pd.date_range("2024-01-02 09:30", periods=n, freq="1min")
    close = pd.Series(np.arange(10, 10 + n), dtype="float64")
    mdf = pd.DataFrame(
        {"datetime": dt, "code": "000001.SZ", "open": close, "high": close + 0.2,
         "low": close - 0.2, "close": close, "volume": 100.0, "amount": 1000.0}
    )
    out = resample_minute(mdf, base_period=1, target_period=5)
    assert len(out) == 4  # 20 根 1 分钟 → 4 根 5 分钟
    assert out["close"].iloc[0] == close.iloc[4]  # 每组取最后一根
