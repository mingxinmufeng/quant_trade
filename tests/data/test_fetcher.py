"""
数据层（存储 / 复权 / 重采样 / 涨跌停回补）单元测试 —— 零网络。

聚焦 ``src.data`` 中可离线验证的纯逻辑：``DataStore`` 读写 + 按需复权、``resample_*``、
``apply_adjust``、``fetcher`` 涨跌停价（``_limit_ref_price`` / ``_limit_prices``）。网络拉取路径不在本测试范围。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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


def test_datastore_none_keeps_cum_factor_for_corporate_actions(tmp_path):
    """none 价格不缩放，但保留累计因子供回测处理除权持仓调整。"""
    from src.data import DataStore

    store = DataStore(tmp_path)
    df = _raw_daily(n=4)
    store.write_raw("000001.SZ", "daily", df)
    store.write_factor(
        "000001.SZ",
        pd.DataFrame({"date": df["date"], "cum_factor": [1.0, 1.0, 2.0, 2.0]}),
    )

    got = store.load("000001.SZ", "daily", adjust="none")
    assert np.allclose(got["close"], df["close"])
    assert np.allclose(got["adj_factor"], 1.0)
    assert got["cum_factor"].tolist() == [1.0, 1.0, 2.0, 2.0]


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


def test_resample_daily_preserves_cum_factor():
    """周/月等重采样须保留累计因子，供 none 回测做除权持仓调整。"""
    from src.data import resample_daily

    df = _raw_daily(n=10)
    df["adj_factor"] = 1.0
    df["cum_factor"] = [1.0] * 5 + [2.0] * 5

    wk = resample_daily(df, "weekly")

    assert "adj_factor" in wk.columns
    assert "cum_factor" in wk.columns
    assert wk["cum_factor"].tolist() == [1.0, 2.0, 2.0]


def test_resample_weekly_bar_date_is_real_trading_day():
    """P2-7：周线 bar 日期取 bin 内真实最后交易日，而非 W-FRI 的周五标签（可能落非交易日）。"""
    from src.data import resample_daily

    # 某周只有周一~周四有数据（周五缺，模拟节假日）
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    df = pd.DataFrame({
        "date": dates, "open": 1.0, "high": 2.0, "low": 0.5,
        "close": [1.0, 2.0, 3.0, 4.0], "volume": 100.0, "amount": 150.0,
    })
    wk = resample_daily(df, "weekly")
    assert len(wk) == 1
    assert wk["date"].iloc[0] == pd.Timestamp("2024-01-04")   # 周四（真实末日），非周五标签
    assert wk["close"].iloc[0] == 4.0                          # 周内最后收盘


def test_limit_ref_price_skips_suspension_and_applies_exdiv():
    """基准价：复牌日取停牌前最后收盘；除权除息日（含停牌期间除权）用参考价。"""
    from src.data.fetcher import _limit_ref_price

    dates = pd.Series(pd.to_datetime([
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09",
    ]))
    close = pd.Series([10.0, np.nan, np.nan, 12.0, 13.0, 6.0])   # 01-03/04 停牌占位
    events = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-04", "2024-01-09"]),
        "fenhong": [5.0, 0.0],   # 每 10 股派 5 元（停牌期间除息）
        "peijia": [0.0, 0.0],
        "song": [0.0, 10.0],     # 每 10 股送转 10 股
        "pei": [0.0, 0.0],
    })
    ref = _limit_ref_price(dates, close, events)
    assert np.isnan(ref[0])
    assert list(ref[1:3]) == [10.0, 9.5]        # 停牌占位行：前收 10；01-04 起除息
    assert ref[3] == 9.5                        # 复牌日：(10 − 0.5) / 1
    assert ref[4] == 12.0
    assert ref[5] == 6.5                        # 13 / (1 + 1)


def test_limit_ref_price_nonstandard_event_is_nan():
    """股改对价等非标准送转事件：交易所不按公式除权（且可能不设限）→ 基准价 NaN，不臆造。"""
    from src.data.fetcher import _limit_ref_price

    dates = pd.Series(pd.to_datetime(["2006-06-09", "2006-06-12", "2006-06-13"]))
    events = pd.DataFrame({
        "date": pd.to_datetime(["2006-06-12"]), "fenhong": [0.0], "peijia": [0.0],
        "song": [0.7], "pei": [0.0], "nonstandard": [True],
    })
    ref = _limit_ref_price(dates, pd.Series([6.92, 6.71, 6.80]), events)
    assert np.isnan(ref[1]) and ref[2] == 6.71


def test_gbbq_nonstandard_event_dates():
    """送转比例与同日总股本变化不符（股改：总股本不变）才算非标准；正常 10 送 10 不算。"""
    from src.data.gbbq import GbbqStore

    rows = [  # market, code, date, category, f1, f2, f3, f4
        (0, "000726", "2006-06-12", 1, 0.0, 0.0, 0.7, 0.0),
        (0, "000726", "2006-06-12", 5, 14196.0, 42243.2, 15130.9, 42243.2),
        (0, "000726", "2007-06-19", 1, 3.718, 0.0, 10.0, 0.0),
        (0, "000726", "2007-06-19", 5, 15130.9, 42243.2, 30279.1, 84486.5),
    ]
    df = pd.DataFrame(rows, columns=[
        "market", "code", "date", "category", "hongli_panqianliutong",
        "peigujia_qianzongguben", "songgu_qianzongguben", "peigu_houzongguben",
    ])
    df["date"] = pd.to_datetime(df["date"])
    g = GbbqStore(tdx_path="__nonexistent__")
    g._loaded = True
    g._df = df
    assert g.nonstandard_event_dates("000726.SZ") == {pd.Timestamp("2006-06-12")}


def test_round_price_half_up():
    """交易所四舍五入到分（numpy round 对 x.xx5 是银行家舍入）。"""
    from src.data.fetcher import _round_price

    assert list(_round_price([10.005, 10.015, 2.345, np.nan])[:3]) == [10.01, 10.02, 2.35]


# ============================================================
# 逐行点位 ST 涨跌停限幅（_limit_pct_series / _post_process / recompute_limits）
# ============================================================


def _offline_fetcher(tmp_path, code, current_name, profile_rows=None):
    """构造离线 DataFetcher：禁停牌、注入合成 profile 更名表与当前名缓存（零网络）。"""
    from src.data.fetcher import DataFetcher
    from src.data.profile import ProfileStore
    from src.data.trading_calendar import TradingCalendar

    cal = TradingCalendar(store_path=tmp_path, auto_load=False)  # 离线日历：工作日近似交易日
    cal._trading_days = pd.bdate_range("1990-12-19", "2027-12-31")
    f = DataFetcher(
        store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False, calendar=cal,
    )
    f._stock_name_cache[code] = current_name  # 预置避免触发网络名称表
    if profile_rows is not None:
        df = pd.DataFrame(profile_rows, columns=["code", "name", "change_date"])
        df["change_date"] = pd.to_datetime(df["change_date"])
        ps = ProfileStore(tdx_path="__nonexistent__")
        ps._loaded = True
        ps._df = df
        f._profile = ps
    return f


def test_limit_pct_series_main_board_pointwise_st(tmp_path):
    """主板逐行点位 ST：ST 时段 ±5%、转回常规名后 ±10%。"""
    f = _offline_fetcher(
        tmp_path, "600000.SH", "浦发银行",
        profile_rows=[("600000.SH", "ST浦发", "2010-01-01")],
    )
    dates = pd.Series(pd.to_datetime(["2009-06-01", "2009-12-31", "2010-01-04", "2025-01-01"]))
    pct = f._limit_pct_series("600000.SH", dates)
    assert list(pct) == [0.05, 0.05, 0.10, 0.10]


def test_limit_pct_series_non_main_board_ignores_st(tmp_path):
    """创业板（改革后）/科创板 ±20%、北交所 ±30%：即便点位名带 ST 也不受影响。"""
    for code, current, expect in [
        ("300001.SZ", "ST特锐", 0.20),
        ("688001.SH", "ST华兴", 0.20),
        ("830799.BJ", "ST艾融", 0.30),
    ]:
        f = _offline_fetcher(
            tmp_path, code, current,
            profile_rows=[(code, "ST曾用", "2022-01-01")],
        )
        dates = pd.Series(pd.to_datetime(["2021-06-01", "2025-01-01"]))
        pct = f._limit_pct_series(code, dates)
        assert list(pct) == [expect, expect], code


def test_limit_pct_series_rule_history(tmp_path):
    """规则沿革：创业板改革前 ±10%（ST ±5%）；主板 ST 2026-07-06 起 ±10%；1996-12-16 前不设限。"""
    f = _offline_fetcher(
        tmp_path, "300001.SZ", "特锐德",
        profile_rows=[("300001.SZ", "*ST特锐", "2019-01-01")],
    )
    dates = pd.Series(pd.to_datetime(["2018-06-01", "2020-08-21", "2020-08-24"]))
    assert list(f._limit_pct_series("300001.SZ", dates)) == [0.05, 0.10, 0.20]

    f = _offline_fetcher(tmp_path, "600000.SH", "*ST浦发", profile_rows=None)
    dates = pd.Series(pd.to_datetime(["1996-12-13", "1996-12-16", "2026-07-03", "2026-07-06"]))
    assert list(f._limit_pct_series("600000.SH", dates)) == [np.inf, 0.10, 0.05, 0.10]

    # 未完成股改（S 前缀）2006-10-09 起 ±5%；退市整理期（含"退"）不按 ST 档
    f = _offline_fetcher(
        tmp_path, "600733.SH", "北汽蓝谷",
        profile_rows=[("600733.SH", "S前锋", "2012-03-16"), ("600733.SH", "前锋退", "2012-04-01")],
    )
    dates = pd.Series(pd.to_datetime(["2006-09-29", "2006-10-09", "2012-03-20"]))
    assert list(f._limit_pct_series("600733.SH", dates)) == [0.10, 0.05, 0.10]


def test_limit_pct_series_fallback_current_name(tmp_path):
    """profile 不可用时主板回退当前名近似：当前名带 ST → 整段 ±5%。"""
    f = _offline_fetcher(tmp_path, "600000.SH", "ST浦发", profile_rows=None)
    assert not f._profile.available
    dates = pd.Series(pd.to_datetime(["2009-06-01", "2025-01-01"]))
    pct = f._limit_pct_series("600000.SH", dates)
    assert list(pct) == [0.05, 0.05]


def test_limit_prices_pointwise_st(tmp_path):
    """_limit_prices：按行日期取对应限幅；乱序输入按原行序返回。"""
    f = _offline_fetcher(
        tmp_path, "600000.SH", "浦发银行",
        profile_rows=[("600000.SH", "ST浦发", "2010-01-01")],
    )
    df = pd.DataFrame({
        "date": pd.to_datetime(["2010-01-04", "2009-12-30", "2009-12-31"]),
        "close": [12.0, 10.0, 11.0],
    })
    lu, ld = f._limit_prices("600000.SH", df)
    assert np.isnan(lu[1])                  # 首日无前收
    assert np.isclose(lu[2], 10.5)          # ST 段 ±5%，前收 10
    assert np.isclose(lu[0], 12.1)          # 常规 ±10%，前收 11
    assert np.isclose(ld[0], 9.9)


def test_limit_prices_bse_rounds_toward_ref(tmp_path):
    """北交所：涨停价向下、跌停价向上取整到分（9.95×1.3=12.935→12.93，9.95×0.7=6.965→6.97）。"""
    f = _offline_fetcher(tmp_path, "920273.BJ", "某北交所股", profile_rows=None)
    df = pd.DataFrame({"date": pd.bdate_range("2023-08-01", periods=8), "close": np.full(8, 9.95)})
    lu, ld = f._limit_prices("920273.BJ", df)
    assert lu[-1] == 12.93 and ld[-1] == 6.97


def test_limit_prices_ipo_window(tmp_path):
    """新股不设限窗口按上市日制度：注册制主板前 5 日 NaN，核准制主板次日即 ±10%。"""
    for list_day, expect_day2 in [("2023-04-10", np.nan), ("2023-04-06", 11.0)]:
        f = _offline_fetcher(tmp_path, "600001.SH", "新股", profile_rows=None)
        dates = pd.bdate_range(list_day, periods=7)
        df = pd.DataFrame({"date": dates, "close": np.full(7, 10.0)})
        lu, _ = f._limit_prices("600001.SH", df)
        if np.isnan(expect_day2):
            assert np.isnan(lu[:5]).all() and lu[5] == 11.0, list_day
        else:
            assert np.isnan(lu[0]) and (lu[1:] == 11.0).all(), list_day


def test_rebuild_daily_meta_writes_meta_only(tmp_path):
    """rebuild_daily_meta 只写加工层（点位 ST 涨跌停），原始日线文件不动；无变化不重写。"""
    f = _offline_fetcher(
        tmp_path, "600000.SH", "浦发银行",
        profile_rows=[("600000.SH", "ST浦发", "2010-01-01")],
    )
    dates = pd.to_datetime(["2009-12-30", "2009-12-31", "2010-01-04", "2010-01-05"])
    close = pd.Series([10.0, 11.0, 12.0, 13.0])
    raw = pd.DataFrame({
        "date": dates, "code": "600000.SH",
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e5, "amount": 1e6, "source": "test",
    })
    f._store.write_raw("600000.SH", "daily", raw)
    raw_mtime = f._store.freq_path("600000.SH", "daily").stat().st_mtime_ns

    assert f.rebuild_daily_meta(["600000.SH"], max_workers=1) == 1
    assert f._store.freq_path("600000.SH", "daily").stat().st_mtime_ns == raw_mtime
    got = f._store.read_view("600000.SH", "daily")
    got = got[got["status"] == "trade"]
    got = got.set_index(got["date"].dt.strftime("%Y-%m-%d"))
    assert list(got.index) == ["2009-12-30", "2009-12-31", "2010-01-04", "2010-01-05"]
    assert pd.isna(got.loc["2009-12-30", "limit_up"])              # 首行无前收
    assert np.isclose(got.loc["2009-12-31", "limit_up"], 10.5)     # ST → ±5%，前收 10
    assert np.isclose(got.loc["2009-12-31", "limit_down"], 9.5)
    assert np.isclose(got.loc["2010-01-04", "limit_up"], 12.1)     # 常规 → ±10%，前收 11
    assert np.isclose(got.loc["2010-01-05", "limit_up"], 13.2)     # 前收 12
    assert f.rebuild_daily_meta(["600000.SH"], max_workers=1) == 0   # 再跑无变化


def test_daily_view_joins_meta_timeline(tmp_path):
    """视图：加工层时间轴上无原始行的日子补 OHLCV 全 NaN 行；无加工层时兜底不报错。"""
    from src.data import DataStore

    store = DataStore(tmp_path)
    raw = _raw_daily(n=3)[["date", "code", "open", "high", "low", "close", "volume", "amount", "source"]]
    raw = raw.drop(index=1).reset_index(drop=True)              # 第 2 天无原始行（停牌）
    store.write_raw("000001.SZ", "daily", raw)
    no_meta = store.read_view("000001.SZ", "daily")
    assert len(no_meta) == 2 and not no_meta["is_suspended"].any()

    meta = pd.DataFrame({
        "date": _raw_daily(n=3)["date"], "is_suspended": [False, True, False],
        "limit_up": [np.nan, 11.0, 11.0], "limit_down": [np.nan, 9.0, 9.0],
        "status": ["trade", "suspend", "trade"],
    })
    store.write_daily_meta("000001.SZ", meta)
    view = store.read_view("000001.SZ", "daily")
    assert len(view) == 3
    assert pd.isna(view.loc[1, "close"]) and bool(view.loc[1, "is_suspended"])
    assert view.loc[1, "code"] == "000001.SZ"
    assert list(view.columns[:9]) == ["date", "code", "open", "high", "low", "close", "volume", "amount", "source"]
    assert np.isclose(store.load("000001.SZ", "daily", adjust="none").loc[2, "limit_up"], 11.0)


def test_name_table_prefers_local_tushare(tmp_path):
    """名称表优先读本地 tushare 基础信息缓存（离线，不联网东财）。"""
    from src.data.fetcher import STOCK_BASIC_TUSHARE_FILE, DataFetcher

    pd.DataFrame({
        "ts_code": ["600000.SH", "000004.SZ", "830799.BJ"],
        "name": ["浦发银行", "*ST国华", "艾融软件"],
    }).to_parquet(tmp_path / STOCK_BASIC_TUSHARE_FILE, index=False)

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False)
    f._stock_name_cache.clear()  # 实例缓存（新实例本就空，此处仅保持显式意图）
    assert f._load_names_from_local() is True
    assert f._get_stock_name("600000.SH") == "浦发银行"
    assert f._get_stock_name("000004.SZ") == "*ST国华"


def test_name_table_local_missing_falls_back(tmp_path):
    """本地基础信息缺失 → _load_names_from_local 返回 False（交由东财降级源）。"""
    from src.data.fetcher import DataFetcher

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False)
    f._stock_name_cache.clear()
    assert f._load_names_from_local() is False


def test_name_cache_isolated_per_store(tmp_path):
    """P2-4：不同 store_path 的两个 fetcher 名称缓存相互隔离（不再进程级 ClassVar 串扰）。"""
    from src.data.fetcher import STOCK_BASIC_TUSHARE_FILE, DataFetcher

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pd.DataFrame({"ts_code": ["600000.SH"], "name": ["浦发银行"]}).to_parquet(
        tmp_path / "a" / STOCK_BASIC_TUSHARE_FILE, index=False)
    pd.DataFrame({"ts_code": ["600000.SH"], "name": ["ST浦发"]}).to_parquet(
        tmp_path / "b" / STOCK_BASIC_TUSHARE_FILE, index=False)

    fa = DataFetcher(store_path=tmp_path / "a", tdx_path="__nonexistent__", suspend_enabled=False)
    fb = DataFetcher(store_path=tmp_path / "b", tdx_path="__nonexistent__", suspend_enabled=False)
    assert fa._get_stock_name("600000.SH") == "浦发银行"
    assert fb._get_stock_name("600000.SH") == "ST浦发"   # 不被 a 实例的缓存串扰


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


# ============================================================
# 加载复权口径：gbbq 优先、外部兜底（P1-7）
# ============================================================


def test_load_factor_source_default_gbbq_preferred(tmp_path):
    """P1-7：加载默认 gbbq 优先（_resolve_use_gbbq(None) → True）。"""
    from src.data.fetcher import DataFetcher

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False)
    assert f._load_factor_source == "gbbq"
    assert f._resolve_use_gbbq(None) is True       # 缺省 → gbbq 优先
    assert f._resolve_use_gbbq("active") is False   # 显式 active → 外部源


def test_datastore_gbbq_preferred_with_external_fallback(tmp_path):
    """use_gbbq=True：有 factors_gbbq/ 用 gbbq；该股无 gbbq 因子则回退 factors/。"""
    from src.data import DataStore

    store = DataStore(tmp_path)
    raw_a = _raw_daily("000001.SZ", n=10)
    store.write_raw("000001.SZ", "daily", raw_a)
    ext = pd.DataFrame({"date": raw_a["date"], "cum_factor": [1.0] * 5 + [2.0] * 5})
    gbb = pd.DataFrame({"date": raw_a["date"], "cum_factor": [1.0] * 5 + [3.0] * 5})
    store.write_factor("000001.SZ", ext, gbbq=False)
    store.write_factor("000001.SZ", gbb, gbbq=True)
    hfq = store.load("000001.SZ", "daily", adjust="hfq", use_gbbq=True)
    assert np.allclose(hfq["close"].to_numpy()[5:], raw_a["close"].to_numpy()[5:] * 3.0), "应优先用 gbbq 因子(×3)"

    # 另一只只有外部因子、无 gbbq → 回退外部
    raw_b = _raw_daily("600519.SH", n=10)
    store.write_raw("600519.SH", "daily", raw_b)
    store.write_factor("600519.SH", pd.DataFrame({"date": raw_b["date"], "cum_factor": [1.0] * 5 + [2.0] * 5}), gbbq=False)
    hfq_b = store.load("600519.SH", "daily", adjust="hfq", use_gbbq=True)
    assert np.allclose(hfq_b["close"].to_numpy()[5:], raw_b["close"].to_numpy()[5:] * 2.0), "无 gbbq 应回退外部因子(×2)"


def test_datastore_qfq_default_anchor_ignores_future_factor(tmp_path):
    """qfq 默认锚点应取数据末日，不能被因子表中的未来公告事件平移历史价格。"""
    from src.data import DataStore

    store = DataStore(tmp_path)
    raw = _raw_daily("000001.SZ", n=2)
    raw["date"] = pd.to_datetime(["2026-07-01", "2026-07-02"])
    store.write_raw("000001.SZ", "daily", raw)
    store.write_factor(
        "000001.SZ",
        pd.DataFrame({
            "date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-06"]),
            "cum_factor": [1.0, 2.0, 4.0],
        }),
        gbbq=True,
    )

    qfq = store.load("000001.SZ", "daily", adjust="qfq", use_gbbq=True)

    assert qfq["close"].iloc[-1] == pytest.approx(raw["close"].iloc[-1])


# ============================================================
# 零成交但有价 ≠ 停牌（P0-4）
# ============================================================


def test_zero_volume_priced_not_suspended(tmp_path):
    """P0-4：有 close 但零成交（北交所/低流动性真实交易日）不应被判停牌、价不被抹。

    停牌名单关闭时，零成交但有价的行交由名单裁定 → 未确认 → 原始层保留该行、
    加工层 status=trade、is_suspended=False。
    """
    from src.data.fetcher import DataFetcher
    from src.data.trading_calendar import TradingCalendar

    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    cal = TradingCalendar(store_path=tmp_path, auto_load=False)
    cal._trading_days = pd.DatetimeIndex(days)

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__",
                    suspend_enabled=False, calendar=cal)
    f._stock_name_cache["830799.BJ"] = "艾融软件"  # 避免触发网络名称表

    src = pd.DataFrame({
        "date": days,
        "open": [10.0, 10.1, 10.2], "high": [10.0, 10.1, 10.2],
        "low": [10.0, 10.1, 10.2], "close": [10.0, 10.1, 10.2],
        "volume": [1e5, 0.0, 1e5],          # 第 2 日零成交但有价
        "amount": [1e6, 0.0, 1e6], "raw_close": [10.0, 10.1, 10.2],
    })
    raw = f._normalize_daily_raw("830799.BJ", src, source="test")
    assert len(raw) == 3, "零成交真实交易日的行不应被丢"
    meta = f._build_daily_meta("830799.BJ", raw, until=days[-1].date())
    assert list(meta["status"]) == ["trade", "trade", "trade"], "零成交但有价不应被判停牌"
    assert not meta["is_suspended"].any()


def test_should_skip_factor_external_uses_raw_advance(tmp_path):
    """外部源(sina)：触发器按"原始日线是否较因子表新增交易日"判定，与 gbbq 无关。

    回归 P0-1：旧逻辑用 gbbq 事件门控外部源——gbbq 不可用时永不跳过、gbbq 滞后于外部源
    时会误跳过真实更新。新逻辑：因子已覆盖到原始最新交易日 → 跳过；原始新增交易日 → 必刷新。
    """
    from src.data.fetcher import DataFetcher

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__",
                    suspend_enabled=False, factor_source="sina")
    assert not f._gbbq.available  # 离线无 gbbq；旧逻辑此时会一律返回 False（永不跳过）

    raw = _raw_daily("600000.SH", n=10)
    f._store.write_raw("600000.SH", "daily", raw)

    # 因子覆盖到与原始同一最新交易日 → 无新交易日 → 跳过（证明外部源有了独立的正确触发器）
    f._store.write_factor("600000.SH", pd.DataFrame({"date": raw["date"], "cum_factor": [1.0] * 10}))
    assert f._should_skip_factor("600000.SH", gbbq=False) is True

    # 因子只覆盖前 8 日、原始已到第 10 日 → 新增交易日 → 必须刷新（不得误跳过）
    f._store.write_factor("600000.SH", pd.DataFrame({"date": raw["date"].iloc[:8], "cum_factor": [1.0] * 8}))
    assert f._should_skip_factor("600000.SH", gbbq=False) is False


def test_should_skip_factor_no_factor_file_never_skips(tmp_path):
    """本地尚无因子表（无基准）→ 不跳过。"""
    from src.data.fetcher import DataFetcher

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__",
                    suspend_enabled=False, factor_source="sina")
    f._store.write_raw("600000.SH", "daily", _raw_daily("600000.SH", n=5))
    assert f._should_skip_factor("600000.SH", gbbq=False) is False


def test_should_skip_factor_disabled_never_skips(tmp_path):
    """factor_skip_via_gbbq=False → 永不跳过（强制每次刷新）。"""
    from src.data.fetcher import DataFetcher

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__",
                    suspend_enabled=False, factor_source="sina", factor_skip_via_gbbq=False)
    raw = _raw_daily("600000.SH", n=5)
    f._store.write_raw("600000.SH", "daily", raw)
    f._store.write_factor("600000.SH", pd.DataFrame({"date": raw["date"], "cum_factor": [1.0] * 5}))
    assert f._should_skip_factor("600000.SH", gbbq=False) is False


def test_should_skip_factor_gbbq_source_needs_gbbq_file(tmp_path):
    """gbbq 口径但 gbbq 文件不可用 → 不跳过（gbbq 源无触发器依据，绝不退化到外部源逻辑）。"""
    from src.data.fetcher import DataFetcher

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__",
                    suspend_enabled=False, factor_source="gbbq")
    raw = _raw_daily("600000.SH", n=5)
    f._store.write_raw("600000.SH", "daily", raw)
    f._store.write_factor("600000.SH", pd.DataFrame({"date": raw["date"], "cum_factor": [1.0] * 5}))
    assert not f._gbbq.available
    assert f._should_skip_factor("600000.SH", gbbq=False) is False


def test_gbbq_snapshot_only_prewarms_event_cache(tmp_path):
    """无通达信源文件但有 gbbq_events.parquet 时，DataFetcher 应使用快照预热事件缓存。"""
    from src.data.fetcher import DataFetcher

    pd.DataFrame({
        "market": [1],
        "code": ["600519"],
        "date": pd.to_datetime(["2025-06-20"]),
        "category": [1],
        "hongli_panqianliutong": [38.0],
        "peigujia_qianzongguben": [0.0],
        "songgu_qianzongguben": [0.0],
        "peigu_houzongguben": [0.0],
    }).to_parquet(tmp_path / "gbbq_events.parquet", index=False)

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False)

    assert f._gbbq.available is True
    f._prewarm_gbbq_event_cache()
    assert f._gbbq_event_cache["600519.SH"] == pd.Timestamp("2025-06-20")


def test_should_skip_factor_gbbq_future_factor_forces_rebuild(tmp_path):
    """已有 gbbq 因子若超出 raw 覆盖末日，说明含未来事件，应强制重算清理。"""
    from src.data.fetcher import DataFetcher

    pd.DataFrame({
        "market": [0],
        "code": ["000001"],
        "date": pd.to_datetime(["2026-07-06"]),
        "category": [1],
        "hongli_panqianliutong": [1.0],
        "peigujia_qianzongguben": [0.0],
        "songgu_qianzongguben": [0.0],
        "peigu_houzongguben": [0.0],
    }).to_parquet(tmp_path / "gbbq_events.parquet", index=False)

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False)
    raw = _raw_daily("000001.SZ", n=2)
    raw["date"] = pd.to_datetime(["2026-07-01", "2026-07-02"])
    f._store.write_raw("000001.SZ", "daily", raw)
    f._store.write_factor(
        "000001.SZ",
        pd.DataFrame({"date": pd.to_datetime(["2026-07-01", "2026-07-06"]), "cum_factor": [1.0, 1.01]}),
        gbbq=True,
    )

    assert f._gbbq.available is True
    assert f._should_skip_factor("000001.SZ", gbbq=True) is False


# ============================================================
# load_daily(auto_fetch=)：基准指数缺则按需补拉（如 000300.SH）
# ============================================================


def test_load_daily_auto_fetch_triggers_update(tmp_path):
    """本地无该代码 + auto_fetch=True → 调 update 落盘后再读到数据。"""
    from src.data import DataStore
    from src.data.fetcher import DataFetcher

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False)
    called: dict = {}

    def fake_update(codes, freqs=("daily",), **kw):
        called["codes"] = list(codes)
        DataStore(tmp_path).write_raw(codes[0], "daily", _raw_daily(code=codes[0]))

    f.update = fake_update  # type: ignore[method-assign]
    df = f.load_daily("000300.SH", "2024-01-01", "2024-02-01", adjust="none", auto_fetch=True)
    assert called["codes"] == ["000300.SH"]      # 触发了补拉
    assert len(df) > 0                            # 补拉后读到数据


def test_load_daily_auto_fetch_skips_when_covered(tmp_path):
    """本地已覆盖到 end → auto_fetch 不应再触发 update。"""
    from src.data import DataStore
    from src.data.fetcher import DataFetcher

    DataStore(tmp_path).write_raw("000300.SH", "daily", _raw_daily(code="000300.SH", n=10))
    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False)
    called: dict = {}

    def fake_update(codes, freqs=("daily",), **kw):
        called["hit"] = True

    f.update = fake_update  # type: ignore[method-assign]
    # _raw_daily 起于 2024-01-02、10 个工作日，end 落在覆盖区间内
    df = f.load_daily("000300.SH", "2024-01-02", "2024-01-10", adjust="none", auto_fetch=True)
    assert "hit" not in called
    assert len(df) > 0
