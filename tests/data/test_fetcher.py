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


# ============================================================
# 逐行点位 ST 涨跌停限幅（_limit_pct_series / _post_process / recompute_limits）
# ============================================================


def _offline_fetcher(tmp_path, code, current_name, profile_rows=None):
    """构造离线 DataFetcher：禁停牌、注入合成 profile 更名表与当前名缓存（零网络）。"""
    from src.data.fetcher import DataFetcher
    from src.data.profile import ProfileStore

    f = DataFetcher(
        store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False,
    )
    DataFetcher._STOCK_NAME_CACHE[code] = current_name  # 预置避免触发网络名称表
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
    """创业板/科创板 ±20%、北交所 ±30%：即便点位名带 ST 也不受影响。"""
    for code, current, expect in [
        ("300001.SZ", "ST特锐", 0.20),
        ("688001.SH", "ST华兴", 0.20),
        ("830799.BJ", "ST艾融", 0.30),
    ]:
        f = _offline_fetcher(
            tmp_path, code, current,
            profile_rows=[(code, "ST曾用", "2010-01-01")],
        )
        dates = pd.Series(pd.to_datetime(["2009-06-01", "2025-01-01"]))
        pct = f._limit_pct_series(code, dates)
        assert list(pct) == [expect, expect], code


def test_limit_pct_series_fallback_current_name(tmp_path):
    """profile 不可用时主板回退当前名近似：当前名带 ST → 整段 ±5%。"""
    f = _offline_fetcher(tmp_path, "600000.SH", "ST浦发", profile_rows=None)
    assert not f._profile.available
    dates = pd.Series(pd.to_datetime(["2009-06-01", "2025-01-01"]))
    pct = f._limit_pct_series("600000.SH", dates)
    assert list(pct) == [0.05, 0.05]


def test_backfill_limit_df_with_pct_series(tmp_path):
    """_backfill_limit_df 接受逐行 pct Series：仅回补 NaN，按行日期取对应限幅。"""
    from src.data.fetcher import _backfill_limit_df

    f = _offline_fetcher(
        tmp_path, "600000.SH", "浦发银行",
        profile_rows=[("600000.SH", "ST浦发", "2010-01-01")],
    )
    df = pd.DataFrame({
        "date": pd.to_datetime(["2009-12-30", "2009-12-31", "2010-01-04"]),
        "close": [10.0, 11.0, 12.0],
        "limit_up": [np.nan, np.nan, np.nan],
        "limit_down": [np.nan, np.nan, np.nan],
    })
    pct = f._limit_pct_series("600000.SH", df["date"])
    out = _backfill_limit_df(df.copy(), pct)
    assert pd.isna(out.loc[0, "limit_up"])              # 首行无前收
    assert np.isclose(out.loc[1, "limit_up"], 10.5)     # ST 段 ±5%，前收 10
    assert np.isclose(out.loc[2, "limit_up"], 12.1)     # 常规 ±10%，前收 11
    assert np.isclose(out.loc[2, "limit_down"], 9.9)


def test_recompute_limits_migrates_historical_st(tmp_path):
    """recompute_limits 用逐行点位 ST 重算已落盘日线限幅，覆盖旧的当前名近似值。"""
    f = _offline_fetcher(
        tmp_path, "600000.SH", "浦发银行",
        profile_rows=[("600000.SH", "ST浦发", "2010-01-01")],
    )
    dates = pd.to_datetime(["2009-12-30", "2009-12-31", "2010-01-04", "2010-01-05"])
    close = pd.Series([10.0, 11.0, 12.0, 13.0])
    raw = pd.DataFrame({
        "date": dates, "code": "600000.SH",
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e5, "amount": 1e6, "is_suspended": False,
        # 旧值：整段按当前名（非 ST）±10% 近似
        "limit_up": (close.shift(1) * 1.1).round(2),
        "limit_down": (close.shift(1) * 0.9).round(2),
        "name": "浦发银行", "source": "test",
    })
    f._store.write_raw("600000.SH", "daily", raw)

    n = f.recompute_limits(["600000.SH"])
    assert n == 1
    got = f._store.read_raw("600000.SH", "daily").sort_values("date").reset_index(drop=True)
    assert pd.isna(got.loc[0, "limit_up"])              # 首行无前收，保持原值（NaN）
    assert np.isclose(got.loc[1, "limit_up"], 10.5)     # 2009-12-31 ST → ±5%，前收 10
    assert np.isclose(got.loc[1, "limit_down"], 9.5)
    assert np.isclose(got.loc[2, "limit_up"], 12.1)     # 2010-01-04 常规 → ±10%，前收 11
    assert np.isclose(got.loc[3, "limit_up"], 13.2)     # 前收 12


def test_name_table_prefers_local_tushare(tmp_path):
    """名称表优先读本地 tushare 基础信息缓存（离线，不联网东财）。"""
    from src.data.fetcher import STOCK_BASIC_TUSHARE_FILE, DataFetcher

    pd.DataFrame({
        "ts_code": ["600000.SH", "000004.SZ", "830799.BJ"],
        "name": ["浦发银行", "*ST国华", "艾融软件"],
    }).to_parquet(tmp_path / STOCK_BASIC_TUSHARE_FILE, index=False)

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False)
    DataFetcher._STOCK_NAME_CACHE.clear()  # 强制从本地源重建
    assert f._load_names_from_local() is True
    assert f._get_stock_name("600000.SH") == "浦发银行"
    assert f._get_stock_name("000004.SZ") == "*ST国华"


def test_name_table_local_missing_falls_back(tmp_path):
    """本地基础信息缺失 → _load_names_from_local 返回 False（交由东财降级源）。"""
    from src.data.fetcher import DataFetcher

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__", suspend_enabled=False)
    DataFetcher._STOCK_NAME_CACHE.clear()
    assert f._load_names_from_local() is False


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


# ============================================================
# _post_process_daily：零成交但有价 ≠ 停牌（P0-4）
# ============================================================


def test_post_process_zero_volume_priced_not_suspended(tmp_path):
    """P0-4：有 close 但零成交（北交所/低流动性真实交易日）不应被判停牌、价不被抹。

    停牌名单关闭时，零成交但有价的行交由名单裁定 → 未确认 → is_suspended=False、
    close 保留、来源不打 :gap（旧逻辑会因 volume<=0 直接判停牌并标 :gap）。
    """
    from src.data.fetcher import DataFetcher
    from src.data.trading_calendar import TradingCalendar

    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    cal = TradingCalendar(store_path=tmp_path, auto_load=False)
    cal._trading_days = pd.DatetimeIndex(days)

    f = DataFetcher(store_path=tmp_path, tdx_path="__nonexistent__",
                    suspend_enabled=False, calendar=cal)
    DataFetcher._STOCK_NAME_CACHE["830799.BJ"] = "艾融软件"  # 避免触发网络名称表

    raw = pd.DataFrame({
        "date": days,
        "open": [10.0, 10.1, 10.2], "high": [10.0, 10.1, 10.2],
        "low": [10.0, 10.1, 10.2], "close": [10.0, 10.1, 10.2],
        "volume": [1e5, 0.0, 1e5],          # 第 2 日零成交但有价
        "amount": [1e6, 0.0, 1e6], "raw_close": [10.0, 10.1, 10.2],
    })
    out = f._post_process_daily("830799.BJ", raw, days[0].date(), days[-1].date(), source="test")
    out = out.sort_values("date").reset_index(drop=True)

    assert bool(out.loc[1, "is_suspended"]) is False, "零成交但有价不应被判停牌"
    assert np.isclose(out.loc[1, "close"], 10.1), "零成交真实交易日的价格不应被抹"
    assert not str(out.loc[1, "source"]).endswith(":gap"), "真实数据行不应被标 :gap（否则会被尾部裁剪）"


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
