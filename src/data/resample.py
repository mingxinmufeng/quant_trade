"""
周期重采样（resample）
======================

只落盘三套原始周期：日线 / 5 分钟 / 1 分钟。其余周期在 load 时由此模块按需生成：
- 日线以上（周/月/季/年）由**日线** resample；
- 分钟周期由 5 分钟（5 的倍数）或 1 分钟生成。
"""

from __future__ import annotations

import pandas as pd

#: 支持 resample 落地的「日线以上」周期 → pandas 偏移别名
#: 周线锚到周五（A 股周线惯例为当周最后交易日，W-FRI 比默认 W=W-SUN 更贴近，
#: 避免 bar 日期落在周日这一非交易日）；月/季/年用期末（ME/QE/YE）。
DAILY_RESAMPLE_RULES = {
    "weekly": "W-FRI",
    "monthly": "ME",
    "quarterly": "QE",
    "yearly": "YE",
}


def resample_daily(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """日线 → 周/月/季/年线（OHLCV 聚合）。``period`` 见 ``DAILY_RESAMPLE_RULES``。"""
    if period == "daily":
        return df.reset_index(drop=True)
    rule = DAILY_RESAMPLE_RULES.get(period)
    if rule is None:
        raise ValueError(f"不支持的日线周期: {period}（支持 {list(DAILY_RESAMPLE_RULES)}）")
    if df.empty:
        return df
    g = df.set_index("date").resample(rule)
    cols = {
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(min_count=1),
        "amount": g["amount"].sum(min_count=1),
    }
    if "adj_factor" in df.columns:
        cols["adj_factor"] = g["adj_factor"].last()
    out = pd.DataFrame(cols).dropna(subset=["close"]).reset_index()
    if "code" in df.columns and not df.empty:
        out["code"] = df["code"].iloc[0]
    return out


def resample_minute(df: pd.DataFrame, base_period: int, target_period: int) -> pd.DataFrame:
    """
    分钟线升采样：按「每日内连续 k 根」分组聚合（k = target/base），符合通达信对齐习惯。
    例：5 分钟 → 15 分钟（k=3）；1 分钟 → 3 分钟（k=3）。bar 时间取该组最后一根。
    """
    if df.empty:
        return df
    if target_period == base_period:
        return df.reset_index(drop=True)
    if target_period % base_period != 0:
        raise ValueError(f"{target_period} 分钟无法由 {base_period} 分钟整除生成")
    k = target_period // base_period
    df = df.sort_values("datetime").reset_index(drop=True)
    day = df["datetime"].dt.normalize()
    grp_idx = df.groupby(day).cumcount() // k
    keys = [day, grp_idx]
    g = df.groupby(keys, sort=True)
    out = pd.DataFrame({
        "datetime": g["datetime"].last().values,
        "open": g["open"].first().values,
        "high": g["high"].max().values,
        "low": g["low"].min().values,
        "close": g["close"].last().values,
        "volume": g["volume"].sum(min_count=1).values,
        "amount": g["amount"].sum(min_count=1).values,
    })
    if "adj_factor" in df.columns:
        out["adj_factor"] = g["adj_factor"].last().values
    if "code" in df.columns:
        out["code"] = df["code"].iloc[0]
    return out.sort_values("datetime").reset_index(drop=True)
