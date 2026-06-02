"""
按需复权（adjust）
==================

把分离落盘的「不复权 OHLCV」与「累计后复权因子表」在加载时合成复权价。
日线 / 分钟线通用，三种模式：

- ``none``：原样返回不复权价（仅补 ``adj_factor`` 列，恒为 1.0 含义上的"原始"）。
- ``hfq`` ：后复权。``价_hfq = 价_raw × cum_factor``。
- ``qfq`` ：前复权。``价_qfq = 价_raw × cum_factor / cum_factor(anchor)``
            = ``价_hfq / cum_factor(anchor)``，即 hfq 除以锚点日的累计因子（标量）。
            ``anchor_date=None`` 时取因子表最后一日（最新口径，会随新分红平移历史）；
            **回测可复现性**：传入固定 ``anchor_date`` 可锁定基准，新分红不再改变
            该日之前的前复权价格。

因子边界对齐
------------
用 ``merge_asof(direction="backward")`` 把因子按日期对齐到行。早于因子表最早一日
的行会得到 NaN —— **用 bfill 补成最早一个因子值**（而非 1.0），保证 hfq 早期段
刻度与后续连续；若因子表整体为空则退化为 1.0（等价不复权）。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Union

import pandas as pd

from ..utils.helpers import parse_date

#: 按因子缩放的价格列（OHLC + 涨跌停价，保证与价格同口径；volume/amount 不缩放）
_PRICE_COLS = ("open", "high", "low", "close", "limit_up", "limit_down")

#: 支持的复权模式
ADJUST_MODES = ("none", "hfq", "qfq")


def align_cum_factor(time_series: pd.Series, factor: pd.DataFrame) -> pd.Series:
    """把因子表 ``DataFrame[date, cum_factor]`` 按 backward 对齐到 ``time_series``。

    返回与 ``time_series`` **同索引、同顺序**的累计因子（早于最早因子日的行 bfill
    成最早因子值；因子表为空则全 1.0）。
    """
    idx = time_series.index
    if factor is None or factor.empty:
        return pd.Series(1.0, index=idx, dtype="float64")
    d = pd.to_datetime(time_series).dt.normalize().astype("datetime64[ns]")
    left = pd.DataFrame({"_d": d.to_numpy()}, index=idx)
    f = factor.rename(columns={"date": "_d", "cum_factor": "_f"}).loc[:, ["_d", "_f"]].copy()
    f["_d"] = pd.to_datetime(f["_d"]).dt.normalize().astype("datetime64[ns]")
    f["_f"] = pd.to_numeric(f["_f"], errors="coerce")
    f = f.dropna(subset=["_d", "_f"]).sort_values("_d")
    if f.empty:
        return pd.Series(1.0, index=idx, dtype="float64")
    left_sorted = left.sort_values("_d")
    merged = pd.merge_asof(left_sorted, f, on="_d", direction="backward")
    merged.index = left_sorted.index
    out = merged["_f"].reindex(idx)
    # 早于最早因子日 → bfill 成最早因子值（不退化为 1.0，保证早期刻度连续）
    earliest = float(f["_f"].iloc[0])
    return out.fillna(earliest).astype("float64")


def cum_factor_at(factor: pd.DataFrame, anchor_date: Optional[Union[str, date, datetime]]) -> float:
    """取锚点日（含之前最近一个因子日）的累计因子；anchor=None 取最后一日。空表返回 1.0。"""
    if factor is None or factor.empty:
        return 1.0
    f = factor.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f["cum_factor"] = pd.to_numeric(f["cum_factor"], errors="coerce")
    f = f.dropna(subset=["date", "cum_factor"]).sort_values("date")
    if f.empty:
        return 1.0
    if anchor_date is None:
        return float(f["cum_factor"].iloc[-1])
    anchor = pd.Timestamp(parse_date(anchor_date)).normalize()
    prior = f[f["date"] <= anchor]
    if prior.empty:
        return float(f["cum_factor"].iloc[0])  # 锚点早于上市：用最早因子
    return float(prior["cum_factor"].iloc[-1])


def apply_adjust(
    df: pd.DataFrame,
    factor: pd.DataFrame,
    mode: str = "hfq",
    *,
    time_col: Optional[str] = None,
    anchor_date: Optional[Union[str, date, datetime]] = None,
) -> pd.DataFrame:
    """把不复权 ``df`` 按 ``factor`` 复权，返回新 df（含 ``adj_factor`` 列）。

    Args:
        df: 不复权行情（含 OHLC + 时间列）。
        factor: ``DataFrame[date, cum_factor]``；空表等价不复权。
        mode: ``none`` / ``hfq`` / ``qfq``。
        time_col: 时间列名；None 时自动探测（``date`` 或 ``datetime``）。
        anchor_date: 仅 ``qfq`` 用；None=因子表最后一日。

    ``adj_factor`` 列语义：实际作用于价格的乘数。
      - none → 1.0
      - hfq  → cum_factor
      - qfq  → cum_factor / cum_factor(anchor)
    """
    mode = (mode or "hfq").strip().lower()
    if mode not in ADJUST_MODES:
        raise ValueError(f"未知复权模式 {mode!r}（支持 {ADJUST_MODES}）")
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        if out is not None and not out.empty:
            out["adj_factor"] = 1.0
        return out

    out = df.copy()
    if time_col is None:
        time_col = "date" if "date" in out.columns else "datetime"

    if mode == "none":
        out["adj_factor"] = 1.0
        return out

    cum = align_cum_factor(out[time_col], factor)
    if mode == "qfq":
        anchor_f = cum_factor_at(factor, anchor_date)
        mult = cum / anchor_f if anchor_f else cum
    else:  # hfq
        mult = cum

    mult = mult.to_numpy(dtype="float64")
    for c in _PRICE_COLS:
        if c in out.columns:
            out[c] = (pd.to_numeric(out[c], errors="coerce").to_numpy(dtype="float64") * mult)
    out["adj_factor"] = mult
    return out
