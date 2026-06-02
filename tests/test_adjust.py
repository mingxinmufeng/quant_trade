"""
按需复权单元测试（无网络 / 无 IO）
====================================

覆盖 ``src.data.adjust``：
  - ``align_cum_factor``   因子 backward 对齐 + 早期段 bfill
  - ``cum_factor_at``      锚点日累计因子
  - ``apply_adjust``       none / hfq / qfq 三模式 + 涨跌停同步缩放
以及 ``src.data.fetcher._backfill_limit_df`` 涨跌停首行回补。

运行：
    python -m pytest tests/test_adjust.py -q
或直接：
    python tests/test_adjust.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.adjust import (  # noqa: E402
    align_cum_factor,
    apply_adjust,
    cum_factor_at,
)
from src.data.fetcher import _backfill_limit_df  # noqa: E402


def _daily(dates, closes, factor_cols=None):
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [100.0] * len(dates), "amount": [1000.0] * len(dates),
    })
    return df


def _factor(dates, vals):
    return pd.DataFrame({"date": pd.to_datetime(dates), "cum_factor": vals})


# ------------------------------------------------------------
# align_cum_factor
# ------------------------------------------------------------

def test_align_backward_and_bfill():
    """因子 backward 对齐；早于最早因子日的行 bfill 成最早因子值（非 1.0）。"""
    ts = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]))
    f = _factor(["2024-01-02", "2024-01-04"], [2.0, 3.0])
    out = align_cum_factor(ts, f)
    # 2024-01-01 早于最早因子日 → bfill 成 2.0；01-03→2.0；01-05→3.0
    assert list(out.values) == [2.0, 2.0, 3.0], out.values


def test_align_empty_factor_is_one():
    ts = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    out = align_cum_factor(ts, pd.DataFrame(columns=["date", "cum_factor"]))
    assert list(out.values) == [1.0, 1.0]


def test_align_preserves_order_when_unsorted():
    ts = pd.Series(pd.to_datetime(["2024-01-05", "2024-01-01", "2024-01-03"]))
    f = _factor(["2024-01-02", "2024-01-04"], [2.0, 3.0])
    out = align_cum_factor(ts, f)
    assert list(out.values) == [3.0, 2.0, 2.0], out.values


# ------------------------------------------------------------
# cum_factor_at
# ------------------------------------------------------------

def test_cum_factor_at_anchor():
    f = _factor(["2024-01-02", "2024-01-04", "2024-01-06"], [2.0, 3.0, 4.0])
    assert cum_factor_at(f, None) == 4.0           # 默认取最后一日
    assert cum_factor_at(f, "2024-01-04") == 3.0   # 锚点当日
    assert cum_factor_at(f, "2024-01-05") == 3.0   # 锚点取之前最近
    assert cum_factor_at(f, "2024-01-01") == 2.0   # 锚点早于上市 → 最早因子
    assert cum_factor_at(pd.DataFrame(columns=["date", "cum_factor"]), None) == 1.0


# ------------------------------------------------------------
# apply_adjust
# ------------------------------------------------------------

def test_apply_none_keeps_raw():
    df = _daily(["2024-01-02", "2024-01-03"], [10.0, 11.0])
    f = _factor(["2024-01-02"], [2.0])
    out = apply_adjust(df, f, mode="none")
    assert list(out["close"]) == [10.0, 11.0]
    assert list(out["adj_factor"]) == [1.0, 1.0]


def test_apply_hfq():
    df = _daily(["2024-01-02", "2024-01-03"], [10.0, 11.0])
    f = _factor(["2024-01-02"], [2.0])
    out = apply_adjust(df, f, mode="hfq")
    assert list(out["close"]) == [20.0, 22.0]
    assert list(out["adj_factor"]) == [2.0, 2.0]


def test_apply_qfq_default_anchor():
    """qfq 默认锚定最后一日：最后一日价格 == 原始价。"""
    df = _daily(["2024-01-02", "2024-01-04"], [10.0, 12.0])
    f = _factor(["2024-01-02", "2024-01-04"], [2.0, 3.0])
    out = apply_adjust(df, f, mode="qfq")
    # cum=[2,3], anchor=3 → mult=[2/3, 1] → close=[10*2/3, 12*1]
    assert abs(out["close"].iloc[0] - 10.0 * 2 / 3) < 1e-9
    assert abs(out["close"].iloc[1] - 12.0) < 1e-9


def test_apply_qfq_fixed_anchor_reproducible():
    """固定锚点：新增因子不改变锚点前价格。"""
    df = _daily(["2024-01-02", "2024-01-04"], [10.0, 12.0])
    f = _factor(["2024-01-02", "2024-01-04"], [2.0, 3.0])
    out = apply_adjust(df, f, mode="qfq", anchor_date="2024-01-02")
    # anchor=2 → mult=[1, 1.5]
    assert abs(out["close"].iloc[0] - 10.0) < 1e-9
    assert abs(out["close"].iloc[1] - 18.0) < 1e-9


def test_apply_scales_limit_columns():
    """涨跌停价随价格同口径缩放（hfq）。"""
    df = _daily(["2024-01-02"], [10.0])
    df["limit_up"] = 11.0
    df["limit_down"] = 9.0
    f = _factor(["2024-01-02"], [2.0])
    out = apply_adjust(df, f, mode="hfq")
    assert out["limit_up"].iloc[0] == 22.0
    assert out["limit_down"].iloc[0] == 18.0


def test_apply_minute_time_col_autodetect():
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-02 09:35", "2024-01-02 09:40"]),
        "open": [10.0, 10.0], "high": [10.0, 10.0], "low": [10.0, 10.0],
        "close": [10.0, 10.0], "volume": [1.0, 1.0], "amount": [10.0, 10.0],
    })
    f = _factor(["2024-01-02"], [2.0])
    out = apply_adjust(df, f, mode="hfq")
    assert list(out["close"]) == [20.0, 20.0]


# ------------------------------------------------------------
# _backfill_limit_df
# ------------------------------------------------------------

def test_backfill_limit_first_row():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "close": [10.0, 11.0, 12.0],
        "limit_up": [np.nan, 12.1, 13.2],
        "limit_down": [np.nan, 9.9, 10.8],
    })
    out = _backfill_limit_df(df, 0.10)
    # 首行无前收 → 保持 NaN；第 2 行用第 1 行 close=10 反推
    assert pd.isna(out["limit_up"].iloc[0])
    assert abs(out["limit_up"].iloc[1] - 12.1) < 1e-9


def _run_all():
    import inspect
    fns = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and inspect.isfunction(obj)
    ]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n全部 {len(fns)} 个测试通过")


if __name__ == "__main__":
    _run_all()
