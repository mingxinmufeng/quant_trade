"""
公共测试 fixtures（全部离线，零网络）
=====================================

把仓库根加入 sys.path，提供合成行情、临时数据仓库、离线交易日历等共享夹具，供
``tests/`` 下各子目录测试复用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture
def make_ohlc():
    """工厂：生成单只股票的合成日线 OHLCV（含涨跌停/停牌列）。"""

    def _make(
        code: str = "000001.SZ",
        n: int = 120,
        start: str = "2024-01-01",
        seed: int = 0,
        v_shape: bool = True,
    ) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        if v_shape:
            trend = np.concatenate([np.linspace(20, 14, n // 2), np.linspace(14, 28, n - n // 2)])
        else:
            trend = np.linspace(15, 25, n)
        close = pd.Series(trend + rng.randn(n) * 0.3).clip(lower=1.0)
        return pd.DataFrame(
            {
                "date": pd.date_range(start, periods=n, freq="B"),
                "code": code,
                "open": close.shift(1).fillna(close.iloc[0]).astype("float64"),
                "high": (close * 1.03).astype("float64"),
                "low": (close * 0.97).astype("float64"),
                "close": close.astype("float64"),
                "volume": pd.Series(rng.randint(int(5e6), int(1e7), n)).astype("float64"),
                "amount": (close * 1e6).astype("float64"),
                "limit_up": (close * 1.1).astype("float64"),
                "limit_down": (close * 0.9).astype("float64"),
                "is_suspended": False,
            }
        )

    return _make


@pytest.fixture
def sample_data(make_ohlc):
    """两只股票的合成数据字典（回测用）。"""
    return {
        "000001.SZ": make_ohlc("000001.SZ", seed=1),
        "600519.SH": make_ohlc("600519.SH", seed=2),
    }


@pytest.fixture
def offline_calendar(tmp_path):
    """把一段工作日写入 ``calendar.parquet`` 并返回 TradingCalendar（不联网）。"""
    from src.data import TradingCalendar

    days = pd.date_range("2020-01-01", "2025-12-31", freq="B")
    pd.DataFrame({"date": days}).to_parquet(tmp_path / "calendar.parquet", index=False)
    return TradingCalendar(store_path=tmp_path, refresh_days=10_000, auto_load=True)
