"""FactorBase 标准化 / 截面标准化单元测试（零网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factor.base import FactorBase


class _Stub(FactorBase):
    factor_name = "stub"

    def compute(self, data):  # pragma: no cover - 测试桩
        return data


def test_normalize_zscore_cross_section():
    """截面 z-score（index=code）：均值≈0，标准差≈1。"""
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=["A", "B", "C", "D"])
    s.index.name = "code"
    out = _Stub().normalize(s, "zscore")
    assert abs(float(out.mean())) < 1e-9
    assert abs(float(out.std(ddof=0)) - 1.0) < 1e-9


def test_normalize_cross_section_is_per_date_no_lookahead():
    """P1-4：normalize_cross_section 逐交易日独立标准化（无未来函数）。

    两个日期量纲差 10 倍，但逐行截面 z-score 后两行结果应完全相同（仅用当日截面统计量）。
    """
    panel = pd.DataFrame(
        {"A": [1.0, 10.0], "B": [2.0, 20.0], "C": [3.0, 30.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    out = _Stub().normalize_cross_section(panel, "zscore")
    expected = [-1.224744871, 0.0, 1.224744871]
    assert np.allclose(out.iloc[0].to_numpy(), expected)
    assert np.allclose(out.iloc[1].to_numpy(), expected)   # 两日同形 → 未串入跨日(未来)信息
    assert list(out.columns) == ["A", "B", "C"]


def test_normalize_constant_and_allnan_safe():
    """常数列 / 全 NaN 安全返回（不抛、不产生 inf）。"""
    const = pd.Series([5.0, 5.0, 5.0], index=["A", "B", "C"])
    out = _Stub().normalize(const, "zscore")
    assert out.notna().all() and not np.isinf(out.to_numpy()).any()

    allnan = pd.Series([np.nan, np.nan], index=["A", "B"])
    out2 = _Stub().normalize(allnan, "zscore")
    assert out2.isna().all()
