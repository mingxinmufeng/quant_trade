"""helpers 绩效指标 / 工具函数边界单元测试（零网络）。"""

from __future__ import annotations

import numpy as np


def test_max_drawdown_normal():
    from src.utils.helpers import calculate_max_drawdown

    assert calculate_max_drawdown([1.0, 1.2, 0.9, 1.1]) < 0
    assert abs(calculate_max_drawdown([1.0, 0.5]) - (-0.5)) < 1e-9


def test_max_drawdown_nonpositive_equity_safe():
    """P2-14：净值含 0/负时不产生 inf/nan（守卫按位置回撤记 0）。"""
    from src.utils.helpers import calculate_max_drawdown

    for curve in ([0.0, 1.0, 0.5], [0.0, 0.0], [-1.0, -2.0], [1.0, 0.0, -0.5]):
        mdd = calculate_max_drawdown(curve)
        assert np.isfinite(mdd), f"{curve} → {mdd} 非有限值"


def test_max_drawdown_short_input():
    from src.utils.helpers import calculate_max_drawdown

    assert calculate_max_drawdown([]) == 0.0
    assert calculate_max_drawdown([1.0]) == 0.0
