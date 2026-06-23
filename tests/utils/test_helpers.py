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


def test_sharpe_near_constant_returns_safe():
    """P2-14：近似常数收益（浮点微噪声）不应除出天文数字（旧 std==0 精确判等会漏，污染调参）。"""
    from src.utils.helpers import calculate_sharpe

    assert calculate_sharpe([0.0] * 50) == 0.0                      # 完全常数 → 0
    noisy = [1e-18 * ((-1) ** i) for i in range(50)]                # 1e-18 级微噪声
    s = calculate_sharpe(noisy)
    assert np.isfinite(s) and abs(s) < 1.0                          # 有限、不爆炸
    assert calculate_sharpe([0.01, -0.005, 0.02, -0.01, 0.015]) != 0.0  # 正常波动仍正常


def test_sharpe_short_input():
    from src.utils.helpers import calculate_sharpe

    assert calculate_sharpe([]) == 0.0
    assert calculate_sharpe([0.01]) == 0.0
