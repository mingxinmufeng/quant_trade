"""optimize 的 walk-forward 窗口切分单元测试（纯函数，零网络/无 optuna 依赖）。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.main import _walk_forward_windows


def _days(n):
    return [d.date() for d in pd.bdate_range("2024-01-01", periods=n)]


def test_walk_forward_expanding_no_leak():
    """folds 折：训练段锚定起点逐折扩张、严格早于测试段（无未来泄漏），测试段连续非重叠到末日。"""
    days = _days(12)
    w = _walk_forward_windows(days, 3)
    assert len(w) == 3
    for tr_s, tr_e, te_s, te_e in w:
        assert tr_s <= tr_e < te_s <= te_e          # 训练严格早于测试
    assert all(win[0] == days[0] for win in w)        # 训练锚定起点
    assert w[0][1] < w[1][1] < w[2][1]                # 训练段逐折扩张
    assert w[0][3] < w[1][2] and w[1][3] < w[2][2]    # 测试段非重叠
    assert w[-1][3] == days[-1]                        # 末折测试到末日


def test_walk_forward_single_fold_is_split():
    """folds=1 → 单次五五分（前半训练、后半测试）。"""
    days = _days(10)
    w = _walk_forward_windows(days, 1)
    assert len(w) == 1
    tr_s, tr_e, te_s, te_e = w[0]
    assert tr_s == days[0] and te_e == days[-1] and tr_e < te_s


def test_walk_forward_rejects_too_few_days():
    with pytest.raises(ValueError):
        _walk_forward_windows(_days(4), 3)


def test_walk_forward_rejects_bad_folds():
    with pytest.raises(ValueError):
        _walk_forward_windows(_days(20), 0)
