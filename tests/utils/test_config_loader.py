"""config_loader 工具函数单元测试（零网络）。"""

from __future__ import annotations

from src.utils.config_loader import _coerce_env_value


def test_coerce_preserves_leading_zero_codes():
    """P1-5：前导零纯数字串保留为字符串（股票代码/版本号），不丢前导零。"""
    assert _coerce_env_value("000001") == "000001"
    assert _coerce_env_value("007") == "007"
    assert _coerce_env_value("600519") == 600519        # 无前导零仍转 int
    assert _coerce_env_value("0") == 0                  # 单个 0 仍是 int 0
    assert _coerce_env_value("-5") == -5
    assert _coerce_env_value("+12") == 12


def test_coerce_other_types_unchanged():
    """常规类型推断不受影响。"""
    assert _coerce_env_value("3.14") == 3.14
    assert _coerce_env_value("true") is True
    assert _coerce_env_value("false") is False
    assert _coerce_env_value("none") is None
    assert _coerce_env_value("[1, 2]") == [1, 2]
    assert _coerce_env_value("tushare") == "tushare"
