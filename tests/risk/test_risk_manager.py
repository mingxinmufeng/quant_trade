"""RiskManager 参数校验 + 费用/裁减 单元测试（零网络）。"""

from __future__ import annotations

import pytest


def test_param_validation_rejects_out_of_range():
    """P2-16：构造期校验——负费率 / 比例 >1 / 负最低佣金 应抛 ValueError。"""
    from src.risk import RiskManager

    with pytest.raises(ValueError):
        RiskManager(commission_rate=-0.001)
    with pytest.raises(ValueError):
        RiskManager(commission_rate=2.5)            # 误把"万2.5"写成 2.5
    with pytest.raises(ValueError):
        RiskManager(max_single_position=1.5)
    with pytest.raises(ValueError):
        RiskManager(total_drawdown_stop=-0.1)
    with pytest.raises(ValueError):
        RiskManager(min_commission=-1.0)


def test_param_validation_allows_boundary_values():
    """边界 0 / 1 合法：熔断线 0=立即停盘、止损线 1=关闭、仓位上限 0/1。"""
    from src.risk import RiskManager

    RiskManager(total_drawdown_stop=0.0)            # 立即熔断
    RiskManager(daily_stop_loss=1.0)                # 实质关闭单日止损
    RiskManager(max_single_position=1.0, max_industry_position=1.0)


def test_apply_commission_buy_sell_min():
    """费用：买入仅佣金、卖出含印花、低额走最低佣金。"""
    from src.risk import RiskManager

    rm = RiskManager()
    assert rm.apply_commission(100_000, is_buy=True) == pytest.approx(100_000 * 0.00025)
    assert rm.apply_commission(100_000, is_buy=False) == pytest.approx(100_000 * (0.00025 + 0.0005))
    assert rm.apply_commission(1000, is_buy=True) == 5.0   # 最低佣金
