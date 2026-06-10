"""
风控层（risk）

    risk_manager  仓位控制 / 止损 / 手续费 / 印花税

外部只通过本接口调用。撮合引擎通过 ``apply_commission`` 计费，回测/实盘循环通过
``check_order`` / ``clip_position_size`` / ``check_drawdown_stop`` 施加风控。
"""

from .risk_manager import RiskManager

__all__ = ["RiskManager"]
