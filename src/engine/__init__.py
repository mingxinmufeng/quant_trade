"""
回测引擎层（engine）

    portfolio    持仓与资金管理（T+1 冻结 / 现金冻结 / 估值 / 已实现盈亏）
    execution    撮合引擎（T+1/涨跌停/滑点/成交量/分红）
    backtester   回测主引擎 + 绩效评估（夏普/回撤/Calmar/胜率/盈亏比/Alpha/Beta）

外部只通过本接口调用。
"""

from .backtester import Backtester, BacktestResult, Trade
from .execution import ExecutionEngine, Order, OrderStatus
from .portfolio import Portfolio, Position

__all__ = [
    "BacktestResult",
    "Backtester",
    "ExecutionEngine",
    "Order",
    "OrderStatus",
    "Portfolio",
    "Position",
    "Trade",
]
