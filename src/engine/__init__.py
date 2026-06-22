"""
回测引擎层（engine）

    portfolio          持仓与资金管理（T+1 冻结 / 现金冻结 / 估值 / 已实现盈亏）
    execution          撮合引擎（T+1/涨跌停/滑点/成交量/分红；intraday 日内夹价）
    backtester         日级回测主引擎 + 绩效评估（夏普/回撤/Calmar/胜率/盈亏比/Alpha/Beta）
    minute_backtester  分钟级回测引擎（复用 portfolio+execution，按 bar 驱动，防未来函数）

外部只通过本接口调用。
"""

from .backtester import Backtester, BacktestResult, Trade
from .execution import ExecutionEngine, Order, OrderStatus
from .minute_backtester import MinuteBacktester
from .portfolio import Portfolio, Position

__all__ = [
    "BacktestResult",
    "Backtester",
    "ExecutionEngine",
    "MinuteBacktester",
    "Order",
    "OrderStatus",
    "Portfolio",
    "Position",
    "Trade",
]
