"""
数据层（data）

公开子模块：
    calendar    交易日历（A 股）
    fetcher     行情数据拉取（多源容灾）            [待生成]
    processor   数据清洗 / 复权 / 对齐               [待生成]
    universe    动态股票池（防幸存者偏差）           [待生成]

注：本 __init__.py 当前仅做 placeholder；统一对外导出（DataFetcher 等）
将在所有子模块完成后于第 12 步最终装配。
"""

from .trading_calendar import TradingCalendar

__all__ = ["TradingCalendar"]
