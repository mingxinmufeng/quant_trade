"""
数据层（data）

公开子模块（重构后，职责解耦）：
    trading_calendar  交易日历（A 股）
    sources           数据源适配器（只返回不复权 OHLCV）
    factors           复权因子提供器（外部单源）
    adjust            按需复权（none/hfq/qfq + anchor_date）
    resample          周期重采样
    storage           本地仓库（原始/因子分离落盘 + hfq 缓存）
    fetcher           编排：采集不复权原始数据 + 刷新因子表
"""

from .adjust import apply_adjust, cum_factor_at
from .factors import FactorProvider
from .fetcher import (
    DataFetcher,
    DataFetchError,
    ProxyConfigError,
    resample_daily,
    resample_minute,
)
from .storage import DAILY_COLUMNS, FREQ_DIRS, MINUTE_COLUMNS, DataStore
from .trading_calendar import TradingCalendar

__all__ = [
    "DAILY_COLUMNS",
    "FREQ_DIRS",
    "MINUTE_COLUMNS",
    "DataFetchError",
    "DataFetcher",
    "DataStore",
    "FactorProvider",
    "ProxyConfigError",
    "TradingCalendar",
    "apply_adjust",
    "cum_factor_at",
    "resample_daily",
    "resample_minute",
]
