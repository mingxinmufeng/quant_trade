"""
数据层（data）

公开子模块（重构后，职责解耦）：
    trading_calendar  交易日历（A 股）
    sources           数据源适配器（只返回不复权 OHLCV）
    factors           复权因子提供器（外部单源）
    adjust            按需复权（none/hfq/qfq + anchor_date）
    resample          周期重采样
    storage           本地仓库（原始/因子分离落盘 + hfq 缓存）
    processor         清洗 / 交易日历对齐 / 面板对齐 / 复权·重采样编排
    universe          动态股票池（防幸存者偏差）
    gbbq              通达信本地权息（复权因子源 + 历史股本序列）
    profile           通达信本地更名史（点位证券简称 / ST）
    fetcher           编排：采集不复权原始数据 + 刷新因子表
"""

from .adjust import apply_adjust, cum_factor_at
from .factors import FactorCalculator, FactorProvider
from .fetcher import (
    DataFetcher,
    DataFetchError,
    ProxyConfigError,
    resample_daily,
    resample_minute,
)
from .gbbq import GbbqStore
from .processor import DataProcessor
from .profile import ProfileStore
from .storage import DAILY_COLUMNS, FREQ_DIRS, MINUTE_COLUMNS, DataStore
from .trading_calendar import TradingCalendar
from .universe import Universe

__all__ = [
    "DAILY_COLUMNS",
    "FREQ_DIRS",
    "MINUTE_COLUMNS",
    "DataFetchError",
    "DataFetcher",
    "DataProcessor",
    "DataStore",
    "FactorCalculator",
    "FactorProvider",
    "GbbqStore",
    "ProfileStore",
    "ProxyConfigError",
    "TradingCalendar",
    "Universe",
    "apply_adjust",
    "cum_factor_at",
    "resample_daily",
    "resample_minute",
]
