"""
因子层（factor）

框架公开、具体 alpha 因子私有。本层提供：
    base         FactorBase 抽象基类（计算 + 标准化 + 面板编排）
    technical    纯技术指标因子（MA/RSI/MACD/布林带/ATR/量能等）
    fundamental  通用财务因子框架（估值/质量/成长/杠杆/规模 + 自定义比率）
"""

from .base import FactorBase
from .fundamental import FundamentalFactor, FundamentalFactors
from .technical import TechnicalFactor, TechnicalFactors

__all__ = [
    "FactorBase",
    "FundamentalFactor",
    "FundamentalFactors",
    "TechnicalFactor",
    "TechnicalFactors",
]
