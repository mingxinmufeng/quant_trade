"""
策略层（strategy）

基类与示例公开，私有策略通过外部加载器接入：
    base      BaseStrategy 抽象基类 + Signal 枚举
    loader    外部私有策略加载器（load_strategy / create_strategy）
    examples  示例策略（ma_rsi 双均线+RSI）

外部只通过本接口调用。私有 alpha 策略放外部仓库，经 ``load_strategy(..., external_path=...)`` 加载。
"""

from .base import BaseStrategy, Signal
from .examples.ma_rsi import MaRsiStrategy
from .loader import StrategyLoadError, create_strategy, list_examples, load_strategy

__all__ = [
    "BaseStrategy",
    "MaRsiStrategy",
    "Signal",
    "StrategyLoadError",
    "create_strategy",
    "list_examples",
    "load_strategy",
]
