"""
内置示例策略包（examples）

永远公开，作为框架用法范例与 ``loader`` 的回退来源。私有 alpha 策略请放外部仓库，
通过 ``strategy.loader.load_strategy(..., external_path=...)`` 加载。
"""

from .ma_rsi import MaRsiStrategy

__all__ = ["MaRsiStrategy"]
