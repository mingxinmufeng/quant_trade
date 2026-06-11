"""
示例策略：双均线 + RSI（MaRsiStrategy）
========================================

经典趋势跟随 + 超买过滤的组合，演示如何基于 :class:`~src.strategy.base.BaseStrategy`
与因子层（:class:`~src.factor.technical.SMAFactor` / :class:`RSIFactor`）实现一个完整、
可回测、可调参（Optuna）的策略。**公开示例，非投资建议。**

参数（默认值见 ``default_params``）
-----------------------------------
- ``fast_period`` (int=10)  快速均线周期
- ``slow_period`` (int=30)  慢速均线周期
- ``rsi_period``  (int=14)  RSI 周期
- ``rsi_upper``   (float=70) RSI 超买阈值
- ``rsi_lower``   (float=30) RSI 超卖阈值（保留参数，便于扩展/调参）

信号逻辑（逐只股票，按交易日）
------------------------------
- **BUY**：快线**上穿**慢线（前一日 ``fast<slow`` 且当日 ``fast>slow``）**且** ``RSI < rsi_upper``；
- **SELL**：快线**下穿**慢线 **或** ``RSI > rsi_upper``；
- **HOLD**：其余。

防未来函数
----------
``T`` 日信号仅用 ``<= T`` 的收盘价计算（均线/RSI 在 ``T`` 日收盘后可得，穿越用 ``T`` 与
``T-1``）。回测引擎在 ``T+1`` 开盘撮合，故本策略不引入未来数据。
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
from loguru import logger

from ...factor.technical import RSIFactor, SMAFactor
from ..base import BaseStrategy, Signal

__all__ = ["MaRsiStrategy"]


class MaRsiStrategy(BaseStrategy):
    """双均线 + RSI 示例策略。"""

    strategy_name = "ma_rsi"

    default_params: ClassVar[dict[str, Any]] = {
        "fast_period": 10,
        "slow_period": 30,
        "rsi_period": 14,
        "rsi_upper": 70.0,
        "rsi_lower": 30.0,
    }

    # ------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not data:
            return self.validate_signals(pd.DataFrame())

        if int(self.fast_period) >= int(self.slow_period):
            logger.warning(
                f"[{self.strategy_name}] fast_period({self.fast_period}) >= "
                f"slow_period({self.slow_period})，双均线退化，请检查参数"
            )

        per_stock: dict[str, pd.Series] = {}
        for code, df in data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            per_stock[code] = self._signals_one(df)

        if not per_stock:
            return self.validate_signals(pd.DataFrame())

        # 各股按日期对齐拼成信号矩阵；缺口由 validate_signals 填 HOLD
        signals = pd.DataFrame(per_stock)
        return self.validate_signals(signals)

    def _signals_one(self, df: pd.DataFrame) -> pd.Series:
        """单只股票的信号序列（index=date, values ∈ {-1,0,1}）。"""
        fast = SMAFactor(int(self.fast_period)).compute(df)
        slow = SMAFactor(int(self.slow_period)).compute(df)
        rsi = RSIFactor(int(self.rsi_period)).compute(df)

        prev_fast, prev_slow = fast.shift(1), slow.shift(1)
        cross_up = (prev_fast < prev_slow) & (fast > slow)
        cross_down = (prev_fast > prev_slow) & (fast < slow)

        buy = cross_up & (rsi < float(self.rsi_upper))
        sell = cross_down | (rsi > float(self.rsi_upper))

        sig = pd.Series(int(Signal.HOLD), index=fast.index, dtype="int64")
        # 由构造可证 buy 与 sell 互斥；为稳健起见显式先 BUY 再 SELL（如有冲突以 SELL 为先，风控优先）
        sig[buy.fillna(False)] = int(Signal.BUY)
        sig[sell.fillna(False)] = int(Signal.SELL)
        return sig

    # ------------------------------------------------------------
    # Optuna 搜索空间
    # ------------------------------------------------------------

    def get_param_space(self, trial: Any) -> dict[str, Any]:
        fast = trial.suggest_int("fast_period", 3, 20)
        slow = trial.suggest_int("slow_period", fast + 5, 120)
        return {
            "fast_period": fast,
            "slow_period": slow,
            "rsi_period": trial.suggest_int("rsi_period", 6, 30),
            "rsi_upper": trial.suggest_float("rsi_upper", 60.0, 85.0),
            "rsi_lower": trial.suggest_float("rsi_lower", 15.0, 40.0),
        }


# ============================================================
# 模块自测  python -m src.strategy.examples.ma_rsi
# ============================================================

if __name__ == "__main__":
    import numpy as np

    from ...utils.helpers import init_logging

    init_logging(level="INFO")
    rng = np.random.RandomState(7)
    n = 120
    # 造一段先跌后涨的价格，制造均线交叉
    trend = np.concatenate([np.linspace(20, 12, n // 2), np.linspace(12, 25, n - n // 2)])
    close = pd.Series(trend + rng.randn(n) * 0.3)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": pd.Series(rng.randint(int(1e5), int(2e5), n)).astype("float64"),
        }
    )
    strat = MaRsiStrategy(fast_period=5, slow_period=20)
    logger.info(f"策略: {strat}")
    sig = strat.generate_signals({"000001.SZ": df, "600519.SH": df})
    counts = sig.apply(pd.Series.value_counts).fillna(0).astype(int)
    logger.info(f"信号计数（按股票）:\n{counts}")
    logger.info(f"取值集合: {sorted(set(sig.to_numpy().ravel()))}")
