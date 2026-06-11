"""
策略抽象基类（BaseStrategy）+ 信号枚举（Signal）
=================================================

所有策略（公开示例 / 私有 alpha）的统一契约。框架与回测引擎只通过本接口交互，
私有策略通过 ``strategy/loader.py`` 从外部仓库加载（物理上不进入本仓库）。

信号规范
--------
策略输出**信号矩阵** ``DataFrame``：``index = 交易日期``、``columns = 股票代码``、
``values ∈ {Signal.SELL=-1, Signal.HOLD=0, Signal.BUY=1}``。**禁止**在策略代码里直接
写 ``-1/0/1`` 字面量，一律用 :class:`Signal` 枚举，提升可读性并避免方向写反。

防未来函数（重要）
------------------
``generate_signals`` 在 ``T`` 日收盘后基于"截至 ``T`` 日"的数据计算信号；回测引擎在
``T+1`` 日开盘撮合（见 ``engine/execution.py``）。因此**策略内严禁使用未来数据**：
计算 ``T`` 日信号只能用 ``<= T`` 的行情。基类提供 :meth:`empty_signals` /
:meth:`validate_signals` 等工具，但不强制时序滞后——这属策略逻辑职责。

超参优化
--------
:meth:`get_param_space` 供 Optuna 调参调用，返回"参数名 → 采样值"的 dict，默认空
（即不调参）。子类覆盖以声明搜索空间，例如::

    def get_param_space(self, trial) -> dict:
        return {
            "fast_period": trial.suggest_int("fast_period", 5, 20),
            "slow_period": trial.suggest_int("slow_period", 20, 60),
        }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, ClassVar

import pandas as pd
from loguru import logger

__all__ = ["BaseStrategy", "Signal"]


class Signal(IntEnum):
    """交易信号枚举（数值可直接进入信号矩阵与回测引擎）。

    - ``BUY  = 1``  买入 / 持有目标多头
    - ``HOLD = 0``  不动
    - ``SELL = -1`` 卖出 / 清仓
    """

    SELL = -1
    HOLD = 0
    BUY = 1


class BaseStrategy(ABC):
    """策略抽象基类。

    子类约定：
    - 设置类属性 ``strategy_name``（用于日志归档与 CLI/loader 指定，应与文件/类名呼应）；
    - 可选设置类属性 ``default_params``（dict，默认参数），实例化时与传入参数深度合并；
    - 实现 :meth:`generate_signals`，返回信号矩阵 ``DataFrame``；
    - 可选覆盖 :meth:`get_param_space` 声明 Optuna 搜索空间。

    Args:
        **params: 策略参数；与 ``default_params`` 合并后存入 ``self.params``，并作为属性
            暴露（如 ``self.fast_period``）。
    """

    #: 策略名（子类应覆盖；默认取类名）
    strategy_name: str = "base"

    #: 默认参数（子类覆盖；实例化时与传入 params 合并）
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        merged: dict[str, Any] = {**type(self).default_params, **params}
        self.params: dict[str, Any] = merged
        # 参数同时暴露为属性，便于子类内 self.fast_period 直接取用
        for key, value in merged.items():
            setattr(self, key, value)
        # 未显式设置 strategy_name 的子类回退到类名（小写）
        if type(self).strategy_name == BaseStrategy.strategy_name:
            self.strategy_name = type(self).__name__.lower()

    # ------------------------------------------------------------
    # 抽象：信号生成
    # ------------------------------------------------------------

    @abstractmethod
    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """生成信号矩阵。

        Args:
            data: ``{code: 单只股票行情 DataFrame}``（含 ``date`` 列或 DatetimeIndex 及
                OHLCV）。回测引擎按交易日喂入"截至当前可见"的数据，**不含未来**。

        Returns:
            ``DataFrame``：``index=交易日期``、``columns=股票代码``、
            ``values ∈ {-1, 0, 1}``（:class:`Signal`）。
        """
        raise NotImplementedError

    # ------------------------------------------------------------
    # 超参优化（Optuna）
    # ------------------------------------------------------------

    def get_param_space(self, trial: Any) -> dict[str, Any]:
        """返回 Optuna 参数搜索空间（参数名 → 采样值）。默认空（不调参）。

        Args:
            trial: ``optuna.Trial``（不在框架硬依赖 optuna，按需传入）。
        """
        return {}

    # ------------------------------------------------------------
    # 参数管理
    # ------------------------------------------------------------

    def set_params(self, **params: Any) -> BaseStrategy:
        """更新参数（同步刷新属性），返回自身以支持链式调用。"""
        self.params.update(params)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self) -> dict[str, Any]:
        """返回当前参数的浅拷贝。"""
        return dict(self.params)

    # ------------------------------------------------------------
    # 信号工具（供子类复用）
    # ------------------------------------------------------------

    @staticmethod
    def empty_signals(dates, codes) -> pd.DataFrame:
        """构造全 ``HOLD`` 的信号矩阵骨架（index=dates, columns=codes）。"""
        idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        df = pd.DataFrame(int(Signal.HOLD), index=idx, columns=list(codes), dtype="int64")
        df.index.name = "date"
        df.columns.name = "code"
        return df

    @staticmethod
    def validate_signals(signals: pd.DataFrame) -> pd.DataFrame:
        """规整信号矩阵：NaN→HOLD、转 int64，并校验取值仅含 {-1,0,1}。

        Returns:
            规整后的新 DataFrame（不修改入参）。

        Raises:
            ValueError: 含 {-1,0,1} 之外的非法信号值。
        """
        if signals is None or not isinstance(signals, pd.DataFrame):
            raise TypeError(f"信号矩阵必须是 DataFrame，got {type(signals).__name__}")
        out = signals.copy()
        out = out.fillna(int(Signal.HOLD))
        try:
            out = out.astype("int64")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"信号矩阵存在无法转为整数的值: {exc}") from exc
        valid = {int(Signal.SELL), int(Signal.HOLD), int(Signal.BUY)}
        bad = set(pd.unique(out.to_numpy().ravel())) - valid
        if bad:
            raise ValueError(f"信号矩阵含非法值 {sorted(bad)}（仅允许 {sorted(valid)}）")
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
        out.index.name = "date"
        out.columns.name = "code"
        return out

    # ------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------

    def __repr__(self) -> str:
        if self.params:
            kv = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"<{type(self).__name__} name={self.strategy_name!r} {kv}>"
        return f"<{type(self).__name__} name={self.strategy_name!r}>"


# ============================================================
# 模块自测  python -m src.strategy.base
# ============================================================

if __name__ == "__main__":
    from ..utils.helpers import init_logging

    init_logging(level="INFO")

    class _BuyAndHold(BaseStrategy):
        """示例：首日全买入并持有。"""

        strategy_name = "buy_and_hold"
        default_params: ClassVar[dict[str, Any]] = {"warmup": 0}

        def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
            codes = list(data.keys())
            dates = pd.to_datetime(next(iter(data.values()))["date"])
            sig = self.empty_signals(dates, codes)
            sig.iloc[self.warmup] = int(Signal.BUY)  # 仅 warmup 当日发买入
            return self.validate_signals(sig)

    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5, freq="B"), "close": range(5)})
    strat = _BuyAndHold(warmup=0)
    logger.info(f"策略: {strat}")
    sig = strat.generate_signals({"000001.SZ": df, "600519.SH": df})
    logger.info(f"信号矩阵:\n{sig.to_string()}")
    logger.info(f"取值集合: {sorted(set(sig.to_numpy().ravel()))}")
