"""
因子抽象基类（FactorBase）
===========================

所有因子（公开技术因子 / 私有 alpha 因子）的统一契约。子类只需实现 :meth:`compute`，
即可获得标准化、横截面编排、缺失值处理等通用能力。

两种计算口径
------------
A 股因子常见两种形态，本基类用同一个 :meth:`compute` 抽象方法统一承载，语义由子类
决定、并在子类文档中声明：

- **时序因子（单只股票）**：``compute(df)`` 接收**单支股票**的 OHLCV 时序
  （``index`` 或 ``date`` 列为交易日），返回 ``pd.Series``（``index=date``）。
  技术指标（MA/RSI/MACD…）属于此类。配合 :meth:`compute_panel` 可把一篮子股票
  拼成面板 ``DataFrame(index=date, columns=code)``。
- **横截面因子**：``compute(df)`` 接收某一**时间截面**的多股票数据，返回
  ``pd.Series``（``index=code``）。配合 :meth:`normalize` 做截面标准化。

防未来函数
----------
因子计算只允许使用"当前及之前"的数据。基类提供 :meth:`shift` 便捷方法，便于子类在
把因子用作"信号"时显式滞后一位（``T`` 日因子用于 ``T+1`` 决策），避免未来函数。
基类本身不强制滞后（因子值本就是 ``T`` 日的客观观测），是否滞后由策略层决定。

标准化
------
:meth:`normalize` 默认 z-score，另支持 ``rank`` / ``minmax`` / ``robust``，并可选
``winsorize``（分位裁尾）抑制极端值。截面标准化对横截面因子（``index=code``）最有意义。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

__all__ = ["FactorBase"]

#: 支持的标准化方法
NORMALIZE_METHODS = ("zscore", "rank", "minmax", "robust")


class FactorBase(ABC):
    """因子抽象基类。

    子类约定：
    - 设置类属性 ``factor_name``（用于日志与结果列命名）；
    - 实现 :meth:`compute`，返回 ``pd.Series``（时序因子 index=date，横截面因子 index=code）；
    - 通过 ``__init__(**params)`` 接收并保存参数（如周期 ``period`` 等）。

    Args:
        **params: 任意因子参数，存入 ``self.params``，并作为属性暴露（如 ``self.period``）。
    """

    #: 因子名（子类应覆盖；默认取类名）
    factor_name: str = "factor"

    def __init__(self, **params) -> None:
        self.params: Dict[str, object] = dict(params)
        # 参数同时暴露为属性，便于子类内 self.period 直接取用
        for key, value in self.params.items():
            setattr(self, key, value)
        # 未显式设置 factor_name 的子类，回退到类名（小写）
        if type(self).factor_name == FactorBase.factor_name:
            self.factor_name = type(self).__name__.lower()

    # ------------------------------------------------------------
    # 抽象：核心计算
    # ------------------------------------------------------------

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值。

        Args:
            data: 时序因子 → 单只股票 OHLCV（含 ``date`` 索引或列）；
                  横截面因子 → 某截面的多股票数据。

        Returns:
            ``pd.Series``：时序因子 ``index=date``；横截面因子 ``index=code``。
            ``name`` 应为 :attr:`factor_name`。
        """
        raise NotImplementedError

    # ------------------------------------------------------------
    # 面板编排（时序因子专用便捷方法）
    # ------------------------------------------------------------

    def compute_panel(
        self,
        data: Mapping[str, pd.DataFrame],
        *,
        skip_errors: bool = True,
    ) -> pd.DataFrame:
        """对一篮子股票逐只 :meth:`compute`，拼成面板。

        Args:
            data: ``{code: 单只股票 OHLCV}``。
            skip_errors: 单只股票计算失败时是否跳过（记 WARNING）。``False`` 则抛出。

        Returns:
            ``DataFrame(index=date, columns=code)``；输入为空时返回空 DataFrame。
        """
        cols: Dict[str, pd.Series] = {}
        for code, df in data.items():
            if df is None or df.empty:
                continue
            try:
                ser = self.compute(df)
            except Exception as exc:  # noqa: BLE001
                if not skip_errors:
                    raise
                logger.warning(f"[{self.factor_name}] 计算 {code} 失败，已跳过: {type(exc).__name__}: {exc}")
                continue
            cols[code] = self._as_dated_series(ser, df)
        if not cols:
            return pd.DataFrame()
        panel = pd.DataFrame(cols).sort_index()
        panel.columns.name = "code"
        panel.index.name = "date"
        return panel

    @staticmethod
    def _as_dated_series(ser: pd.Series, source: pd.DataFrame) -> pd.Series:
        """把 compute 结果对齐到以交易日为索引的 Series（容忍 RangeIndex + date 列）。"""
        if isinstance(ser.index, pd.DatetimeIndex):
            return ser
        if "date" in source.columns and len(source) == len(ser):
            idx = pd.to_datetime(source["date"]).to_numpy()
            return pd.Series(ser.to_numpy(), index=idx, name=ser.name)
        if isinstance(source.index, pd.DatetimeIndex) and len(source) == len(ser):
            return pd.Series(ser.to_numpy(), index=source.index, name=ser.name)
        return ser

    # ------------------------------------------------------------
    # 标准化
    # ------------------------------------------------------------

    def normalize(
        self,
        factor: pd.Series,
        method: str = "zscore",
        *,
        winsorize: Optional[float] = None,
    ) -> pd.Series:
        """标准化因子值（默认 z-score）。

        Args:
            factor: 因子 Series（横截面因子 index=code 时最有意义）。
            method: ``zscore`` / ``rank`` / ``minmax`` / ``robust``。
                - ``zscore``：``(x - mean) / std``；
                - ``rank``：排名归一到 ``[0, 1]``；
                - ``minmax``：线性缩放到 ``[0, 1]``；
                - ``robust``：``(x - median) / IQR``（抗异常值）。
            winsorize: 分位裁尾比例（如 ``0.01`` 表示按 1%/99% 分位裁剪），``None`` 不裁。

        Returns:
            标准化后的 Series（保持原 index 与 name）；全 NaN 或常数列安全返回。
        """
        if method not in NORMALIZE_METHODS:
            raise ValueError(f"未知标准化方法 {method!r}（支持 {NORMALIZE_METHODS}）")
        s = pd.to_numeric(factor, errors="coerce").astype("float64")
        if winsorize is not None:
            s = self._winsorize(s, winsorize)

        valid = s.dropna()
        if valid.empty:
            return s

        if method == "zscore":
            std = valid.std(ddof=0)
            out = (s - valid.mean()) / std if std and std > 0 else s - valid.mean()
        elif method == "rank":
            out = s.rank(pct=True)
        elif method == "minmax":
            lo, hi = valid.min(), valid.max()
            out = (s - lo) / (hi - lo) if hi > lo else s * 0.0
        else:  # robust
            med = valid.median()
            q75, q25 = valid.quantile(0.75), valid.quantile(0.25)
            iqr = q75 - q25
            out = (s - med) / iqr if iqr and iqr > 0 else s - med
        out.name = factor.name
        return out

    @staticmethod
    def _winsorize(s: pd.Series, ratio: float) -> pd.Series:
        """按 ``[ratio, 1-ratio]`` 分位裁剪极端值。"""
        if not (0.0 < ratio < 0.5):
            raise ValueError(f"winsorize 比例须在 (0, 0.5)，got {ratio}")
        valid = s.dropna()
        if valid.empty:
            return s
        lo, hi = valid.quantile(ratio), valid.quantile(1.0 - ratio)
        return s.clip(lower=lo, upper=hi)

    # ------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------

    @staticmethod
    def shift(factor: pd.Series, periods: int = 1) -> pd.Series:
        """因子滞后 ``periods`` 期（``T`` 日因子用于 ``T+periods`` 决策，防未来函数）。"""
        return factor.shift(periods)

    @staticmethod
    def validate_input(data: pd.DataFrame, required: Sequence[str]) -> None:
        """校验输入 DataFrame 含必要列，缺失则抛 ``KeyError``。"""
        if data is None or not isinstance(data, pd.DataFrame):
            raise TypeError(f"因子输入必须是 DataFrame，got {type(data).__name__}")
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise KeyError(f"因子输入缺少必要列 {missing}（实际列 {list(data.columns)}）")

    # ------------------------------------------------------------
    # 便捷调用 / 展示
    # ------------------------------------------------------------

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        """``factor(df)`` 等价于 ``factor.compute(df)``。"""
        return self.compute(data)

    def __repr__(self) -> str:
        if self.params:
            kv = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"<{type(self).__name__} name={self.factor_name!r} {kv}>"
        return f"<{type(self).__name__} name={self.factor_name!r}>"


# ============================================================
# 模块自测  python -m src.factor.base
# ============================================================

if __name__ == "__main__":
    from ..utils.helpers import init_logging

    init_logging(level="INFO")

    class _CloseMA(FactorBase):
        """示例：N 日收盘均线（时序因子，index=date）。"""

        factor_name = "close_ma"

        def compute(self, data: pd.DataFrame) -> pd.Series:
            self.validate_input(data, ["close"])
            period = int(self.params.get("period", 5))
            ser = data["close"].rolling(period, min_periods=1).mean()
            ser.name = self.factor_name
            return ser

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": dates, "close": np.arange(10, 20, dtype="float64")})

    f = _CloseMA(period=3)
    logger.info(f"因子: {f}")
    out = f.compute(df)
    logger.info(f"compute 结果:\n{out.to_string()}")

    # 面板
    panel = f.compute_panel({"000001.SZ": df, "600519.SH": df.assign(close=df['close'] * 2)})
    logger.info(f"面板 shape={panel.shape}, 列={list(panel.columns)}")

    # 横截面标准化（取最后一行作截面）
    cross = panel.iloc[-1]
    cross.index.name = "code"
    logger.info(f"截面 z-score:\n{f.normalize(cross, 'zscore').to_string()}")
