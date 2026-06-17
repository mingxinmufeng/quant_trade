"""
纯技术指标因子（technical）
============================

基于 ``pandas-ta-classic``（原版 ``pandas-ta`` 的社区维护分支）封装常用技术指标为
:class:`~src.factor.base.FactorBase` 子类。每个因子的 :meth:`compute` 接收**单只股票**
的 OHLCV 时序 DataFrame，返回 ``pd.Series``（``index=date``，与输入对齐）。

实现的指标（Prompt 要求）：
    SMAFactor            简单移动平均
    EMAFactor            指数移动平均
    RSIFactor            相对强弱指标
    MACDFactor           MACD（输出 hist/macd/signal 之一）
    BollingerBandsFactor 布林带（输出 percent/bandwidth/upper/middle/lower 之一）
    ATRFactor            平均真实波幅
    VolumeMAFactor       成交量移动平均

设计要点
--------
- **单一输出契约**：``FactorBase.compute`` 约定返回单列 Series；MACD / 布林带等多输出
  指标通过 ``output`` 参数选择其中一路（默认取最常用作因子的那一路）。
- **列名版本鲁棒**：``pandas-ta`` 多输出列名带参数后缀（如 ``MACDh_12_26_9``），本模块
  按**前缀**匹配选列，兼容不同版本/参数组合。
- **数据不足安全**：样本长度不足以计算指标时返回全 ``NaN`` Series（不抛异常）。
- **统一门面** :class:`TechnicalFactors`：提供注册表 + 工厂方法，按名创建因子实例。

依赖导入兼容：优先 ``pandas_ta_classic``，回退原版 ``pandas_ta``（二者 API 一致）。
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from loguru import logger

from .base import FactorBase

__all__ = [
    "ATRFactor",
    "BollingerBandsFactor",
    "EMAFactor",
    "MACDFactor",
    "RSIFactor",
    "SMAFactor",
    "TechnicalFactor",
    "TechnicalFactors",
    "VolumeMAFactor",
]

# ------------------------------------------------------------
# pandas-ta 导入（classic 分支优先，回退原版）
# ------------------------------------------------------------
try:
    import pandas_ta_classic as ta  # type: ignore
except ImportError:  # pragma: no cover - 视环境而定
    try:
        import pandas_ta as ta  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "technical.py 需要 pandas-ta-classic（或 pandas-ta）：\n"
            "  pip install pandas-ta-classic"
        ) from exc


def _pick_prefix(df: pd.DataFrame, prefix: str) -> pd.Series | None:
    """从多输出 DataFrame 中按列名前缀取第一列（精确前缀匹配）。"""
    for col in df.columns:
        if str(col).upper().startswith(prefix.upper()):
            return df[col]
    return None


# ============================================================
# 技术因子基类
# ============================================================


class TechnicalFactor(FactorBase):
    """技术因子基类：约束输入为单只股票 OHLCV，输出对齐交易日索引的单列 Series。

    子类只需实现 :meth:`_calc`（返回与输入等长的 ndarray/Series），基类负责输入校验、
    date 索引对齐与命名。
    """

    #: 计算所需列（子类覆盖）
    required_columns = ("close",)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        self.validate_input(data, self.required_columns)
        result = self._calc(data)
        ser = self._to_series(result, data)
        ser.name = self.factor_name
        # 交易日索引对齐统一交给基类 _as_dated_series（结果恒与 data 等长，走按位置对齐）
        return self._as_dated_series(ser, data)

    def _calc(self, data: pd.DataFrame):
        """子类实现：返回指标值（Series / ndarray / None）。"""
        raise NotImplementedError

    @staticmethod
    def _to_series(result, data: pd.DataFrame) -> pd.Series:
        """把指标结果规整为与 ``data`` 等长的 float Series（None / 长度不符 → 全 NaN）。"""
        if result is None:
            return pd.Series(np.nan, index=data.index, dtype="float64")
        if isinstance(result, pd.Series):
            return result.astype("float64").reset_index(drop=True).set_axis(data.index)
        arr = np.asarray(result, dtype="float64")
        if arr.shape[0] != len(data):
            return pd.Series(np.nan, index=data.index, dtype="float64")
        return pd.Series(arr, index=data.index, dtype="float64")


# ============================================================
# 单输出指标
# ============================================================


class SMAFactor(TechnicalFactor):
    """简单移动平均（SMA）。参数：``period``（默认 20）。"""

    required_columns = ("close",)

    def __init__(self, period: int = 20) -> None:
        super().__init__(period=int(period))
        self.factor_name = f"sma_{int(period)}"

    def _calc(self, data: pd.DataFrame):
        return ta.sma(data["close"].astype("float64"), length=self.params["period"])


class EMAFactor(TechnicalFactor):
    """指数移动平均（EMA）。参数：``period``（默认 20）。"""

    required_columns = ("close",)

    def __init__(self, period: int = 20) -> None:
        super().__init__(period=int(period))
        self.factor_name = f"ema_{int(period)}"

    def _calc(self, data: pd.DataFrame):
        return ta.ema(data["close"].astype("float64"), length=self.params["period"])


class RSIFactor(TechnicalFactor):
    """相对强弱指标（RSI）。参数：``period``（默认 14）。"""

    required_columns = ("close",)

    def __init__(self, period: int = 14) -> None:
        super().__init__(period=int(period))
        self.factor_name = f"rsi_{int(period)}"

    def _calc(self, data: pd.DataFrame):
        return ta.rsi(data["close"].astype("float64"), length=self.params["period"])


class ATRFactor(TechnicalFactor):
    """平均真实波幅（ATR）。参数：``period``（默认 14）。需 high/low/close。"""

    required_columns = ("high", "low", "close")

    def __init__(self, period: int = 14) -> None:
        super().__init__(period=int(period))
        self.factor_name = f"atr_{int(period)}"

    def _calc(self, data: pd.DataFrame):
        return ta.atr(
            data["high"].astype("float64"),
            data["low"].astype("float64"),
            data["close"].astype("float64"),
            length=self.params["period"],
        )


class VolumeMAFactor(TechnicalFactor):
    """成交量移动平均（Volume MA）。参数：``period``（默认 20）。需 volume。"""

    required_columns = ("volume",)

    def __init__(self, period: int = 20) -> None:
        super().__init__(period=int(period))
        self.factor_name = f"volma_{int(period)}"

    def _calc(self, data: pd.DataFrame):
        return ta.sma(data["volume"].astype("float64"), length=self.params["period"])


# ============================================================
# 多输出指标（通过 output 选择一路）
# ============================================================


class MACDFactor(TechnicalFactor):
    """MACD。参数：``fast=12, slow=26, signal=9, output``。

    ``output`` ∈ {``hist`` (默认，MACD 柱=快慢线差减信号线), ``macd`` (DIF), ``signal`` (DEA)}。
    """

    required_columns = ("close",)
    _PREFIX: ClassVar[dict[str, str]] = {"macd": "MACD_", "signal": "MACDs_", "hist": "MACDh_"}

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, output: str = "hist") -> None:
        output = str(output).lower()
        if output not in self._PREFIX:
            raise ValueError(f"MACD output 须为 {tuple(self._PREFIX)}，got {output!r}")
        super().__init__(fast=int(fast), slow=int(slow), signal=int(signal), output=output)
        self.factor_name = f"macd_{output}_{int(fast)}_{int(slow)}_{int(signal)}"

    def _calc(self, data: pd.DataFrame):
        df = ta.macd(
            data["close"].astype("float64"),
            fast=self.params["fast"],
            slow=self.params["slow"],
            signal=self.params["signal"],
        )
        if df is None:
            return None
        return _pick_prefix(df, self._PREFIX[self.params["output"]])


class BollingerBandsFactor(TechnicalFactor):
    """布林带（Bollinger Bands）。参数：``period=20, std=2.0, output``。

    ``output`` ∈ {``percent`` (默认 %B=价格在带内相对位置), ``bandwidth`` (带宽),
    ``upper``, ``middle``, ``lower``}。
    """

    required_columns = ("close",)
    _PREFIX: ClassVar[dict[str, str]] = {
        "lower": "BBL_",
        "middle": "BBM_",
        "upper": "BBU_",
        "bandwidth": "BBB_",
        "percent": "BBP_",
    }

    def __init__(self, period: int = 20, std: float = 2.0, output: str = "percent") -> None:
        output = str(output).lower()
        if output not in self._PREFIX:
            raise ValueError(f"BBands output 须为 {tuple(self._PREFIX)}，got {output!r}")
        super().__init__(period=int(period), std=float(std), output=output)
        self.factor_name = f"bb_{output}_{int(period)}_{float(std)}"

    def _calc(self, data: pd.DataFrame):
        df = ta.bbands(
            data["close"].astype("float64"),
            length=self.params["period"],
            std=self.params["std"],
        )
        if df is None:
            return None
        return _pick_prefix(df, self._PREFIX[self.params["output"]])


# ============================================================
# 统一门面：注册表 + 工厂
# ============================================================


class TechnicalFactors:
    """技术因子统一门面：按名创建因子实例。

    用法::

        f = TechnicalFactors.create("rsi", period=14)
        series = f.compute(stock_df)

        # 或便捷工厂
        f = TechnicalFactors.rsi(14)

        # 列出全部可用因子名
        TechnicalFactors.available()
    """

    #: 因子名 → 类（名称与 config/CLI 字符串对齐，全小写）
    REGISTRY: ClassVar[dict[str, type[TechnicalFactor]]] = {
        "sma": SMAFactor,
        "ema": EMAFactor,
        "rsi": RSIFactor,
        "macd": MACDFactor,
        "bbands": BollingerBandsFactor,
        "atr": ATRFactor,
        "volume_ma": VolumeMAFactor,
    }

    @classmethod
    def create(cls, name: str, **params) -> TechnicalFactor:
        """按注册名创建因子实例。未知名抛 ``KeyError``。"""
        key = str(name).lower().strip()
        if key not in cls.REGISTRY:
            raise KeyError(f"未知技术因子 {name!r}（可用 {sorted(cls.REGISTRY)}）")
        return cls.REGISTRY[key](**params)

    @classmethod
    def available(cls) -> list[str]:
        """返回全部可用因子名（已排序）。"""
        return sorted(cls.REGISTRY)

    # ---- 便捷工厂 ----
    @staticmethod
    def sma(period: int = 20) -> SMAFactor:
        return SMAFactor(period=period)

    @staticmethod
    def ema(period: int = 20) -> EMAFactor:
        return EMAFactor(period=period)

    @staticmethod
    def rsi(period: int = 14) -> RSIFactor:
        return RSIFactor(period=period)

    @staticmethod
    def macd(fast: int = 12, slow: int = 26, signal: int = 9, output: str = "hist") -> MACDFactor:
        return MACDFactor(fast=fast, slow=slow, signal=signal, output=output)

    @staticmethod
    def bbands(period: int = 20, std: float = 2.0, output: str = "percent") -> BollingerBandsFactor:
        return BollingerBandsFactor(period=period, std=std, output=output)

    @staticmethod
    def atr(period: int = 14) -> ATRFactor:
        return ATRFactor(period=period)

    @staticmethod
    def volume_ma(period: int = 20) -> VolumeMAFactor:
        return VolumeMAFactor(period=period)


# ============================================================
# 模块自测  python -m src.factor.technical
# ============================================================

if __name__ == "__main__":
    from ..utils.helpers import init_logging

    init_logging(level="INFO")
    rng = np.random.RandomState(42)
    n = 60
    close = pd.Series(np.linspace(10, 20, n) + rng.randn(n) * 0.3)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": pd.Series(rng.randint(int(1e5), int(2e5), n)).astype("float64"),
        }
    )
    for name in TechnicalFactors.available():
        f = TechnicalFactors.create(name)
        s = f.compute(df)
        logger.info(f"{f.factor_name:<22} 尾值={s.iloc[-1]:.4f} 有效={int(s.notna().sum())}/{len(s)}")
