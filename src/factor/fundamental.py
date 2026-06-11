"""
通用财务因子框架（fundamental）
================================

提供**横截面财务因子**的公开框架：估值 / 盈利质量 / 成长 / 杠杆 / 规模五大类常用
因子，以及自定义比率因子 :class:`RatioFactor`。具体 alpha 因子（私有）通过继承
:class:`FundamentalFactor` 在外部私有仓库实现。

输入约定（重要：防未来函数 / 幸存者偏差）
------------------------------------------
每个因子的 :meth:`compute` 接收一个**时间点横截面** DataFrame：

    index = code（股票代码），columns = 标准财务字段（见 :data:`STANDARD_FIELDS`）

调用方（数据层 / 策略层）有责任保证该截面是**点位数据**——即每只股票取的财务值，其
**公告日（pub_date）必须 <= as_of_date**，且股票池来自 ``Universe.get_tradable_stocks``。
本层只做"给定干净截面 → 算因子"，不负责取数（取数属数据层职责，与 technical.py 一致）。

字段方向（higher_is_better）
----------------------------
每个因子声明 :attr:`higher_is_better`，表示"因子值越大是否越优"。估值类统一返回
**收益率口径**（如盈利收益率 = 1/PE），从而与质量/成长一致地"越大越便宜/越好"；
杠杆、规模类则 ``higher_is_better=False``。:meth:`score` 据此把任意因子翻成统一
"越大越优"的标准化分值，便于多因子合成。

横截面标准化沿用 :meth:`FactorBase.normalize`（z-score / rank / minmax / robust + winsorize）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pandas as pd
from loguru import logger

from .base import FactorBase

__all__ = [
    "STANDARD_FIELDS",
    "BookToMarketFactor",
    "DebtToAssetFactor",
    "DividendYieldFactor",
    "EarningsYieldFactor",
    "FundamentalFactor",
    "FundamentalFactors",
    "GrossMarginFactor",
    "LogMarketCapFactor",
    "NetMarginFactor",
    "ProfitGrowthFactor",
    "ROAFactor",
    "ROEFactor",
    "RatioFactor",
    "RevenueGrowthFactor",
    "SalesYieldFactor",
]

#: 标准财务字段（横截面 DataFrame 的列名约定；数据层取数后归一到这些名字）
STANDARD_FIELDS: dict[str, str] = {
    # 估值（原始比率，越小越便宜）
    "pe": "市盈率（TTM）",
    "pb": "市净率（MRQ）",
    "ps": "市销率（TTM）",
    "dividend_yield": "股息率",
    # 盈利质量（越大越好）
    "roe": "净资产收益率",
    "roa": "总资产收益率",
    "gross_margin": "毛利率",
    "net_margin": "净利率",
    # 成长（同比，越大越好）
    "revenue_yoy": "营收同比增速",
    "profit_yoy": "净利润同比增速",
    # 杠杆（越小越稳健）
    "debt_to_asset": "资产负债率",
    # 规模（总市值，元）
    "total_mv": "总市值",
    # 每股
    "eps": "每股收益",
    "bps": "每股净资产",
}


def _safe_inverse(s: pd.Series) -> pd.Series:
    """收益率口径：取倒数，分母 <= 0（亏损/无意义）置 NaN。"""
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    out = 1.0 / x.where(x > 0)
    return out


# ============================================================
# 财务因子基类
# ============================================================


class FundamentalFactor(FactorBase):
    """横截面财务因子基类。

    子类约定：
    - ``required_fields``：所需标准字段；
    - ``higher_is_better``：因子值越大是否越优（用于 :meth:`score` 统一方向）；
    - 实现 :meth:`_calc(data) -> pd.Series`（index=code）。
    """

    #: 计算所需的标准字段（子类覆盖）
    required_fields: Sequence[str] = ()

    #: 因子值越大是否越优（用于 score 方向统一）
    higher_is_better: bool = True

    def compute(self, data: pd.DataFrame) -> pd.Series:
        self.validate_input(data, self.required_fields)
        ser = self._calc(data)
        ser = pd.to_numeric(ser, errors="coerce").astype("float64")
        ser.name = self.factor_name
        return ser

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def score(
        self,
        data: pd.DataFrame,
        method: str = "zscore",
        *,
        winsorize: float | None = 0.01,
    ) -> pd.Series:
        """计算并标准化为"越大越优"的截面分值（``higher_is_better=False`` 自动取负）。"""
        raw = self.compute(data)
        norm = self.normalize(raw, method=method, winsorize=winsorize)
        return norm if self.higher_is_better else -norm


# ============================================================
# 自定义比率因子（框架内通用扩展点）
# ============================================================


class RatioFactor(FundamentalFactor):
    """通用比率因子：``numerator / denominator``（denominator<=0 可选置 NaN）。

    便于在框架内快速定义私有 alpha 的派生比率，无需新建类。
    """

    def __init__(
        self,
        numerator: str,
        denominator: str,
        name: str | None = None,
        higher_is_better: bool = True,
        drop_nonpositive_denom: bool = True,
    ) -> None:
        super().__init__(
            numerator=numerator,
            denominator=denominator,
            drop_nonpositive_denom=bool(drop_nonpositive_denom),
        )
        self.required_fields = (numerator, denominator)
        self.higher_is_better = bool(higher_is_better)
        self.factor_name = name or f"ratio_{numerator}_over_{denominator}"

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        num = pd.to_numeric(data[self.params["numerator"]], errors="coerce").astype("float64")
        den = pd.to_numeric(data[self.params["denominator"]], errors="coerce").astype("float64")
        if self.params["drop_nonpositive_denom"]:
            den = den.where(den > 0)
        else:
            den = den.replace(0.0, np.nan)
        return num / den


# ============================================================
# 估值类（统一返回收益率口径：越大越便宜/越优）
# ============================================================


class EarningsYieldFactor(FundamentalFactor):
    """盈利收益率 = 1 / PE（PE<=0 置 NaN）。价值因子，越大越便宜。"""

    factor_name = "earnings_yield"
    required_fields = ("pe",)
    higher_is_better = True

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return _safe_inverse(data["pe"])


class BookToMarketFactor(FundamentalFactor):
    """账面市值比 = 1 / PB（PB<=0 置 NaN）。经典价值因子，越大越便宜。"""

    factor_name = "book_to_market"
    required_fields = ("pb",)
    higher_is_better = True

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return _safe_inverse(data["pb"])


class SalesYieldFactor(FundamentalFactor):
    """销售收益率 = 1 / PS（PS<=0 置 NaN）。越大越便宜。"""

    factor_name = "sales_yield"
    required_fields = ("ps",)
    higher_is_better = True

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return _safe_inverse(data["ps"])


class DividendYieldFactor(FundamentalFactor):
    """股息率（已是收益率口径）。越大越优。"""

    factor_name = "dividend_yield"
    required_fields = ("dividend_yield",)
    higher_is_better = True

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(data["dividend_yield"], errors="coerce").astype("float64")


# ============================================================
# 盈利质量类（越大越好）
# ============================================================


class ROEFactor(FundamentalFactor):
    """净资产收益率（ROE）。越大越好。"""

    factor_name = "roe"
    required_fields = ("roe",)
    higher_is_better = True

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return data["roe"]


class ROAFactor(FundamentalFactor):
    """总资产收益率（ROA）。越大越好。"""

    factor_name = "roa"
    required_fields = ("roa",)
    higher_is_better = True

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return data["roa"]


class GrossMarginFactor(FundamentalFactor):
    """毛利率。越大越好。"""

    factor_name = "gross_margin"
    required_fields = ("gross_margin",)
    higher_is_better = True

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return data["gross_margin"]


class NetMarginFactor(FundamentalFactor):
    """净利率。越大越好。"""

    factor_name = "net_margin"
    required_fields = ("net_margin",)
    higher_is_better = True

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return data["net_margin"]


# ============================================================
# 成长类（越大越好）
# ============================================================


class RevenueGrowthFactor(FundamentalFactor):
    """营收同比增速。越大越好。"""

    factor_name = "revenue_growth"
    required_fields = ("revenue_yoy",)
    higher_is_better = True

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return data["revenue_yoy"]


class ProfitGrowthFactor(FundamentalFactor):
    """净利润同比增速。越大越好。"""

    factor_name = "profit_growth"
    required_fields = ("profit_yoy",)
    higher_is_better = True

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return data["profit_yoy"]


# ============================================================
# 杠杆 / 规模（越小越优）
# ============================================================


class DebtToAssetFactor(FundamentalFactor):
    """资产负债率。越小越稳健（higher_is_better=False）。"""

    factor_name = "debt_to_asset"
    required_fields = ("debt_to_asset",)
    higher_is_better = False

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        return data["debt_to_asset"]


class LogMarketCapFactor(FundamentalFactor):
    """对数总市值（规模因子）。A 股小市值溢价 → 越小越优（higher_is_better=False）。"""

    factor_name = "log_market_cap"
    required_fields = ("total_mv",)
    higher_is_better = False

    def _calc(self, data: pd.DataFrame) -> pd.Series:
        mv = pd.to_numeric(data["total_mv"], errors="coerce").astype("float64")
        return np.log(mv.where(mv > 0))


# ============================================================
# 统一门面：注册表 + 工厂
# ============================================================


class FundamentalFactors:
    """财务因子统一门面：按名创建因子实例（私有 alpha 因子可在外部 register 扩展）。

    用法::

        f = FundamentalFactors.create("roe")
        scores = f.score(cross_section_df)            # 越大越优的标准化分值
        FundamentalFactors.available()                # 列出全部因子名
        FundamentalFactors.register("my_alpha", MyAlphaFactor)  # 注册私有因子
    """

    REGISTRY: ClassVar[dict[str, type[FundamentalFactor]]] = {
        "earnings_yield": EarningsYieldFactor,
        "book_to_market": BookToMarketFactor,
        "sales_yield": SalesYieldFactor,
        "dividend_yield": DividendYieldFactor,
        "roe": ROEFactor,
        "roa": ROAFactor,
        "gross_margin": GrossMarginFactor,
        "net_margin": NetMarginFactor,
        "revenue_growth": RevenueGrowthFactor,
        "profit_growth": ProfitGrowthFactor,
        "debt_to_asset": DebtToAssetFactor,
        "log_market_cap": LogMarketCapFactor,
    }

    @classmethod
    def create(cls, name: str, **params) -> FundamentalFactor:
        """按注册名创建因子实例。未知名抛 ``KeyError``。"""
        key = str(name).lower().strip()
        if key not in cls.REGISTRY:
            raise KeyError(f"未知财务因子 {name!r}（可用 {sorted(cls.REGISTRY)}）")
        return cls.REGISTRY[key](**params)

    @classmethod
    def register(cls, name: str, factor_cls: type[FundamentalFactor]) -> None:
        """注册自定义/私有财务因子类（必须继承 FundamentalFactor）。"""
        if not (isinstance(factor_cls, type) and issubclass(factor_cls, FundamentalFactor)):
            raise TypeError(f"{factor_cls!r} 必须是 FundamentalFactor 的子类")
        cls.REGISTRY[str(name).lower().strip()] = factor_cls

    @classmethod
    def available(cls) -> list[str]:
        """返回全部可用因子名（已排序）。"""
        return sorted(cls.REGISTRY)

    @classmethod
    def ratio(
        cls,
        numerator: str,
        denominator: str,
        name: str | None = None,
        higher_is_better: bool = True,
    ) -> RatioFactor:
        """便捷创建自定义比率因子。"""
        return RatioFactor(numerator, denominator, name=name, higher_is_better=higher_is_better)


# ============================================================
# 模块自测  python -m src.factor.fundamental
# ============================================================

if __name__ == "__main__":
    from ..utils.helpers import init_logging

    init_logging(level="INFO")
    # 构造一个点位横截面（index=code）
    cs = pd.DataFrame(
        {
            "pe": [10.0, 25.0, -5.0, 8.0],            # 第3只亏损 → 盈利收益率 NaN
            "pb": [1.2, 4.0, 2.0, 0.9],
            "ps": [2.0, 8.0, 3.0, 1.5],
            "dividend_yield": [0.03, 0.005, 0.0, 0.04],
            "roe": [0.18, 0.09, -0.05, 0.22],
            "roa": [0.08, 0.04, -0.02, 0.10],
            "gross_margin": [0.45, 0.30, 0.20, 0.50],
            "net_margin": [0.15, 0.08, -0.03, 0.18],
            "revenue_yoy": [0.20, 0.05, -0.10, 0.30],
            "profit_yoy": [0.25, 0.02, -0.50, 0.35],
            "debt_to_asset": [0.40, 0.65, 0.80, 0.35],
            "total_mv": [5e10, 2e11, 8e9, 3e10],
        },
        index=["000001.SZ", "600519.SH", "000002.SZ", "300750.SZ"],
    )
    cs.index.name = "code"

    for name in FundamentalFactors.available():
        f = FundamentalFactors.create(name)
        raw = f.compute(cs)
        sc = f.score(cs)
        logger.info(
            f"{f.factor_name:<16} better={'↑' if f.higher_is_better else '↓'} "
            f"raw_valid={int(raw.notna().sum())}/{len(raw)} "
            f"top={sc.idxmax() if sc.notna().any() else 'N/A'}"
        )
