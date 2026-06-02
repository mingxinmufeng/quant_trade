"""
本地数据仓库（storage）
=======================

职责：parquet 读写 + 复权缓存。**原始价与因子分离落盘**：

落盘布局::

    {store}/daily/{code}.parquet          # 不复权日线 OHLCV + 涨跌停/停牌/名称/来源
    {store}/min1/{code}.parquet           # 不复权 1 分钟 OHLCV
    {store}/min5/{code}.parquet           # 不复权 5 分钟 OHLCV
    {store}/factors/{code}.parquet        # date → cum_factor（外部源，覆盖刷新）
    {store}/adjusted/{freq}/{code}.parquet  # 后复权(hfq)缓存（按版本戳重算）

复权缓存策略（用户拍板）
------------------------
- **只缓存 hfq 一份到硬盘**；``qfq`` 由 hfq 除以锚点日累计因子（标量）现场派生，
  ``none`` 直接读原始，二者都不落缓存。
- **版本戳重算，不按时间删除**：hfq 缓存文件的 mtime ≥ ``原始文件`` 与 ``因子文件``
  的 mtime 时视为有效；任一源更新（mtime 变新）即重算覆盖。既不浪费也不读脏数据。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from loguru import logger

from ..utils.helpers import ensure_dir
from .adjust import apply_adjust, cum_factor_at
from .factors import FACTOR_COLUMNS

# ============================================================
# 字段规范（**原始不复权**落盘；复权在 load 时按需合成）
# ============================================================

#: 日线落盘字段（不复权 OHLCV + 元数据）
DAILY_COLUMNS: Dict[str, str] = {
    "date": "datetime64[ns]",
    "code": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "amount": "float64",
    "is_suspended": "bool",
    "limit_up": "float64",
    "limit_down": "float64",
    "name": "string",
    "source": "string",  # 行级数据来源，便于溯源/回滚
}

#: 分钟线落盘字段（不复权 OHLCV；仅 min1 / min5 落盘）
MINUTE_COLUMNS: Dict[str, str] = {
    "datetime": "datetime64[ns]",
    "code": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "amount": "float64",
    "source": "string",
}

#: 落盘周期目录名
FREQ_DIRS = {"daily": "daily", "min5": "min5", "min1": "min1"}


def _time_col(freq: str) -> str:
    return "date" if freq == "daily" else "datetime"


class DataStore:
    """parquet 读写 + 复权缓存的本地仓库。"""

    def __init__(self, store_path: Union[str, Path] = "data_store") -> None:
        self._store_path = Path(store_path)
        self._dirs = {f: self._store_path / d for f, d in FREQ_DIRS.items()}
        for d in self._dirs.values():
            ensure_dir(d)
        self._factor_dir = ensure_dir(self._store_path / "factors")
        self._adjusted_dir = ensure_dir(self._store_path / "adjusted")

    # ------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------

    def freq_path(self, code: str, freq: str) -> Path:
        return self._dirs[freq] / f"{code}.parquet"

    def factor_path(self, code: str) -> Path:
        return self._factor_dir / f"{code}.parquet"

    def adjusted_path(self, code: str, freq: str) -> Path:
        return ensure_dir(self._adjusted_dir / freq) / f"{code}.parquet"

    # ------------------------------------------------------------
    # 原始数据读写
    # ------------------------------------------------------------

    def read_raw(self, code: str, freq: str) -> Optional[pd.DataFrame]:
        path = self.freq_path(code, freq)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            tc = _time_col(freq)
            df[tc] = pd.to_datetime(df[tc])
            if "source" not in df.columns:
                df["source"] = "legacy"
            df["source"] = df["source"].fillna("legacy").astype("string")
            return df.sort_values(tc).reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"读取本地缓存 {path} 失败: {exc}（视为不存在重拉）")
            return None

    def write_raw(self, code: str, freq: str, df: pd.DataFrame) -> None:
        path = self.freq_path(code, freq)
        ensure_dir(path.parent)
        df.to_parquet(path, index=False, compression="snappy")

    # ------------------------------------------------------------
    # 因子读写
    # ------------------------------------------------------------

    def read_factor(self, code: str) -> pd.DataFrame:
        path = self.factor_path(code)
        if not path.exists():
            return pd.DataFrame(columns=list(FACTOR_COLUMNS.keys()))
        try:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df["cum_factor"] = pd.to_numeric(df["cum_factor"], errors="coerce")
            return df.dropna(subset=["date", "cum_factor"]).sort_values("date").reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"读取因子文件 {path} 失败: {exc}")
            return pd.DataFrame(columns=list(FACTOR_COLUMNS.keys()))

    def write_factor(self, code: str, df: pd.DataFrame) -> None:
        """覆盖写因子表（新架构下因子直接刷新，无需自愈比对）。"""
        if df is None or df.empty:
            return
        path = self.factor_path(code)
        ensure_dir(path.parent)
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out["cum_factor"] = pd.to_numeric(out["cum_factor"], errors="coerce")
        out = out.dropna(subset=["date", "cum_factor"]).sort_values("date").reset_index(drop=True)
        out[["date", "cum_factor"]].to_parquet(path, index=False, compression="snappy")

    # ------------------------------------------------------------
    # 复权加载（hfq 缓存 + qfq/none 派生）
    # ------------------------------------------------------------

    def _hfq_cache_valid(self, code: str, freq: str) -> bool:
        """hfq 缓存有效 ⇔ 缓存存在且其 mtime ≥ 原始与因子文件的 mtime。"""
        cache = self.adjusted_path(code, freq)
        if not cache.exists():
            return False
        cache_mt = cache.stat().st_mtime
        raw = self.freq_path(code, freq)
        if raw.exists() and raw.stat().st_mtime > cache_mt:
            return False
        factor = self.factor_path(code)
        if factor.exists() and factor.stat().st_mtime > cache_mt:
            return False
        return True

    def _build_hfq_cache(self, code: str, freq: str) -> Optional[pd.DataFrame]:
        """重算并落盘 hfq 缓存；原始缺失返回 None。"""
        raw = self.read_raw(code, freq)
        if raw is None or raw.empty:
            return None
        factor = self.read_factor(code)
        hfq = apply_adjust(raw, factor, mode="hfq", time_col=_time_col(freq))
        path = self.adjusted_path(code, freq)
        ensure_dir(path.parent)
        hfq.to_parquet(path, index=False, compression="snappy")
        logger.debug(f"[{code}|{freq}] 重算 hfq 缓存（{len(hfq)} 行）")
        return hfq

    def get_hfq(self, code: str, freq: str) -> Optional[pd.DataFrame]:
        """取后复权数据（命中有效缓存直接读，否则重算覆盖）。"""
        if self._hfq_cache_valid(code, freq):
            try:
                df = pd.read_parquet(self.adjusted_path(code, freq))
                tc = _time_col(freq)
                df[tc] = pd.to_datetime(df[tc])
                return df.sort_values(tc).reset_index(drop=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[{code}|{freq}] 读 hfq 缓存失败，重算: {exc}")
        return self._build_hfq_cache(code, freq)

    def load(
        self,
        code: str,
        freq: str,
        adjust: str = "hfq",
        *,
        anchor_date: Optional[Union[str, date, datetime]] = None,
    ) -> Optional[pd.DataFrame]:
        """加载某周期数据并按 ``adjust`` 复权。

        - ``none``：原始不复权。
        - ``hfq`` ：读/重算 hfq 缓存。
        - ``qfq`` ：hfq 除以锚点日累计因子（标量），现场派生不落缓存。
        """
        mode = (adjust or "hfq").strip().lower()
        tc = _time_col(freq)
        if mode == "none":
            raw = self.read_raw(code, freq)
            if raw is None:
                return None
            return apply_adjust(raw, self.read_factor(code), mode="none", time_col=tc)

        hfq = self.get_hfq(code, freq)
        if hfq is None:
            return None
        if mode == "hfq":
            return hfq
        if mode == "qfq":
            anchor_f = cum_factor_at(self.read_factor(code), anchor_date)
            if not anchor_f or anchor_f == 1.0:
                return hfq
            out = hfq.copy()
            for c in ("open", "high", "low", "close", "limit_up", "limit_down", "adj_factor"):
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce") / anchor_f
            return out
        raise ValueError(f"未知复权模式 {mode!r}（支持 none/hfq/qfq）")
