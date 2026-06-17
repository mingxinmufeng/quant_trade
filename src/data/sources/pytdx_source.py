"""
通达信本地源（读 vipdoc：日线 + 5 分钟 + 1 分钟），零网络、零 pytdx 依赖。

行情解析由自洽模块 :mod:`src.data.sources.tdx_reader` 完成，本模块只负责 ``DataSource``
接口、vipdoc 路径定位与区间裁剪。解析失败（文件损坏/未知板块）即返回空，由上层多源
fallback（akshare/baostock/tushare）接管，不再回退到 pytdx 库。

量纲：``tdx_reader`` 已把个股日线 volume 统一为「股」（pytdx 原版为「手」），并修正其对
科创板 ``sh68``、北交所 ``bj`` 的漏判/不支持。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ...utils.helpers import format_code
from . import tdx_reader
from .base import DataSourceBase

#: 通达信自动寻径候选目录
_TDX_AUTO_DISCOVERY_CANDIDATES = (
    r"C:\new_tdx", r"D:\new_tdx", r"E:\new_tdx", r"F:\new_tdx",
    r"C:\zd_tdx", r"D:\zd_tdx", r"C:\TDX", r"D:\TDX",
    r"C:\Program Files\new_tdx", r"C:\Program Files (x86)\new_tdx",
    r"D:\Program Files\new_tdx", r"D:\Program Files (x86)\new_tdx",
    r"C:\通达信", r"D:\通达信",
)


def _auto_discover_tdx_path() -> str | None:
    """扫描常见通达信安装位置，返回首个含 ``vipdoc`` 的有效路径。"""
    for candidate in _TDX_AUTO_DISCOVERY_CANDIDATES:
        if Path(candidate, "vipdoc").exists():
            return candidate
    return None


class PytdxLocalSource(DataSourceBase):
    """读取本地通达信 vipdoc 不复权数据（日线 / 5 分钟 / 1 分钟），零网络。"""

    name = "pytdx"
    supports_minute = True

    def __init__(self, tdx_path: str | None = None) -> None:
        resolved = (tdx_path or "").strip() or _auto_discover_tdx_path()
        if not resolved:
            raise RuntimeError(
                "未配置通达信路径且自动寻径失败；请在 config.yaml/config.private.yaml "
                "设置 data.tdx_path，或安装通达信到常见目录。"
            )
        if not Path(resolved, "vipdoc").exists():
            raise RuntimeError(f"通达信目录无效（未找到 vipdoc）: {resolved}")
        self._tdx_path = str(resolved)
        logger.debug(f"PytdxLocalSource 使用通达信路径: {self._tdx_path}")

    # ------------------------------------------------------------
    # vipdoc 路径定位
    # ------------------------------------------------------------

    def _vipdoc_path(self, market: str, subdir: str, fname: str) -> Path | None:
        """拼接 vipdoc 文件路径，不存在则返回 None。"""
        p = Path(self._tdx_path) / "vipdoc" / market / subdir / fname
        return p if p.exists() else None

    def _resolve_daily_file(self, code: str) -> Path | None:
        """根据代码定位日线 .day 文件（含北交所）。"""
        sym = format_code(code).split(".")[0].lower()
        if sym.startswith(("sh", "sz", "bj")):
            market = sym[:2]
            bare = sym[2:]
        elif sym.startswith("88"):
            market, bare = "sh", sym
        else:
            market = format_code(code).split(".")[1].lower()
            bare = sym
        return self._vipdoc_path(market, "lday", f"{market}{bare}.day")

    def _resolve_min_file(self, code: str, period: str) -> Path | None:
        """根据代码和周期定位分钟线文件。"""
        sym = format_code(code).split(".")[0].lower()
        if sym.startswith(("sh", "sz", "bj")):
            market = sym[:2]
            bare = sym[2:]
        else:
            market = format_code(code).split(".")[1].lower()
            bare = sym
        if str(period) == "5":
            return self._vipdoc_path(market, "fzline", f"{market}{bare}.lc5")
        return self._vipdoc_path(market, "minline", f"{market}{bare}.lc1")

    # ------------------------------------------------------------
    # 取数（tdx_reader 解析；失败返回空，由上层源 fallback 接管）
    # ------------------------------------------------------------

    def fetch_daily(self, code: str, start: date, end: date) -> pd.DataFrame:
        fpath = self._resolve_daily_file(code)
        if fpath is None:
            return pd.DataFrame()
        try:
            raw = tdx_reader.read_day(str(fpath), start=start)
        except Exception as exc:
            logger.debug(f"[{code}] tdx_reader 日线解析失败，返回空交由上层 fallback: {exc}")
            return pd.DataFrame()
        return self._slice(raw, start, end, time_col="date")

    def fetch_minute(
        self, code: str, period: str, start: date, end: date
    ) -> pd.DataFrame:
        fpath = self._resolve_min_file(code, period)
        if fpath is None:
            return pd.DataFrame()
        try:
            raw = tdx_reader.read_lc(str(fpath), start=start)
        except Exception as exc:
            logger.debug(f"[{code}] tdx_reader 分钟解析失败，返回空交由上层 fallback: {exc}")
            return pd.DataFrame()
        return self._slice(raw, start, end, time_col="datetime")

    # ------------------------------------------------------------
    # 规范化裁剪
    # ------------------------------------------------------------

    @staticmethod
    def _slice(raw, start: date, end: date, time_col: str) -> pd.DataFrame:
        """通达信读出的 DataFrame → 统一不复权 OHLCV，并按区间过滤。"""
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        df = raw.reset_index()
        # 统一时间列名：通达信索引可能叫 datetime / date
        ts_src = "datetime" if "datetime" in df.columns else (
            "date" if "date" in df.columns else df.columns[0]
        )
        df = df.rename(columns={ts_src: time_col})
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        cols = ["open", "high", "low", "close", "volume", "amount"]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[[time_col, *cols]].dropna(subset=[time_col])
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) + pd.Timedelta(days=1)  # 含 end 当日全部分钟
        df = df[(df[time_col] >= s) & (df[time_col] < e)].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame()
        df["raw_close"] = df["close"].astype("float64")
        return df
