"""
Mootdx 本地源（读通达信 vipdoc：日线 + 5 分钟 + 1 分钟），零网络。

含北交所(.BJ)修复：mootdx 自带 ``Reader`` 的 ``find_path`` 无法定位北交所文件
（未给 bj 文件加 ``bj`` 前缀，且 92 开头代码被误判为 sh）。本模块通过
``_build_bj_reader`` 子类覆写 ``find_path`` 修复，不改动已安装的 mootdx 包。
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from ...utils.helpers import format_code
from .base import DataSourceBase

#: 通达信自动寻径候选目录
_TDX_AUTO_DISCOVERY_CANDIDATES = (
    r"C:\new_tdx", r"D:\new_tdx", r"E:\new_tdx", r"F:\new_tdx",
    r"C:\zd_tdx", r"D:\zd_tdx", r"C:\TDX", r"D:\TDX",
    r"C:\Program Files\new_tdx", r"C:\Program Files (x86)\new_tdx",
    r"D:\Program Files\new_tdx", r"D:\Program Files (x86)\new_tdx",
    r"C:\通达信", r"D:\通达信",
)


def _auto_discover_tdx_path() -> Optional[str]:
    """扫描常见通达信安装位置，返回首个含 ``vipdoc`` 的有效路径。"""
    for candidate in _TDX_AUTO_DISCOVERY_CANDIDATES:
        if Path(candidate, "vipdoc").exists():
            return candidate
    return None


def _build_bj_reader(tdx_path: str):
    """
    构造一个修复北交所识别的 mootdx StdReader 子类实例。

    修复点：
    - 用本仓库 ``format_code`` 判定市场（正确覆盖 43/83/87/88/92 → BJ，
      避免 mootdx ``get_stock_market`` 把 92 开头误判为 sh）；
    - 对 sh/sz/**bj** 三个市场都给文件名加市场前缀
      （mootdx 原版只对 sh/sz 加前缀，导致 ``vipdoc/bj/lday/810011.day``
      无法命中真实文件 ``bj810011.day``）；
    - 扩展日线 bar reader 的 ``get_security_type``，识别 ``bj`` 交易所
      （tdxpy/mootdx 原版对 bj 文件抛 ``NotImplementedError``）。北交所股票
      价格/成交量系数与 A 股一致（[0.01, 0.01]）。
    """
    from mootdx.contrib.compat import MooTdxDailyBarReader  # noqa: WPS433
    from mootdx.reader import StdReader  # noqa: WPS433
    from mootdx.utils import to_data  # noqa: WPS433

    class _BJDailyBarReader(MooTdxDailyBarReader):  # type: ignore[misc]
        SECURITY_TYPE = list(MooTdxDailyBarReader.SECURITY_TYPE) + ["BJ_A_STOCK"]
        SECURITY_COEFFICIENT = {
            **MooTdxDailyBarReader.SECURITY_COEFFICIENT,
            "BJ_A_STOCK": [0.01, 0.01],
        }

        def get_security_type(self, fname):
            if str(fname[-12:-10]).lower() == "bj":
                return "BJ_A_STOCK"
            return super().get_security_type(fname)

    class BJAwareReader(StdReader):  # type: ignore[misc]
        def daily(self, symbol=None, **kwargs):
            stem = Path(symbol).stem
            vipdoc = self.find_path(symbol=stem, subdir="lday", suffix="day")
            reader = _BJDailyBarReader()
            result = reader.get_df(str(vipdoc)) if vipdoc else None
            return to_data(result, symbol=stem, **kwargs)

        def find_path(self, symbol=None, subdir="lday", suffix=None, **kwargs):
            sym = str(symbol)
            # 通达信板块指数 88**** 放在 sh 目录下（沿用 mootdx 特例）
            if "#" in sym:
                market = "ds"
            elif sym.startswith("88"):
                market = "sh"
            else:
                market = format_code(sym).split(".")[1].lower()  # sh/sz/bj

            if market in ("sh", "sz", "bj"):
                bare = sym.lower()
                for m in ("sh", "sz", "bj"):
                    if bare.startswith(m):
                        bare = bare[len(m):]
                sym = market + bare

            suffix = suffix if isinstance(suffix, list) else [suffix]
            if kwargs.get("debug"):
                return market, sym, suffix

            for ex_ in suffix:
                ex_ = str(ex_).strip(".")
                vipdoc = Path(self.tdxdir) / "vipdoc" / market / subdir / f"{sym}.{ex_}"
                if Path(vipdoc).exists():
                    return vipdoc
            return None

    return BJAwareReader(tdxdir=tdx_path)


class MootdxLocalSource(DataSourceBase):
    """读取本地通达信 vipdoc 不复权数据（日线 / 5 分钟 / 1 分钟），零网络。"""

    name = "mootdx"
    supports_minute = True

    def __init__(self, tdx_path: Optional[str] = None) -> None:
        resolved = (tdx_path or "").strip() or _auto_discover_tdx_path()
        if not resolved:
            raise RuntimeError(
                "未配置通达信路径且自动寻径失败；请在 config.yaml/config.private.yaml "
                "设置 data.tdx_path，或安装通达信到常见目录。"
            )
        if not Path(resolved, "vipdoc").exists():
            raise RuntimeError(f"通达信目录无效（未找到 vipdoc）: {resolved}")
        self._tdx_path = str(resolved)
        logger.debug(f"MootdxLocalSource 使用通达信路径: {self._tdx_path}")
        self._reader = None  # lazy
        self._reader_lock = threading.Lock()

    def _get_reader(self):
        if self._reader is None:
            with self._reader_lock:
                if self._reader is None:
                    self._reader = _build_bj_reader(self._tdx_path)
        return self._reader

    def fetch_daily(self, code: str, start: date, end: date) -> pd.DataFrame:
        symbol = format_code(code).split(".")[0]
        raw = self._get_reader().daily(symbol=symbol)
        return self._slice(raw, start, end, time_col="date")

    def fetch_minute(
        self, code: str, period: str, start: date, end: date
    ) -> pd.DataFrame:
        symbol = format_code(code).split(".")[0]
        # mootdx: suffix=5 → fzline(5分钟); suffix=1 → minline(1分钟)
        suffix = 5 if str(period) == "5" else 1
        raw = self._get_reader().minute(symbol=symbol, suffix=suffix)
        return self._slice(raw, start, end, time_col="datetime")

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
        df = df[[time_col] + cols].dropna(subset=[time_col])
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) + pd.Timedelta(days=1)  # 含 end 当日全部分钟
        df = df[(df[time_col] >= s) & (df[time_col] < e)].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame()
        df["raw_close"] = df["close"].astype("float64")
        return df
