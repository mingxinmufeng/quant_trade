"""
复权因子提供器（外部单源）
==========================

统一拉取「日期 → 累计后复权因子(``cum_factor``)」表，按股票缓存。

**单源锁定**（保证因子刻度一致，避免多源归一化差异）：仅使用 ``source`` 指定的
一个因子源，**不**在多源间 fallback。
  - ``sina``    ak.stock_zh_a_daily(adjust="hfq-factor")
  - ``tushare`` pro.adj_factor
  - ``em``      东财 hfq_close / raw_close 比值（ak.stock_zh_a_hist）

该源暂时失败 → 返回空表（不缓存空结果，后续可重试）。

设计说明
--------
新架构下原始价与因子**分离落盘**，因子表每次刷新直接覆盖即可，因此**不再需要**
旧版的"零容差因子自愈"。本类只负责"取到最新的累计因子表"。

未来自算因子（点2）：可新增 ``FactorCalculator``（由通达信 gbbq 等公司行为事件
按复权公式累乘），产出**同一张** ``DataFrame[date, cum_factor]``，与外部源互验，
下游 ``adjust`` 无感知。当前先不实现自算公式，仅保留扩展点。
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.helpers import format_code
from .sources.base import ProxyConfigError, _ak_call

#: 因子表落盘 / 内存 schema
FACTOR_COLUMNS = {"date": "datetime64[ns]", "cum_factor": "float64"}

#: em 源两次请求之间的基础间隔（秒）
_EM_CALL_SLEEP = 0.5


class FactorProvider:
    """统一拉取「日期 → 累计后复权因子」表，按股票缓存，单源锁定不 fallback。"""

    def __init__(self, source: str = "sina", jitter: float = 0.0) -> None:
        self._cache: Dict[str, pd.DataFrame] = {}
        self._jitter = max(0.0, float(jitter))
        self._pro = None  # tushare lazy
        self._source = (source or "sina").strip().lower()
        # 多线程：每只股票一把锁，保证同一代码只拉取一次因子（避免并发重复请求）
        self._lock = threading.Lock()
        self._code_locks: Dict[str, threading.Lock] = {}

    def _code_lock(self, code: str) -> threading.Lock:
        with self._lock:
            lk = self._code_locks.get(code)
            if lk is None:
                lk = threading.Lock()
                self._code_locks[code] = lk
            return lk

    def _source_fn(self):
        mapping = {
            "sina": self._from_sina,
            "tushare": self._from_tushare,
            "em": self._from_em,
        }
        fn = mapping.get(self._source)
        if fn is None:
            raise ValueError(
                f"未知复权因子源 {self._source!r}（支持 {sorted(mapping)}）"
            )
        return fn

    def get_factor(self, code: str) -> pd.DataFrame:
        """返回 DataFrame[date, cum_factor]（已按 date 升序）；失败返回空表。

        多线程安全：同一代码加锁串行，命中缓存直接返回，避免并发重复网络请求。
        """
        std = format_code(code)
        if std in self._cache:
            return self._cache[std]
        with self._code_lock(std):
            if std in self._cache:  # 等锁期间已被别的线程填充
                return self._cache[std]
            fn = self._source_fn()
            try:
                df = fn(std)
            except ProxyConfigError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[{std}] 复权因子源 {self._source} 失败: {exc}")
                df = None
            if df is not None and not df.empty:
                df = df.sort_values("date").reset_index(drop=True)
                self._cache[std] = df
                return df
        logger.warning(
            f"[{std}] 复权因子主源 {self._source} 暂时失败"
            f"（不缓存空结果，后续可重试）"
        )
        return pd.DataFrame(columns=list(FACTOR_COLUMNS.keys()))

    def _sleep(self, base: float = _EM_CALL_SLEEP) -> None:
        """防风控节流：基础间隔 + 随机抖动（多线程下错峰，降低被限频概率）。"""
        time.sleep(base + (random.uniform(0, self._jitter) if self._jitter else 0.0))

    def _from_sina(self, code: str) -> pd.DataFrame:
        import akshare as ak  # noqa: WPS433
        self._sleep()  # 新浪源防风控：请求前节流
        num, mkt = code.split(".")
        df = _ak_call(
            ak.stock_zh_a_daily, code=code,
            symbol=f"{mkt.lower()}{num}", adjust="hfq-factor",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
        df["cum_factor"] = pd.to_numeric(df["hfq_factor"], errors="coerce")
        return df[["date", "cum_factor"]].dropna()

    def _from_tushare(self, code: str) -> pd.DataFrame:
        if self._pro is None:
            with self._lock:
                if self._pro is None:
                    token = os.environ.get("TUSHARE_TOKEN", "").strip()
                    if not token:
                        return pd.DataFrame()
                    import tushare as ts  # noqa: WPS433
                    ts.set_token(token)
                    self._pro = ts.pro_api()
        df = self._pro.adj_factor(ts_code=code)
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["trade_date"]).dt.normalize().astype("datetime64[ns]")
        df["cum_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
        return df[["date", "cum_factor"]].dropna()

    def _from_em(self, code: str) -> pd.DataFrame:
        import akshare as ak  # noqa: WPS433
        symbol = code.split(".")[0]
        hfq = _ak_call(
            ak.stock_zh_a_hist, code=code,
            symbol=symbol, period="daily", adjust="hfq",
        )
        time.sleep(_EM_CALL_SLEEP + (random.uniform(0, self._jitter) if self._jitter else 0))
        raw = _ak_call(
            ak.stock_zh_a_hist, code=code,
            symbol=symbol, period="daily", adjust="",
        )
        if hfq is None or hfq.empty or raw is None or raw.empty:
            return pd.DataFrame()
        h = hfq.rename(columns={"日期": "date", "收盘": "close_hfq"})[["date", "close_hfq"]]
        r = raw.rename(columns={"日期": "date", "收盘": "close_raw"})[["date", "close_raw"]]
        m = pd.merge(h, r, on="date", how="inner")
        m["date"] = pd.to_datetime(m["date"]).dt.normalize().astype("datetime64[ns]")
        m["cum_factor"] = np.where(
            pd.to_numeric(m["close_raw"], errors="coerce") > 0,
            pd.to_numeric(m["close_hfq"], errors="coerce")
            / pd.to_numeric(m["close_raw"], errors="coerce"),
            1.0,
        )
        return m[["date", "cum_factor"]].dropna()
