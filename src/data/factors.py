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

自算因子：``FactorCalculator``（见本文件下方）由通达信 gbbq 除权除息事件 + 本地不复权
收盘价按复权公式累乘，产出**同一张** ``DataFrame[date, cum_factor]``，下游 ``adjust``
无感知。完全离线、零网络请求，可作为可选因子源或与外部源互验。
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections.abc import Callable

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
        self._cache: dict[str, pd.DataFrame] = {}
        self._jitter = max(0.0, float(jitter))
        self._pro = None  # tushare lazy
        self._source = (source or "sina").strip().lower()
        # 多线程：每只股票一把锁，保证同一代码只拉取一次因子（避免并发重复请求）
        self._lock = threading.Lock()
        self._code_locks: dict[str, threading.Lock] = {}

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
            except Exception as exc:
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
        import akshare as ak
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
                    import tushare as ts
                    ts.set_token(token)
                    self._pro = ts.pro_api()
        df = self._pro.adj_factor(ts_code=code)
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["trade_date"]).dt.normalize().astype("datetime64[ns]")
        df["cum_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
        return df[["date", "cum_factor"]].dropna()

    def _from_em(self, code: str) -> pd.DataFrame:
        import akshare as ak
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


class FactorCalculator:
    """由通达信 ``gbbq`` 除权除息事件 + 本地不复权收盘价**自算**累计后复权因子。

    与 ``FactorProvider`` 同接口（``get_factor(code) -> DataFrame[date, cum_factor]``）、
    同 schema，可作为下游无感知的可选因子源（完全离线，零网络请求）。

    刻度约定
    --------
    以**最早一个交易日**为基准 ``cum_factor = 1.0``；每遇一个除权除息日 ``D``，按

        除权参考价 ex = (前收盘 - 每股现金分红 + 每股配股×配股价) / (1 + 每股送转 + 每股配股)
        当日因子增量 = 前收盘 / ex
        cum_factor(D) = cum_factor(D⁻) × (前收盘 / ex)

    累乘。产出只含**变化点**（基准日 + 各除权日）的稀疏因子表，非除权日由
    ``adjust.align_cum_factor`` 的 backward 对齐自动向后填充，结果与逐日表等价。

    与外部源（sina/tushare）的**绝对刻度可能不同**（基准取最早日=1.0），但任意两日之间
    的比值一致，故 hfq 收益序列等价。
    """

    def __init__(
        self,
        gbbq_store,
        raw_close_loader: Callable[[str], pd.DataFrame | None],
    ) -> None:
        """
        Args:
            gbbq_store: ``src.data.gbbq.GbbqStore`` 实例（提供 ``events(code)``）。
            raw_close_loader: ``code -> DataFrame[date, close]``（**不复权**日线，可含其它列）。
        """
        self._gbbq = gbbq_store
        self._load_raw = raw_close_loader
        self._cache: dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()
        self._code_locks: dict[str, threading.Lock] = {}

    def _code_lock(self, code: str) -> threading.Lock:
        with self._lock:
            lk = self._code_locks.get(code)
            if lk is None:
                lk = threading.Lock()
                self._code_locks[code] = lk
            return lk

    def get_factor(self, code: str) -> pd.DataFrame:
        """返回 DataFrame[date, cum_factor]（按 date 升序）；无原始价/无事件返回空表。"""
        std = format_code(code)
        if std in self._cache:
            return self._cache[std]
        with self._code_lock(std):
            if std in self._cache:
                return self._cache[std]
            raw = self._load_raw(std)
            events = self._gbbq.events(std)
            df = self._compute(raw, events)
            if df is not None and not df.empty:
                self._cache[std] = df
            return df

    @staticmethod
    def _compute(raw: pd.DataFrame | None, events: pd.DataFrame) -> pd.DataFrame:
        empty = pd.DataFrame(columns=list(FACTOR_COLUMNS.keys()))
        if raw is None or raw.empty or "close" not in raw.columns or "date" not in raw.columns:
            return empty
        r = raw[["date", "close"]].copy()
        r["date"] = pd.to_datetime(r["date"]).dt.normalize()
        r["close"] = pd.to_numeric(r["close"], errors="coerce")
        r = r.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if r.empty:
            return empty

        closes = r.set_index("date")["close"]
        # 基准行：最早一日 cum_factor = 1.0
        rows = [(r["date"].iloc[0], 1.0)]
        cum = 1.0

        if events is not None and not events.empty:
            ev = events.sort_values("date")
            for _, row in ev.iterrows():
                d = pd.Timestamp(row["date"]).normalize()
                prior = closes[closes.index < d]
                if prior.empty:
                    continue  # 除权日早于本地最早行情，无前收可用，跳过
                p_prev = float(prior.iloc[-1])
                if p_prev <= 0:
                    continue
                div = float(row.get("fenhong", 0.0)) / 10.0   # 每股现金分红
                song = float(row.get("song", 0.0)) / 10.0     # 每股送转
                pei = float(row.get("pei", 0.0)) / 10.0       # 每股配股
                peijia = float(row.get("peijia", 0.0))        # 配股价
                denom = 1.0 + song + pei
                if denom <= 0:
                    continue
                ex = (p_prev - div + pei * peijia) / denom
                if ex <= 0:
                    continue
                cum *= p_prev / ex
                rows.append((d, cum))

        out = pd.DataFrame(rows, columns=["date", "cum_factor"])
        out = (
            out.drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        return out
