"""
Akshare 源（日线 + 分钟；内部二次容灾：东财→新浪→腾讯），返回**不复权** OHLCV。

日线内部容灾：东财 ``stock_zh_a_hist`` → 新浪 ``stock_zh_a_daily`` →
              腾讯 ``stock_zh_a_hist_tx``。
分钟内部容灾：东财 ``stock_zh_a_hist_min_em`` → 新浪 ``stock_zh_a_minute``。
长区间（首拉）按年分片串行并 sleep 防风控。
"""

from __future__ import annotations

import random
import time
from datetime import date
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from ...utils.helpers import format_code
from .base import DataSourceBase, ProxyConfigError, _ak_call

#: akshare 防风控参数（长区间首拉分片）
AK_CHUNK_THRESHOLD_DAYS = 800   # 区间超过该天数才启用年分片
AK_CHUNK_YEARS = 3              # 每个分片覆盖年数
AK_CHUNK_SLEEP = 1.5           # 分片之间间隔秒数
AK_CALL_SLEEP = 0.5            # 单股内多次请求之间间隔秒数


class AkshareSource(DataSourceBase):
    name = "akshare"
    supports_minute = True

    def __init__(self, jitter: float = 0.0) -> None:
        self._jitter = max(0.0, float(jitter))

    def _sleep(self, base: float) -> None:
        time.sleep(base + (random.uniform(0, self._jitter) if self._jitter else 0.0))

    # ---------------- 日线 ----------------

    def fetch_daily(self, code: str, start: date, end: date) -> pd.DataFrame:
        span = (end - start).days
        if span <= AK_CHUNK_THRESHOLD_DAYS:
            return self._daily_failover(code, start, end)

        logger.info(
            f"[{code}] akshare 首拉保护：区间 {span} 天 > {AK_CHUNK_THRESHOLD_DAYS}，"
            f"按 {AK_CHUNK_YEARS} 年分片串行拉取"
        )
        parts: List[pd.DataFrame] = []
        cur, first = start, True
        while cur <= end:
            seg_end = min(date(cur.year + AK_CHUNK_YEARS, 1, 1) - pd.Timedelta(days=1), end)
            if not first:
                self._sleep(AK_CHUNK_SLEEP)
            first = False
            seg = self._daily_failover(code, cur, seg_end)
            if seg is not None and not seg.empty:
                parts.append(seg)
            cur = seg_end + pd.Timedelta(days=1)
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        return out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    def _daily_failover(self, code: str, start: date, end: date) -> pd.DataFrame:
        """单区间日线：东财→新浪→腾讯 顺序容灾。"""
        for fn in (self._daily_em, self._daily_sina, self._daily_tx):
            try:
                df = fn(code, start, end)
            except ProxyConfigError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[{code}] akshare 日线子源 {fn.__name__} 失败: {exc}")
                continue
            if df is not None and not df.empty:
                logger.debug(f"[{code}] akshare 日线命中子源 {fn.__name__}（{len(df)} 行）")
                return df
        return pd.DataFrame()

    def _daily_em(self, code: str, start: date, end: date) -> pd.DataFrame:
        import akshare as ak  # noqa: WPS433
        symbol = format_code(code).split(".")[0]
        df = _ak_call(
            ak.stock_zh_a_hist, code=code,
            symbol=symbol, period="daily",
            start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
            adjust="",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        rename = {
            "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
            "最低": "low", "成交量": "volume", "成交额": "amount",
        }
        df = df.rename(columns=rename)
        return self._finalize_daily(df)

    def _daily_sina(self, code: str, start: date, end: date) -> pd.DataFrame:
        import akshare as ak  # noqa: WPS433
        self._sleep(AK_CALL_SLEEP)
        df = _ak_call(
            ak.stock_zh_a_daily, code=code,
            symbol=self._sina_symbol(code),
            start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
            adjust="",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # 新浪列: date, open, high, low, close, volume, amount, ...
        return self._finalize_daily(df)

    def _daily_tx(self, code: str, start: date, end: date) -> pd.DataFrame:
        import akshare as ak  # noqa: WPS433
        self._sleep(AK_CALL_SLEEP)
        df = _ak_call(
            ak.stock_zh_a_hist_tx, code=code,
            symbol=self._sina_symbol(code),
            start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"),
            adjust="",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # 腾讯列: date, open, close, high, low, amount(实为成交量/手)；无成交额
        df = df.rename(columns={"amount": "volume"})
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100.0  # 手→股
        df["amount"] = np.nan
        return self._finalize_daily(df)

    @staticmethod
    def _finalize_daily(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        for c in ("open", "high", "low", "close", "volume", "amount"):
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
        out = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
        out["raw_close"] = out["close"].astype("float64")
        return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ---------------- 分钟 ----------------

    def fetch_minute(
        self, code: str, period: str, start: date, end: date
    ) -> pd.DataFrame:
        for fn in (self._minute_em, self._minute_sina):
            try:
                df = fn(code, period, start, end)
            except ProxyConfigError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[{code}] akshare {period}分钟子源 {fn.__name__} 失败: {exc}")
                continue
            if df is not None and not df.empty:
                return df
        return pd.DataFrame()

    def _minute_em(self, code: str, period: str, start: date, end: date) -> pd.DataFrame:
        import akshare as ak  # noqa: WPS433
        symbol = format_code(code).split(".")[0]
        df = _ak_call(
            ak.stock_zh_a_hist_min_em, code=code,
            symbol=symbol, period=str(period),
            start_date=f"{start:%Y-%m-%d} 09:00:00",
            end_date=f"{end:%Y-%m-%d} 15:30:00",
            adjust="",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        rename = {
            "时间": "datetime", "开盘": "open", "收盘": "close", "最高": "high",
            "最低": "low", "成交量": "volume", "成交额": "amount",
        }
        df = df.rename(columns=rename)
        return self._finalize_minute(df)

    def _minute_sina(self, code: str, period: str, start: date, end: date) -> pd.DataFrame:
        import akshare as ak  # noqa: WPS433
        self._sleep(AK_CALL_SLEEP)
        df = _ak_call(
            ak.stock_zh_a_minute, code=code,
            symbol=self._sina_symbol(code), period=str(period), adjust="",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # 新浪列: day, open, high, low, close, volume
        df = df.rename(columns={"day": "datetime"})
        df["amount"] = np.nan
        out = self._finalize_minute(df)
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) + pd.Timedelta(days=1)
        return out[(out["datetime"] >= s) & (out["datetime"] < e)].reset_index(drop=True)

    @staticmethod
    def _finalize_minute(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for c in ("open", "high", "low", "close", "volume", "amount"):
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
        out = df[["datetime", "open", "high", "low", "close", "volume", "amount"]].copy()
        out["raw_close"] = out["close"].astype("float64")
        return out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    @staticmethod
    def _sina_symbol(code: str) -> str:
        num, mkt = format_code(code).split(".")
        return f"{mkt.lower()}{num}"
