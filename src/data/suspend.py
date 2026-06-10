"""
停牌名单 Provider（按交易日整市场快照）
=======================================

职责：给定某个交易日，返回当天**全市场停牌（无全日成交）股票代码集合**，用于在
增量更新时**权威判定停牌**，替代"OHLCV 缺口反推"的二义性（缺口既可能是真停牌，
也可能是数据源当天尚未更新/源滞后）。

源优先级（可配置）
------------------
1. **东方财富**（主源，免 token，akshare ``stock_tfp_em(date=...)``）：
   数据中心-特色数据-两市停复牌，单次返回指定交易日的停复牌快照。
2. **tushare**（兜底，需 ``TUSHARE_TOKEN``，``pro.suspend_d(suspend_type='S', ...)``）：
   按交易日整市场返回全日停牌名单；免费账号限频但**每个交易日仅需 1 次调用**。

设计要点
--------
- **按交易日整市场**：一次调用拿到当天整个市场的停牌名单，与股票数无关。
- **落盘缓存**：``{store}/suspend/{YYYYMMDD}.parquet``；命中缓存即 0 网络调用，
  天然规避 tushare 1 小时 1 次的限频。
- **lookback 限频**：仅对"近 ``lookback_days`` 天内"的未缓存交易日联网拉取；更久远
  的历史日直接返回空集（由 fetcher 的缺口启发式兜底），避免首拉跨千日时海量请求。
- **线程安全**：内存缓存 + 一把锁，供 ``DataFetcher`` 多线程更新共享。

返回的代码统一为 ``format_code`` 标准格式（``XXXXXX.SH/SZ/BJ``）。
"""

from __future__ import annotations

import os
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Sequence, Union

import pandas as pd
from loguru import logger

from ..utils.helpers import ensure_dir, format_code
from .sources.base import _ak_call

#: 默认停牌名单源优先级（东财主源 → tushare 兜底）
DEFAULT_SUSPEND_SOURCES = ("eastmoney", "tushare")

#: 默认仅对近 N 天内未缓存的交易日联网拉取（更久远历史回退缺口启发式）
DEFAULT_SUSPEND_LOOKBACK_DAYS = 30


def _to_date(d: Union[str, date, datetime, pd.Timestamp]) -> date:
    """归一为 ``datetime.date``。"""
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    return pd.Timestamp(d).date()


def _norm_code(raw) -> Optional[str]:
    """东财/ tushare 原始代码 → ``format_code`` 标准格式；无法解析返回 None。"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    # 纯数字补足 6 位（东财个别接口可能丢前导 0）
    if "." not in s and s.isdigit():
        s = s.zfill(6)
    try:
        return format_code(s)
    except Exception:  # noqa: BLE001
        return None


class SuspendProvider:
    """按交易日的全市场停牌名单提供者（东财主源 + tushare 兜底 + 落盘缓存）。"""

    def __init__(
        self,
        store_path: Union[str, Path] = "data_store",
        sources: Sequence[str] = DEFAULT_SUSPEND_SOURCES,
        lookback_days: int = DEFAULT_SUSPEND_LOOKBACK_DAYS,
        enabled: bool = True,
    ) -> None:
        self._dir = ensure_dir(Path(store_path) / "suspend")
        self._sources = tuple(s.strip().lower() for s in sources if s and s.strip())
        self._lookback_days = max(0, int(lookback_days))
        self._enabled = bool(enabled)
        self._mem: Dict[str, FrozenSet[str]] = {}
        self._lock = threading.Lock()
        self._tushare_pro = None  # lazy
        self._tushare_failed = False

    # ------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------

    def get_suspended_set(self, trade_date: Union[str, date, datetime, pd.Timestamp]) -> FrozenSet[str]:
        """返回 ``trade_date`` 当日全市场停牌代码集合（标准代码）。

        命中内存/磁盘缓存直接返回；超出 lookback 的久远历史返回空集（不联网）。
        任何源失败时返回空集且不落缓存（下次重试）。
        """
        if not self._enabled:
            return frozenset()
        d = _to_date(trade_date)
        key = d.strftime("%Y%m%d")

        cached = self._mem.get(key)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._mem.get(key)
            if cached is not None:
                return cached

            disk = self._load_cache(key)
            if disk is not None:
                self._mem[key] = disk
                return disk

            # lookback 限频：久远历史不联网，交给 fetcher 缺口启发式
            if (date.today() - d).days > self._lookback_days:
                self._mem[key] = frozenset()
                return frozenset()

            result = self._fetch(d, key)
            self._mem[key] = result
            return result

    def is_suspended(self, code: str, trade_date) -> bool:
        """该股在 ``trade_date`` 是否停牌（便捷封装）。"""
        return format_code(code) in self.get_suspended_set(trade_date)

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------
    # 缓存读写
    # ------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._dir / f"{key}.parquet"

    def _load_cache(self, key: str) -> Optional[FrozenSet[str]]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if "code" not in df.columns:
                return frozenset()
            return frozenset(str(c) for c in df["code"].dropna().tolist())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"读取停牌缓存 {path} 失败（视为未缓存）: {exc}")
            return None

    def _write_cache(self, key: str, codes: FrozenSet[str], source: str) -> None:
        path = self._cache_path(key)
        try:
            df = pd.DataFrame({"code": sorted(codes)})
            df["source"] = source
            df.to_parquet(path, index=False, compression="snappy")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"写停牌缓存 {path} 失败: {exc}")

    # ------------------------------------------------------------
    # 取数（按源优先级）
    # ------------------------------------------------------------

    def _fetch(self, d: date, key: str) -> FrozenSet[str]:
        for src in self._sources:
            try:
                if src == "eastmoney":
                    codes = self._fetch_eastmoney(d, key)
                elif src == "tushare":
                    codes = self._fetch_tushare(d, key)
                else:
                    logger.debug(f"未知停牌源 {src}，跳过")
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"停牌源 {src} 拉取 {key} 失败: {type(exc).__name__}: {exc}")
                continue
            if codes is not None:
                self._write_cache(key, codes, src)
                logger.debug(f"停牌名单 {key} 命中源 {src}（{len(codes)} 只）")
                return codes
        logger.warning(f"停牌名单 {key} 所有源均失败/无数据，本次返回空集（不缓存，下次重试）")
        return frozenset()

    def _fetch_eastmoney(self, d: date, key: str) -> Optional[FrozenSet[str]]:
        """东方财富两市停复牌：``ak.stock_tfp_em(date='YYYYMMDD')``。"""
        import akshare as ak  # noqa: WPS433

        df = _ak_call(ak.stock_tfp_em, date=key)
        if df is None:
            return None
        if df.empty:
            return frozenset()  # 当日确无停牌（权威空集，可缓存）
        col = "代码" if "代码" in df.columns else df.columns[0]
        codes = {c for c in (_norm_code(v) for v in df[col].tolist()) if c}
        return frozenset(codes)

    def _fetch_tushare(self, d: date, key: str) -> Optional[FrozenSet[str]]:
        """tushare 全日停牌：``pro.suspend_d(suspend_type='S', trade_date='YYYYMMDD')``。"""
        pro = self._get_tushare_pro()
        if pro is None:
            return None
        df = pro.suspend_d(suspend_type="S", trade_date=key)
        if df is None:
            return None
        if df.empty:
            return frozenset()
        col = "ts_code" if "ts_code" in df.columns else df.columns[0]
        codes = {c for c in (_norm_code(v) for v in df[col].tolist()) if c}
        return frozenset(codes)

    def _get_tushare_pro(self):
        if self._tushare_pro is not None:
            return self._tushare_pro
        if self._tushare_failed:
            return None
        token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if not token:
            logger.debug("未配置 TUSHARE_TOKEN，停牌 tushare 兜底不可用")
            self._tushare_failed = True
            return None
        try:
            import tushare as ts  # noqa: WPS433
            ts.set_token(token)
            self._tushare_pro = ts.pro_api()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"初始化 tushare 失败，停牌兜底不可用: {exc}")
            self._tushare_failed = True
            return None
        return self._tushare_pro
