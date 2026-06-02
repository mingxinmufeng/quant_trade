"""
A 股行情拉取器（多源容灾 + 多周期 + 增量更新 + 原始/因子分离落盘）
====================================================================

设计（重构后）
--------------
1. **职责单一**：``fetcher`` 只负责**采集与加工不复权原始数据**并落盘，
   外加**刷新复权因子表**。复权本身由 ``src.data.adjust`` 在 load 时按需合成
   （见 ``DataStore.load``），``fetcher`` 不再把复权价烤进文件。
2. **原始价与因子分离**（核心）：
   - ``{store}/daily|min5|min1/{code}.parquet`` 存**不复权** OHLCV + 涨跌停/停牌等元数据；
   - ``{store}/factors/{code}.parquet`` 存 ``date→cum_factor``，每次覆盖刷新；
   - 因此**不再需要因子自愈**：原始价永不被改写，因子表直接刷新即可。
3. **拉取优先级**：``mootdx``（通达信本地盘）→ ``akshare`` → ``baostock`` → ``tushare``。
   各源仅返回不复权 OHLCV（见 ``src.data.sources``）。
4. **增量更新**：日线 / 5 分钟 / 1 分钟三套原始周期；其余周期 load 时 resample。
5. **多线程**：``update`` 按股票并行（``max_workers``）；baostock 等非线程安全源已加锁。
6. **复权因子单源锁定**（见 ``FactorProvider``）：默认 sina，可选 tushare/em，不 fallback。
7. **代理**：发生代理错误抛 ``ProxyConfigError`` 提示关闭代理。
8. **北交所(.BJ)修复**：见 ``src.data.sources.mootdx_source._build_bj_reader``。

公开接口::

    DataFetcher.update(codes=None, freqs=("daily","min5","min1")) -> None
    DataFetcher.load_daily(code, start, end, period="daily", adjust="hfq") -> pd.DataFrame
    DataFetcher.load_minute(code, period, start, end, adjust="hfq") -> pd.DataFrame
    DataFetcher.load_batch(codes, start, end, adjust="hfq") -> dict[str, pd.DataFrame]
    DataFetcher.list_market_codes(include_bse=True) -> list[str]

``adjust`` ∈ {"none","hfq","qfq"}；``qfq`` 可传 ``anchor_date`` 锚定（回测可复现）。
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.helpers import (
    Timer,
    format_code,
    get_limit_pct,
    parse_date,
    retry,
)
from .factors import FactorProvider
from .resample import (  # 重导出，保持向后兼容
    DAILY_RESAMPLE_RULES,
    resample_daily,
    resample_minute,
)
from .sources import (
    DEFAULT_SOURCES,
    AkshareSource,
    BaostockSource,
    DataFetchError,
    DataSourceBase,
    MootdxLocalSource,
    ProxyConfigError,
    TushareSource,
    _ak_call,
    _auto_discover_tdx_path,
    build_source,
)
from .storage import (  # 重导出，保持向后兼容
    DAILY_COLUMNS,
    FREQ_DIRS,
    MINUTE_COLUMNS,
    DataStore,
)
from .trading_calendar import TradingCalendar

#: 默认重试参数
DEFAULT_RETRY_TIMES = 3
DEFAULT_RETRY_DELAYS: Sequence[float] = (1, 2, 4, 8)

#: 默认并行线程数（按股票并行；IO 密集，>1 即可显著提速）
DEFAULT_MAX_WORKERS = 4

#: A 股最早可拉取日期
EARLIEST_DATE = date(1990, 12, 19)

__all__ = [
    "DAILY_COLUMNS",
    "MINUTE_COLUMNS",
    "FREQ_DIRS",
    "DEFAULT_SOURCES",
    "DataFetcher",
    "DataFetchError",
    "ProxyConfigError",
    "FactorProvider",
    "resample_daily",
    "resample_minute",
]


# ============================================================
# 涨跌停回补（纯函数；原始价口径，不涉及复权）
# ============================================================


def _backfill_limit_df(df: pd.DataFrame, limit_pct: float) -> pd.DataFrame:
    """回补 ``limit_up`` / ``limit_down`` 的 NaN（典型为增量批次首行缺前收）。

    原始价口径：用前一行（不复权）``close`` 作为参考收盘价反推。仅填 NaN 单元，
    不覆盖已有值；首行无前收则保持 NaN。
    """
    if df is None or df.empty:
        return df
    if "limit_up" not in df.columns or "limit_down" not in df.columns:
        return df
    need = df["limit_up"].isna() | df["limit_down"].isna()
    if not need.any():
        return df
    df = df.sort_values("date").reset_index(drop=True)
    ref = pd.to_numeric(df["close"], errors="coerce")
    prev_ref = ref.shift(1)
    lu = (prev_ref * (1.0 + limit_pct)).round(2)
    ld = (prev_ref * (1.0 - limit_pct)).round(2)
    fill_up = df["limit_up"].isna() & lu.notna()
    fill_dn = df["limit_down"].isna() & ld.notna()
    df.loc[fill_up, "limit_up"] = lu[fill_up]
    df.loc[fill_dn, "limit_down"] = ld[fill_dn]
    return df


# ============================================================
# 主类：DataFetcher
# ============================================================


class DataFetcher:
    """多源容灾 + 多周期 + 增量更新 + 原始/因子分离落盘的行情拉取器。"""

    _STOCK_NAME_CACHE: Dict[str, str] = {}

    def __init__(
        self,
        store_path: Union[str, Path] = "data_store",
        sources: Sequence[str] = DEFAULT_SOURCES,
        retry_times: int = DEFAULT_RETRY_TIMES,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
        calendar: Optional[TradingCalendar] = None,
        tdx_path: Optional[str] = None,
        jitter: float = 0.3,
        factor_source: str = "sina",
        max_workers: int = DEFAULT_MAX_WORKERS,
        selfheal_window: Optional[int] = None,  # 已废弃：原始/因子分离后无需自愈
    ) -> None:
        import threading

        self._store = DataStore(store_path)
        self._store_path = Path(store_path)
        self._source_names = tuple(sources)
        self._retry_times = int(retry_times)
        self._retry_delays = list(retry_delays)
        self._calendar = calendar or TradingCalendar(store_path=self._store_path)
        self._tdx_path = (tdx_path or "").strip() or None
        self._jitter = max(0.0, float(jitter))
        self._max_workers = max(1, int(max_workers))
        if selfheal_window is not None:
            logger.debug("selfheal_window 参数已废弃（原始/因子分离后无需自愈），忽略")
        self._sources_cache: Dict[str, Optional[DataSourceBase]] = {}
        self._factors = FactorProvider(source=factor_source, jitter=self._jitter)
        # 多线程：保护源懒构建与股票名称表的并发初始化
        self._source_lock = threading.Lock()
        self._name_lock = threading.Lock()

    # ------------------------------------------------------------
    # 公开接口：更新
    # ------------------------------------------------------------

    def update(
        self,
        codes: Optional[Sequence[str]] = None,
        freqs: Sequence[str] = ("daily", "min5", "min1"),
        throttle: float = 0.3,
        progress_every: int = 200,
        max_workers: Optional[int] = None,
    ) -> None:
        """增量更新本地原始缓存（日线 / 5 分钟 / 1 分钟）并刷新复权因子表。

        Args:
            codes: 股票代码列表；None=更新本地 daily 目录已存在的全部股票。
            freqs: 要更新的周期子集。
            throttle: 每只股票处理完 sleep 秒数（防风控）。
            progress_every: 进度打印间隔。
            max_workers: 并行线程数；None=用实例默认。
        """
        for f in freqs:
            if f not in FREQ_DIRS:
                raise ValueError(f"未知周期 {f}（支持 {list(FREQ_DIRS)}）")

        if codes is None:
            codes = sorted(p.stem for p in self._store._dirs["daily"].glob("*.parquet"))
            if not codes:
                logger.warning("update(codes=None) 未发现本地缓存；请显式传入 codes")
                return

        codes = [format_code(c) for c in codes]
        ordered_freqs = tuple(f for f in ("daily", "min5", "min1") if f in freqs)
        total = len(codes)
        workers = max(1, int(max_workers if max_workers is not None else self._max_workers))
        workers = min(workers, total)

        logger.info(
            f"开始增量更新 | 股票数 {total} | 周期 {ordered_freqs} | "
            f"并发 {workers} | 源优先级 {self._source_names}"
        )
        # 预热共享只读资源，避免线程内并发首建（名称表 + mootdx reader）
        self._ensure_stock_name_table()
        if "mootdx" in self._source_names:
            try:
                self._get_source("mootdx")
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"mootdx 源预热跳过: {exc}")

        success, fail = 0, 0
        with Timer("数据更新"):
            if workers <= 1:
                for idx, code in enumerate(codes, start=1):
                    ok = self._update_code(code, ordered_freqs)
                    success += int(ok)
                    fail += int(not ok)
                    if progress_every and idx % progress_every == 0:
                        logger.info(f"进度 {idx}/{total} | 成功 {success} | 失败 {fail}")
                    if throttle > 0 and idx < total:
                        time.sleep(throttle + (random.uniform(0, self._jitter) if self._jitter else 0))
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fetch") as ex:
                    futures = {
                        ex.submit(self._update_code, code, ordered_freqs, throttle): code
                        for code in codes
                    }
                    try:
                        for done, fut in enumerate(as_completed(futures), start=1):
                            ok = fut.result()  # ProxyConfigError 在此抛出并中断
                            success += int(ok)
                            fail += int(not ok)
                            if progress_every and done % progress_every == 0:
                                logger.info(f"进度 {done}/{total} | 成功 {success} | 失败 {fail}")
                    except ProxyConfigError:
                        for f in futures:
                            f.cancel()
                        raise
        logger.info(f"更新完成 | 成功 {success} | 失败 {fail}")

    def _update_code(
        self, code: str, freqs: Sequence[str], throttle: float = 0.0,
    ) -> bool:
        """更新单只股票：先刷新因子表，再逐周期更新原始数据（线程池工作单元）。

        因子表与原始价分离，互不影响顺序；返回该股票是否全部周期成功。
        代理错误向上抛出以中断整个任务。
        """
        ok = True
        # 1) 刷新复权因子表（失败不阻塞原始数据更新；保留旧因子文件）
        self._update_factor(code)
        # 2) 逐周期更新原始（不复权）数据
        for freq in freqs:
            try:
                self._update_one(code, freq)
            except ProxyConfigError:
                raise  # 代理错误是全局性的，立即中断并提示
            except Exception as exc:  # 单只单周期失败不阻塞整体
                ok = False
                logger.error(f"[{code}|{freq}] 更新失败: {type(exc).__name__}: {exc}")
        if throttle > 0:
            time.sleep(throttle + (random.uniform(0, self._jitter) if self._jitter else 0))
        return ok

    def _update_factor(self, code: str) -> None:
        """拉取并覆盖写复权因子表；源暂时失败则保留旧因子文件、不报错。"""
        try:
            factor = self._factors.get_factor(code)
        except ProxyConfigError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{code}] 复权因子刷新失败（保留旧因子）: {exc}")
            return
        if factor is not None and not factor.empty:
            self._store.write_factor(code, factor)
            logger.debug(f"[{code}] 因子表已刷新（{len(factor)} 行）")

    # ------------------------------------------------------------
    # 公开接口：加载（按需复权）
    # ------------------------------------------------------------

    def load_daily(
        self,
        code: str,
        start: Union[str, date, datetime],
        end: Union[str, date, datetime],
        period: str = "daily",
        adjust: str = "hfq",
        anchor_date: Optional[Union[str, date, datetime]] = None,
    ) -> pd.DataFrame:
        """加载日线或日线以上周期（周/月/季/年由 daily resample 生成）。

        ``adjust`` ∈ {"none","hfq","qfq"}；``qfq`` 可传 ``anchor_date`` 锚定。
        """
        code = format_code(code)
        df = self._store.load(code, "daily", adjust, anchor_date=anchor_date)
        if df is None:
            raise FileNotFoundError(f"本地无 {code} 日线；请先 fetcher.update(['{code}'])")
        s, e = pd.Timestamp(parse_date(start)), pd.Timestamp(parse_date(end))
        df = df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)
        if period != "daily":
            df = resample_daily(df, period)
            df = df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)
        return df

    def load_minute(
        self,
        code: str,
        period: str,
        start: Union[str, date, datetime],
        end: Union[str, date, datetime],
        adjust: str = "hfq",
        anchor_date: Optional[Union[str, date, datetime]] = None,
    ) -> pd.DataFrame:
        """加载分钟线（按需复权）。

        - period == "1" / "5"：直接读落盘。
        - 5 的倍数（15/30/60）：由 min5 resample。
        - 其它（如 2/3/10）：由 min1 resample。
        """
        code = format_code(code)
        p = int(period)
        if p == 1 or p == 5:
            df = self._store.load(code, f"min{p}", adjust, anchor_date=anchor_date)
            if df is None:
                raise FileNotFoundError(f"本地无 {code} {p}分钟数据；请先 update")
        elif p % 5 == 0 and self._store.read_raw(code, "min5") is not None:
            base = self._store.load(code, "min5", adjust, anchor_date=anchor_date)
            df = resample_minute(base, 5, p)
        else:
            base = self._store.load(code, "min1", adjust, anchor_date=anchor_date)
            if base is None:
                raise FileNotFoundError(f"本地无 {code} 1分钟数据，无法生成 {p}分钟；请先 update")
            df = resample_minute(base, 1, p)
        s, e = pd.Timestamp(parse_date(start)), pd.Timestamp(parse_date(end)) + pd.Timedelta(days=1)
        return df[(df["datetime"] >= s) & (df["datetime"] < e)].reset_index(drop=True)

    def load_batch(
        self,
        codes: Sequence[str],
        start: Union[str, date, datetime],
        end: Union[str, date, datetime],
        adjust: str = "hfq",
        anchor_date: Optional[Union[str, date, datetime]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """批量加载日线；缺失的代码跳过并 WARNING。"""
        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            std = format_code(code)
            try:
                result[std] = self.load_daily(std, start, end, adjust=adjust, anchor_date=anchor_date)
            except FileNotFoundError as exc:
                logger.warning(str(exc))
        return result

    def list_market_codes(self, include_bse: bool = True) -> List[str]:
        """返回全市场 A 股代码清单（akshare ``stock_info_a_code_name``，独立于本地包）。"""
        self._ensure_stock_name_table()
        codes = list(self._STOCK_NAME_CACHE.keys())
        if not include_bse:
            codes = [c for c in codes if not c.endswith(".BJ")]
        return sorted(set(codes))

    # ------------------------------------------------------------
    # 内部：单股单周期增量更新（只采集不复权原始数据）
    # ------------------------------------------------------------

    def _update_one(self, code: str, freq: str) -> None:
        existing = self._store.read_raw(code, freq)
        time_col = "date" if freq == "daily" else "datetime"

        last_trading = (
            date.today() if self._calendar.is_trading_day(date.today())
            else self._calendar.previous_trading_day(date.today(), n=1)
        )
        # 交易日但尚未到该频率数据产生时间时，回退到上一交易日
        if self._calendar.is_trading_day(date.today()):
            now = datetime.now()
            cutoff = (16, 0) if freq == "daily" else (9, 30)
            if (now.hour, now.minute) < cutoff:
                last_trading = self._calendar.previous_trading_day(date.today(), n=1)

        first_pull = existing is None or existing.empty
        if not first_pull:
            local_max = existing[time_col].max()
            local_max_date = local_max.date() if hasattr(local_max, "date") else local_max
            if freq == "daily":
                if local_max_date >= last_trading:
                    logger.debug(f"[{code}|{freq}] 已最新（{local_max_date}），跳过")
                    return
                start = self._calendar.next_trading_day(local_max_date, n=1)
            else:
                # 分钟带状态增量：末日「完整」（末根 bar 收于 15:00）→ 从下一交易日开始；
                # 「不完整」（盘中拉取/半截数据）→ 重拉末日补齐。
                ts_max = pd.Timestamp(local_max)
                day_complete = (ts_max.hour, ts_max.minute) >= (15, 0)
                if day_complete and local_max_date >= last_trading:
                    logger.debug(f"[{code}|{freq}] 已最新且末日完整（{ts_max}），跳过")
                    return
                if day_complete:
                    start = self._calendar.next_trading_day(local_max_date, n=1)
                else:
                    logger.debug(
                        f"[{code}|{freq}] 末日 {local_max_date} 不完整"
                        f"（末根 {ts_max.time()}），重拉当天补齐"
                    )
                    start = local_max_date
        else:
            start = EARLIEST_DATE if freq == "daily" else date(2000, 1, 1)
        end = last_trading
        if start > end:
            return

        logger.debug(f"[{code}|{freq}] 拉取 {start}~{end} | {'首拉' if first_pull else '增量'}")
        if freq == "daily":
            new_df, src = self._fetch_with_fallback(code, start, end, kind="daily")
        else:
            period = "5" if freq == "min5" else "1"
            new_df, src = self._fetch_with_fallback(code, start, end, kind=period)
        if new_df.empty:
            logger.debug(f"[{code}|{freq}] 区间内所有源返回空")
            return

        # 不做复权：原始 OHLCV 直接规整入库（复权在 load 时按需合成）
        if freq == "daily":
            new_df = self._post_process_daily(code, new_df, start, end, source=src)
        else:
            new_df = self._post_process_minute(code, new_df, source=src)

        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[time_col], keep="last")
            combined = combined.sort_values(time_col).reset_index(drop=True)
        else:
            combined = new_df

        # 涨跌停首行回补：增量批次首行因缺前收导致 limit_up/down 为 NaN，
        # 拼接后用前一行（不复权）收盘价反推补齐。
        if freq == "daily":
            combined = _backfill_limit_df(combined, self._limit_pct_for(code))

        self._store.write_raw(code, freq, combined)
        logger.info(
            f"[{code}|{freq}] 更新成功 | +{len(new_df)} 行 | 末: {combined[time_col].max()}"
        )

    def _limit_pct_for(self, code: str) -> float:
        """该股票的涨跌停比例（依赖 ST 状态 + 板块）。"""
        try:
            name = self._get_stock_name(code)
        except Exception:  # noqa: BLE001
            name = ""
        is_st = bool(name) and ("ST" in name.upper())
        return get_limit_pct(code, is_st=is_st)

    # ------------------------------------------------------------
    # 内部：fallback 调度
    # ------------------------------------------------------------

    def _fetch_with_fallback(
        self, code: str, start: date, end: date, kind: str,
    ) -> Tuple[pd.DataFrame, str]:
        """按源优先级串行尝试。``kind`` ∈ {"daily","1","5"}。"""
        names = tuple(self._source_names)
        fetch_exc: Optional[Exception] = None
        last_exc: Optional[Exception] = None
        did_fetch = False

        for src_name in names:
            try:
                src = self._get_source(src_name)
            except Exception as exc:  # 源不可用
                logger.warning(f"[{code}] 源 {src_name} 不可用: {exc}")
                last_exc = exc
                continue

            if kind != "daily" and not src.supports_minute:
                continue

            def _call():
                if kind == "daily":
                    return src.fetch_daily(code, start, end)
                return src.fetch_minute(code, kind, start, end)

            fetch = retry(max_attempts=self._retry_times, delays=self._retry_delays)(_call)
            try:
                df = fetch()
                if df is None:
                    df = pd.DataFrame()
            except ProxyConfigError:
                raise
            except Exception as exc:  # noqa: BLE001
                fetch_exc = exc
                last_exc = exc
                logger.warning(f"[{code}] 源 {src_name} 重试耗尽: {type(exc).__name__}: {exc}")
                continue

            did_fetch = True
            if df.empty:
                logger.debug(f"[{code}] 源 {src_name} 返回空，降级下一源")
                continue
            return df, src_name

        if fetch_exc is not None:
            raise DataFetchError(f"[{code}] 所有源 {names} 均失败") from fetch_exc
        if did_fetch:
            return pd.DataFrame(), ""
        raise DataFetchError(f"[{code}] 所有源 {names} 均不可用") from last_exc

    def _get_source(self, name: str) -> DataSourceBase:
        if name not in self._sources_cache:
            with self._source_lock:
                if name not in self._sources_cache:
                    self._sources_cache[name] = build_source(
                        name, tdx_path=self._tdx_path, jitter=self._jitter
                    )
        src = self._sources_cache[name]
        assert src is not None
        return src

    # ------------------------------------------------------------
    # 内部：后处理（不复权原始价口径）
    # ------------------------------------------------------------

    def _post_process_daily(
        self, code: str, df: pd.DataFrame, start: date, end: date, source: str = "",
    ) -> pd.DataFrame:
        """日线规整为 DAILY_COLUMNS（不复权）：交易日对齐、停牌补行、涨跌停、名称、来源。"""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)

        all_days = self._calendar.get_trading_days(start, end)
        full_idx = pd.DatetimeIndex(pd.to_datetime(all_days))
        df = df.set_index("date").reindex(full_idx)
        df.index.name = "date"
        df = df.reset_index()

        susp_col = df.pop("_suspended") if "_suspended" in df.columns else None

        is_suspended = df["close"].isna() | (df["volume"].fillna(0) <= 0)
        if susp_col is not None:
            is_suspended = is_suspended | susp_col.fillna(False).astype(bool)
        df["is_suspended"] = is_suspended.astype(bool)

        # 涨跌停：用不复权前收反推（原始价口径）
        prev_ref = df["close"].shift(1)
        try:
            stock_name = self._get_stock_name(code)
        except Exception:  # noqa: BLE001
            stock_name = ""
        is_st = bool(stock_name) and ("ST" in stock_name.upper())
        limit_pct = get_limit_pct(code, is_st=is_st)
        df["limit_up"] = (prev_ref * (1.0 + limit_pct)).round(2)
        df["limit_down"] = (prev_ref * (1.0 - limit_pct)).round(2)

        df["code"] = format_code(code)
        df["name"] = stock_name
        df["source"] = source or "unknown"
        df.loc[df["close"].isna(), "source"] = f"{source or 'unknown'}:gap"

        for col in DAILY_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[list(DAILY_COLUMNS.keys())]
        return self._cast_types(df, DAILY_COLUMNS)

    def _post_process_minute(self, code: str, df: pd.DataFrame, source: str = "") -> pd.DataFrame:
        """分钟线规整为 MINUTE_COLUMNS（不复权）。"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["code"] = format_code(code)
        df["source"] = source or "unknown"
        for col in MINUTE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[list(MINUTE_COLUMNS.keys())]
        return self._cast_types(df, MINUTE_COLUMNS).sort_values("datetime").reset_index(drop=True)

    @staticmethod
    def _cast_types(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
        for col, dtype in schema.items():
            if dtype.startswith("datetime"):
                df[col] = pd.to_datetime(df[col])
            elif dtype == "string":
                df[col] = df[col].fillna("").astype("string")
            elif dtype == "bool":
                df[col] = df[col].fillna(False).astype(bool)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        return df

    # ------------------------------------------------------------
    # 内部：股票名称（ST 判定）
    # ------------------------------------------------------------

    def _ensure_stock_name_table(self) -> None:
        if self._STOCK_NAME_CACHE:
            return
        with self._name_lock:
            if self._STOCK_NAME_CACHE:  # 等锁期间已被别的线程填充
                return
            try:
                import akshare as ak  # noqa: WPS433
                df = _ak_call(ak.stock_info_a_code_name)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        self._STOCK_NAME_CACHE[format_code(str(row["code"]))] = str(row["name"])
            except ProxyConfigError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"加载股票名称表失败: {exc}")

    def _get_stock_name(self, code: str) -> str:
        std = format_code(code)
        if std in self._STOCK_NAME_CACHE:
            return self._STOCK_NAME_CACHE[std]
        self._ensure_stock_name_table()
        return self._STOCK_NAME_CACHE.get(std, "")

    def __repr__(self) -> str:
        return (
            f"<DataFetcher store={self._store_path} sources={self._source_names} "
            f"workers={self._max_workers}>"
        )


# ============================================================
# 命令行 / 数据源自测入口
# ============================================================
#
# 用法（仓库根目录执行）：
#   python -m src.data.fetcher --selftest          # 各数据源连通性自测（默认）
#   python -m src.data.fetcher --codes 000001.SZ 600519.SH 830799.BJ
#   python -m src.data.fetcher --codes 000001.SZ --freqs daily --max-workers 1 只更日线串行
#   python -m src.data.fetcher                      # 全市场增量（日线+5min+1min）


def _selftest(tdx_path: Optional[str] = None, level: str = "INFO") -> int:
    """逐源连通性自测：验证每个数据源在样例股票上能否取到有效数据。"""
    from datetime import timedelta

    from ..utils.helpers import init_logging
    init_logging(level=level)

    end = date.today()
    start = end - timedelta(days=30)
    start_5d = end - timedelta(days=5)
    start_3d = end - timedelta(days=3)
    sample_daily = "000001.SZ"
    sample_bj = "830799.BJ"
    ok_all = True

    def _try(label: str, fn) -> None:
        nonlocal ok_all
        try:
            df = fn()
            n = 0 if df is None else len(df)
            if n > 0:
                logger.success(f"✓ {label}: {n} 行")
            else:
                logger.warning(f"… {label}: 返回空（可能无数据/未下载本地包）")
        except ProxyConfigError as exc:
            ok_all = False
            logger.error(f"✗ {label}: 代理错误 → {exc}")
        except Exception as exc:  # noqa: BLE001
            ok_all = False
            logger.error(f"✗ {label}: {type(exc).__name__}: {exc}")

    logger.info("===== 数据源自测开始 =====")

    # 1) mootdx 本地（含北交所 BJ 修复验证）
    try:
        mootdx = MootdxLocalSource(tdx_path=tdx_path)
        _try("mootdx 日线 000001", lambda: mootdx.fetch_daily(sample_daily, start, end))
        _try("mootdx 5分钟 000001", lambda: mootdx.fetch_minute(sample_daily, "5", start_5d, end))
        _try("mootdx 1分钟 000001", lambda: mootdx.fetch_minute(sample_daily, "1", start_3d, end))
        _try("mootdx 北交所 BJ 修复", lambda: mootdx.fetch_daily(sample_bj, start, end))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"mootdx 不可用（未安装通达信/缺包）: {exc}")

    # 2) akshare（内部 东财/新浪/腾讯 容灾）
    ak_src = AkshareSource(jitter=0.3)
    _try("akshare 日线", lambda: ak_src.fetch_daily(sample_daily, start, end))
    _try("akshare 5分钟", lambda: ak_src.fetch_minute(sample_daily, "5", start, end))
    _try("akshare 1分钟", lambda: ak_src.fetch_minute(sample_daily, "1", start_3d, end))

    # 3) 复权因子统一拉取
    fp = FactorProvider(jitter=0.3)
    _try("复权因子 000001", lambda: fp.get_factor(sample_daily))

    # 4) baostock
    try:
        _try("baostock 日线", lambda: BaostockSource().fetch_daily(sample_daily, start, end))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"baostock 不可用: {exc}")

    # 5) tushare（需 token）
    try:
        _try("tushare 日线", lambda: TushareSource().fetch_daily(sample_daily, start, end))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"tushare 跳过: {exc}")

    # 6) resample 自测（纯本地，不依赖网络）
    try:
        idx = pd.date_range("2024-01-01 09:35", periods=48, freq="5min")
        demo = pd.DataFrame({
            "datetime": idx, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
            "volume": 100.0, "amount": 150.0,
        })
        r15 = resample_minute(demo, 5, 15)
        assert len(r15) == 16, f"5min→15min 期望 16 根, 实际 {len(r15)}"
        logger.success(f"✓ resample 5min→15min: {len(r15)} 行")
        dd = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=20, freq="B"),
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
            "volume": 100.0, "amount": 150.0,
        })
        rw = resample_daily(dd, "weekly")
        logger.success(f"✓ resample daily→weekly: {len(rw)} 行")
    except Exception as exc:  # noqa: BLE001
        ok_all = False
        logger.error(f"✗ resample 自测失败: {exc}")

    logger.info("===== 数据源自测结束 =====")
    return 0 if ok_all else 1


def _main() -> int:
    import argparse
    from ..utils.helpers import init_logging

    parser = argparse.ArgumentParser(description="A 股多周期数据拉取 / 增量更新 / 数据源自测")
    parser.add_argument("--selftest", action="store_true", help="逐数据源连通性自测后退出")
    parser.add_argument("--codes", nargs="*", default=None, help="指定代码（留空=akshare 全市场）")
    parser.add_argument("--freqs", nargs="*", default=["daily", "min5", "min1"], help="更新周期子集")
    parser.add_argument("--throttle", type=float, default=0.3, help="每只股票间 sleep 秒数")
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"并行线程数（按股票并行，默认 {DEFAULT_MAX_WORKERS}；1=串行）",
    )
    parser.add_argument("--store-path", default="data_store", help="数据仓库根目录")
    parser.add_argument("--tdx-path", default=None, help="通达信目录（留空自动寻径）")
    parser.add_argument("--no-bse", action="store_true", help="全市场清单剔除北交所")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"],
        help="日志级别；DEBUG 可看到完整容灾链路",
    )
    args = parser.parse_args()

    if args.selftest:
        return _selftest(tdx_path=args.tdx_path, level=args.log_level)

    init_logging(level=args.log_level)
    fetcher = DataFetcher(
        store_path=args.store_path, tdx_path=args.tdx_path, max_workers=args.max_workers,
    )
    logger.info(f"实例化: {fetcher}")

    if args.codes:
        codes = [format_code(c) for c in args.codes]
    else:
        codes = fetcher.list_market_codes(include_bse=not args.no_bse)
        if not codes:
            logger.error("未能获取全市场代码清单（网络/接口异常），终止")
            return 1
        logger.info(f"全市场代码数: {len(codes)}（含北交所={not args.no_bse}）")

    fetcher.update(
        codes, freqs=tuple(args.freqs), throttle=args.throttle,
        max_workers=args.max_workers,
    )
    logger.success("数据拉取/更新完成")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
