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
3. **拉取优先级**：``pytdx``（通达信本地盘）→ ``akshare`` → ``baostock`` → ``tushare``。
   各源仅返回不复权 OHLCV（见 ``src.data.sources``）。
4. **增量更新**：日线 / 5 分钟 / 1 分钟三套原始周期；其余周期 load 时 resample。
5. **多线程**：``update`` 按股票并行（``max_workers``）；baostock 等非线程安全源已加锁。
6. **复权因子单源锁定**（见 ``FactorProvider``）：默认 sina，可选 tushare/em，不 fallback。
7. **代理**：发生代理错误抛 ``ProxyConfigError`` 提示关闭代理。
8. **北交所(.BJ)修复**：见 ``src.data.sources.pytdx_source._build_daily_reader``。

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
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import ClassVar

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
from .factors import FactorCalculator, FactorProvider
from .gbbq import GbbqStore
from .resample import (  # 重导出，保持向后兼容
    resample_daily,
    resample_minute,
)
from .sources import (
    DEFAULT_SOURCES,
    AkshareSource,
    BaostockSource,
    DataFetchError,
    DataSourceBase,
    ProxyConfigError,
    PytdxLocalSource,
    TushareSource,
    _ak_call,
    build_source,
)
from .storage import (  # 重导出，保持向后兼容
    DAILY_COLUMNS,
    FREQ_DIRS,
    MINUTE_COLUMNS,
    DataStore,
)
from .suspend import (
    DEFAULT_SUSPEND_LOOKBACK_DAYS,
    DEFAULT_SUSPEND_SOURCES,
    SuspendProvider,
)
from .trading_calendar import TradingCalendar

#: 默认重试参数
DEFAULT_RETRY_TIMES = 3
DEFAULT_RETRY_DELAYS: Sequence[float] = (1, 2, 4, 8)

#: 默认并行线程数（按股票并行；pytdx 为本地文件 IO，可用更多线程）
DEFAULT_MAX_WORKERS = 16

#: A 股最早可拉取日期
EARLIEST_DATE = date(1990, 12, 19)

__all__ = [
    "DAILY_COLUMNS",
    "DEFAULT_SOURCES",
    "FREQ_DIRS",
    "MINUTE_COLUMNS",
    "DataFetchError",
    "DataFetcher",
    "FactorCalculator",
    "FactorProvider",
    "GbbqStore",
    "ProxyConfigError",
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

    _STOCK_NAME_CACHE: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        store_path: str | Path = "data_store",
        sources: Sequence[str] = DEFAULT_SOURCES,
        retry_times: int = DEFAULT_RETRY_TIMES,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
        calendar: TradingCalendar | None = None,
        tdx_path: str | None = None,
        jitter: float = 0.3,
        factor_source: str = "sina",
        max_workers: int = DEFAULT_MAX_WORKERS,
        factor_skip_via_gbbq: bool = True,
        load_factor_source: str = "active",
        suspend_sources: Sequence[str] = DEFAULT_SUSPEND_SOURCES,
        suspend_lookback_days: int = DEFAULT_SUSPEND_LOOKBACK_DAYS,
        suspend_enabled: bool = True,
        selfheal_window: int | None = None,  # 已废弃：原始/因子分离后无需自愈
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
        self._sources_cache: dict[str, DataSourceBase | None] = {}
        # 本地 gbbq（权息）：自算因子的数据来源、因子刷新触发器；事件快照落盘到 store 根下
        self._gbbq = GbbqStore(
            tdx_path=self._tdx_path,
            snapshot_path=self._store_path / "gbbq_events.parquet",
        )
        self._factor_skip_via_gbbq = bool(factor_skip_via_gbbq)
        self._factor_source = (factor_source or "sina").strip().lower()
        # gbbq 自算因子引擎：始终构建（即便默认走外部源，也并存落盘 factors_gbbq/ 供对比）
        self._gbbq_calc = FactorCalculator(
            self._gbbq, raw_close_loader=lambda c: self._store.read_raw(c, "daily"),
        )
        # 生效因子引擎：gbbq=本地自算；其余=外部源（FactorProvider）
        self._factor_engine = (
            self._gbbq_calc if self._factor_source == "gbbq"
            else FactorProvider(source=self._factor_source, jitter=self._jitter)
        )
        self._factors = self._factor_engine  # 向后兼容别名
        # 加载（复权）默认采用哪套因子：'active'=factors/（生效源），'gbbq'=factors_gbbq/
        self._load_factor_source = (load_factor_source or "active").strip().lower()
        # 停牌名单 provider（东财主源 + tushare 兜底，按交易日落盘缓存）：
        # 增量更新时权威判定停牌，替代"OHLCV 缺口反推"的二义性
        self._suspend = SuspendProvider(
            store_path=self._store_path,
            sources=suspend_sources,
            lookback_days=suspend_lookback_days,
            enabled=suspend_enabled,
        )
        # 多线程：保护源懒构建与股票名称表的并发初始化
        self._source_lock = threading.Lock()
        self._name_lock = threading.Lock()
        # update() 时预热：全市场 gbbq 最近除权除息日字典（避免逐股 O(N) 过滤）
        self._gbbq_event_cache: dict[str, pd.Timestamp | None] = {}

    def _resolve_use_gbbq(self, factor_source: str | None) -> bool:
        """解析加载所用因子口径：None→实例默认；'gbbq'→True，其余→False。"""
        src = (factor_source if factor_source is not None else self._load_factor_source)
        return str(src).strip().lower() == "gbbq"

    # ------------------------------------------------------------
    # 公开接口：更新
    # ------------------------------------------------------------

    def update(
        self,
        codes: Sequence[str] | None = None,
        freqs: Sequence[str] = ("daily", "min5", "min1"),
        throttle: float = 0.0,
        progress_every: int = 200,
        max_workers: int | None = None,
    ) -> None:
        """增量更新本地原始缓存（日线 / 5 分钟 / 1 分钟）并刷新复权因子表。

        Args:
            codes: 股票代码列表；None=更新本地 daily 目录已存在的全部股票。
            freqs: 要更新的周期子集。
            throttle: 每只股票处理完 sleep 秒数（防网络源风控；pytdx 本地源设 0 即可）。
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
        # 预热共享只读资源，避免线程内并发首建（名称表 + pytdx reader）
        self._ensure_stock_name_table()
        if "pytdx" in self._source_names:
            try:
                self._get_source("pytdx")
            except Exception as exc:
                logger.debug(f"pytdx 源预热跳过: {exc}")
        # gbbq 事件快照：解析一次并按版本戳落盘（供本次及后续进程免重复解析）
        if self._gbbq.available:
            try:
                self._gbbq.save_snapshot()
            except Exception as exc:
                logger.debug(f"gbbq 事件快照写盘跳过: {exc}")
            # 预热全市场除权除息日字典（一次 groupby，_should_skip_factor 走 O(1) 缓存）
            self._prewarm_gbbq_event_cache()
        # 预热停牌名单（lookback 窗口内交易日批量预取，消除多线程竞争）
        self._prewarm_suspend_cache()

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
        """更新单只股票：先逐周期更新原始数据，再刷新因子表（线程池工作单元）。

        原始价先行，gbbq 自算因子才有最新前收可用；外部源因子不依赖原始价，顺序无碍。
        返回该股票是否全部周期成功；代理错误向上抛出以中断整个任务。
        """
        ok = True
        # 1) 逐周期更新原始（不复权）数据
        for freq in freqs:
            try:
                self._update_one(code, freq)
            except ProxyConfigError:
                raise  # 代理错误是全局性的，立即中断并提示
            except Exception as exc:  # 单只单周期失败不阻塞整体
                ok = False
                logger.error(f"[{code}|{freq}] 更新失败: {type(exc).__name__}: {exc}")
        # 2) 刷新复权因子表（失败不阻塞原始数据更新；保留旧因子文件）
        self._update_factor(code)
        if throttle > 0:
            time.sleep(throttle + (random.uniform(0, self._jitter) if self._jitter else 0))
        return ok

    def _prewarm_suspend_cache(self) -> None:
        """预热停牌名单缓存（lookback 窗口内所有交易日串行预取）。

        在并行更新前一次性把近期停牌名单加载到内存，消除多线程首次访问时的
        锁竞争与重复网络调用（SuspendProvider 内部有内存缓存，命中后 O(1)）。
        """
        if not self._suspend.enabled:
            return
        try:
            from datetime import timedelta
            end = date.today()
            start = end - timedelta(days=self._suspend._lookback_days)
            days = self._calendar.get_trading_days(start, end)
            for d in days:
                self._suspend.get_suspended_set(d)
            logger.debug(f"停牌缓存预热完成（{len(days)} 个交易日）")
        except Exception as exc:
            logger.debug(f"停牌缓存预热失败，不影响主流程: {exc}")

    def _prewarm_gbbq_event_cache(self) -> None:
        """预热全市场 gbbq 除权除息最近日期字典（一次向量化扫描，update() 入口调用）。"""
        if not self._gbbq.available:
            return
        try:
            self._gbbq_event_cache = self._gbbq.last_event_dates_all()
            logger.debug(f"gbbq 事件缓存预热完成（{len(self._gbbq_event_cache)} 只股票）")
        except Exception as exc:
            logger.debug(f"gbbq 事件缓存预热失败，回退逐股查询: {exc}")
            self._gbbq_event_cache = {}

    def _should_skip_factor(self, code: str, gbbq: bool = False) -> bool:
        """增量触发器：本地已有因子表且 gbbq 显示无新除权除息事件 → 因子不会变，跳过刷新。

        gbbq 是全市场权息本地文件；``cum_factor`` 仅在除权除息日变化。因此当某股最近一次
        除权除息日不晚于已落盘因子表覆盖的最大日期时，重新（全量）拉取外部源毫无意义。
        ``gbbq=True`` 时针对 ``factors_gbbq/`` 的并存记录做同样判断。

        保守起见，下列情况**不跳过**（保持原有每次刷新行为）：
        - 关闭了该优化（``factor_skip_via_gbbq=False``）；
        - 本地尚无对应因子表（无基准）；
        - gbbq 文件不可用（无触发器依据）。
        """
        if not self._factor_skip_via_gbbq:
            return False
        if not self._gbbq.available:
            return False
        existing = self._store.read_factor(code, gbbq=gbbq)
        if existing is None or existing.empty:
            return False
        try:
            # 优先走预热缓存（O(1)），缓存未建时回退逐股查询
            if self._gbbq_event_cache:
                last_ev = self._gbbq_event_cache.get(code)
            else:
                last_ev = self._gbbq.last_event_date(code)
        except Exception as exc:
            logger.debug(f"[{code}] gbbq 事件查询失败，不跳过因子刷新: {exc}")
            return False
        stored_max = pd.to_datetime(existing["date"]).max()
        # 无任何除权事件，或最近事件不晚于已落盘覆盖日 → 因子不会变化
        return last_ev is None or last_ev <= stored_max

    def _update_factor(self, code: str) -> None:
        """刷新复权因子表：生效因子写 ``factors/``，并把 gbbq 自算因子并存写 ``factors_gbbq/``。

        两者均受 gbbq 增量触发器约束（无新权息事件则跳过）；源暂时失败则保留旧文件、不报错。
        """
        self._refresh_active_factor(code)
        self._refresh_gbbq_factor(code)

    def _refresh_active_factor(self, code: str) -> None:
        """刷新生效因子表（``factors/``，来自 ``factor_source`` 指定的引擎）。"""
        if self._should_skip_factor(code, gbbq=False):
            logger.debug(f"[{code}] gbbq 无新权息事件，跳过生效因子刷新（沿用旧因子）")
            return
        try:
            factor = self._factor_engine.get_factor(code)
        except ProxyConfigError:
            raise
        except Exception as exc:
            logger.warning(f"[{code}] 复权因子刷新失败（保留旧因子）: {exc}")
            return
        if factor is not None and not factor.empty:
            self._store.write_factor(code, factor)
            logger.debug(f"[{code}] 生效因子表已刷新（{len(factor)} 行）")

    def _refresh_gbbq_factor(self, code: str) -> None:
        """并存写 gbbq 自算因子到 ``factors_gbbq/``（用于长期交叉对比）。

        生效源本就是 gbbq 时无需重复（``factors/`` 已是同一份）；gbbq 不可用时跳过。
        """
        if not self._gbbq.available or self._factor_source == "gbbq":
            return
        if self._should_skip_factor(code, gbbq=True):
            return
        try:
            factor = self._gbbq_calc.get_factor(code)
        except Exception as exc:
            logger.debug(f"[{code}] gbbq 自算因子刷新失败（保留旧记录）: {exc}")
            return
        if factor is not None and not factor.empty:
            self._store.write_factor(code, factor, gbbq=True)
            logger.debug(f"[{code}] gbbq 自算因子已并存（factors_gbbq, {len(factor)} 行）")

    # ------------------------------------------------------------
    # 公开接口：加载（按需复权）
    # ------------------------------------------------------------

    def load_daily(
        self,
        code: str,
        start: str | date | datetime,
        end: str | date | datetime,
        period: str = "daily",
        adjust: str = "hfq",
        anchor_date: str | date | datetime | None = None,
        factor_source: str | None = None,
    ) -> pd.DataFrame:
        """加载日线或日线以上周期（周/月/季/年由 daily resample 生成）。

        ``adjust`` ∈ {"none","hfq","qfq"}；``qfq`` 可传 ``anchor_date`` 锚定。
        ``factor_source`` ∈ {None,"active","gbbq"}：None=实例默认；"gbbq"=用 gbbq 自算因子复权。
        """
        code = format_code(code)
        use_gbbq = self._resolve_use_gbbq(factor_source)
        df = self._store.load(code, "daily", adjust, anchor_date=anchor_date, use_gbbq=use_gbbq)
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
        start: str | date | datetime,
        end: str | date | datetime,
        adjust: str = "hfq",
        anchor_date: str | date | datetime | None = None,
        factor_source: str | None = None,
    ) -> pd.DataFrame:
        """加载分钟线（按需复权）。

        - period == "1" / "5"：直接读落盘。
        - 5 的倍数（15/30/60）：由 min5 resample。
        - 其它（如 2/3/10）：由 min1 resample。

        ``factor_source`` ∈ {None,"active","gbbq"}：None=实例默认；"gbbq"=用 gbbq 自算因子复权。
        """
        code = format_code(code)
        use_gbbq = self._resolve_use_gbbq(factor_source)
        p = int(period)
        if p == 1 or p == 5:
            df = self._store.load(code, f"min{p}", adjust, anchor_date=anchor_date, use_gbbq=use_gbbq)
            if df is None:
                raise FileNotFoundError(f"本地无 {code} {p}分钟数据；请先 update")
        elif p % 5 == 0 and self._store.read_raw(code, "min5") is not None:
            base = self._store.load(code, "min5", adjust, anchor_date=anchor_date, use_gbbq=use_gbbq)
            df = resample_minute(base, 5, p)
        else:
            base = self._store.load(code, "min1", adjust, anchor_date=anchor_date, use_gbbq=use_gbbq)
            if base is None:
                raise FileNotFoundError(f"本地无 {code} 1分钟数据，无法生成 {p}分钟；请先 update")
            df = resample_minute(base, 1, p)
        s, e = pd.Timestamp(parse_date(start)), pd.Timestamp(parse_date(end)) + pd.Timedelta(days=1)
        return df[(df["datetime"] >= s) & (df["datetime"] < e)].reset_index(drop=True)

    def load_batch(
        self,
        codes: Sequence[str],
        start: str | date | datetime,
        end: str | date | datetime,
        adjust: str = "hfq",
        anchor_date: str | date | datetime | None = None,
        factor_source: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """批量加载日线；缺失的代码跳过并 WARNING。"""
        result: dict[str, pd.DataFrame] = {}
        for code in codes:
            std = format_code(code)
            try:
                result[std] = self.load_daily(
                    std, start, end, adjust=adjust, anchor_date=anchor_date,
                    factor_source=factor_source,
                )
            except FileNotFoundError as exc:
                logger.warning(str(exc))
        return result

    def list_market_codes(self, include_bse: bool = True) -> list[str]:
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

        # 前置短路：增量区间内每个交易日都被权威停牌名单确认停牌 → 当天（整段）停牌，
        # 直接跳过取数，避免对每个源做无谓的 fallback。不推进游标，复牌后整段一并补拉。
        if self._range_fully_suspended(code, start, end):
            logger.debug(f"[{code}|{freq}] 区间 {start}~{end} 整段确认停牌，跳过增量更新")
            return

        logger.debug(f"[{code}|{freq}] 拉取 {start}~{end} | {'首拉' if first_pull else '增量'}")
        # 不做复权：原始 OHLCV 直接规整入库（复权在 load 时按需合成）
        if freq == "daily":
            new_df, src = self._fetch_with_fallback(code, start, end, kind="daily")
            # 日线即便区间全空也不立即返回：先按交易日历补全区间，再用权威停牌名单
            # 判定缺口性质——确认停牌→落停牌占位行；未确认（疑似数据滞后）→裁掉尾部待重拉。
            new_df = self._post_process_daily(code, new_df, start, end, source=src)
            new_df = self._trim_unconfirmed_trailing(new_df)
            if new_df.empty:
                logger.debug(f"[{code}|{freq}] 区间内无数据且无确认停牌，跳过（不推进游标，下次重拉）")
                return
        else:
            period = "5" if freq == "min5" else "1"
            new_df, src = self._fetch_with_fallback(code, start, end, kind=period)
            if new_df.empty:
                logger.debug(f"[{code}|{freq}] 区间内所有源返回空")
                return
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
        except Exception:
            name = ""
        is_st = bool(name) and ("ST" in name.upper())
        return get_limit_pct(code, is_st=is_st)

    # ------------------------------------------------------------
    # 内部：fallback 调度
    # ------------------------------------------------------------

    def _fetch_with_fallback(
        self, code: str, start: date, end: date, kind: str,
    ) -> tuple[pd.DataFrame, str]:
        """按源优先级串行尝试。``kind`` ∈ {"daily","1","5"}。"""
        names = tuple(self._source_names)
        fetch_exc: Exception | None = None
        last_exc: Exception | None = None
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

            def _call(src=src):
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
            except Exception as exc:
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
        """日线规整为 DAILY_COLUMNS（不复权）：交易日对齐、停牌补行、涨跌停、名称、来源。

        缺口行（reindex 出的 NaN 行）按**权威停牌名单**区分来源标签：
        - 名单确认停牌 → ``{source}:suspend``（真停牌占位行，可推进增量游标）；
        - 未确认（疑似数据滞后/源未更新）→ ``{source}:gap``（由调用方裁掉尾部待重拉）。
        区间整段为空时同样按交易日历补全为全 NaN 行后再走上述判定。
        """
        df = df.copy()
        if df.empty or "date" not in df.columns:
            df = pd.DataFrame(columns=[
                "date", "open", "high", "low", "close", "volume", "amount", "raw_close",
            ])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)

        all_days = self._calendar.get_trading_days(start, end)
        full_idx = pd.DatetimeIndex(pd.to_datetime(all_days))
        df = df.set_index("date").reindex(full_idx)
        df.index.name = "date"
        df = df.reset_index()

        susp_col = df.pop("_suspended") if "_suspended" in df.columns else None

        missing = df["close"].isna() | (df["volume"].fillna(0) <= 0)
        is_suspended = missing.copy()
        if susp_col is not None:
            is_suspended = is_suspended | susp_col.fillna(False).astype(bool)
        df["is_suspended"] = is_suspended.astype(bool)

        # 权威停牌确认（按交易日整市场名单）：仅对缺口行判定，区分 :suspend / :gap
        confirmed = self._confirmed_suspended_mask(code, df["date"], missing)

        # 涨跌停：用不复权前收反推（原始价口径）
        prev_ref = df["close"].shift(1)
        try:
            stock_name = self._get_stock_name(code)
        except Exception:
            stock_name = ""
        is_st = bool(stock_name) and ("ST" in stock_name.upper())
        limit_pct = get_limit_pct(code, is_st=is_st)
        df["limit_up"] = (prev_ref * (1.0 + limit_pct)).round(2)
        df["limit_down"] = (prev_ref * (1.0 - limit_pct)).round(2)

        df["code"] = format_code(code)
        df["name"] = stock_name
        df["source"] = source or "unknown"
        base_src = source or "unknown"
        df.loc[missing & confirmed, "source"] = f"{base_src}:suspend"
        df.loc[missing & ~confirmed, "source"] = f"{base_src}:gap"

        for col in DAILY_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[list(DAILY_COLUMNS.keys())]
        return self._cast_types(df, DAILY_COLUMNS)

    def _range_fully_suspended(self, code: str, start: date, end: date) -> bool:
        """增量区间 ``[start, end]`` 内每个交易日是否都被权威停牌名单确认停牌。

        用于取数前置短路：整段都确认停牌则跳过 ``_fetch_with_fallback``，避免无谓的
        多源 fallback。任一交易日**未被名单确认停牌**（含名单未启用、超出 lookback
        返回空集、源失败返回空集等情形）即返回 ``False``，回退到正常取数流程，
        保证不会因名单不可用而漏拉真实数据。
        """
        if not self._suspend.enabled:
            return False
        days = self._calendar.get_trading_days(start, end)
        if not days:
            return False
        std = format_code(code)
        for d in days:
            try:
                if std not in self._suspend.get_suspended_set(d):
                    return False
            except Exception as exc:
                logger.debug(f"[{std}] 停牌名单查询 {d} 失败，按未停牌处理: {exc}")
                return False
        return True

    def _confirmed_suspended_mask(
        self, code: str, dates: pd.Series, missing: pd.Series,
    ) -> pd.Series:
        """对缺口行用权威停牌名单确认是否为真停牌。

        仅对 ``missing`` 为 True 的行查询停牌 provider（provider 内部按交易日整市场
        缓存，超出 lookback 的久远历史返回空集，不联网）。返回与 ``dates`` 同索引的
        布尔 Series：True=名单确认停牌。
        """
        confirmed = pd.Series(False, index=dates.index)
        if not self._suspend.enabled:
            return confirmed
        std = format_code(code)
        for i in dates.index[missing.fillna(False).to_numpy()]:
            d = dates.at[i]
            if pd.isna(d):
                continue
            try:
                if std in self._suspend.get_suspended_set(d):
                    confirmed.at[i] = True
            except Exception as exc:
                logger.debug(f"[{std}] 停牌名单查询 {d} 失败: {exc}")
        return confirmed

    @staticmethod
    def _trim_unconfirmed_trailing(df: pd.DataFrame) -> pd.DataFrame:
        """裁掉尾部连续的"未确认缺口行"（source 以 ``:gap`` 结尾）。

        这些行可能只是数据源当天尚未更新/源滞后，并非真停牌；裁掉后不写入、不推进
        增量游标，下次更新会重新拉取。已被权威名单确认的停牌行（``:suspend``）及真实
        数据行不受影响。中间（被真实数据包夹）的缺口行保留（历史停牌占位）。
        """
        if df is None or df.empty or "source" not in df.columns:
            return df
        df = df.sort_values("date").reset_index(drop=True)
        src = df["source"].astype("string").fillna("")
        is_gap = src.str.endswith(":gap").to_numpy()
        keep_n = len(df)
        for flag in is_gap[::-1]:
            if flag:
                keep_n -= 1
            else:
                break
        if keep_n == len(df):
            return df
        return df.iloc[:keep_n].reset_index(drop=True)

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
    def _cast_types(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
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
                import akshare as ak
                df = _ak_call(ak.stock_info_a_code_name)
                if df is not None and not df.empty:
                    # 向量化构建，比 iterrows() 快 10x+
                    self._STOCK_NAME_CACHE.update({
                        format_code(str(c)): str(n)
                        for c, n in zip(df["code"].tolist(), df["name"].tolist(), strict=True)
                    })
            except ProxyConfigError:
                raise
            except Exception as exc:
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


def _selftest(tdx_path: str | None = None, level: str = "INFO") -> int:
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
        except Exception as exc:
            ok_all = False
            logger.error(f"✗ {label}: {type(exc).__name__}: {exc}")

    logger.info("===== 数据源自测开始 =====")

    # 1) pytdx 本地（含北交所 BJ 修复验证）
    try:
        pytdx = PytdxLocalSource(tdx_path=tdx_path)
        _try("pytdx 日线 000001", lambda: pytdx.fetch_daily(sample_daily, start, end))
        _try("pytdx 5分钟 000001", lambda: pytdx.fetch_minute(sample_daily, "5", start_5d, end))
        _try("pytdx 1分钟 000001", lambda: pytdx.fetch_minute(sample_daily, "1", start_3d, end))
        _try("pytdx 北交所 BJ 修复", lambda: pytdx.fetch_daily(sample_bj, start, end))
    except Exception as exc:
        logger.warning(f"pytdx 不可用（未安装通达信/缺包）: {exc}")

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
    except Exception as exc:
        logger.warning(f"baostock 不可用: {exc}")

    # 5) tushare（需 token）
    try:
        _try("tushare 日线", lambda: TushareSource().fetch_daily(sample_daily, start, end))
    except Exception as exc:
        logger.warning(f"tushare 跳过: {exc}")

    # 5b) 停牌名单（东财主源 + tushare 兜底）：用历史日 2020-03-12（已知有停牌）验证
    try:
        import tempfile
        susp = SuspendProvider(
            store_path=tempfile.mkdtemp(prefix="susp_selftest_"),
            lookback_days=10_000_000,  # 放开 lookback 以便拉取历史日
        )
        _try("停牌名单 2020-03-12", lambda: pd.Series(sorted(susp.get_suspended_set("2020-03-12"))))
    except Exception as exc:
        logger.warning(f"停牌名单自测跳过: {exc}")

    # 6) gbbq 本地权息 + 自算因子（FactorCalculator）
    gbbq = GbbqStore(tdx_path=tdx_path)
    if not gbbq.available:
        logger.warning(
            f"… gbbq 文件不可用（{gbbq.gbbq_file()}）；"
            f"未安装通达信或未下载权息数据时跳过 gbbq 自测"
        )
    else:
        _try("gbbq 除权除息事件 000001", lambda: gbbq.events(sample_daily))

        def _gbbq_factor_check() -> pd.DataFrame:
            # 用近年日线 close 作不复权 raw，自算因子做连通性 + sanity 校验
            raw = ak_src.fetch_daily(sample_daily, date(2018, 1, 1), end)
            calc = FactorCalculator(gbbq, raw_close_loader=lambda c: raw)
            factor = calc.get_factor(sample_daily)
            if factor is not None and not factor.empty:
                cf = pd.to_numeric(factor["cum_factor"], errors="coerce")
                assert (cf > 0).all(), "自算 cum_factor 出现非正值"
                assert cf.is_monotonic_increasing or len(cf) <= 1, "后复权因子应单调不减"
            return factor

        _try("gbbq 自算因子 000001（FactorCalculator）", _gbbq_factor_check)

    # 7) resample 自测（纯本地，不依赖网络）
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
    except Exception as exc:
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
        "--factor-source", default="sina",
        choices=["sina", "tushare", "em", "gbbq"],
        help="复权因子源：gbbq=本地权息自算（离线）；其余为外部源（默认 sina）",
    )
    parser.add_argument(
        "--no-factor-skip", action="store_true",
        help="关闭 gbbq 增量触发器（强制每次刷新因子，即使无新权息事件）",
    )
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
        factor_source=args.factor_source,
        factor_skip_via_gbbq=not args.no_factor_skip,
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
