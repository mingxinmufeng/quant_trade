"""
A 股行情拉取器（多源容灾 + 多周期 + 增量更新 + 原始/因子分离落盘）
====================================================================

设计（重构后）
--------------
1. **职责单一**：``fetcher`` 只负责**采集与加工不复权原始数据**并落盘，
   外加**刷新复权因子表**。复权本身由 ``src.data.adjust`` 在 load 时按需合成
   （见 ``DataStore.load``），``fetcher`` 不再把复权价烤进文件。
2. **原始价与因子分离**（核心）：
   - ``{store}/daily|min5|min1/{code}.parquet`` 只存数据源返回的**不复权**成交行（OHLCV + 来源）；
   - ``{store}/daily_meta/{code}.parquet`` 日线加工层：交易日时间轴 + 停牌/状态 + 涨跌停价
     （由原始日线、停牌名单、权息、更名史派生，``rebuild_daily_meta`` 可整体重算）；
   - ``{store}/factors/{code}.parquet`` 存 ``date→cum_factor``，每次覆盖刷新；
   - 因此**不再需要因子自愈**：原始价永不被改写，因子表直接刷新即可。
3. **拉取优先级**：``pytdx``（通达信本地盘）→ ``tushare``（官方，需 token；缺 token
   自动从源链剔除）→ ``akshare`` → ``baostock``。
   各源仅返回不复权 OHLCV（见 ``src.data.sources``）。
4. **增量更新**：日线 / 5 分钟 / 1 分钟三套原始周期；其余周期 load 时 resample。
5. **多线程**：``update`` 按股票并行（``max_workers``）；baostock 等非线程安全源已加锁。
6. **复权因子单源锁定**（见 ``FactorProvider``）：默认 sina，可选 tushare/em，不 fallback。
7. **代理**：发生代理错误抛 ``ProxyConfigError`` 提示关闭代理。
8. **北交所(.BJ)/科创板修复 + 成交量「股」口径**：见自洽解析模块 ``src.data.sources.tdx_reader``。

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
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.helpers import (
    Timer,
    format_code,
    get_limit_pct,
    is_limit_5pct_name,
    parse_date,
    retry,
)
from .factors import FactorCalculator, FactorProvider
from .gbbq import GbbqStore
from .profile import ProfileStore
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
from .sources.tushare_pool import get_tushare_tokens
from .storage import (  # 重导出，保持向后兼容
    DAILY_COLUMNS,
    DAILY_META_COLUMNS,
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
from .universe import is_st_name

#: 默认重试参数
DEFAULT_RETRY_TIMES = 3
DEFAULT_RETRY_DELAYS: Sequence[float] = (1, 2, 4, 8)

#: 默认并行线程数（按股票并行；pytdx 为本地文件 IO，可用更多线程）
DEFAULT_MAX_WORKERS = 16

#: A 股最早可拉取日期
EARLIEST_DATE = date(1990, 12, 19)

#: 本地离线源名（退市股分钟线仅用这些源，不联网；见 _fetch_with_fallback 的 local_only）
LOCAL_SOURCE_NAMES = frozenset({"pytdx"})

#: 本地全市场基础信息缓存（tushare，含在市+退市当前简称）；股票名称表的稳定离线主源
STOCK_BASIC_TUSHARE_FILE = "stock_basic_tushare.parquet"

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
# 涨跌停价（纯函数；原始价口径，不涉及复权）
# ============================================================


def _round_price(x) -> np.ndarray:
    """价格四舍五入到分（交易所口径；numpy ``round`` 是银行家舍入，x.xx5 会舍错）。"""
    x = np.asarray(x, dtype="float64")
    return np.floor(x * 100.0 + 0.5 + 1e-7) / 100.0


def _limit_ref_price(
    dates: pd.Series, close: pd.Series, events: pd.DataFrame | None = None,
) -> np.ndarray:
    """逐行涨跌停**计算基准价**（交易所"前收盘价"口径），与 ``dates`` 同序。

    - 前收 = 本行之前**最近一个有价行**的收盘价（跨过停牌占位行，复牌日不再缺失）；
    - 前一有价日（不含）到本行（含）之间若有除权除息事件（gbbq ``events``，每 10 股口径），
      依次换算为除权除息参考价 ``(前收 − 派息 + 配股价 × 配股比例) / (1 + 送转比例 + 配股比例)``；
    - ``events`` 可带布尔列 ``nonstandard``（见 ``GbbqStore.nonstandard_event_dates``：股改对价/
      定向转增等，交易所不按标准公式除权且可能不设限）→ 受影响行基准价为 NaN（不臆造）；
    - 结果按分四舍五入。首个有价行之前无前收 → NaN。
    """
    d = pd.to_datetime(dates).dt.normalize().to_numpy()
    c = pd.to_numeric(close, errors="coerce")
    ref = c.ffill().shift(1).to_numpy(dtype="float64", copy=True)
    prev_day = pd.Series(d).where(c.notna().to_numpy()).ffill().shift(1).to_numpy()
    if events is not None and not events.empty:
        has_prev = ~pd.isna(prev_day)
        for ev in events.sort_values("date").itertuples(index=False):
            ev_day = np.datetime64(pd.Timestamp(ev.date).normalize())
            hit = has_prev & (d >= ev_day)
            hit[hit] = prev_day[hit] < ev_day
            if not hit.any():
                continue
            if getattr(ev, "nonstandard", False):
                ref[hit] = np.nan
                continue
            div, song, pei = ev.fenhong / 10.0, ev.song / 10.0, ev.pei / 10.0
            ex = (ref[hit] - div + pei * ev.peijia) / (1.0 + song + pei)
            ref[hit] = np.where(ex > 0, ex, np.nan)
    return _round_price(ref)


# ============================================================
# 主类：DataFetcher
# ============================================================


class DataFetcher:
    """多源容灾 + 多周期 + 增量更新 + 原始/因子分离落盘的行情拉取器。"""

    def __init__(
        self,
        store_path: str | Path = "data_store",
        sources: Sequence[str] = DEFAULT_SOURCES,
        retry_times: int = DEFAULT_RETRY_TIMES,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
        calendar: TradingCalendar | None = None,
        tdx_path: str | None = None,
        jitter: float = 0.3,
        factor_source: str = "tushare",
        max_workers: int = DEFAULT_MAX_WORKERS,
        factor_skip_via_gbbq: bool = True,
        load_factor_source: str = "gbbq",
        suspend_sources: Sequence[str] = DEFAULT_SUSPEND_SOURCES,
        suspend_lookback_days: int = DEFAULT_SUSPEND_LOOKBACK_DAYS,
        suspend_enabled: bool = True,
        selfheal_window: int | None = None,  # 已废弃：原始/因子分离后无需自愈
    ) -> None:
        import threading

        self._store = DataStore(store_path)
        self._store_path = Path(store_path)
        # 无 TUSHARE_TOKEN 时把 tushare 从源链一次性剔除（否则每只股票都构建失败刷警告）
        source_names = tuple(sources)
        if "tushare" in source_names and not get_tushare_tokens():
            logger.warning("未配置 TUSHARE_TOKEN，源链中的 tushare 已禁用（其余源兜底）")
            source_names = tuple(n for n in source_names if n != "tushare")
        self._source_names = source_names
        self._retry_times = int(retry_times)
        self._retry_delays = list(retry_delays)
        self._calendar = calendar or TradingCalendar(store_path=self._store_path)
        self._tdx_path = (tdx_path or "").strip() or None
        self._jitter = max(0.0, float(jitter))
        self._max_workers = max(1, int(max_workers))
        if selfheal_window is not None:
            logger.debug("selfheal_window 参数已废弃（原始/因子分离后无需自愈），忽略")
        self._sources_cache: dict[str, DataSourceBase | None] = {}
        # 本地 gbbq（权息）：自算因子的数据来源、因子刷新触发器；事件快照落盘到 store 根下。
        # tushare 补充事件表按股票兜底（TDX 会清除退市股记录，见 sources.tushare_meta）
        self._gbbq = GbbqStore(
            tdx_path=self._tdx_path,
            snapshot_path=self._store_path / "gbbq_events.parquet",
            supplement_path=self._store_path / "gbbq_events_tushare.parquet",
        )
        # 本地 profile（更名史）：点位证券简称 / ST 判定来源；更名快照落盘到 store 根下。
        # tushare 补充更名表按股票兜底（退市股 profile.dat 覆盖率 0%）
        self._profile = ProfileStore(
            tdx_path=self._tdx_path,
            snapshot_path=self._store_path / "profile_names.parquet",
            supplement_path=self._store_path / "profile_names_tushare.parquet",
        )
        self._factor_skip_via_gbbq = bool(factor_skip_via_gbbq)
        self._factor_source = (factor_source or "tushare").strip().lower()
        # 外部因子源默认 tushare（官方 adj_factor）；无 token 降级 sina（免 token），
        # 保证单源锁定语义下无 token 用户仍有可用外部因子源
        if self._factor_source == "tushare" and not get_tushare_tokens():
            logger.warning("factor_source=tushare 但未配置 TUSHARE_TOKEN，外部因子源降级为 sina")
            self._factor_source = "sina"
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
        # 加载（复权）默认采用哪套因子口径：
        #   'gbbq'  = gbbq 自算因子优先、外部源(factors/)兜底（默认，回测推荐）；
        #   'active'= 仅用生效外部源(factors/)。
        # 默认 gbbq 优先：gbbq 自算因子以"行情最早一日=1.0"累乘、**全history 覆盖无早期缺口**，
        # 避免外部源(sina/tushare)早期因子缺失导致的 align_cum_factor bfill 早段绝对价失真
        # （P1-7）。换代码股 gbbq 按当前代码归并、记录完整（见 gbbq.events 实测注释），无副作用。
        self._load_factor_source = (load_factor_source or "gbbq").strip().lower()
        # 停牌名单 provider（tushare 主源 + 东财兜底，按交易日落盘缓存）：
        # 增量更新时权威判定停牌，替代"OHLCV 缺口反推"的二义性
        self._suspend = SuspendProvider(
            store_path=self._store_path,
            sources=suspend_sources,
            lookback_days=suspend_lookback_days,
            enabled=suspend_enabled,
        )
        # 全市场「代码→当前简称」/「代码→退市日」缓存：**实例级**（按 store_path 隔离）。
        # 早期为 ClassVar(进程级共享)，但其内容依赖 store_path 下的本地基础信息文件，跨不同
        # store 的实例共享会串扰（A 实例读到 B 实例 store 的名称/退市表），故改为实例属性。
        self._stock_name_cache: dict[str, str] = {}
        self._delist_date_cache: dict[str, date] = {}
        self._list_date_cache: dict[str, pd.Timestamp] | None = None
        # 多线程：保护源懒构建与股票名称表的并发初始化
        self._source_lock = threading.Lock()
        self._name_lock = threading.Lock()
        self._delist_lock = threading.Lock()
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
        # profile 更名快照：解析一次并按版本戳落盘（供 Universe 点位 ST 判定免重复解析）
        if self._profile.available:
            try:
                self._profile.save_snapshot()
            except Exception as exc:
                logger.debug(f"profile 更名快照写盘跳过: {exc}")
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
        """增量触发器：本地已有因子表且可判定"因子不会变" → 跳过刷新。

        **按因子源口径分流触发器**（关键：外部源不能用 gbbq 事件当依据）：

        - **gbbq 口径**（``gbbq=True``，或生效源本就是 gbbq）：用 gbbq 全市场权息文件——
          该股最近除权除息日不晚于已落盘因子覆盖日则跳过（同源自洽，gbbq 文件既是数据源
          又是触发器）。
        - **外部源口径**（sina/tushare/em）：**绝不**以 gbbq 事件为依据——本地通达信权息
          可能滞后于外部源，会误判"无新事件"从而静默跳过真实的因子更新（产出错误复权价）。
          改用"本地原始日线是否较因子表**新增了交易日**"：``cum_factor`` 仅在除权日（必为
          交易日）新增行，故无新交易日即不可能有新因子；有新交易日则必须刷新。

        保守起见，下列情况**不跳过**（保持每次刷新行为）：
        - 关闭了该优化（``factor_skip_via_gbbq=False``）；
        - 本地尚无对应因子表（无基准）；
        - gbbq 口径但 gbbq 文件不可用 / 外部口径但本地无原始日线（无触发器依据）。
        """
        if not self._factor_skip_via_gbbq:
            return False
        existing = self._store.read_factor(code, gbbq=gbbq)
        if existing is None or existing.empty:
            return False
        stored_max = pd.to_datetime(existing["date"]).max()
        if pd.isna(stored_max):
            return False

        # gbbq 口径（gbbq 文件，或生效源本就是 gbbq）：用 gbbq 事件触发（同源自洽）
        if gbbq or self._factor_source == "gbbq":
            if not self._gbbq.available:
                return False
            raw = self._store.read_raw(code, "daily")
            if raw is not None and not raw.empty and "date" in raw.columns:
                raw_max = pd.to_datetime(raw["date"], errors="coerce").max()
                if pd.notna(raw_max) and stored_max > raw_max:
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
            # 无任何除权事件，或最近事件不晚于已落盘覆盖日 → 因子不会变化
            return last_ev is None or last_ev <= stored_max

        # 外部源口径：原始日线未较因子表新增交易日 → 不可能有新因子（与 gbbq 文件无关）
        raw = self._store.read_raw(code, "daily")
        if raw is None or raw.empty or "date" not in raw.columns:
            return False
        raw_max = pd.to_datetime(raw["date"]).max()
        return pd.notna(raw_max) and raw_max <= stored_max

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

    @staticmethod
    def _daily_needs_fetch(df: pd.DataFrame | None, end: pd.Timestamp) -> bool:
        """auto_fetch 判定：本地无数据 / 为空 / 最新日未覆盖到 ``end`` 即需补拉。"""
        if df is None or df.empty or "date" not in df.columns:
            return True
        last = pd.to_datetime(df["date"], errors="coerce").max()
        return pd.isna(last) or last < end

    def load_daily(
        self,
        code: str,
        start: str | date | datetime,
        end: str | date | datetime,
        period: str = "daily",
        adjust: str = "hfq",
        anchor_date: str | date | datetime | None = None,
        factor_source: str | None = None,
        auto_fetch: bool = False,
    ) -> pd.DataFrame:
        """加载日线或日线以上周期（周/月/季/年由 daily resample 生成）。

        ``adjust`` ∈ {"none","hfq","qfq"}；``qfq`` 可传 ``anchor_date`` 锚定。
        ``factor_source`` ∈ {None,"active","gbbq"}：None=实例默认；"gbbq"=用 gbbq 自算因子复权。
        ``auto_fetch=True``：本地缺该代码或未覆盖到 ``end`` 时，先 ``update`` 补全再读
        （供回测基准指数等"按需落盘"场景，如 ``000300.SH``；pytdx 本地源近乎瞬时）。
        """
        code = format_code(code)
        use_gbbq = self._resolve_use_gbbq(factor_source)
        s, e = pd.Timestamp(parse_date(start)), pd.Timestamp(parse_date(end))
        df = self._store.load(code, "daily", adjust, anchor_date=anchor_date, use_gbbq=use_gbbq)
        if auto_fetch and self._daily_needs_fetch(df, e):
            logger.info(f"[{code}] 本地日线缺失/未覆盖至 {e.date()}，自动 update 补全")
            try:
                self.update([code], freqs=("daily",))
            except Exception as exc:
                logger.warning(f"[{code}] auto_fetch update 失败: {exc}")
            df = self._store.load(code, "daily", adjust, anchor_date=anchor_date, use_gbbq=use_gbbq)
        if df is None:
            raise FileNotFoundError(f"本地无 {code} 日线；请先 fetcher.update(['{code}'])")
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
        codes = list(self._stock_name_cache.keys())
        if not include_bse:
            codes = [c for c in codes if not c.endswith(".BJ")]
        return sorted(set(codes))

    def rebuild_daily_meta(
        self, codes: Sequence[str] | None = None, max_workers: int | None = None,
    ) -> int:
        """按最新规则整体重算日线加工层 ``daily_meta/``（原始日线不动）。

        时间轴、停牌状态、涨跌停价的口径见 ``_build_daily_meta``。规则或其输入（权息、
        更名史、停牌名单、基础信息）变化后运行即可。``codes=None`` → 本地全部日线。
        返回内容有变化而重写的股票数。
        """
        if codes is None:
            codes = sorted(p.stem for p in self._store._dirs["daily"].glob("*.parquet"))
        codes = [format_code(c) for c in codes]
        # 解析一次更名/权息快照，供点位 ST 判定与除权参考价
        for store in (self._profile, self._gbbq):
            if store.available:
                try:
                    store.save_snapshot()
                except Exception as exc:
                    logger.debug(f"{type(store).__name__} 快照写盘跳过: {exc}")
        self._prewarm_suspend_cache()

        def _one(code: str) -> bool:
            raw = self._store.read_raw(code, "daily")
            if raw is None or raw.empty:
                return False
            return self._refresh_daily_meta(code, raw, since=None)

        workers = max(1, int(max_workers if max_workers is not None else self._max_workers))
        n = 0
        with Timer("日线加工层重算"):
            if workers <= 1:
                n = sum(_one(c) for c in codes)
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="meta") as ex:
                    for done, changed in enumerate(ex.map(_one, codes), start=1):
                        n += int(changed)
                        if done % 500 == 0:
                            logger.info(f"加工层重算进度 {done}/{len(codes)} | 有变化 {n}")
        logger.info(f"日线加工层重算完成 | {len(codes)} 只，重写 {n} 只")
        return n

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
        # 退市股：历史已冻结。本地已有数据 → 跳过（不增量、不联网），避免每次都对「末次交易日
        # →退市日」的停牌/退市整理期空隙重试联网（刷屏 + 风控）。仅**首拉**（本地尚无）时才取数：
        # 日线走完整源链（pytdx 无则 akshare/baostock/tushare 兜底，覆盖北交所退市股）；分钟线
        # 只走本地（外部源均无退市股历史分钟）。退市整理期如需补全由 backfill 脚本一次性处理。
        delist_d = self._delist_date(code)
        if delist_d is not None and not first_pull:
            logger.debug(f"[{code}|{freq}] 退市股本地已有数据，历史冻结，跳过")
            return
        if not first_pull:
            local_max = existing[time_col].max()
            local_max_date = local_max.date() if hasattr(local_max, "date") else local_max
            if freq == "daily":
                # 游标 = 原始日线最后一个成交日（原始层不含停牌占位行；停牌由停牌名单判定，见下）
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
        # 退市股首拉：把 end 截到退市日（退市后无数据）。日线走完整源链（含 tushare 兜底），
        # 分钟线只走本地（akshare/baostock/tushare 均无退市股历史分钟，联网纯属无效请求）。
        if delist_d is not None and end > delist_d:
            end = delist_d
        if start > end:
            return
        local_only = delist_d is not None and freq != "daily"
        meta_since = pd.Timestamp(start)  # 加工层从原游标之后开始重算

        # 前置短路：跳过区间开头被权威停牌名单确认停牌的交易日（停牌期不取数，避免对每个源
        # 做无谓的 fallback）；整段确认停牌则不取数。日线仍刷新加工层，把确认的停牌日记入时间轴。
        fetch_start = self._first_unconfirmed_day(code, start, end)
        if fetch_start is None:
            logger.debug(f"[{code}|{freq}] 区间 {start}~{end} 整段确认停牌，跳过取数")
            if freq == "daily" and not first_pull:
                self._refresh_daily_meta(code, existing, since=meta_since, until=end)
            return
        start = fetch_start

        logger.debug(f"[{code}|{freq}] 拉取 {start}~{end} | {'首拉' if first_pull else '增量'}")
        # 不做复权：原始 OHLCV 直接规整入库（复权在 load 时按需合成）
        if freq == "daily":
            new_df, src = self._fetch_with_fallback(
                code, start, end, kind="daily", local_only=local_only,
            )
            # 原始层只落数据源返回的成交行；缺口（停牌/源滞后）不落占位行，由加工层按停牌名单
            # 标注，未确认的尾部缺口不进时间轴、游标不推进，下次重拉。
            new_df = self._normalize_daily_raw(code, new_df, source=src)
            if new_df.empty:
                logger.debug(f"[{code}|{freq}] 区间内无成交数据（不推进游标，下次重拉）")
                if not first_pull:
                    self._refresh_daily_meta(code, existing, since=meta_since, until=end)
                return
        else:
            period = "5" if freq == "min5" else "1"
            new_df, src = self._fetch_with_fallback(
                code, start, end, kind=period, local_only=local_only,
            )
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

        self._store.write_raw(code, freq, combined)
        # 日线加工层（停牌/涨跌停/时间轴）：按完整历史重算原游标之后的部分，之前的行不动
        if freq == "daily":
            self._refresh_daily_meta(
                code, combined, since=None if first_pull else meta_since, until=end,
            )
        logger.info(
            f"[{code}|{freq}] 更新成功 | +{len(new_df)} 行 | 末: {combined[time_col].max()}"
        )

    def _limit_pct_for(self, code: str) -> float:
        """该股票的涨跌停比例（**当前名** ST 近似的标量兜底）。

        仅在不需要逐行点位 ST 时使用；日线落盘的历史限幅一律走 ``_limit_pct_series``
        逐行按行日期的点位 ST 判定（主板历史 ST 时段 ±5%）。
        """
        try:
            name = self._get_stock_name(code)
        except Exception:
            name = ""
        return get_limit_pct(code, is_st=is_st_name(name))

    def _limit_pct_series(
        self, code: str, dates: pd.Series, traded: pd.Series | None = None,
    ) -> pd.Series:
        """逐行按行日期的涨跌停限幅比例（与 ``dates`` 同索引；不设限为 ``inf``）。

        规则沿革见 ``get_limit_pct``（1996-12-16 前不设限、创业板 2020-08-24 前 ±10%、
        主板 ST 2026-07-06 起 ±10% 等）。ST 按 ``ProfileStore.name_at`` 逐日点位判定
        （曾用名优先、当前名兜底），ST 状态随时间变化，故必须逐行算；``profile.dat``
        不可用时回退到**当前名近似**（与 Universe 口径一致）。

        ``traded``（与 ``dates`` 同索引的"有价行"布尔列，须按日期升序）给出时，再按上市日
        所处制度套用新股不设限窗口：上市日取本地基础信息 ``list_date``，缺失时取首个有价行。
        """
        try:
            current_name = self._get_stock_name(code)
        except Exception:
            current_name = ""
        use_profile = self._profile is not None and self._profile.available
        norm = pd.to_datetime(dates).dt.normalize()
        pct_cache: dict[pd.Timestamp, float] = {}

        def _pct_at(ts: pd.Timestamp) -> float:
            if pd.isna(ts):  # reindex 出的占位行无日期：当前名、今天的规则兜底
                return get_limit_pct(code, is_st=is_limit_5pct_name(current_name, date.today()))
            cached = pct_cache.get(ts)
            if cached is None:
                former = self._profile.name_at(code, ts) if use_profile else None
                is_st = is_limit_5pct_name(former if former is not None else current_name, ts)
                cached = get_limit_pct(code, is_st=is_st, trade_date=ts)
                pct_cache[ts] = cached
            return cached

        pct = norm.map(_pct_at).astype("float64")
        if traded is not None:
            pct = self._apply_ipo_window(code, norm, traded, pct)
        return pct

    def _apply_ipo_window(
        self, code: str, dates: pd.Series, traded: pd.Series, pct: pd.Series,
    ) -> pd.Series:
        """新股上市后前 N 个有价交易日（按上市日所处制度，见 ``get_limit_pct``）置为不设限。"""
        tr = traded.to_numpy(dtype=bool)
        if not tr.any():
            return pct
        list_ts = self._list_date(code)
        first_traded = dates[tr].iloc[0]
        if list_ts is None:
            list_ts = first_traded
        elif first_traded > list_ts + pd.Timedelta(days=14):
            return pct  # 本地行情不从上市日开始（如迁入/缺早期数据），无法定位新股窗口
        listed = tr & (dates >= list_ts).to_numpy()
        ordinal = np.where(listed, np.cumsum(listed), 0)
        out = pct.copy()
        for k in range(1, 6):
            rows = ordinal == k
            if rows.any() and get_limit_pct(
                code, trading_days_since_list=k, list_date=list_ts,
            ) == float("inf"):
                out[rows] = float("inf")
        return out

    def _limit_prices(
        self, code: str, df: pd.DataFrame, rows: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """按完整历史 ``df``（``date``/``close``，停牌行 close 为 NaN）逐行算
        ``(limit_up, limit_down)``，与 ``df`` 行序一一对应。

        基准价见 ``_limit_ref_price``（跨停牌取前收 + gbbq 除权除息参考价），限幅见
        ``_limit_pct_series``（规则沿革 + 点位 ST + 新股窗口）；按分四舍五入。
        不设限或无前收的行为 NaN（下游执行层据此不夹价）。``rows``（与 ``df`` 同序的
        布尔数组）给出时只算这些行（省去全历史逐日规则判定），其余为 NaN。
        """
        order = np.argsort(pd.to_datetime(df["date"]).to_numpy(), kind="stable")
        s = df.iloc[order].reset_index(drop=True)
        close = pd.to_numeric(s["close"], errors="coerce")
        ref = _limit_ref_price(s["date"], close, self._limit_events(code))
        dates = pd.to_datetime(s["date"]).dt.normalize()
        if rows is None:
            pct = self._limit_pct_series(code, dates)
        else:
            sel = np.asarray(rows, dtype=bool)[order]
            pct = self._limit_pct_series(code, dates[sel]).reindex(dates.index)
        pct = self._apply_ipo_window(code, dates, close.notna(), pct).to_numpy()
        pct = np.where(np.isfinite(pct), pct, np.nan)
        up, dn = ref * (1.0 + pct), ref * (1.0 - pct)
        lu = np.empty(len(s))
        ld = np.empty(len(s))
        if format_code(code).endswith(".BJ"):
            # 北交所：涨停价向下、跌停价向上取至分（限价不超出 ±30%；交易所公布值实测）
            lu[order] = np.floor(up * 100.0 + 1e-6) / 100.0
            ld[order] = np.ceil(dn * 100.0 - 1e-6) / 100.0
        else:
            lu[order] = _round_price(up)
            ld[order] = _round_price(dn)
        return lu, ld

    def _limit_events(self, code: str) -> pd.DataFrame | None:
        """涨跌停基准价用的除权除息事件（gbbq），带 ``nonstandard`` 标记；不可用返回 None。"""
        try:
            if not self._gbbq.available:
                return None
            events = self._gbbq.events(code)
            if events.empty:
                return events
            special = self._gbbq.nonstandard_event_dates(code)
            return events.assign(nonstandard=events["date"].isin(special))
        except Exception as exc:
            logger.debug(f"[{code}] gbbq 事件读取失败，涨跌停按未除权前收计算: {exc}")
            return None

    def _list_date(self, code: str) -> pd.Timestamp | None:
        """该股票的上市日（本地 tushare 基础信息 ``list_date``；缺失返回 None）。"""
        if self._list_date_cache is None:
            with self._delist_lock:
                if self._list_date_cache is None:
                    cache: dict[str, pd.Timestamp] = {}
                    path = self._store_path / STOCK_BASIC_TUSHARE_FILE
                    try:
                        df = pd.read_parquet(path, columns=["ts_code", "list_date"])
                        ld = pd.to_datetime(
                            df["list_date"].astype("string"), format="%Y%m%d", errors="coerce",
                        )
                        for c, d in zip(df["ts_code"].tolist(), ld.tolist(), strict=True):
                            if pd.notna(d):
                                cache[format_code(str(c))] = pd.Timestamp(d)
                    except Exception as exc:
                        logger.debug(f"读取上市日表失败（新股窗口用首个有价行近似）: {exc}")
                    self._list_date_cache = cache
        return self._list_date_cache.get(format_code(code))

    # ------------------------------------------------------------
    # 内部：fallback 调度
    # ------------------------------------------------------------

    def _fetch_with_fallback(
        self, code: str, start: date, end: date, kind: str, local_only: bool = False,
    ) -> tuple[pd.DataFrame, str]:
        """按源优先级串行尝试。``kind`` ∈ {"daily","1","5"}。

        ``local_only=True``（退市股分钟线）时只用本地离线源（``LOCAL_SOURCE_NAMES``），
        过滤掉所有联网源；过滤后无源则返回空（视作无数据，不报错、不联网）。
        """
        names = tuple(self._source_names)
        if local_only:
            names = tuple(n for n in names if n in LOCAL_SOURCE_NAMES)
            if not names:
                return pd.DataFrame(), ""
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

    def _normalize_daily_raw(self, code: str, df: pd.DataFrame, source: str = "") -> pd.DataFrame:
        """数据源返回的日线规整为原始层 ``DAILY_COLUMNS``（不复权，只保留成交数据行）。

        丢弃无任何价量的空行，以及数据源自带停牌标记（``_suspended``，如 baostock 的
        tradestatus=0 前收占位行）的行：停牌由加工层按停牌名单标注，不落原始层。
        零成交但有价的行保留（北交所/低流动性真实交易日，是否停牌交给加工层裁定）。
        """
        cols = list(DAILY_COLUMNS)
        if df is None or df.empty or "date" not in df.columns:
            return self._cast_types(pd.DataFrame(columns=cols), DAILY_COLUMNS)
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if "_suspended" in df.columns:
            df = df[~df["_suspended"].fillna(False).astype(bool)]
        price_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        df = df[df[price_cols].notna().any(axis=1)]
        df["code"] = format_code(code)
        df["source"] = source or "unknown"
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[cols].drop_duplicates(subset=["date"], keep="last")
        return self._cast_types(df, DAILY_COLUMNS).sort_values("date").reset_index(drop=True)

    def _first_unconfirmed_day(self, code: str, start: date, end: date) -> date | None:
        """``[start, end]`` 内第一个**未被**权威停牌名单确认停牌的交易日；整段确认停牌返回 None。

        增量游标据此跳过停牌期：停牌日不取数（避免对每个源做无谓 fallback）。名单未启用、
        超出 lookback 且无缓存、源失败等无法确认的日子一律视为"未确认"，照常取数，
        保证不会因名单不可用而漏拉真实数据。区间内无交易日返回 ``start``（交给取数流程）。
        """
        days = self._calendar.get_trading_days(start, end)
        if not days or not self._suspend.enabled:
            return start
        std = format_code(code)
        for d in days:
            try:
                if std not in self._suspend.get_suspended_set(d):
                    return d
            except Exception as exc:
                logger.debug(f"[{std}] 停牌名单查询 {d} 失败，按未停牌处理: {exc}")
                return d
        return None

    def _confirmed_suspended_mask(
        self, code: str, dates: pd.Series, missing: pd.Series,
    ) -> pd.Series:
        """对缺口行用权威停牌名单确认是否为真停牌。

        仅对 ``missing`` 为 True 的行查询停牌 provider（provider 内部按交易日整市场
        缓存，超出 lookback 的久远历史只读磁盘缓存、不联网）。返回与 ``dates`` 同索引的
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

    def _build_daily_meta(
        self,
        code: str,
        raw: pd.DataFrame,
        until: date | None = None,
        compute_from: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """由原始日线派生加工层 ``DAILY_META_COLUMNS``（交易日时间轴 + 停牌/状态 + 涨跌停价）。

        时间轴：首个原始行 → 最后一个原始行之间的全部交易日，再接上其后**连续被停牌名单
        确认停牌**的交易日（至 ``until``，默认最近交易日；不越过退市日）。未确认的尾部缺口
        不进时间轴（可能只是数据源滞后），与增量游标口径一致。

        每行状态：
        - ``trade``：有收盘价，且非"零成交 + 名单确认停牌"；
        - ``suspend``：名单确认停牌（无原始行，或零成交行）；
        - ``missing``：无原始行（或收盘价缺失）且名单未确认。
        ``is_suspended`` = 非 ``trade``。涨跌停价见 ``_limit_prices``。

        ``compute_from`` 给出时只为该日及之后的行计算涨跌停（其余为 NaN），供增量调用方
        只替换新行、省去全历史的逐日规则判定。
        """
        cols = list(DAILY_META_COLUMNS)
        if raw is None or raw.empty:
            return self._cast_types(pd.DataFrame(columns=cols), DAILY_META_COLUMNS)
        r = raw.copy()
        r["date"] = pd.to_datetime(r["date"]).dt.normalize()
        r = r.drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
        first, last_raw = r.index.min(), r.index.max()

        days = pd.DatetimeIndex(pd.to_datetime(self._calendar.get_trading_days(first.date(), last_raw.date())))
        timeline = days.union(r.index)

        # 尾部：原始行之后连续被名单确认停牌的交易日
        if until is None:
            today = date.today()
            until = today if self._calendar.is_trading_day(today) else self._calendar.previous_trading_day(today, n=1)
        delist = self._delist_date(code)
        tail_end = min(until, delist - timedelta(days=1)) if delist else until
        tail_days = []
        if tail_end > last_raw.date() and self._suspend.enabled:
            tail_start = self._calendar.next_trading_day(last_raw.date(), n=1)
            if tail_start <= tail_end:
                tail_days = self._calendar.get_trading_days(tail_start, tail_end)
        if tail_days:
            std = format_code(code)
            tail = []
            for d in tail_days:
                try:
                    if std not in self._suspend.get_suspended_set(d):
                        break
                except Exception:
                    break
                tail.append(pd.Timestamp(d))
            if tail:
                timeline = timeline.union(pd.DatetimeIndex(tail))

        m = pd.DataFrame({"date": timeline})
        close = pd.to_numeric(r["close"], errors="coerce").reindex(timeline)
        volume = pd.to_numeric(r["volume"], errors="coerce").reindex(timeline)
        priced = close.notna().to_numpy()
        zero_vol = priced & (volume.fillna(0).to_numpy() <= 0)
        confirmed = self._confirmed_suspended_mask(
            code, m["date"], pd.Series(~priced | zero_vol, index=m.index),
        ).to_numpy()
        trade = priced & ~(zero_vol & confirmed)
        m["status"] = np.where(trade, "trade", np.where(confirmed, "suspend", "missing"))
        m["is_suspended"] = ~trade
        rows = None if compute_from is None else (m["date"] >= compute_from).to_numpy()
        m["limit_up"], m["limit_down"] = self._limit_prices(
            code, pd.DataFrame({"date": m["date"], "close": close.to_numpy()}), rows=rows,
        )
        return self._cast_types(m[cols], DAILY_META_COLUMNS)

    def _refresh_daily_meta(
        self,
        code: str,
        raw: pd.DataFrame,
        since: pd.Timestamp | None = None,
        until: date | None = None,
    ) -> bool:
        """重算并落盘该股日线加工层；返回是否有变化（无变化不写盘，保留 mtime）。

        ``since`` 给出且已有加工层时：``since`` 之前的旧行原样保留，之后的行按完整历史重算
        （增量更新用）；否则整体重算（``rebuild_daily_meta`` / 首拉）。
        """
        old = self._store.read_daily_meta(code)
        incremental = since is not None and old is not None and not old.empty
        new = self._build_daily_meta(code, raw, until=until, compute_from=since if incremental else None)
        if incremental:
            new = pd.concat(
                [old[old["date"] < since], new[new["date"] >= since]], ignore_index=True,
            )
            new = self._cast_types(new[list(DAILY_META_COLUMNS)], DAILY_META_COLUMNS)
        if old is not None and len(old) == len(new):
            o = self._cast_types(old[list(DAILY_META_COLUMNS)].copy(), DAILY_META_COLUMNS)
            if o.reset_index(drop=True).equals(new.reset_index(drop=True)):
                return False
        self._store.write_daily_meta(code, new)
        return True

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
                # 统一到 ns 精度（parquet 往返可能是 us），否则新旧帧比较会因精度不同误判有变化
                df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
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
        """构建全市场「代码→当前简称」表（ST 判定用）。

        **本地稳定源优先**：先读本地 tushare 基础信息缓存（``stock_basic_tushare.parquet``，
        含在市+退市当前名，离线）；缺失/读取失败才**降级**到东财（akshare
        ``stock_info_a_code_name``，实时联网，偶发连接重置不稳定）。
        """
        if self._stock_name_cache:
            return
        with self._name_lock:
            if self._stock_name_cache:  # 等锁期间已被别的线程填充
                return
            if self._load_names_from_local():
                return
            self._load_names_from_akshare()

    def _load_names_from_local(self) -> bool:
        """从本地 tushare 基础信息缓存构建名称表（离线主源）。返回是否成功填充。"""
        path = self._store_path / STOCK_BASIC_TUSHARE_FILE
        if not path.exists():
            logger.debug(f"本地基础信息缓存不存在（{path}），降级东财名称表")
            return False
        try:
            df = pd.read_parquet(path, columns=["ts_code", "name"])
        except Exception as exc:
            logger.warning(f"读取本地基础信息 {path} 失败，降级东财: {exc}")
            return False
        if df is None or df.empty:
            return False
        self._stock_name_cache.update({
            format_code(str(c)): str(n)
            for c, n in zip(df["ts_code"].tolist(), df["name"].tolist(), strict=True)
            if pd.notna(c) and pd.notna(n) and str(n).strip()
        })
        if not self._stock_name_cache:
            return False
        logger.debug(
            f"股票名称表已从本地基础信息加载: {len(self._stock_name_cache)} 只 | {path}"
        )
        return True

    def _load_names_from_akshare(self) -> None:
        """降级源：东财全市场代码+当前名（实时联网，偶发不稳）。"""
        try:
            import akshare as ak
            df = _ak_call(ak.stock_info_a_code_name)
            if df is not None and not df.empty:
                # 向量化构建，比 iterrows() 快 10x+
                self._stock_name_cache.update({
                    format_code(str(c)): str(n)
                    for c, n in zip(df["code"].tolist(), df["name"].tolist(), strict=True)
                })
        except ProxyConfigError:
            raise
        except Exception as exc:
            logger.warning(f"加载股票名称表失败（东财降级源）: {exc}")

    def _get_stock_name(self, code: str) -> str:
        std = format_code(code)
        if std in self._stock_name_cache:
            return self._stock_name_cache[std]
        self._ensure_stock_name_table()
        return self._stock_name_cache.get(std, "")

    def _ensure_delist_table(self) -> None:
        """构建全市场「代码→退市日」表（仅退市股，来自本地 tushare 基础信息）。

        退市股历史在退市日后即冻结；增量更新据此把 ``end`` 截到退市日，避免对已退市股
        做"退市日→今天"的无效拉取（曾因退市股拉超长空区间降级东财、触发系统代理错误而
        中断整个全市场更新）。退市日前的联网 fallback 仍保留（用于补退市整理期数据）。

        本地基础信息缺失/无 ``delist_date`` 列时留空（不截断，行为同旧版）。
        """
        if self._delist_date_cache:
            return
        with self._delist_lock:
            if self._delist_date_cache:  # 等锁期间已被别的线程填充
                return
            path = self._store_path / STOCK_BASIC_TUSHARE_FILE
            if not path.exists():
                logger.debug(f"本地基础信息缺失（{path}），退市股 end 不截断")
                return
            try:
                df = pd.read_parquet(path, columns=["ts_code", "list_status", "delist_date"])
            except Exception as exc:
                logger.debug(f"读取退市日表失败（退市股 end 不截断）: {exc}")
                return
            if df is None or df.empty:
                return
            # 退市判定：状态为 D，**或** 已填退市日（tushare 对退市整理期/刚摘牌的边界股
            # 常出现 delist_date 已填、list_status 仍 L 的口径滞后，如 688287.SH 摘牌于
            # 20260610 却仍标 L）。只要有退市日即视为退市并据此截断，避免对其无效联网拉取。
            status = df["list_status"].astype("string").str.upper()
            delist_raw = df["delist_date"].astype("string").str.strip()
            has_delist = delist_raw.notna() & ~delist_raw.isin(["", "nan", "NaN", "None"])
            d = df[(status == "D") | has_delist]
            for code, raw in zip(d["ts_code"].tolist(), d["delist_date"].tolist(), strict=True):
                dt = pd.to_datetime(str(raw), format="%Y%m%d", errors="coerce")
                if pd.notna(dt):
                    self._delist_date_cache[format_code(str(code))] = dt.date()
            logger.debug(f"退市日表已加载: {len(self._delist_date_cache)} 只退市股 | {path}")

    def _delist_date(self, code: str) -> date | None:
        """该股票的退市日（已退市且有 delist_date 时返回，否则 None）。"""
        self._ensure_delist_table()
        return self._delist_date_cache.get(format_code(code))

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
        "--max-workers", type=int, default=None,
        help=f"并行线程数（按股票并行，1=串行）；缺省读 config.data.max_workers，无配置则 {DEFAULT_MAX_WORKERS}",
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
        "--rebuild-meta", "--recompute-limits", dest="rebuild_meta", action="store_true",
        help="按最新规则重算日线加工层 daily_meta（时间轴/停牌/涨跌停）后退出；原始日线不动、不拉新数据",
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
    # 并发数单一事实来源：CLI 显式 > config.data.max_workers > 代码默认（与 src.main fetch 一致）
    if args.max_workers is not None:
        workers = args.max_workers
    else:
        try:
            from ..utils.config_loader import load_config
            workers = int(load_config().data.max_workers)
        except Exception:
            workers = DEFAULT_MAX_WORKERS
    fetcher = DataFetcher(
        store_path=args.store_path, tdx_path=args.tdx_path, max_workers=workers,
        factor_source=args.factor_source,
        factor_skip_via_gbbq=not args.no_factor_skip,
    )
    logger.info(f"实例化: {fetcher}")

    if args.rebuild_meta:
        codes = [format_code(c) for c in args.codes] if args.codes else None
        n = fetcher.rebuild_daily_meta(codes)
        logger.success(f"日线加工层重算完成（重写 {n} 只）")
        return 0

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
        max_workers=workers,
    )
    logger.success("数据拉取/更新完成")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
