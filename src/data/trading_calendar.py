"""
A 股交易日历
============
基于 akshare `tool_trade_date_hist_sina()` 拉取 1990-01 至当前年末的所有交易日，
并落盘到 ``data_store/calendar.parquet`` 作为离线缓存。

特性：
- 优先读本地 parquet；缓存过期或缺失才回源
- 缓存过期阈值由 ``config.data.calendar_refresh_days`` 控制（默认 7 天）
- 远程拉取套用 ``utils.helpers.retry`` 指数退避；全部失败时若有本地缓存则
  WARNING 后继续使用旧缓存（保证断网可用），无缓存才抛 ``RuntimeError``
- 查询接口基于 ``np.searchsorted``，单次复杂度 O(log N)

公开接口（与 Prompt 一致）：
    is_trading_day(d)              -> bool
    next_trading_day(d, n=1)       -> date
    previous_trading_day(d, n=1)   -> date
    get_trading_days(start, end)   -> list[date]

使用示例：
    >>> from a_share_quant_pro.data import TradingCalendar
    >>> cal = TradingCalendar()
    >>> cal.is_trading_day(date(2024, 1, 1))
    False
    >>> cal.next_trading_day(date(2024, 1, 1))
    datetime.date(2024, 1, 2)
"""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from ..utils.helpers import ensure_dir, parse_date, retry

# ============================================================
# 常量
# ============================================================

#: 缓存文件相对于 store_path 的路径
_CALENDAR_FILE = "calendar.parquet"

#: 远程拉取单次重试间等待秒数（指数退避）
_RETRY_DELAYS = [1, 2, 4]

#: 远程拉取最大重试次数
_RETRY_TIMES = 3


# ============================================================
# 主类
# ============================================================


class TradingCalendar:
    """
    A 股交易日历。

    Args:
        store_path: 数据仓库根目录，缓存写入 ``{store_path}/calendar.parquet``。
                    默认 ``"data_store"``。
        refresh_days: 缓存过期阈值（天）。文件 mtime 距今 > 此值则刷新。
                      默认 7。
        auto_load: 构造时立即加载/刷新缓存。设 False 可延迟（用于测试）。
                   默认 True。

    Attributes:
        trading_days: ``pd.DatetimeIndex``，已排序去重的全部交易日（含历史 + 未来发布日历）。
    """

    def __init__(
        self,
        store_path: str | Path = "data_store",
        refresh_days: int = 7,
        auto_load: bool = True,
    ) -> None:
        self._store_path = Path(store_path)
        self._refresh_days = int(refresh_days)
        self._cache_file = self._store_path / _CALENDAR_FILE
        self._trading_days: pd.DatetimeIndex = pd.DatetimeIndex([])
        if auto_load:
            self.load()

    # ------------------------------------------------------------
    # 加载与缓存
    # ------------------------------------------------------------

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        """全部交易日（已排序去重）"""
        return self._trading_days

    def load(self, force_refresh: bool = False) -> None:
        """
        加载交易日历。

        策略：
        1. ``force_refresh=True`` → 直接回源
        2. 缓存不存在 → 回源
        3. 缓存 mtime 距今 > refresh_days → 回源；回源失败但缓存可用则 WARNING 沿用
        4. 否则直接读缓存

        Raises:
            RuntimeError: 无本地缓存且远程拉取全部重试失败。
        """
        cache_exists = self._cache_file.exists()
        cache_stale = cache_exists and self._is_cache_stale()

        if cache_exists and not force_refresh and not cache_stale:
            self._load_from_cache()
            return

        # 需要回源（首次/强制/过期）
        try:
            self._refresh_remote()
        except Exception as exc:
            if cache_exists:
                logger.warning(
                    f"远程交易日历拉取失败（{type(exc).__name__}: {exc}），"
                    f"沿用本地缓存 {self._cache_file}"
                )
                self._load_from_cache()
            else:
                raise RuntimeError(
                    f"无法加载交易日历：本地缓存不存在且远程拉取失败（{exc}）"
                ) from exc

    def _is_cache_stale(self) -> bool:
        """判断缓存是否过期（基于 mtime）"""
        try:
            mtime = self._cache_file.stat().st_mtime
        except OSError:
            return True
        age_days = (time.time() - mtime) / 86400.0
        return age_days > self._refresh_days

    def _load_from_cache(self) -> None:
        """从 parquet 读缓存"""
        df = pd.read_parquet(self._cache_file)
        days = self._normalize_days(df["date"])
        self._trading_days = days
        logger.debug(
            f"交易日历缓存已加载 | 文件: {self._cache_file} | "
            f"区间: {days[0].date()} ~ {days[-1].date()} | 天数: {len(days)}"
        )

    def _refresh_remote(self) -> None:
        """从远程拉取并写入缓存"""
        days = self._normalize_days(self._fetch_remote())
        if days.empty:
            raise RuntimeError("远程返回空日历")
        self._trading_days = days
        self._save_cache(days)
        logger.info(
            f"交易日历已刷新 | 区间: {days[0].date()} ~ {days[-1].date()} | "
            f"天数: {len(days)} | 缓存: {self._cache_file}"
        )

    @retry(max_attempts=_RETRY_TIMES, delays=_RETRY_DELAYS)
    def _fetch_remote(self) -> pd.Series:
        """
        从 akshare 拉取交易日历。

        Returns:
            pd.Series of date-like values（日期列）。

        Note:
            akshare 的 ``tool_trade_date_hist_sina()`` 返回 DataFrame，
            列名为 ``trade_date``，类型为 datetime.date。
        """
        # 延迟 import，避免未联网环境下纯本地缓存测试也强制依赖 akshare
        import akshare as ak

        df = ak.tool_trade_date_hist_sina()
        if df is None or df.empty:
            raise RuntimeError("akshare 返回空 DataFrame")

        # 兼容字段名：trade_date 是当前主流，历史版本可能为 date
        col = "trade_date" if "trade_date" in df.columns else "date"
        if col not in df.columns:
            raise RuntimeError(
                f"akshare 返回的 DataFrame 缺少日期列，实际列: {list(df.columns)}"
            )
        return df[col]

    def _save_cache(self, days: pd.DatetimeIndex) -> None:
        """写 parquet 缓存"""
        ensure_dir(self._store_path)
        df = pd.DataFrame({"date": days})
        df.to_parquet(self._cache_file, index=False)

    @staticmethod
    def _normalize_days(values) -> pd.DatetimeIndex:
        """
        将任意日期序列统一规整为：去重 + 升序 + 标准化到 00:00:00 的 DatetimeIndex。
        """
        idx = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).dropna()
        idx = idx.normalize()  # 去除时分秒
        idx = idx.unique().sort_values()
        return idx

    # ------------------------------------------------------------
    # 公开查询接口
    # ------------------------------------------------------------

    def is_trading_day(self, d: str | date | datetime) -> bool:
        """
        判断指定日期是否为交易日。

        Args:
            d: 任意可被 ``parse_date`` 解析的日期。

        Returns:
            True/False
        """
        self._ensure_loaded()
        ts = pd.Timestamp(parse_date(d))
        # searchsorted 精确匹配
        pos = self._trading_days.searchsorted(ts, side="left")
        return bool(pos < len(self._trading_days) and self._trading_days[pos] == ts)

    def next_trading_day(
        self, d: str | date | datetime, n: int = 1
    ) -> date:
        """
        返回距 d 之后的第 n 个交易日。

        Args:
            d: 起始日期（不要求是交易日；不被计入计数）。
            n: 偏移天数，必须 ≥ 1。

        Returns:
            ``datetime.date`` 对象。

        Raises:
            ValueError: n < 1
            IndexError: 超出日历末尾。
        """
        if n < 1:
            raise ValueError(f"n 必须 >= 1, got {n}")
        self._ensure_loaded()
        ts = pd.Timestamp(parse_date(d))
        # 严格大于 d 的第一个位置
        pos = int(self._trading_days.searchsorted(ts, side="right"))
        target = pos + (n - 1)
        if target >= len(self._trading_days):
            raise IndexError(
                f"超出日历范围：基准 {ts.date()} 之后第 {n} 个交易日不在缓存内 "
                f"（缓存末尾 {self._trading_days[-1].date()}）"
            )
        return self._trading_days[target].date()

    def previous_trading_day(
        self, d: str | date | datetime, n: int = 1
    ) -> date:
        """
        返回距 d 之前的第 n 个交易日。

        Args:
            d: 起始日期（不要求是交易日；不被计入计数）。
            n: 偏移天数，必须 ≥ 1。

        Returns:
            ``datetime.date`` 对象。

        Raises:
            ValueError: n < 1
            IndexError: 超出日历起点。
        """
        if n < 1:
            raise ValueError(f"n 必须 >= 1, got {n}")
        self._ensure_loaded()
        ts = pd.Timestamp(parse_date(d))
        # 严格小于 d 的最后一个位置
        pos = int(self._trading_days.searchsorted(ts, side="left")) - 1
        target = pos - (n - 1)
        if target < 0:
            raise IndexError(
                f"超出日历范围：基准 {ts.date()} 之前第 {n} 个交易日不在缓存内 "
                f"（缓存起点 {self._trading_days[0].date()}）"
            )
        return self._trading_days[target].date()

    def get_trading_days(
        self,
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> list[date]:
        """
        返回闭区间 ``[start, end]`` 内的全部交易日列表。

        Args:
            start: 起始日期（含）。
            end:   结束日期（含）。

        Returns:
            按时间升序的 ``date`` 列表；区间无交易日返回空列表。

        Raises:
            ValueError: ``start > end``。
        """
        self._ensure_loaded()
        s = pd.Timestamp(parse_date(start))
        e = pd.Timestamp(parse_date(end))
        if s > e:
            raise ValueError(f"start({s.date()}) 必须 <= end({e.date()})")
        left = int(self._trading_days.searchsorted(s, side="left"))
        right = int(self._trading_days.searchsorted(e, side="right"))
        return [ts.date() for ts in self._trading_days[left:right]]

    # ------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """惰性加载兜底：构造时 auto_load=False 时由首次查询触发"""
        if len(self._trading_days) == 0:
            self.load()

    def __repr__(self) -> str:
        if len(self._trading_days) == 0:
            return f"<TradingCalendar empty store={self._store_path}>"
        return (
            f"<TradingCalendar "
            f"{self._trading_days[0].date()}~{self._trading_days[-1].date()} "
            f"days={len(self._trading_days)}>"
        )


# ============================================================
# 模块自测  python -m src.data.trading_calendar
# ============================================================


if __name__ == "__main__":
    from ..utils.helpers import init_logging

    init_logging(level="DEBUG")

    cal = TradingCalendar(store_path="data_store", refresh_days=7)
    logger.info(f"日历: {cal}")

    # 已知非交易日：2024-01-01 元旦
    assert cal.is_trading_day("2024-01-01") is False
    # 2024-01-02 周二，正常交易日
    assert cal.is_trading_day("2024-01-02") is True
    logger.success("✓ is_trading_day")

    nxt = cal.next_trading_day("2024-01-01")
    assert nxt == date(2024, 1, 2), nxt
    nxt5 = cal.next_trading_day("2024-01-01", n=5)
    logger.info(f"2024-01-01 之后第 5 个交易日: {nxt5}")
    logger.success("✓ next_trading_day")

    prev = cal.previous_trading_day("2024-01-02")
    logger.info(f"2024-01-02 之前 1 个交易日: {prev}")
    logger.success("✓ previous_trading_day")

    days = cal.get_trading_days("2024-01-01", "2024-01-10")
    logger.info(f"2024-01-01~10 区间交易日: {days}")
    assert all(cal.is_trading_day(d) for d in days)
    assert days == sorted(days)
    logger.success("✓ get_trading_days")

    # 边界异常
    try:
        cal.next_trading_day("2024-01-01", n=0)
    except ValueError:
        logger.success("✓ next_trading_day(n=0) 正确报错")

    try:
        cal.get_trading_days("2024-12-31", "2024-01-01")
    except ValueError:
        logger.success("✓ get_trading_days(start>end) 正确报错")

    logger.success("所有测试通过！")
