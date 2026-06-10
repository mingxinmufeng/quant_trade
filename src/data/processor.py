"""
数据清洗 / 对齐 / 复权编排（processor）
=======================================

在「原始价与因子分离落盘」的新架构下，复权（``adjust``）、重采样（``resample``）、
停牌名单（``suspend``）均已拆为独立模块并在 ``DataStore.load`` / ``DataFetcher``
中按需调用。本模块不重复造轮子，而是把这些散落的能力**编排**成一条面向使用方
（因子层 / 策略层 / 回测引擎）的标准流水线，并补齐其中**唯一缺位**的一环 ——
**OHLCV 清洗**与**交易日历对齐 / 多股票面板对齐**。

职责
----
1. **清洗 clean()**：单只股票 OHLCV 去重、排序、按统一 schema 规整类型、剔除非法
   数值（负价/零价、``high<low`` 等），并按 A 股口径处理停牌行（OHLCV 置 NaN）。
2. **对齐 align_to_calendar()**：日线按 ``TradingCalendar`` reindex 到 ``[start,end]``
   的完整交易日序列，缺失交易日补行并标记停牌（**不臆造价格**，仅填 NaN），缺口
   写日志（与 fetcher 的"不自动补脏数据"原则一致）。
3. **面板对齐 align_panel()**：把 ``{code: df}`` 透视为 ``{field: DataFrame}``
   （``index=date, columns=code``），供因子 / 回测向量化使用。
4. **复权 adjust() / 重采样 resample()**：对 ``src.data.adjust`` 与
   ``src.data.resample`` 的**薄封装**，提供单一调用入口（不改变其语义）。
5. **流水线 process()**：``clean → (对齐) → 复权``（默认复权在 load 时已完成，故
   ``adjust='none'`` 不二次复权），返回可直接喂给上层的干净 DataFrame。

设计要点
--------
- **幂等**：对已清洗的数据再次 ``clean`` 结果不变。
- **不臆造数据**：对齐补出的交易日只填 NaN + ``is_suspended=True``，绝不前向填充
  价格（避免引入未来函数 / 脏数据）。
- **零强依赖网络**：``TradingCalendar`` 可注入；不传则惰性构造（读本地缓存）。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.helpers import parse_date
from .adjust import ADJUST_MODES, apply_adjust
from .resample import DAILY_RESAMPLE_RULES, resample_daily, resample_minute
from .storage import DAILY_COLUMNS, MINUTE_COLUMNS
from .trading_calendar import TradingCalendar

__all__ = ["DataProcessor"]

#: 参与"非法数值→NaN"校验与复权缩放的价格列
_PRICE_COLS = ("open", "high", "low", "close")

#: 停牌时应被置为 NaN 的列（价格 + 量额；元数据 is_suspended/limit_*/name 保留）
_SUSPEND_NAN_COLS = ("open", "high", "low", "close", "volume", "amount")


def _time_col(freq: str) -> str:
    """日线时间列为 ``date``，分钟线为 ``datetime``。"""
    return "date" if freq == "daily" else "datetime"


def _schema(freq: str) -> Dict[str, str]:
    """返回对应周期的落盘字段 schema。"""
    return DAILY_COLUMNS if freq == "daily" else MINUTE_COLUMNS


class DataProcessor:
    """OHLCV 清洗 / 交易日历对齐 / 复权 / 重采样的编排器。

    Args:
        calendar: 交易日历实例；``None`` 时按 ``store_path`` 惰性构造（读本地缓存）。
        store_path: 仅在需要惰性构造 ``TradingCalendar`` 时使用的数据仓库根目录。
        mask_suspended: 清洗时是否把停牌行（``is_suspended=True``）的 OHLCV / 量额置 NaN。
            A 股口径默认 ``True``（停牌日无有效成交价）。
        drop_zero_price: 是否把 ``<=0`` 的价格视为非法并置 NaN（默认 ``True``）。
    """

    def __init__(
        self,
        calendar: Optional[TradingCalendar] = None,
        store_path: Union[str, Path] = "data_store",
        mask_suspended: bool = True,
        drop_zero_price: bool = True,
    ) -> None:
        self._calendar = calendar
        self._store_path = store_path
        self._mask_suspended = bool(mask_suspended)
        self._drop_zero_price = bool(drop_zero_price)

    # ------------------------------------------------------------
    # 交易日历（惰性）
    # ------------------------------------------------------------

    @property
    def calendar(self) -> TradingCalendar:
        """交易日历（首次访问时按 ``store_path`` 构造）。"""
        if self._calendar is None:
            self._calendar = TradingCalendar(store_path=self._store_path)
        return self._calendar

    # ------------------------------------------------------------
    # 1. 清洗
    # ------------------------------------------------------------

    def clean(self, df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
        """清洗单只股票的 OHLCV（去重 / 排序 / 类型规整 / 非法值剔除 / 停牌处理）。

        Args:
            df: 单只股票行情（含时间列 ``date`` 或 ``datetime``）。
            freq: ``daily`` / ``min5`` / ``min1``（决定时间列与 schema）。

        Returns:
            新的 DataFrame（原对象不被修改）。空输入原样返回。
        """
        if df is None or df.empty:
            return df.copy() if df is not None else pd.DataFrame()

        tc = _time_col(freq)
        if tc not in df.columns:
            raise KeyError(f"清洗 {freq} 需要时间列 {tc!r}，实际列: {list(df.columns)}")

        out = df.copy()
        # 时间列规整 + 丢弃无法解析的时间行
        out[tc] = pd.to_datetime(out[tc], errors="coerce")
        if freq == "daily":
            out[tc] = out[tc].dt.normalize()
        bad_time = out[tc].isna()
        if bad_time.any():
            logger.debug(f"清洗：丢弃 {int(bad_time.sum())} 行无法解析的时间")
            out = out[~bad_time]

        # 去重：同一时间保留最后一条（最新拉取覆盖旧值），再升序
        out = out.drop_duplicates(subset=[tc], keep="last").sort_values(tc).reset_index(drop=True)

        # 类型规整：按 schema 强制 dtype（缺失列不补，避免改变上游契约）
        out = self._coerce_dtypes(out, freq)

        # 非法数值 → NaN
        out = self._sanitize_values(out)

        # 停牌行处理（A 股口径：停牌日 OHLCV / 量额无效）
        if self._mask_suspended and "is_suspended" in out.columns:
            susp = out["is_suspended"].fillna(False).astype(bool)
            if susp.any():
                for c in _SUSPEND_NAN_COLS:
                    if c in out.columns:
                        out.loc[susp, c] = np.nan

        return out.reset_index(drop=True)

    def _coerce_dtypes(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """按 schema 把已存在的列转成目标 dtype（缺失列跳过）。"""
        schema = _schema(freq)
        for col, dtype in schema.items():
            if col not in df.columns:
                continue
            if dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dtype == "bool":
                df[col] = df[col].fillna(False).astype(bool)
            elif dtype == "string":
                df[col] = df[col].astype("string")
        return df

    def _sanitize_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """剔除非法 OHLCV：非正价格、量额为负、``high/low`` 越界。"""
        price_cols = [c for c in _PRICE_COLS if c in df.columns]

        # 非正价格 → NaN（停牌/异常）
        if self._drop_zero_price:
            for c in price_cols:
                df.loc[df[c] <= 0, c] = np.nan

        # 成交量 / 成交额为负 → NaN（数据源异常）
        for c in ("volume", "amount"):
            if c in df.columns:
                df.loc[df[c] < 0, c] = np.nan

        # high/low 一致性：high 应 ≥ max(o,c)，low 应 ≤ min(o,c)，且 high ≥ low
        if {"high", "low"}.issubset(df.columns):
            oc_cols = [c for c in ("open", "close") if c in df.columns]
            if oc_cols:
                df["high"] = df[["high", *oc_cols]].max(axis=1)
                df["low"] = df[["low", *oc_cols]].min(axis=1)
            bad_hl = df["high"] < df["low"]
            if bad_hl.any():
                logger.debug(f"清洗：{int(bad_hl.sum())} 行 high<low，已置 NaN")
                df.loc[bad_hl, ["high", "low"]] = np.nan
        return df

    # ------------------------------------------------------------
    # 2. 交易日历对齐（仅日线）
    # ------------------------------------------------------------

    def align_to_calendar(
        self,
        df: pd.DataFrame,
        start: Optional[Union[str, date, datetime]] = None,
        end: Optional[Union[str, date, datetime]] = None,
        code: Optional[str] = None,
    ) -> pd.DataFrame:
        """把日线 reindex 到 ``[start, end]`` 的完整交易日序列。

        缺失的交易日补出一行：``is_suspended=True``，OHLCV / 量额为 NaN（**不前向填充
        价格**），``code`` / ``name`` 沿用已知值，``source`` 标记为 ``aligned``。
        区间内的缺口数量写 DEBUG 日志。

        Args:
            df: 已清洗的日线（含 ``date`` 列）。
            start: 起始日期（含）；``None`` 取数据最早日。
            end: 结束日期（含）；``None`` 取数据最晚日。
            code: 用于日志标识，可选。

        Returns:
            按交易日升序、索引重置的 DataFrame。
        """
        if df is None or df.empty:
            return df.copy() if df is not None else pd.DataFrame()
        if "date" not in df.columns:
            raise KeyError("align_to_calendar 仅适用于含 'date' 列的日线数据")

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        s = parse_date(start) if start is not None else out["date"].min().date()
        e = parse_date(end) if end is not None else out["date"].max().date()

        days = self.calendar.get_trading_days(s, e)
        if not days:
            logger.warning(f"[{code or '?'}] 区间 {s}~{e} 无交易日，对齐返回空")
            return out.iloc[0:0].reset_index(drop=True)

        target = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
        out = out.set_index("date").reindex(target)
        out.index.name = "date"

        # 缺口统计：原本无数据、被本次对齐补出的交易日
        missing_mask = out["close"].isna() if "close" in out.columns else out.isna().all(axis=1)
        n_missing = int(missing_mask.sum())
        if n_missing:
            logger.debug(
                f"[{code or '?'}] 交易日历对齐：{s}~{e} 共 {len(target)} 个交易日，"
                f"补出 {n_missing} 个缺口（标记停牌，不补价）"
            )

        # 标记新增行：停牌 + 沿用静态字段（不沿用价格）
        if "is_suspended" in out.columns:
            out.loc[missing_mask, "is_suspended"] = True
            out["is_suspended"] = out["is_suspended"].fillna(True).astype(bool)
        for static_col in ("code", "name"):
            if static_col in out.columns:
                out[static_col] = out[static_col].ffill().bfill()
        if code is not None and "code" in out.columns:
            out["code"] = out["code"].fillna(code)
        if "source" in out.columns:
            out.loc[missing_mask, "source"] = "aligned"
            out["source"] = out["source"].astype("string")

        return out.reset_index()

    # ------------------------------------------------------------
    # 3. 多股票面板对齐
    # ------------------------------------------------------------

    def align_panel(
        self,
        data: Dict[str, pd.DataFrame],
        fields: Sequence[str] = ("close",),
        start: Optional[Union[str, date, datetime]] = None,
        end: Optional[Union[str, date, datetime]] = None,
        use_calendar: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """把 ``{code: 日线df}`` 透视为 ``{field: DataFrame(index=date, columns=code)}``。

        统一日期索引：``use_calendar=True`` 时取 ``[start,end]`` 的交易日（缺省由数据
        覆盖区间推断）；否则取所有股票出现过的日期并集。**不前向填充**，缺失为 NaN。

        Args:
            data: 每只股票一张日线表（含 ``date`` 列）。
            fields: 需要提取的字段，如 ``("open","close","volume")``。
            start / end: 统一索引区间，可选。
            use_calendar: 是否用交易日历生成统一索引（推荐，规避个股停牌导致的索引漂移）。

        Returns:
            ``{field: DataFrame}``；输入为空时返回各 field 的空表。
        """
        fields = tuple(fields)
        if not data:
            return {f: pd.DataFrame() for f in fields}

        frames = {c: df for c, df in data.items() if df is not None and not df.empty}
        if not frames:
            return {f: pd.DataFrame() for f in fields}

        # 统一日期索引
        if use_calendar:
            mins = [pd.to_datetime(df["date"]).min() for df in frames.values()]
            maxs = [pd.to_datetime(df["date"]).max() for df in frames.values()]
            s = parse_date(start) if start is not None else min(mins).date()
            e = parse_date(end) if end is not None else max(maxs).date()
            idx = pd.DatetimeIndex([pd.Timestamp(d) for d in self.calendar.get_trading_days(s, e)])
        else:
            all_dates = pd.DatetimeIndex(
                np.concatenate([pd.to_datetime(df["date"]).values for df in frames.values()])
            )
            idx = all_dates.normalize().unique().sort_values()
            if start is not None:
                idx = idx[idx >= pd.Timestamp(parse_date(start))]
            if end is not None:
                idx = idx[idx <= pd.Timestamp(parse_date(end))]

        result: Dict[str, pd.DataFrame] = {}
        for field in fields:
            cols = {}
            for code, df in frames.items():
                if field not in df.columns:
                    continue
                ser = df.set_index(pd.to_datetime(df["date"]).dt.normalize())[field]
                ser = ser[~ser.index.duplicated(keep="last")]
                cols[code] = ser.reindex(idx)
            panel = pd.DataFrame(cols, index=idx)
            panel.index.name = "date"
            result[field] = panel.sort_index()
        return result

    # ------------------------------------------------------------
    # 4. 复权 / 重采样（薄封装，委托既有纯函数）
    # ------------------------------------------------------------

    def adjust(
        self,
        df: pd.DataFrame,
        factor: pd.DataFrame,
        mode: str = "hfq",
        *,
        freq: str = "daily",
        anchor_date: Optional[Union[str, date, datetime]] = None,
    ) -> pd.DataFrame:
        """按因子表复权（委托 ``src.data.adjust.apply_adjust``）。

        ``mode`` ∈ {``none``, ``hfq``, ``qfq``}；``qfq`` 可传 ``anchor_date`` 锚定。
        """
        if mode not in ADJUST_MODES:
            raise ValueError(f"未知复权模式 {mode!r}（支持 {ADJUST_MODES}）")
        return apply_adjust(df, factor, mode=mode, time_col=_time_col(freq), anchor_date=anchor_date)

    def resample(self, df: pd.DataFrame, period: str, freq: str = "daily") -> pd.DataFrame:
        """周期重采样（委托 ``src.data.resample``）。

        - ``freq='daily'``：``period`` ∈ ``DAILY_RESAMPLE_RULES``（weekly/monthly/...）。
        - ``freq='min5'/'min1'``：``period`` 为目标分钟数（如 ``15``、``30``）。
        """
        if freq == "daily":
            if period != "daily" and period not in DAILY_RESAMPLE_RULES:
                raise ValueError(f"不支持的日线周期 {period!r}（支持 {list(DAILY_RESAMPLE_RULES)} 或 'daily'）")
            return resample_daily(df, period)
        base = int(freq.replace("min", ""))
        return resample_minute(df, base, int(period))

    # ------------------------------------------------------------
    # 5. 端到端流水线
    # ------------------------------------------------------------

    def process(
        self,
        df: pd.DataFrame,
        freq: str = "daily",
        *,
        align: bool = False,
        start: Optional[Union[str, date, datetime]] = None,
        end: Optional[Union[str, date, datetime]] = None,
        code: Optional[str] = None,
    ) -> pd.DataFrame:
        """标准流水线：``clean`` →（可选 ``align_to_calendar``）。

        复权默认已在 ``DataFetcher.load_*`` / ``DataStore.load`` 阶段完成，故本流水线
        不二次复权（如需复权请显式调用 :meth:`adjust`）。

        Args:
            df: 单只股票行情。
            freq: ``daily`` / ``min5`` / ``min1``。
            align: 是否对齐交易日历（仅 ``daily`` 有意义）。
            start / end / code: 透传给 :meth:`align_to_calendar`。
        """
        cleaned = self.clean(df, freq=freq)
        if align and freq == "daily":
            cleaned = self.align_to_calendar(cleaned, start=start, end=end, code=code)
        return cleaned

    def __repr__(self) -> str:  # pragma: no cover - 仅调试展示
        return (
            f"<DataProcessor mask_suspended={self._mask_suspended} "
            f"drop_zero_price={self._drop_zero_price}>"
        )


# ============================================================
# 模块自测  python -m src.data.processor
# ============================================================

if __name__ == "__main__":
    from ..utils.helpers import init_logging

    init_logging(level="DEBUG")
    proc = DataProcessor(mask_suspended=True)

    demo = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"],
            "code": ["000001.SZ"] * 4,
            "open": [10.0, 10.0, 0.0, 11.0],   # 第3行 0 价 → NaN
            "high": [10.5, 10.5, 9.0, 11.5],   # 第3行 high<low 触发修正
            "low": [9.8, 9.8, 10.0, 10.8],
            "close": [10.2, 10.2, 10.5, 11.2],
            "volume": [1e5, 1e5, 0.0, 2e5],
            "amount": [1e6, 1e6, 0.0, 2e6],
            "is_suspended": [False, False, True, False],
            "name": ["平安银行"] * 4,
            "source": ["pytdx"] * 4,
        }
    )
    cleaned = proc.clean(demo, freq="daily")
    logger.info(f"清洗后:\n{cleaned}")
    logger.info(f"停牌行 close 是否 NaN: {bool(cleaned.loc[cleaned['date'] == pd.Timestamp('2024-01-03'), 'close'].isna().all())}")
