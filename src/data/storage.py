"""
本地数据仓库（storage）
=======================

职责：parquet 读写 + 复权缓存。**原始数据与加工信息分离落盘**：

落盘布局::

    {store}/daily/{code}.parquet          # 原始日线：只存数据源返回的成交行（不复权 OHLCV + 来源）
    {store}/daily_meta/{code}.parquet     # 日线加工层：交易日时间轴 + 停牌/状态 + 涨跌停价（可重算）
    {store}/min1/{code}.parquet           # 不复权 1 分钟 OHLCV
    {store}/min5/{code}.parquet           # 不复权 5 分钟 OHLCV
    {store}/factors/{code}.parquet        # date → cum_factor（生效因子，覆盖刷新）
    {store}/factors_gbbq/{code}.parquet   # date → cum_factor（gbbq 自算并存记录，供交叉对比）
    {store}/gbbq_events.parquet           # 全市场权息事件快照（update 时按版本戳落盘）
    {store}/profile_names.parquet         # 全市场更名史快照（点位 ST 判定，update 时落盘）
    {store}/stock_basic.parquet           # 全市场基础信息（在市+退市，Universe 防幸存者偏差）
    {store}/adjusted/{freq}/{code}.parquet       # 后复权(hfq)缓存（生效因子，按版本戳重算）
    {store}/adjusted_gbbq/{freq}/{code}.parquet  # 后复权(hfq)缓存（gbbq 因子口径，隔离）

复权缓存策略（用户拍板）
------------------------
- **只缓存 hfq 一份到硬盘**；``qfq`` 由 hfq 除以锚点日累计因子（标量）现场派生，
  ``none`` 直接读原始，二者都不落缓存。
- **版本戳重算，不按时间删除**：hfq 缓存文件的 mtime ≥ ``原始文件`` / ``因子文件`` /
  （日线）``加工层文件`` 的 mtime 时视为有效；任一源更新（mtime 变新）即重算覆盖。

日线视图（``read_view`` / ``load``）
------------------------------------
原始日线不含停牌占位行与任何加工列；读取时按 ``daily_meta`` 的交易日时间轴外连接原始行，
补回 ``is_suspended`` / ``limit_up`` / ``limit_down`` / ``status``（停牌、缺数据的日子是 OHLCV
全 NaN 的行，与旧版占位行口径一致）。加工层由 ``DataFetcher`` 在更新时生成，
``DataFetcher.rebuild_daily_meta`` 可随时按最新规则整体重算，原始文件不动。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from ..utils.helpers import ensure_dir
from .adjust import align_cum_factor, apply_adjust, cum_factor_at
from .factors import FACTOR_COLUMNS

# ============================================================
# 字段规范（**原始不复权**落盘；复权在 load 时按需合成）
# ============================================================

#: 原始日线落盘字段（只存数据源返回的成交行：不复权 OHLCV + 行级来源）
DAILY_COLUMNS: dict[str, str] = {
    "date": "datetime64[ns]",
    "code": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "amount": "float64",
    "source": "string",  # 行级数据来源，便于溯源/回滚
}

#: 日线加工层字段（``daily_meta/``；由原始日线 + 交易日历 + 停牌名单 + 权息 + 更名史派生）
DAILY_META_COLUMNS: dict[str, str] = {
    "date": "datetime64[ns]",
    # 当日不可交易：无成交数据（停牌或缺数据），或零成交且停牌名单确认停牌
    "is_suspended": "bool",
    "limit_up": "float64",    # 涨停价（不设限 / 无法确定 → NaN）
    "limit_down": "float64",  # 跌停价
    "status": "string",       # trade=有成交数据 | suspend=停牌名单确认停牌 | missing=无数据且未确认
}

#: 日线加工层目录名
DAILY_META_DIR = "daily_meta"


def build_daily_view(
    raw: pd.DataFrame | None, meta: pd.DataFrame | None, code: str = "",
) -> pd.DataFrame | None:
    """原始日线 + 加工层 → 日线视图（交易日时间轴，列 = DAILY_COLUMNS + 加工列）。

    按日期外连接：加工层时间轴上无原始行的日子（停牌/缺数据）补 OHLCV 全 NaN 的行；
    加工层缺失或滞后（原始行不在时间轴上）时，这些行 ``is_suspended`` 按"有无收盘价"兜底、
    涨跌停为 NaN。
    """
    if raw is None:
        return None
    meta_cols = [c for c in DAILY_META_COLUMNS if c != "date"]
    if meta is None or meta.empty:
        # 无加工层：兜底（尚未迁移的旧格式仓库原始文件自带这些列，原样沿用）
        out = raw.copy()
        if "is_suspended" not in out.columns:
            out["is_suspended"] = out["close"].isna()
        for c in ("limit_up", "limit_down"):
            if c not in out.columns:
                out[c] = float("nan")
        if "status" not in out.columns:
            out["status"] = pd.array([pd.NA] * len(out), dtype="string")
        return out
    raw_cols = [c for c in raw.columns if c not in meta_cols]
    out = meta[["date", *meta_cols]].merge(raw[raw_cols], on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    stale = out["is_suspended"].isna()
    if stale.any():
        out.loc[stale, "is_suspended"] = out.loc[stale, "close"].isna()
    out["is_suspended"] = out["is_suspended"].astype(bool)
    if "code" in out.columns:
        out["code"] = out["code"].fillna(code or (raw["code"].iloc[0] if len(raw) else ""))
    extra = [c for c in out.columns if c not in DAILY_COLUMNS and c not in meta_cols]
    return out[[*[c for c in DAILY_COLUMNS if c in out.columns], *meta_cols, *extra]]

#: 分钟线落盘字段（不复权 OHLCV；仅 min1 / min5 落盘）
MINUTE_COLUMNS: dict[str, str] = {
    "datetime": "datetime64[ns]",
    "code": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "amount": "float64",
    "source": "string",
}

#: 落盘周期目录名
FREQ_DIRS = {"daily": "daily", "min5": "min5", "min1": "min1"}


def _time_col(freq: str) -> str:
    return "date" if freq == "daily" else "datetime"


class DataStore:
    """parquet 读写 + 复权缓存的本地仓库。"""

    def __init__(self, store_path: str | Path = "data_store") -> None:
        self._store_path = Path(store_path)
        self._dirs = {f: self._store_path / d for f, d in FREQ_DIRS.items()}
        for d in self._dirs.values():
            ensure_dir(d)
        self._meta_dir = ensure_dir(self._store_path / DAILY_META_DIR)
        self._factor_dir = ensure_dir(self._store_path / "factors")
        # gbbq 自算因子并存目录（与外部源 factors/ 互不覆盖，便于长期交叉对比）
        self._factor_gbbq_dir = ensure_dir(self._store_path / "factors_gbbq")
        self._adjusted_dir = ensure_dir(self._store_path / "adjusted")
        # gbbq 因子口径的 hfq 缓存（与 adjusted/ 隔离，避免两种因子来源互相污染缓存）
        self._adjusted_gbbq_dir = ensure_dir(self._store_path / "adjusted_gbbq")

    # ------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------

    def freq_path(self, code: str, freq: str) -> Path:
        return self._dirs[freq] / f"{code}.parquet"

    def daily_meta_path(self, code: str) -> Path:
        return self._meta_dir / f"{code}.parquet"

    def factor_path(self, code: str, gbbq: bool = False) -> Path:
        base = self._factor_gbbq_dir if gbbq else self._factor_dir
        return base / f"{code}.parquet"

    def adjusted_path(self, code: str, freq: str, gbbq: bool = False) -> Path:
        base = self._adjusted_gbbq_dir if gbbq else self._adjusted_dir
        return ensure_dir(base / freq) / f"{code}.parquet"

    # ------------------------------------------------------------
    # 生效因子选择（gbbq 口径优先用 factors_gbbq/，缺失则回退 factors/）
    # ------------------------------------------------------------

    def _effective_factor_path(self, code: str, use_gbbq: bool) -> Path:
        """加载所用的因子文件路径：use_gbbq 且 factors_gbbq/ 有该股 → 用之，否则 factors/。"""
        if use_gbbq:
            p = self.factor_path(code, gbbq=True)
            if p.exists():
                return p
        return self.factor_path(code, gbbq=False)

    def _effective_factor(self, code: str, use_gbbq: bool) -> pd.DataFrame:
        """加载所用的因子表（同上回退逻辑）。"""
        if use_gbbq:
            f = self.read_factor(code, gbbq=True)
            if f is not None and not f.empty:
                return f
        return self.read_factor(code, gbbq=False)

    # ------------------------------------------------------------
    # 原始数据读写
    # ------------------------------------------------------------

    def read_raw(self, code: str, freq: str) -> pd.DataFrame | None:
        """原始数据（数据源返回的行，不含停牌占位与加工列）。日线要带停牌/涨跌停请用 ``read_view``。"""
        path = self.freq_path(code, freq)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            tc = _time_col(freq)
            df[tc] = pd.to_datetime(df[tc])
            if "source" not in df.columns:
                df["source"] = "legacy"
            df["source"] = df["source"].fillna("legacy").astype("string")
            return df.sort_values(tc).reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"读取本地缓存 {path} 失败: {exc}（视为不存在重拉）")
            return None

    def write_raw(self, code: str, freq: str, df: pd.DataFrame) -> None:
        path = self.freq_path(code, freq)
        ensure_dir(path.parent)
        df.to_parquet(path, index=False, compression="snappy")

    # ------------------------------------------------------------
    # 日线加工层读写 + 视图
    # ------------------------------------------------------------

    def read_daily_meta(self, code: str) -> pd.DataFrame | None:
        path = self.daily_meta_path(code)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"读取日线加工层 {path} 失败: {exc}（视为缺失）")
            return None

    def write_daily_meta(self, code: str, df: pd.DataFrame) -> None:
        path = self.daily_meta_path(code)
        ensure_dir(path.parent)
        df[list(DAILY_META_COLUMNS)].to_parquet(path, index=False, compression="snappy")

    def read_view(self, code: str, freq: str) -> pd.DataFrame | None:
        """不复权视图：日线 = 原始行按加工层时间轴外连接（含停牌行与 is_suspended/涨跌停/status）；
        分钟 = 原始行。"""
        raw = self.read_raw(code, freq)
        if freq != "daily":
            return raw
        return build_daily_view(raw, self.read_daily_meta(code), code)

    # ------------------------------------------------------------
    # 因子读写
    # ------------------------------------------------------------

    def read_factor(self, code: str, gbbq: bool = False) -> pd.DataFrame:
        path = self.factor_path(code, gbbq=gbbq)
        if not path.exists():
            return pd.DataFrame(columns=list(FACTOR_COLUMNS.keys()))
        try:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df["cum_factor"] = pd.to_numeric(df["cum_factor"], errors="coerce")
            return df.dropna(subset=["date", "cum_factor"]).sort_values("date").reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"读取因子文件 {path} 失败: {exc}")
            return pd.DataFrame(columns=list(FACTOR_COLUMNS.keys()))

    def write_factor(self, code: str, df: pd.DataFrame, gbbq: bool = False) -> None:
        """覆盖写因子表（新架构下因子直接刷新，无需自愈比对）。

        ``gbbq=True`` 写入 ``factors_gbbq/``（gbbq 自算因子并存记录），否则写 ``factors/``（生效因子）。
        """
        if df is None or df.empty:
            return
        path = self.factor_path(code, gbbq=gbbq)
        ensure_dir(path.parent)
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out["cum_factor"] = pd.to_numeric(out["cum_factor"], errors="coerce")
        out = out.dropna(subset=["date", "cum_factor"]).sort_values("date").reset_index(drop=True)
        out[["date", "cum_factor"]].to_parquet(path, index=False, compression="snappy")

    # ------------------------------------------------------------
    # 复权加载（hfq 缓存 + qfq/none 派生）
    # ------------------------------------------------------------

    def _hfq_cache_valid(self, code: str, freq: str, use_gbbq: bool = False) -> bool:
        """hfq 缓存有效 ⇔ 缓存存在且其 mtime ≥ 原始与（生效）因子文件的 mtime。"""
        cache = self.adjusted_path(code, freq, gbbq=use_gbbq)
        if not cache.exists():
            return False
        cache_mt = cache.stat().st_mtime
        raw = self.freq_path(code, freq)
        if raw.exists() and raw.stat().st_mtime > cache_mt:
            return False
        if freq == "daily":
            meta = self.daily_meta_path(code)
            if meta.exists() and meta.stat().st_mtime > cache_mt:
                return False
        factor = self._effective_factor_path(code, use_gbbq)
        return not (factor.exists() and factor.stat().st_mtime > cache_mt)

    def _build_hfq_cache(self, code: str, freq: str, use_gbbq: bool = False) -> pd.DataFrame | None:
        """重算并落盘 hfq 缓存；原始缺失返回 None。"""
        raw = self.read_view(code, freq)
        if raw is None or raw.empty:
            return None
        factor = self._effective_factor(code, use_gbbq)
        hfq = apply_adjust(raw, factor, mode="hfq", time_col=_time_col(freq))
        path = self.adjusted_path(code, freq, gbbq=use_gbbq)
        ensure_dir(path.parent)
        hfq.to_parquet(path, index=False, compression="snappy")
        logger.debug(
            f"[{code}|{freq}] 重算 hfq 缓存（{len(hfq)} 行"
            f"{'，gbbq 口径' if use_gbbq else ''}）"
        )
        return hfq

    def get_hfq(self, code: str, freq: str, use_gbbq: bool = False) -> pd.DataFrame | None:
        """取后复权数据（命中有效缓存直接读，否则重算覆盖）。"""
        if self._hfq_cache_valid(code, freq, use_gbbq):
            try:
                df = pd.read_parquet(self.adjusted_path(code, freq, gbbq=use_gbbq))
                tc = _time_col(freq)
                df[tc] = pd.to_datetime(df[tc])
                return df.sort_values(tc).reset_index(drop=True)
            except Exception as exc:
                logger.warning(f"[{code}|{freq}] 读 hfq 缓存失败，重算: {exc}")
        return self._build_hfq_cache(code, freq, use_gbbq)

    def load(
        self,
        code: str,
        freq: str,
        adjust: str = "hfq",
        *,
        anchor_date: str | date | datetime | None = None,
        use_gbbq: bool = False,
    ) -> pd.DataFrame | None:
        """加载某周期数据并按 ``adjust`` 复权。

        - ``none``：原始不复权。
        - ``hfq`` ：读/重算 hfq 缓存。
        - ``qfq`` ：hfq 除以锚点日累计因子（标量），现场派生不落缓存。

        ``use_gbbq=True`` 用 gbbq 自算因子（``factors_gbbq/``）复权，缓存隔离在
        ``adjusted_gbbq/``；该股无 gbbq 因子时回退到生效因子 ``factors/``。
        """
        mode = (adjust or "hfq").strip().lower()
        tc = _time_col(freq)
        if mode == "none":
            raw = self.read_view(code, freq)
            if raw is None:
                return None
            factor = self._effective_factor(code, use_gbbq)
            out = apply_adjust(raw, factor, mode="none", time_col=tc)
            # Keep adj_factor as the actual price multiplier for "none" (=1.0), but
            # expose the cumulative factor so backtests can handle corporate actions
            # on raw prices without treating ex-right gaps as losses.
            if out is not None and not out.empty:
                out["cum_factor"] = align_cum_factor(out[tc], factor)
            return out

        hfq = self.get_hfq(code, freq, use_gbbq)
        if hfq is None:
            return None
        if mode == "hfq":
            return hfq
        if mode == "qfq":
            effective_anchor = anchor_date
            if effective_anchor is None and not hfq.empty and tc in hfq.columns:
                effective_anchor = pd.to_datetime(hfq[tc]).max()
            anchor_f = cum_factor_at(self._effective_factor(code, use_gbbq), effective_anchor)
            if not anchor_f or anchor_f == 1.0:
                return hfq
            out = hfq.copy()
            for c in ("open", "high", "low", "close", "limit_up", "limit_down", "adj_factor"):
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce") / anchor_f
            return out
        raise ValueError(f"未知复权模式 {mode!r}（支持 none/hfq/qfq）")
