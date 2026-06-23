"""
动态股票池（universe）—— 防幸存者偏差核心
============================================

给定一个**历史时间点** ``as_of_date``，返回该时点"真实可交易"的 A 股清单，
**严禁用当前数据倒推历史**（否则会引入幸存者偏差：把今天才上市的股票算进过去、
或把已退市股票从历史里抹掉，导致回测虚高）。

数据基础：全市场股票基础信息表
--------------------------------
落盘缓存 ``{store}/stock_basic.parquet``，字段::

    code         str            标准代码 XXXXXX.SH/SZ/BJ
    name         str            证券简称（含 ST/*ST 标识，**当前名**；历史时点名见下方）
    list_date    datetime64[ns] 上市日期
    delist_date  datetime64[ns] 退市/终止上市日期（在市为 NaT）
    exchange     str            SH / SZ / BJ
    market_type  str            主板 / 创业板 / 科创板 / 北交所
    industry     str            行业（可空，按数据源能力填充）

来源（akshare，按交易所拼装）：
- 在市：``stock_info_sh_name_code`` / ``stock_info_sz_name_code`` / ``stock_info_bj_name_code``
- 退市：``stock_info_sh_delist`` / ``stock_info_sz_delist``
- 列名随 akshare 版本波动，本模块用**关键字模糊匹配**列，尽量兼容多版本。

点位 ST 判定（防幸存者偏差的名称口径）
--------------------------------------
akshare/csindex 只能拿到**当前证券简称**，无法回溯"某历史日该股是否带 ST"。本模块用
**通达信本地更名史** ``profile.dat``（见 :class:`~src.data.profile.ProfileStore`）补齐这一
点位维度：``get_tradable_stocks`` 的 ``exclude_st`` 过滤优先按 ``as_of_date`` 的**点位曾
用名**判定 ST，仅当 profile.dat 不可用（未装通达信/无文件）时才回退到**当前名近似**。
配合本就严格点位化的 ``list_date`` / ``delist_date`` 上市/退市过滤，整条股票池口径
**统一防幸存者偏差**（不再用今天的 ST 状态倒推历史）。

指数成分股（防幸存者偏差的指数口径）
------------------------------------
``get_index_components(index_code, as_of_date)`` 从
``{store}/index_weights/{index_code}/{YYYYMMDD}.parquet`` 读取**调仓日快照**，返回
距 ``as_of_date`` 最近且不晚于它的快照成分股。csindex 接口只提供**当前**权重，故
历史快照需通过 :meth:`refresh_index_components` 定期落盘逐步积累（首次只有今日快照）。

公开接口（与 Prompt 一致）::

    Universe.get_tradable_stocks(as_of_date) -> list[str]
    Universe.get_index_components(index_code, as_of_date) -> list[str]
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.helpers import ensure_dir, format_code, parse_date, retry
from .profile import ProfileStore
from .sources.base import _ak_call

__all__ = ["Universe"]

#: 基础信息缓存文件名（相对 store_path）
BASIC_INFO_FILE = "stock_basic.parquet"

#: 更名史快照文件名（相对 store_path；ProfileStore 落盘，供点位 ST 判定）
PROFILE_SNAPSHOT_FILE = "profile_names.parquet"

#: 指数成分股快照目录名（相对 store_path）
INDEX_WEIGHTS_DIR = "index_weights"

#: 基础信息表字段 schema
BASIC_COLUMNS: dict[str, str] = {
    "code": "string",
    "name": "string",
    "list_date": "datetime64[ns]",
    "delist_date": "datetime64[ns]",
    "exchange": "string",
    "market_type": "string",
    "industry": "string",
}

#: ST / 退市 关键字（当前名近似过滤）
_ST_TOKENS = ("ST", "*ST", "退")

#: 远程拉取重试
_RETRY_TIMES = 3
_RETRY_DELAYS = [1, 2, 4]


# ============================================================
# 纯函数工具
# ============================================================


def _pick_col(df: pd.DataFrame, *keyword_groups: Sequence[str]) -> str | None:
    """按关键字组在 df 列名中模糊匹配第一个命中的列名（兼容 akshare 多版本列名）。

    每个 ``keyword_groups`` 是一组"且"关系关键字：列名需**同时包含**该组全部关键字。
    多组之间是"或"：按顺序返回第一组命中的列。无命中返回 ``None``。
    """
    cols = [str(c) for c in df.columns]
    for group in keyword_groups:
        for c in cols:
            if all(k in c for k in group):
                return c
    return None


def _classify_market(code: str) -> str:
    """按标准代码前缀判定板块类型。"""
    base, _, suffix = code.partition(".")
    if suffix == "BJ":
        return "北交所"
    if base.startswith(("688", "689")):
        return "科创板"
    if base.startswith(("300", "301")):
        return "创业板"
    return "主板"


def is_st_name(name: str | None) -> bool:
    """证券简称是否含 ST / *ST / 退（点位/当前名通用：仅判断给定的这个名字）。"""
    if not name:
        return False
    s = str(name).upper().replace(" ", "")
    return ("ST" in s) or ("退" in str(name))


# ============================================================
# 主类
# ============================================================


class Universe:
    """动态股票池（防幸存者偏差）。

    Args:
        store_path: 数据仓库根目录；基础信息缓存写入 ``{store}/stock_basic.parquet``。
        min_list_days: 上市最少**自然日**数（避开新股波动期）。默认 60。
        exclude_st: 是否剔除 ST/*ST/退。默认 True。剔除口径见 ``use_profile_st``。
        exclude_new_ipo: 是否启用 ``min_list_days`` 新股窗口过滤。False 时仅要求"已上市"。
            默认 True。
        use_profile_st: ST 过滤是否按**点位曾用名**（通达信 profile.dat）判定。默认 True；
            profile.dat 不可用时自动回退到**当前名近似**。
        tdx_path: 通达信安装目录（供 ProfileStore 读 profile.dat）；留空则自动寻径。
        refresh_days: 基础信息缓存过期阈值（天）。默认 7。
        auto_load: 构造时立即加载/刷新基础信息表。设 False 可延迟（用于测试）。默认 True。
    """

    def __init__(
        self,
        store_path: str | Path = "data_store",
        min_list_days: int = 60,
        exclude_st: bool = True,
        exclude_new_ipo: bool = True,
        use_profile_st: bool = True,
        tdx_path: str | None = None,
        refresh_days: int = 7,
        auto_load: bool = True,
    ) -> None:
        self._store_path = Path(store_path)
        self._min_list_days = max(0, int(min_list_days))
        self._exclude_st = bool(exclude_st)
        self._exclude_new_ipo = bool(exclude_new_ipo)
        self._use_profile_st = bool(use_profile_st)
        self._tdx_path = (tdx_path or "").strip() or None
        self._refresh_days = int(refresh_days)
        self._cache_file = self._store_path / BASIC_INFO_FILE
        self._weights_dir = self._store_path / INDEX_WEIGHTS_DIR
        self._basic: pd.DataFrame = pd.DataFrame(columns=list(BASIC_COLUMNS.keys()))
        self._profile: ProfileStore | None = None  # 点位曾用名（lazy）
        if auto_load:
            self.load()

    # ------------------------------------------------------------
    # 从配置构造
    # ------------------------------------------------------------

    @classmethod
    def from_config(cls, config, auto_load: bool = True) -> Universe:
        """从 ``load_config()`` 的结果构造（读取 ``data.store_path`` 与 ``universe.*``）。"""
        data_cfg = config.get("data", {}) if hasattr(config, "get") else {}
        uni_cfg = config.get("universe", {}) if hasattr(config, "get") else {}
        return cls(
            store_path=data_cfg.get("store_path", "data_store"),
            min_list_days=uni_cfg.get("min_list_days", 60),
            exclude_st=uni_cfg.get("exclude_st", True),
            exclude_new_ipo=uni_cfg.get("exclude_new_ipo", True),
            use_profile_st=uni_cfg.get("use_profile_st", True),
            tdx_path=data_cfg.get("tdx_path") or None,
            auto_load=auto_load,
        )

    # ------------------------------------------------------------
    # 基础信息表：加载 / 缓存 / 刷新
    # ------------------------------------------------------------

    @property
    def basic_info(self) -> pd.DataFrame:
        """全市场基础信息表（已加载）。"""
        self._ensure_loaded()
        return self._basic

    def load(self, force_refresh: bool = False) -> None:
        """加载基础信息表。

        策略（与 ``TradingCalendar`` 一致）：
        1. ``force_refresh`` → 回源；
        2. 缓存缺失 → 回源；
        3. 缓存过期（mtime 距今 > refresh_days）→ 回源，失败但有缓存则 WARNING 沿用；
        4. 否则直接读缓存。

        Raises:
            RuntimeError: 无本地缓存且远程拉取失败。
        """
        cache_exists = self._cache_file.exists()
        cache_stale = cache_exists and self._is_cache_stale()

        if cache_exists and not force_refresh and not cache_stale:
            self._load_from_cache()
            return

        try:
            df = self._fetch_basic_info_remote()
            if df.empty:
                raise RuntimeError("远程基础信息为空")
            self._basic = df
            self._save_cache(df)
            logger.info(
                f"股票基础信息已刷新 | 总数 {len(df)} | "
                f"在市 {int(df['delist_date'].isna().sum())} | 缓存 {self._cache_file}"
            )
        except Exception as exc:
            if cache_exists:
                logger.warning(
                    f"远程基础信息拉取失败（{type(exc).__name__}: {exc}），沿用本地缓存 {self._cache_file}"
                )
                self._load_from_cache()
            else:
                raise RuntimeError(
                    f"无法加载股票基础信息：本地缓存不存在且远程拉取失败（{exc}）"
                ) from exc

    def _is_cache_stale(self) -> bool:
        try:
            mtime = self._cache_file.stat().st_mtime
        except OSError:
            return True
        return (time.time() - mtime) / 86400.0 > self._refresh_days

    def _load_from_cache(self) -> None:
        df = pd.read_parquet(self._cache_file)
        self._basic = self._coerce_basic(df)
        n_active = int(self._basic["delist_date"].isna().sum())
        logger.debug(
            f"基础信息缓存已加载 | 文件 {self._cache_file} | 总数 {len(self._basic)} | 在市 {n_active}"
        )

    def _save_cache(self, df: pd.DataFrame) -> None:
        ensure_dir(self._store_path)
        df.to_parquet(self._cache_file, index=False, compression="snappy")

    @staticmethod
    def _coerce_basic(df: pd.DataFrame) -> pd.DataFrame:
        """把任意来源的基础信息规整到 ``BASIC_COLUMNS`` schema。"""
        out = df.copy()
        for col, dtype in BASIC_COLUMNS.items():
            if col not in out.columns:
                out[col] = pd.Series([pd.NaT] * len(out)) if dtype.startswith("datetime") else pd.NA
            if dtype.startswith("datetime"):
                out[col] = pd.to_datetime(out[col], errors="coerce")
            else:
                out[col] = out[col].astype("string")
        return out[list(BASIC_COLUMNS.keys())].drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)

    # ------------------------------------------------------------
    # 远程拉取（在市 + 退市，按交易所拼装）
    # ------------------------------------------------------------

    def _fetch_basic_info_remote(self) -> pd.DataFrame:
        """拉取并合并在市 + 退市股票基础信息。"""
        frames: list[pd.DataFrame] = []
        # 在市
        for fetch in (self._fetch_listed_sh, self._fetch_listed_sz, self._fetch_listed_bj):
            try:
                part = fetch()
                if part is not None and not part.empty:
                    frames.append(part)
            except Exception as exc:
                logger.warning(f"在市清单 {fetch.__name__} 拉取失败: {type(exc).__name__}: {exc}")
        # 退市（delist_date 非空）
        delist_frames: list[pd.DataFrame] = []
        for fetch in (self._fetch_delist_sh, self._fetch_delist_sz):
            try:
                part = fetch()
                if part is not None and not part.empty:
                    delist_frames.append(part)
            except Exception as exc:
                logger.warning(f"退市清单 {fetch.__name__} 拉取失败: {type(exc).__name__}: {exc}")
        # 并入 tushare 退市清单（含北交所/老三板，补 akshare 退市接口未覆盖部分；与回填总清单口径一致）
        ts_delist = self._load_tushare_delisted()
        if not ts_delist.empty:
            delist_frames.append(ts_delist)
            logger.debug(f"并入 tushare 退市清单 {len(ts_delist)} 条")

        if not frames and not delist_frames:
            raise RuntimeError("在市与退市清单均拉取失败")

        merged = pd.concat(frames + delist_frames, ignore_index=True) if (frames or delist_frames) else pd.DataFrame()
        # 同一代码若既在在市表又在退市表，保留含 delist_date 的记录
        merged = merged.sort_values("delist_date", na_position="first")
        merged = merged.drop_duplicates(subset=["code"], keep="last")
        return self._coerce_basic(merged)

    # ---- 在市清单 ----

    @retry(max_attempts=_RETRY_TIMES, delays=_RETRY_DELAYS)
    def _fetch_listed_sh(self) -> pd.DataFrame:
        import akshare as ak

        df = _ak_call(ak.stock_info_sh_name_code)
        return self._norm_listed(df, default_suffix="SH")

    @retry(max_attempts=_RETRY_TIMES, delays=_RETRY_DELAYS)
    def _fetch_listed_sz(self) -> pd.DataFrame:
        import akshare as ak

        df = _ak_call(ak.stock_info_sz_name_code)
        return self._norm_listed(df, default_suffix="SZ")

    @retry(max_attempts=_RETRY_TIMES, delays=_RETRY_DELAYS)
    def _fetch_listed_bj(self) -> pd.DataFrame:
        import akshare as ak

        df = _ak_call(ak.stock_info_bj_name_code)
        return self._norm_listed(df, default_suffix="BJ")

    def _norm_listed(self, df: pd.DataFrame, default_suffix: str) -> pd.DataFrame:
        """把交易所在市清单规整为标准列（list_date 有值，delist_date=NaT）。"""
        if df is None or df.empty:
            return pd.DataFrame()
        code_col = _pick_col(df, ["代码"], ["证券代码"], ["A股代码"], ["公司代码"])
        name_col = _pick_col(df, ["简称"], ["名称"], ["A股简称"])
        list_col = _pick_col(df, ["上市", "日期"], ["A股上市日期"], ["上市日期"], ["上市"])
        ind_col = _pick_col(df, ["行业"], ["所属行业"])
        if code_col is None:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["code"] = df[code_col].astype(str).map(lambda s: self._safe_format_code(s, default_suffix))
        out["name"] = df[name_col].astype(str) if name_col else ""
        out["list_date"] = pd.to_datetime(df[list_col], errors="coerce") if list_col else pd.NaT
        out["delist_date"] = pd.NaT
        out["industry"] = df[ind_col].astype(str) if ind_col else ""
        out = out.dropna(subset=["code"])
        out["exchange"] = out["code"].str.split(".").str[-1]
        out["market_type"] = out["code"].map(_classify_market)
        return out

    # ---- 退市清单 ----

    @retry(max_attempts=_RETRY_TIMES, delays=_RETRY_DELAYS)
    def _fetch_delist_sh(self) -> pd.DataFrame:
        import akshare as ak

        df = _ak_call(ak.stock_info_sh_delist)
        return self._norm_delist(df, default_suffix="SH")

    @retry(max_attempts=_RETRY_TIMES, delays=_RETRY_DELAYS)
    def _fetch_delist_sz(self) -> pd.DataFrame:
        import akshare as ak

        # 深市退市接口需指定终止上市公司
        try:
            df = _ak_call(ak.stock_info_sz_delist, symbol="终止上市公司")
        except TypeError:
            df = _ak_call(ak.stock_info_sz_delist)
        return self._norm_delist(df, default_suffix="SZ")

    def _norm_delist(self, df: pd.DataFrame, default_suffix: str) -> pd.DataFrame:
        """把退市清单规整为标准列（delist_date 有值）。"""
        if df is None or df.empty:
            return pd.DataFrame()
        code_col = _pick_col(df, ["代码"], ["证券代码"], ["公司代码"])
        name_col = _pick_col(df, ["简称"], ["名称"])
        list_col = _pick_col(df, ["上市", "日期"], ["上市日期"])
        delist_col = _pick_col(
            df, ["终止上市", "日期"], ["退市", "日期"], ["暂停上市", "日期"], ["终止上市"], ["退市"]
        )
        if code_col is None:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["code"] = df[code_col].astype(str).map(lambda s: self._safe_format_code(s, default_suffix))
        out["name"] = df[name_col].astype(str) if name_col else ""
        out["list_date"] = pd.to_datetime(df[list_col], errors="coerce") if list_col else pd.NaT
        out["delist_date"] = pd.to_datetime(df[delist_col], errors="coerce") if delist_col else pd.NaT
        out["industry"] = ""
        out = out.dropna(subset=["code"])
        out["exchange"] = out["code"].str.split(".").str[-1]
        out["market_type"] = out["code"].map(_classify_market)
        return out

    def _load_tushare_delisted(self) -> pd.DataFrame:
        """并入 tushare ``stock_basic`` 的退市清单（含北交所/老三板，补 akshare 退市接口未覆盖部分）。

        读本地 ``{store}/stock_basic_tushare.parquet`` 的 ``list_status='D'`` 行，规整到
        ``BASIC_COLUMNS`` 口径；文件不存在 / 无合法记录时返回空表（仅作退市补充，不替代
        akshare 在市来源）。脏码（非 ``XXXXXX.SH/SZ/BJ``，如 ``T600018.SH``）一并剔除。
        """
        path = self._store_path / "stock_basic_tushare.parquet"
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning(f"读取 tushare 基础信息 {path} 失败: {exc}")
            return pd.DataFrame()
        if "list_status" not in df.columns or "ts_code" not in df.columns:
            return pd.DataFrame()
        d = df[df["list_status"].astype(str) == "D"].copy()
        if d.empty:
            return pd.DataFrame()
        d = d[d["ts_code"].astype(str).str.match(r"^\d{6}\.(SH|SZ|BJ)$")]
        if d.empty:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["code"] = d["ts_code"].astype(str)
        out["name"] = d["name"].astype(str) if "name" in d.columns else ""
        out["list_date"] = (
            pd.to_datetime(d["list_date"].astype(str), format="%Y%m%d", errors="coerce")
            if "list_date" in d.columns else pd.NaT
        )
        out["delist_date"] = (
            pd.to_datetime(d["delist_date"].astype(str), format="%Y%m%d", errors="coerce")
            if "delist_date" in d.columns else pd.NaT
        )
        out["industry"] = d["industry"].astype(str) if "industry" in d.columns else ""
        out["exchange"] = out["code"].str.split(".").str[-1]
        out["market_type"] = out["code"].map(_classify_market)
        return out.reset_index(drop=True)

    @staticmethod
    def _safe_format_code(raw: str, default_suffix: str) -> str | None:
        """把交易所原始代码归一为标准格式；纯 6 位无后缀时按交易所补后缀。"""
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none"):
            return None
        # 抽取 6 位数字（部分接口带空格/前缀）
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) >= 6:
            digits = digits[:6]
        else:
            return None
        try:
            std = format_code(digits)
        except Exception:
            return None
        # format_code 对北交所/科创板等已能判定；若仍异常按 default_suffix 兜底
        if not (std.endswith(".SH") or std.endswith(".SZ") or std.endswith(".BJ")):
            std = f"{digits}.{default_suffix}"
        return std

    # ------------------------------------------------------------
    # 公开接口 1：可交易股票池
    # ------------------------------------------------------------

    def get_tradable_stocks(self, as_of_date: str | date | datetime) -> list[str]:
        """返回 ``as_of_date`` 时点真实可交易的股票代码列表（严防幸存者偏差）。

        过滤顺序：
          1. 已上市且满足新股窗口：``list_date <= as_of_date - min_list_days``
             （``exclude_new_ipo=False`` 时窗口取 0，仅要求 ``list_date <= as_of_date``）；
          2. 未退市：``delist_date`` 为空 或 ``delist_date > as_of_date``；
          3. ``exclude_st=True`` 时剔除 ``as_of_date`` 时点名含 ST/*ST/退 的股票
             （点位曾用名优先，profile.dat 不可用时回退当前名，见模块说明）。
        """
        self._ensure_loaded()
        as_of = pd.Timestamp(parse_date(as_of_date)).normalize()
        df = self._basic
        if df.empty:
            logger.warning("基础信息表为空，可交易股票池返回空列表")
            return []

        list_date = pd.to_datetime(df["list_date"], errors="coerce")
        delist_date = pd.to_datetime(df["delist_date"], errors="coerce")

        window = self._min_list_days if self._exclude_new_ipo else 0
        list_cutoff = as_of - timedelta(days=window)
        # 1. 上市满窗口（list_date 缺失视为不满足，保守剔除）
        listed_ok = list_date.notna() & (list_date <= list_cutoff)
        # 2. 未退市
        not_delisted = delist_date.isna() | (delist_date > as_of)

        mask = listed_ok & not_delisted
        # 3. ST 过滤（点位曾用名优先，profile.dat 不可用则回退当前名）
        if self._exclude_st:
            mask = mask & (~self._st_mask_at(df, as_of))

        codes = df.loc[mask, "code"].dropna().astype(str).tolist()
        logger.debug(
            f"get_tradable_stocks({as_of.date()}) → {len(codes)} 只 "
            f"(min_list_days={window}, exclude_st={self._exclude_st})"
        )
        return sorted(set(codes))

    # ------------------------------------------------------------
    # 点位证券简称 / ST（曾用名 profile.dat + 当前名兜底）
    # ------------------------------------------------------------

    def _get_profile(self) -> ProfileStore | None:
        """惰性构造 ProfileStore（点位曾用名源）；``use_profile_st=False`` 时返回 None。"""
        if not self._use_profile_st:
            return None
        if self._profile is None:
            self._profile = ProfileStore(
                tdx_path=self._tdx_path,
                snapshot_path=self._store_path / PROFILE_SNAPSHOT_FILE,
            )
        return self._profile

    def _current_name(self, std_code: str) -> str:
        """基础信息表里的当前证券简称；无记录返回空串。"""
        self._ensure_loaded()
        row = self._basic.loc[self._basic["code"] == std_code, "name"]
        return str(row.iloc[0]) if not row.empty else ""

    def name_at(self, code: str, as_of_date: str | date | datetime) -> str:
        """返回某股在 ``as_of_date`` 的**点位证券简称**：曾用名(profile.dat)优先，否则当前名。"""
        std = format_code(code)
        prof = self._get_profile()
        if prof is not None and prof.available:
            former = prof.name_at(std, as_of_date)
            if former is not None:
                return former
        return self._current_name(std)

    def is_st_at(self, code: str, as_of_date: str | date | datetime) -> bool:
        """某股在 ``as_of_date`` 是否为 ST/*ST/退（点位名判定）。"""
        return is_st_name(self.name_at(code, as_of_date))

    def _st_mask_at(self, df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """``as_of`` 时点的 ST 布尔掩码（与 ``df`` 行对齐）。

        以当前名为基线，用 profile.dat 的点位曾用名**覆盖**当日仍处于曾用名阶段的股票；
        profile.dat 不可用时即纯当前名近似（向后兼容）。
        """
        prof = self._get_profile()
        codes = df["code"].astype(str)
        if prof is None or not prof.available:
            return df["name"].map(is_st_name).astype(bool)
        name_map = dict(zip(codes, df["name"].astype(str), strict=True))
        for code, former in prof.names_at(as_of).items():
            if code in name_map:
                name_map[code] = former
        return codes.map(lambda c: is_st_name(name_map.get(c))).astype(bool)

    # ------------------------------------------------------------
    # 公开接口 2：指数成分股（点位快照）
    # ------------------------------------------------------------

    def get_index_components(
        self, index_code: str, as_of_date: str | date | datetime
    ) -> list[str]:
        """返回 ``index_code`` 在 ``as_of_date`` 的成分股（取最近且不晚于该日的快照）。

        从 ``{store}/index_weights/{index_code}/{YYYYMMDD}.parquet`` 读取调仓日快照。
        若无任何 ``<= as_of_date`` 的快照，返回空列表并 WARNING（需先用
        :meth:`refresh_index_components` 积累快照）。
        """
        as_of = pd.Timestamp(parse_date(as_of_date)).normalize()
        std_index = self._norm_index_code(index_code)
        snap_dir = self._weights_dir / std_index
        if not snap_dir.exists():
            logger.warning(
                f"指数 {std_index} 无成分股快照目录 {snap_dir}；"
                f"请先调用 refresh_index_components('{index_code}') 落盘快照"
            )
            return []

        snapshots = self._list_snapshots(snap_dir)
        eligible = [(d, p) for d, p in snapshots if d <= as_of]
        if not eligible:
            logger.warning(
                f"指数 {std_index} 在 {as_of.date()} 前无可用快照"
                f"（最早快照 {snapshots[0][0].date() if snapshots else 'N/A'}）"
            )
            return []
        snap_date, snap_path = eligible[-1]
        codes = self._read_snapshot_codes(snap_path)
        logger.debug(
            f"get_index_components({std_index}, {as_of.date()}) → 取快照 {snap_date.date()}，{len(codes)} 只"
        )
        return sorted(set(codes))

    def refresh_index_components(
        self, index_code: str, as_of: str | date | datetime | None = None
    ) -> list[str]:
        """拉取 ``index_code`` **当前**成分股权重并落盘为一份快照（文件名=快照日）。

        csindex 接口只提供当前权重，无法回溯历史；本方法用于**定期**调用以逐步积累
        点位快照。``as_of`` 指定快照日期（默认今日）。返回成分股代码列表。
        """
        std_index = self._norm_index_code(index_code)
        snap_date = pd.Timestamp(parse_date(as_of)).normalize() if as_of else pd.Timestamp.today().normalize()
        df = self._fetch_index_weight_remote(std_index)
        if df is None or df.empty:
            logger.warning(f"指数 {std_index} 当前成分股拉取为空，未落盘快照")
            return []
        snap_dir = ensure_dir(self._weights_dir / std_index)
        path = snap_dir / f"{snap_date.strftime('%Y%m%d')}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        codes = self._read_snapshot_codes(path)
        logger.info(f"指数 {std_index} 成分股快照已落盘 {path}（{len(codes)} 只）")
        return sorted(set(codes))

    @retry(max_attempts=_RETRY_TIMES, delays=_RETRY_DELAYS)
    def _fetch_index_weight_remote(self, std_index: str) -> pd.DataFrame:
        """拉取中证指数当前成分股权重（``ak.index_stock_cons_weight_csindex``）。"""
        import akshare as ak

        bare = std_index.split(".")[0]
        df = _ak_call(ak.index_stock_cons_weight_csindex, symbol=bare)
        if df is None or df.empty:
            return pd.DataFrame()
        code_col = _pick_col(df, ["成分券代码"], ["成份券代码"], ["证券代码"], ["代码"])
        name_col = _pick_col(df, ["成分券名称"], ["成份券名称"], ["证券名称"], ["简称"])
        w_col = _pick_col(df, ["权重"])
        if code_col is None:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["code"] = df[code_col].astype(str).map(lambda s: self._safe_format_code(s, "SH"))
        out["name"] = df[name_col].astype(str) if name_col else ""
        out["weight"] = pd.to_numeric(df[w_col], errors="coerce") if w_col else np.nan
        return out.dropna(subset=["code"]).reset_index(drop=True)

    # ---- 快照辅助 ----

    @staticmethod
    def _norm_index_code(index_code: str) -> str:
        """指数代码归一为**纯 6 位数字**（作快照目录键）。

        消除"带/不带后缀"导致同一指数目录分裂：``000300`` / ``000300.SH`` / ``000300.SZ``
        / ``sh000300`` 全部落到同一目录 ``000300``。csindex 拉取本就只用数字部分（见
        :meth:`_fetch_index_weight_remote` 的 ``bare``），目录键用数字即可。非 6 位数字
        （罕见/异常输入）原样返回，不强行编造后缀。
        """
        s = str(index_code).strip().upper()
        digits = "".join(ch for ch in s if ch.isdigit())
        return digits if len(digits) == 6 else s

    @staticmethod
    def _list_snapshots(snap_dir: Path) -> list[tuple]:
        """返回 ``[(snapshot_date, path), ...]`` 按日期升序（文件名 YYYYMMDD）。"""
        out: list[tuple] = []
        for p in snap_dir.glob("*.parquet"):
            try:
                d = pd.Timestamp(datetime.strptime(p.stem, "%Y%m%d"))
            except ValueError:
                continue
            out.append((d, p))
        return sorted(out, key=lambda t: t[0])

    @staticmethod
    def _read_snapshot_codes(path: Path) -> list[str]:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning(f"读取指数快照 {path} 失败: {exc}")
            return []
        if "code" not in df.columns:
            return []
        return [str(c) for c in df["code"].dropna().tolist()]

    # ------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._basic is None or self._basic.empty:
            self.load()

    def __repr__(self) -> str:
        n = len(self._basic) if self._basic is not None else 0
        return (
            f"<Universe stocks={n} min_list_days={self._min_list_days} "
            f"exclude_st={self._exclude_st} exclude_new_ipo={self._exclude_new_ipo}>"
        )


# ============================================================
# 模块自测  python -m src.data.universe
# ============================================================

if __name__ == "__main__":
    from ..utils.helpers import init_logging

    init_logging(level="INFO")
    uni = Universe()
    today = date.today()
    codes = uni.get_tradable_stocks(today)
    logger.info(f"今日可交易股票数: {len(codes)}（示例 {codes[:5]}）")
    # 十年前同口径，验证点位过滤（数量通常明显更少）
    past = date(today.year - 10, today.month, min(today.day, 28))
    logger.info(f"{past} 可交易股票数: {len(uni.get_tradable_stocks(past))}")
