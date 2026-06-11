"""
通达信本地权息（gbbq）读取与公司行为事件解析
==============================================

通达信把**全市场所有股票**的「股本变迁 / 权息资料」存在一个本地文件::

    {tdx}/T0002/hq_cache/gbbq

一次读取即可拿到所有除权除息事件，**零网络请求**。本模块把它解析成按股票可查的
事件表，供两个用途：

1. ``FactorCalculator``（见 ``factors.py``）：用除权除息事件 + 本地不复权收盘价
   **自算累计后复权因子**，完全离线。
2. **增量触发器**：``last_event_date(code)`` 给出某股最近一次除权除息日；若它不晚于
   已落盘因子表的覆盖日，则因子不可能变化，``fetcher`` 可据此**跳过外部源的全量拉取**。

gbbq 二进制结构（``pytdx.reader.GbbqReader``，逐条 ``<B7sIBffff``）：

    market(0=SZ,1=SH,2=BJ) | code(6位) | datetime(YYYYMMDD 整数) | category(类别) |
    f1=hongli_panqianliutong | f2=peigujia_qianzongguben |
    f3=songgu_qianzongguben | f4=peigu_houzongguben

仅 ``category == 1``（除权除息）影响价格，此时四个浮点含义为（每 10 股口径）：

    f1 = 每 10 股现金分红(元)   f2 = 配股价(元)
    f3 = 每 10 股送转股         f4 = 每 10 股配股
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
from loguru import logger

from ..utils.helpers import format_code
from .sources import _auto_discover_tdx_path

#: gbbq 类别：除权除息（唯一影响复权的类别）
GBBQ_CATEGORY_EXDIV = 1

#: 标准代码后缀 → gbbq market 编码
_MARKET_BY_SUFFIX = {"SZ": 0, "SH": 1, "BJ": 2}

#: 解析后的事件表 schema（每 10 股口径的原始字段，未除以 10）
EVENT_COLUMNS = ("date", "fenhong", "peijia", "song", "pei")

#: 事件快照落盘列（归一化后；足够 events() 查询使用）
_SNAPSHOT_COLUMNS = [
    "market", "code", "date", "category",
    "hongli_panqianliutong", "peigujia_qianzongguben",
    "songgu_qianzongguben", "peigu_houzongguben",
]


class GbbqStore:
    """本地 gbbq 文件的解析与按股票查询（进程内只解析一次，线程安全）。

    可选**事件快照**（``snapshot_path``）：把解析后的全市场事件落盘为单个 parquet，
    下次按**版本戳**（快照 mtime ≥ 源 gbbq mtime 即有效）直接读取，省去重复解析
    （源文件约 19 万行）；版本戳与 ``storage`` 的 hfq 缓存一致。写盘由调用方在
    ``update`` 时通过 ``save_snapshot()`` 触发。
    """

    def __init__(
        self,
        tdx_path: str | None = None,
        snapshot_path: str | Path | None = None,
    ) -> None:
        self._tdx_path = (tdx_path or "").strip() or _auto_discover_tdx_path()
        self._snapshot_path = Path(snapshot_path) if snapshot_path else None
        self._df: pd.DataFrame | None = None
        self._loaded = False
        self._loaded_from_snapshot = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------
    # 路径 / 可用性
    # ------------------------------------------------------------

    def gbbq_file(self) -> Path | None:
        if not self._tdx_path:
            return None
        return Path(self._tdx_path) / "T0002" / "hq_cache" / "gbbq"

    @property
    def available(self) -> bool:
        """gbbq 文件是否存在（不触发解析）。"""
        f = self.gbbq_file()
        return f is not None and f.exists()

    # ------------------------------------------------------------
    # 解析（懒加载，进程内一次）
    # ------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            # 优先读有效快照（免重复解析源文件）
            if self._snapshot_path and self._snapshot_valid():
                snap = self._read_snapshot()
                if snap is not None and not snap.empty:
                    self._df = snap
                    self._loaded_from_snapshot = True
                    logger.debug(
                        f"gbbq 读取事件快照: {len(snap)} 行 | {self._snapshot_path}"
                    )
                    return
            self._df = self._read_and_normalize()
            n = 0 if self._df is None else len(self._df)
            logger.debug(f"gbbq 解析完成: {n} 行权息事件")

    # ------------------------------------------------------------
    # 事件快照（版本戳读/写）
    # ------------------------------------------------------------

    def _snapshot_valid(self) -> bool:
        """快照有效 ⇔ 快照存在、源 gbbq 存在，且快照 mtime ≥ 源 mtime。"""
        if not self._snapshot_path or not self._snapshot_path.exists():
            return False
        src = self.gbbq_file()
        if src is None or not src.exists():
            return False
        return self._snapshot_path.stat().st_mtime >= src.stat().st_mtime

    def _read_snapshot(self) -> pd.DataFrame | None:
        try:
            df = pd.read_parquet(self._snapshot_path)
            df["date"] = pd.to_datetime(df["date"])
            df["market"] = pd.to_numeric(df["market"], errors="coerce")
            df["category"] = pd.to_numeric(df["category"], errors="coerce")
            return df
        except Exception as exc:
            logger.warning(f"读取 gbbq 事件快照失败（将回退解析源文件）: {exc}")
            return None

    def save_snapshot(self, force: bool = False) -> Path | None:
        """把解析后的事件落盘为快照；已最新则跳过。返回快照路径或 None。

        典型在 ``update`` 流程中调用一次。``force=True`` 强制重写。
        """
        if not self._snapshot_path:
            return None
        self._load()
        if self._df is None or self._df.empty:
            return None
        if not force and self._snapshot_valid():
            logger.debug(f"gbbq 事件快照已最新，跳过写盘: {self._snapshot_path}")
            return self._snapshot_path
        # 写盘前对比旧快照，统计本次新增事件数（供 update 摘要）
        old = self._read_snapshot() if self._snapshot_path.exists() else None
        cols = [c for c in _SNAPSHOT_COLUMNS if c in self._df.columns]
        from ..utils.helpers import ensure_dir  # 局部导入避免循环
        ensure_dir(self._snapshot_path.parent)
        self._df[cols].to_parquet(self._snapshot_path, index=False, compression="snappy")
        n = len(self._df)
        if old is None or old.empty:
            logger.info(f"gbbq 事件快照首次生成: {n} 条事件 | {self._snapshot_path}")
        else:
            added, delta = self._snapshot_diff(old, self._df)
            logger.info(
                f"gbbq 事件快照已更新: {n} 条事件（较上次新增 {added} 条，净变化 {delta:+d} 行）"
                f" | {self._snapshot_path}"
            )
        return self._snapshot_path

    @staticmethod
    def _event_keys(df: pd.DataFrame) -> set:
        """事件唯一键集合 {(market, code, date, category)}，用于快照前后求差。"""
        if df is None or df.empty:
            return set()
        d = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        return set(
            zip(
                pd.to_numeric(df["market"], errors="coerce"),
                df["code"].astype(str),
                d,
                pd.to_numeric(df["category"], errors="coerce"),
                strict=True,
            )
        )

    @classmethod
    def _snapshot_diff(cls, old: pd.DataFrame, new: pd.DataFrame):
        """返回 (新增事件条数, 净行数变化)。新增=新键集相对旧键集的增量。"""
        old_keys, new_keys = cls._event_keys(old), cls._event_keys(new)
        added = len(new_keys - old_keys)
        delta = len(new) - len(old)
        return added, delta

    def _read_and_normalize(self) -> pd.DataFrame:
        f = self.gbbq_file()
        if f is None or not f.exists():
            logger.debug(f"gbbq 文件不存在（tdx_path={self._tdx_path}）")
            return pd.DataFrame()
        try:
            from pytdx.reader import GbbqReader
        except ImportError:
            logger.warning("未安装 pytdx，无法读取 gbbq（gbbq 因子源/触发器不可用）")
            return pd.DataFrame()
        try:
            df = GbbqReader().get_df(str(f))
        except Exception as exc:
            logger.warning(f"读取 gbbq 失败: {exc}")
            return pd.DataFrame()
        return self._normalize(df)

    @staticmethod
    def _clean_code(x) -> str:
        if isinstance(x, bytes):
            x = x.decode("ascii", "ignore")
        return str(x).strip().strip("\x00")[:6]

    @classmethod
    def _normalize(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["code"] = out["code"].map(cls._clean_code)
        out["market"] = pd.to_numeric(out["market"], errors="coerce")
        out["category"] = pd.to_numeric(out["category"], errors="coerce")
        out["date"] = pd.to_datetime(
            pd.to_numeric(out["datetime"], errors="coerce").astype("Int64").astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        return out.dropna(subset=["date", "market", "category"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------

    def events(self, code: str) -> pd.DataFrame:
        """返回某股全部**除权除息**事件 ``DataFrame[date, fenhong, peijia, song, pei]``。

        字段均为「每 10 股」原始口径（未除以 10）；按 ``date`` 升序、去重。无事件返回空表。
        """
        self._load()
        empty = pd.DataFrame(columns=list(EVENT_COLUMNS))
        if self._df is None or self._df.empty:
            return empty
        std = format_code(code)
        if "." not in std:
            return empty
        num, suffix = std.split(".")
        market = _MARKET_BY_SUFFIX.get(suffix)
        if market is None:
            return empty
        # 全市场（含北交所）按当前代码直查：实测表明新代码已含完整除权记录（老代码为其子集）
        sub = self._df[
            (self._df["market"] == market)
            & (self._df["code"] == num)
            & (self._df["category"] == GBBQ_CATEGORY_EXDIV)
        ]
        if sub.empty:
            return empty
        out = sub.rename(
            columns={
                "hongli_panqianliutong": "fenhong",
                "peigujia_qianzongguben": "peijia",
                "songgu_qianzongguben": "song",
                "peigu_houzongguben": "pei",
            }
        )[list(EVENT_COLUMNS)].copy()
        for c in ("fenhong", "peijia", "song", "pei"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return (
            out.drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

    def last_event_date(self, code: str) -> pd.Timestamp | None:
        """某股最近一次除权除息日；无事件返回 None。"""
        ev = self.events(code)
        if ev.empty:
            return None
        return pd.Timestamp(ev["date"].max())

    def last_event_dates_all(self) -> dict[str, pd.Timestamp | None]:
        """返回全市场所有股票最近除权除息日字典 {std_code: pd.Timestamp}。

        一次 groupby 向量化扫描，避免逐股调用 last_event_date() 重复全表过滤
        (5000 只股票 x 全表 boolean filter -> 一次 groupby)。供 DataFetcher
        在 update() 前预热缓存，使 _should_skip_factor 降至 O(1) 查询。
        """
        self._load()
        if self._df is None or self._df.empty:
            return {}
        _suffix_by_market = {v: k for k, v in _MARKET_BY_SUFFIX.items()}
        exdiv = self._df[self._df["category"] == GBBQ_CATEGORY_EXDIV]
        if exdiv.empty:
            return {}
        grp = (
            exdiv.groupby(["market", "code"])["date"]
            .max()
            .reset_index()
        )
        result = {}
        for row in grp.itertuples(index=False):
            suffix = _suffix_by_market.get(int(row.market))
            if suffix:
                std = f"{str(row.code).zfill(6)}.{suffix}"
                result[std] = pd.Timestamp(row.date)
        return result
