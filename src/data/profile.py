"""
通达信本地「更名信息」(profile.dat) 读取与点位证券简称
======================================================

通达信把**全市场所有股票**的「曾用名历史」存在一个本地文件::

    {tdx}/T0002/hq_cache/profile.dat

一次读取即可拿到所有更名记录，**零网络请求**。每条记录给出某只股票的一个**原名称**
及其**更名日**（该名称被替换的日期），由此可重建任意历史时点的证券简称，进而做
**点位级 ST 判定**——这正是免费数据源原本拿不到、只能用今值近似的那块（防幸存者
偏差：严禁用今天的 ST 状态倒推历史）。

二进制结构（定长 **64 字节/条**，小端；实测全文件日期合法率 100%）::

    offset 0       1 字节   标志位 (0x00)
    offset 1..6    6 字节   代码 (ASCII 数字，如 "000001")
    offset 7       1 字节   0x00
    offset 8..16   9 字节   原名称 (GBK，0x00 补齐)
    offset 17..20  4 字节   更名日 uint32 (YYYYMMDD)
    offset 21..63  余下补 0

语义：记录 ``(原名称, 更名日)`` 表示该名称**生效至更名日当天（不含）**，更名日起启用
下一个名称。**当前名不在本文件**（由基础信息表 ``stock_basic`` 提供）。例如 000001::

    profile.dat: (深发展A, 20061009) (S深发展A, 20070620) (深发展A, 20120802)
    当前名(基础信息表): 平安银行
    → <20061009 深发展A | [20061009,20070620) S深发展A
      | [20070620,20120802) 深发展A | >=20120802 平安银行

代码归一：profile.dat 仅存 6 位代码、无交易所标识，按 ``format_code`` 依前缀推断
（沪/深/北），与全项目口径一致。

本模块只负责「曾用名史」与「点位曾用名查询」；ST 判定（``is_st_name``）与当前名兜底由
:class:`~src.data.universe.Universe` 组合完成，避免循环依赖。
"""

from __future__ import annotations

import struct
import threading
from pathlib import Path

import pandas as pd
from loguru import logger

from ..utils.helpers import format_code, parse_date
from .sources import _auto_discover_tdx_path

__all__ = ["PROFILE_COLUMNS", "ProfileStore"]

#: 解析后的更名表 schema
PROFILE_COLUMNS = ("code", "name", "change_date")

#: profile.dat 定长记录字节数
_RECORD_SIZE = 64

#: 合法更名日范围（剔除脏记录）
_MIN_DATE, _MAX_DATE = 19900101, 21001231


class ProfileStore:
    """本地 profile.dat 的解析与按股票查询点位曾用名（进程内只解析一次，线程安全）。

    与 :class:`~src.data.gbbq.GbbqStore` 同构：可选**快照**（``snapshot_path``）把解析后
    的全市场更名表落盘为单个 parquet，下次按**版本戳**（快照 mtime ≥ 源 profile.dat
    mtime 即有效）直接读取，省去重复解析。写盘由调用方在 ``update`` 时通过
    :meth:`save_snapshot` 触发。

    Args:
        tdx_path: 通达信安装目录；留空则自动寻径。
        snapshot_path: 更名表快照 parquet 路径；None 则不走快照（每进程解析源文件）。
    """

    def __init__(
        self,
        tdx_path: str | None = None,
        snapshot_path: str | Path | None = None,
    ) -> None:
        self._tdx_path = (tdx_path or "").strip() or _auto_discover_tdx_path()
        self._snapshot_path = Path(snapshot_path) if snapshot_path else None
        self._df: pd.DataFrame | None = None
        self._hist: dict[str, list[tuple[pd.Timestamp, str]]] | None = None
        self._loaded = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------
    # 路径 / 可用性
    # ------------------------------------------------------------

    def profile_file(self) -> Path | None:
        if not self._tdx_path:
            return None
        return Path(self._tdx_path) / "T0002" / "hq_cache" / "profile.dat"

    @property
    def available(self) -> bool:
        """已解析出数据，或 profile.dat 存在，或已有快照（不触发解析）。"""
        if self._loaded and self._df is not None and not self._df.empty:
            return True
        f = self.profile_file()
        if f is not None and f.exists():
            return True
        return bool(self._snapshot_path and self._snapshot_path.exists())

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
            if self._snapshot_path and self._snapshot_valid():
                snap = self._read_snapshot()
                if snap is not None and not snap.empty:
                    self._df = snap
                    logger.debug(
                        f"profile 读取更名快照: {len(snap)} 行 | {self._snapshot_path}"
                    )
                    return
            self._df = self._read_and_normalize()
            n = 0 if self._df is None else len(self._df)
            logger.debug(f"profile 解析完成: {n} 条更名记录")

    def _read_and_normalize(self) -> pd.DataFrame:
        f = self.profile_file()
        empty = pd.DataFrame(columns=list(PROFILE_COLUMNS))
        if f is None or not f.exists():
            logger.debug(f"profile.dat 文件不存在（tdx_path={self._tdx_path}）")
            return empty
        try:
            data = f.read_bytes()
        except OSError as exc:
            logger.warning(f"读取 profile.dat 失败: {exc}")
            return empty
        recs: list[tuple[str, str, int]] = []
        n_full = len(data) - (len(data) % _RECORD_SIZE)
        for i in range(0, n_full, _RECORD_SIZE):
            r = data[i : i + _RECORD_SIZE]
            code6 = r[1:7].decode("ascii", "ignore").strip("\x00").strip()
            if len(code6) != 6 or not code6.isdigit():
                continue
            name = r[8:17].split(b"\x00")[0].decode("gbk", "ignore").strip()
            if not name:
                continue
            date = struct.unpack("<I", r[17:21])[0]
            if not (_MIN_DATE <= date <= _MAX_DATE):
                continue
            try:
                std = format_code(code6)
            except Exception:
                continue
            recs.append((std, name, date))
        if not recs:
            return empty
        df = pd.DataFrame(recs, columns=list(PROFILE_COLUMNS))
        df["change_date"] = pd.to_datetime(
            df["change_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
        return df.dropna(subset=["change_date"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # 快照（版本戳读/写，与 gbbq 一致）
    # ------------------------------------------------------------

    def _snapshot_valid(self) -> bool:
        """快照有效 ⇔ 快照存在、源 profile.dat 存在，且快照 mtime ≥ 源 mtime。

        源文件缺失但快照在 → 视为有效（沿用历史快照，便于无 TDX 环境只带快照运行）。
        """
        if not self._snapshot_path or not self._snapshot_path.exists():
            return False
        src = self.profile_file()
        if src is None or not src.exists():
            return True
        return self._snapshot_path.stat().st_mtime >= src.stat().st_mtime

    def _read_snapshot(self) -> pd.DataFrame | None:
        try:
            df = pd.read_parquet(self._snapshot_path)
            df["change_date"] = pd.to_datetime(df["change_date"])
            return df
        except Exception as exc:
            logger.warning(f"读取 profile 更名快照失败（将回退解析源文件）: {exc}")
            return None

    def save_snapshot(self, force: bool = False) -> Path | None:
        """把解析后的更名表落盘为快照；已最新则跳过。返回快照路径或 None。"""
        if not self._snapshot_path:
            return None
        self._load()
        if self._df is None or self._df.empty:
            return None
        if not force and self._snapshot_valid():
            logger.debug(f"profile 更名快照已最新，跳过写盘: {self._snapshot_path}")
            return self._snapshot_path
        from ..utils.helpers import ensure_dir  # 局部导入避免循环

        ensure_dir(self._snapshot_path.parent)
        cols = list(PROFILE_COLUMNS)
        self._df[cols].to_parquet(self._snapshot_path, index=False, compression="snappy")
        logger.info(
            f"profile 更名快照已更新: {len(self._df)} 条更名记录 | {self._snapshot_path}"
        )
        return self._snapshot_path

    # ------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------

    def _history(self) -> dict[str, list[tuple[pd.Timestamp, str]]]:
        """``{std_code: [(更名日, 原名称), ...]}``（按更名日升序）。进程内构建一次。"""
        self._load()
        if self._hist is not None:
            return self._hist
        h: dict[str, list[tuple[pd.Timestamp, str]]] = {}
        if self._df is not None and not self._df.empty:
            ordered = self._df.sort_values("change_date")
            for code, grp in ordered.groupby("code"):
                h[str(code)] = list(
                    zip(
                        pd.to_datetime(grp["change_date"]),
                        grp["name"].astype(str),
                        strict=True,
                    )
                )
        self._hist = h
        return h

    def former_names(self, code: str) -> pd.DataFrame:
        """某股全部曾用名 ``DataFrame[change_date, name]``（按更名日升序）；无则空表。"""
        std = format_code(code)
        hist = self._history().get(std)
        if not hist:
            return pd.DataFrame(columns=["change_date", "name"])
        return pd.DataFrame(hist, columns=["change_date", "name"])

    def name_at(self, code: str, as_of) -> str | None:
        """``as_of`` 时点的**曾用名**；若该时点已是当前名（晚于所有更名日）返回 None。

        返回 None 表示"用当前名"——当前名不在 profile.dat，由调用方（Universe）兜底。
        """
        std = format_code(code)
        hist = self._history().get(std)
        if not hist:
            return None
        as_of_ts = pd.Timestamp(parse_date(as_of)).normalize()
        for change_date, name in hist:  # 升序：第一个"更名日 > as_of"的名称即点位名
            if as_of_ts < change_date:
                return name
        return None

    def names_at(self, as_of) -> dict[str, str]:
        """全市场在 ``as_of`` 时点仍处于**曾用名**阶段的股票 ``{std_code: 曾用名}``。

        供 Universe 批量覆盖当前名做点位 ST 过滤（已是当前名的股票不在结果里）。
        """
        as_of_ts = pd.Timestamp(parse_date(as_of)).normalize()
        out: dict[str, str] = {}
        for code, hist in self._history().items():
            for change_date, name in hist:
                if as_of_ts < change_date:
                    out[code] = name
                    break
        return out

    def __repr__(self) -> str:
        n = len(self._df) if self._df is not None else 0
        return f"<ProfileStore records={n} tdx={self._tdx_path!r}>"
