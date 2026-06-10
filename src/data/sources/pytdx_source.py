"""
Pytdx 本地源（读通达信 vipdoc：日线 + 5 分钟 + 1 分钟），零网络。

含北交所(.BJ)修复：pytdx 原版 ``TdxDailyBarReader.get_security_type`` 不识别
``bj`` 市场前缀，对 bj 文件抛 ``NotImplementedError``。本模块通过子类扩展
``get_security_type`` 修复，不改动已安装的 pytdx 包。
"""

from __future__ import annotations

import struct
import threading
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from ...utils.helpers import format_code
from .base import DataSourceBase

#: 通达信定长记录字节数（日线 .day / 分钟 .lc1/.lc5 均为 32 字节/条）
_RECORD_SIZE = 32

#: 日线 .day 记录的二进制布局（与 pytdx ``'<IIIIIfII'`` 等价）：
#: date(YYYYMMDD,u4), open/high/low/close(u4), amount(f4), volume(u4), reserved(u4)。
#: OHLC 需乘以 ``coefficient[0]``、volume 乘以 ``coefficient[1]``（见 SECURITY_COEFFICIENT），
#: 与 ``TdxDailyBarReader._df_convert`` 完全一致，保证与全量解析口径相同。
_DAILY_DT = np.dtype([
    ("date", "<u4"), ("open", "<u4"), ("high", "<u4"), ("low", "<u4"),
    ("close", "<u4"), ("amount", "<f4"), ("volume", "<u4"), ("_rsv", "<u4"),
])

#: 分钟 .lc1/.lc5 记录布局（与 pytdx ``'<HHfffffII'`` 等价）：
#: date(packed,u2), time(packed,u2), open/high/low/close/amount(f4), volume(u4), reserved(u4)。
#: 价格/量为真实值，无系数。
_MIN_DT = np.dtype([
    ("date", "<u2"), ("time", "<u2"), ("open", "<f4"), ("high", "<f4"),
    ("low", "<f4"), ("close", "<f4"), ("amount", "<f4"), ("volume", "<u4"),
    ("_rsv", "<u4"),
])

#: 通达信自动寻径候选目录
_TDX_AUTO_DISCOVERY_CANDIDATES = (
    r"C:\new_tdx", r"D:\new_tdx", r"E:\new_tdx", r"F:\new_tdx",
    r"C:\zd_tdx", r"D:\zd_tdx", r"C:\TDX", r"D:\TDX",
    r"C:\Program Files\new_tdx", r"C:\Program Files (x86)\new_tdx",
    r"D:\Program Files\new_tdx", r"D:\Program Files (x86)\new_tdx",
    r"C:\通达信", r"D:\通达信",
)


def _auto_discover_tdx_path() -> Optional[str]:
    """扫描常见通达信安装位置，返回首个含 ``vipdoc`` 的有效路径。"""
    for candidate in _TDX_AUTO_DISCOVERY_CANDIDATES:
        if Path(candidate, "vipdoc").exists():
            return candidate
    return None


def _build_daily_reader():
    """
    构造支持北交所的 pytdx ``TdxDailyBarReader`` 子类实例。

    修复点：pytdx 原版 ``get_security_type`` 只识别 ``sz``/``sh`` 两个交易所前缀，
    对 ``bj`` 文件抛 ``NotImplementedError``。子类在原逻辑前先判断 ``bj`` 前缀，
    识别为 ``BJ_A_STOCK``（价格/成交量系数与 A 股一致：[0.01, 0.01]）。
    """
    from pytdx.reader import TdxDailyBarReader  # noqa: WPS433

    class _BJAwareDailyBarReader(TdxDailyBarReader):  # type: ignore[misc]
        SECURITY_TYPE = list(TdxDailyBarReader.SECURITY_TYPE) + ["BJ_A_STOCK"]
        SECURITY_COEFFICIENT = {
            **TdxDailyBarReader.SECURITY_COEFFICIENT,
            "BJ_A_STOCK": [0.01, 0.01],
        }

        def get_security_type(self, fname):
            lower = str(fname[-12:-10]).lower()
            if lower == "bj":
                return "BJ_A_STOCK"
            if lower == "sh":
                code = str(fname[-10:-4])  # 去掉 .day 等后缀后的 6 位代码
                if code.startswith("68"):
                    return "SH_A_STOCK"
            return super().get_security_type(fname)

    return _BJAwareDailyBarReader()


class PytdxLocalSource(DataSourceBase):
    """读取本地通达信 vipdoc 不复权数据（日线 / 5 分钟 / 1 分钟），零网络。"""

    name = "pytdx"
    supports_minute = True

    def __init__(self, tdx_path: Optional[str] = None) -> None:
        resolved = (tdx_path or "").strip() or _auto_discover_tdx_path()
        if not resolved:
            raise RuntimeError(
                "未配置通达信路径且自动寻径失败；请在 config.yaml/config.private.yaml "
                "设置 data.tdx_path，或安装通达信到常见目录。"
            )
        if not Path(resolved, "vipdoc").exists():
            raise RuntimeError(f"通达信目录无效（未找到 vipdoc）: {resolved}")
        self._tdx_path = str(resolved)
        logger.debug(f"PytdxLocalSource 使用通达信路径: {self._tdx_path}")
        self._daily_reader = None  # lazy
        self._min_reader = None  # lazy
        self._reader_lock = threading.Lock()

    def _get_daily_reader(self):
        if self._daily_reader is None:
            with self._reader_lock:
                if self._daily_reader is None:
                    self._daily_reader = _build_daily_reader()
        return self._daily_reader

    def _get_min_reader(self):
        if self._min_reader is None:
            with self._reader_lock:
                if self._min_reader is None:
                    from pytdx.reader import TdxLCMinBarReader  # noqa: WPS433
                    self._min_reader = TdxLCMinBarReader()
        return self._min_reader

    def _vipdoc_path(self, market: str, subdir: str, fname: str) -> Optional[Path]:
        """拼接 vipdoc 文件路径，不存在则返回 None。"""
        p = Path(self._tdx_path) / "vipdoc" / market / subdir / fname
        return p if p.exists() else None

    def _resolve_daily_file(self, code: str) -> Optional[Path]:
        """根据代码定位日线 .day 文件（含北交所）。"""
        sym = format_code(code).split(".")[0].lower()
        if sym.startswith(("sh", "sz", "bj")):
            market = sym[:2]
            bare = sym[2:]
        elif sym.startswith("88"):
            market, bare = "sh", sym
        else:
            market = format_code(code).split(".")[1].lower()
            bare = sym
        return self._vipdoc_path(market, "lday", f"{market}{bare}.day")

    def _resolve_min_file(self, code: str, period: str) -> Optional[Path]:
        """根据代码和周期定位分钟线文件。"""
        sym = format_code(code).split(".")[0].lower()
        if sym.startswith(("sh", "sz", "bj")):
            market = sym[:2]
            bare = sym[2:]
        else:
            market = format_code(code).split(".")[1].lower()
            bare = sym
        if str(period) == "5":
            return self._vipdoc_path(market, "fzline", f"{market}{bare}.lc5")
        return self._vipdoc_path(market, "minline", f"{market}{bare}.lc1")

    def fetch_daily(self, code: str, start: date, end: date) -> pd.DataFrame:
        fpath = self._resolve_daily_file(code)
        if fpath is None:
            return pd.DataFrame()
        # 增量读取：二分定位 start 偏移，只解析尾部记录（避免全量解析整张 .day）。
        # 任何异常（如未知证券类型）回退到 pytdx 全量解析，保证不丢数据。
        try:
            raw = self._read_tail(fpath, start, kind="daily")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[{code}] pytdx 日线尾部增量读取失败，回退全量解析: {exc}")
            raw = self._get_daily_reader().get_df(str(fpath))
        return self._slice(raw, start, end, time_col="date")

    def fetch_minute(
        self, code: str, period: str, start: date, end: date
    ) -> pd.DataFrame:
        fpath = self._resolve_min_file(code, period)
        if fpath is None:
            return pd.DataFrame()
        try:
            raw = self._read_tail(fpath, start, kind="min")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[{code}] pytdx 分钟尾部增量读取失败，回退全量解析: {exc}")
            raw = self._get_min_reader().get_df(str(fpath))
        return self._slice(raw, start, end, time_col="datetime")

    # ------------------------------------------------------------
    # 增量尾部读取（二分定位 + 向量化解析）
    # ------------------------------------------------------------

    def _read_tail(self, fpath: Path, start: date, kind: str) -> pd.DataFrame:
        """只解析 ``date >= start`` 的尾部记录，返回与 ``get_df`` 同构的 DataFrame。

        通达信日/分钟文件均为定长记录且按时间升序，故可二分定位首条 ``>= start``
        的记录偏移，再一次性读取到文件尾。首拉（``start`` 远早于数据）会定位到偏移 0，
        等价于全量读取（且向量化解析比 pytdx 的逐行 Python 循环更快）。

        返回的 DataFrame 以 ``date`` 命名的 ``DatetimeIndex`` 为索引，列为
        ``open/high/low/close/amount/volume``，与 ``TdxDailyBarReader.get_df`` /
        ``TdxLCMinBarReader.get_df`` 输出口径一致，可直接喂给 ``_slice``。
        """
        n = fpath.stat().st_size // _RECORD_SIZE
        if n == 0:
            return pd.DataFrame()
        target = self._start_key(start, kind)
        with open(fpath, "rb") as f:
            first = self._lower_bound(f, n, target, kind)
            if first >= n:
                return pd.DataFrame()
            f.seek(first * _RECORD_SIZE)
            buf = f.read((n - first) * _RECORD_SIZE)
        if kind == "daily":
            return self._daily_buf_to_df(buf, str(fpath))
        return self._min_buf_to_df(buf)

    @staticmethod
    def _start_key(start: date, kind: str) -> int:
        """构造与文件首字段（日期）同口径的二分比较键。

        日线：``YYYYMMDD`` 整数。分钟：通达信打包日期 ``(年-2004)*2048+月*100+日``。
        早于数据起点的 ``start`` 会得到偏小（甚至为负）的键，使二分定位到偏移 0（全量）。
        """
        if kind == "daily":
            return start.year * 10000 + start.month * 100 + start.day
        return (start.year - 2004) * 2048 + start.month * 100 + start.day

    @staticmethod
    def _lower_bound(f, n: int, target: int, kind: str) -> int:
        """二分查找首条 ``date >= target`` 的记录序号（记录按时间升序）。

        分钟记录按 ``datetime`` 升序 ⇒ 打包日期字段单调非减，按日期键二分即可定位到
        ``start`` 当天的首条记录（当天全部分钟均被纳入，由 ``_slice`` 精确裁剪）。
        """
        fmt, size = ("<I", 4) if kind == "daily" else ("<H", 2)
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            f.seek(mid * _RECORD_SIZE)
            key = struct.unpack(fmt, f.read(size))[0]
            if key < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _daily_buf_to_df(self, buf: bytes, fpath: str) -> pd.DataFrame:
        """日线字节缓冲 → DataFrame（向量化，应用与 pytdx 一致的价量系数）。"""
        if not buf:
            return pd.DataFrame()
        arr = np.frombuffer(buf, dtype=_DAILY_DT)
        reader = self._get_daily_reader()
        sec_type = reader.get_security_type(fpath)
        coef = reader.SECURITY_COEFFICIENT[sec_type]  # 未知类型→KeyError，上层回退全量
        idx = pd.to_datetime(arr["date"].astype("int64").astype(str), format="%Y%m%d")
        df = pd.DataFrame({
            "open": arr["open"].astype("float64") * coef[0],
            "high": arr["high"].astype("float64") * coef[0],
            "low": arr["low"].astype("float64") * coef[0],
            "close": arr["close"].astype("float64") * coef[0],
            "amount": arr["amount"].astype("float64"),
            "volume": arr["volume"].astype("float64") * coef[1],
        })
        df.index = pd.DatetimeIndex(idx, name="date")
        return df

    @staticmethod
    def _min_buf_to_df(buf: bytes) -> pd.DataFrame:
        """分钟字节缓冲 → DataFrame（向量化解包打包日期/时间）。"""
        if not buf:
            return pd.DataFrame()
        arr = np.frombuffer(buf, dtype=_MIN_DT)
        d = arr["date"].astype("int64")
        t = arr["time"].astype("int64")
        idx = pd.to_datetime(pd.DataFrame({
            "year": d // 2048 + 2004,
            "month": (d % 2048) // 100,
            "day": (d % 2048) % 100,
            "hour": t // 60,
            "minute": t % 60,
        }))
        df = pd.DataFrame({
            "open": arr["open"].astype("float64"),
            "high": arr["high"].astype("float64"),
            "low": arr["low"].astype("float64"),
            "close": arr["close"].astype("float64"),
            "amount": arr["amount"].astype("float64"),
            "volume": arr["volume"].astype("float64"),
        })
        df.index = pd.DatetimeIndex(idx, name="date")
        return df

    @staticmethod
    def _slice(raw, start: date, end: date, time_col: str) -> pd.DataFrame:
        """通达信读出的 DataFrame → 统一不复权 OHLCV，并按区间过滤。"""
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        df = raw.reset_index()
        # 统一时间列名：通达信索引可能叫 datetime / date
        ts_src = "datetime" if "datetime" in df.columns else (
            "date" if "date" in df.columns else df.columns[0]
        )
        df = df.rename(columns={ts_src: time_col})
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        cols = ["open", "high", "low", "close", "volume", "amount"]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[[time_col] + cols].dropna(subset=[time_col])
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) + pd.Timedelta(days=1)  # 含 end 当日全部分钟
        df = df[(df[time_col] >= s) & (df[time_col] < e)].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame()
        df["raw_close"] = df["close"].astype("float64")
        return df
