"""Baostock 备用日线源，返回不复权 OHLCV（adjustflag=3；不覆盖北交所）。"""

from __future__ import annotations

import threading
from datetime import date
from typing import List

import pandas as pd

from ...utils.helpers import format_code, to_baostock_code
from .base import DataSourceBase


class BaostockSource(DataSourceBase):
    """baostock 备用日线源，返回不复权 OHLCV（不支持北交所）。

    **线程安全**：baostock 基于进程级全局 socket，并发查询会串扰；故所有
    登录/查询都串行化到一把类级锁 ``_LOCK`` 上（多线程更新时 baostock 退化为
    串行，但它仅作为 mootdx/akshare 失败后的兜底源，命中率低，影响可忽略）。
    """

    name = "baostock"

    #: 类级锁：baostock 全局状态非线程安全，所有实例共享串行化
    _LOCK = threading.Lock()

    def __init__(self) -> None:
        self._logged_in = False

    def _ensure_login(self) -> None:
        if self._logged_in:
            return
        import baostock as bs  # noqa: WPS433
        rs = bs.login()
        if getattr(rs, "error_code", "0") != "0":
            raise RuntimeError(f"baostock 登录失败: {rs.error_msg}")
        self._logged_in = True

    def fetch_daily(self, code: str, start: date, end: date) -> pd.DataFrame:
        import baostock as bs  # noqa: WPS433
        if format_code(code).endswith(".BJ"):
            return pd.DataFrame()  # baostock 不覆盖北交所
        with self._LOCK:
            self._ensure_login()
            fields = "date,open,high,low,close,volume,amount,tradestatus"
            rs = bs.query_history_k_data_plus(
                to_baostock_code(code), fields,
                start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"),
                frequency="d", adjustflag="3",  # 3=不复权
            )
            if getattr(rs, "error_code", "0") != "0":
                raise RuntimeError(f"baostock 查询失败: {rs.error_msg}")
            rows: List[List[str]] = []
            while rs.next():
                rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=fields.split(","))
        for c in ("open", "high", "low", "close", "volume", "amount"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        out = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
        out["raw_close"] = out["close"].astype("float64")
        out["_suspended"] = df["tradestatus"].astype(str) == "0"
        return out.sort_values("date").reset_index(drop=True)
