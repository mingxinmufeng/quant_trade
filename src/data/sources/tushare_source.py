"""Tushare pro 备用日线源，返回不复权 OHLCV（需 TUSHARE_TOKEN）。"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

import pandas as pd

from ...utils.helpers import format_code
from .base import DataSourceBase


class TushareSource(DataSourceBase):
    name = "tushare"

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token or os.environ.get("TUSHARE_TOKEN", "").strip()
        if not self._token:
            raise RuntimeError("tushare 数据源需要 TUSHARE_TOKEN 环境变量或显式 token")
        import tushare as ts  # noqa: WPS433
        ts.set_token(self._token)
        self._pro = ts.pro_api()

    def fetch_daily(self, code: str, start: date, end: date) -> pd.DataFrame:
        ts_code = format_code(code)
        df = self._pro.daily(
            ts_code=ts_code,
            start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        out = pd.DataFrame({
            "date": df["date"],
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
            "volume": df["vol"].astype("float64") * 100.0,    # 手→股
            "amount": df["amount"].astype("float64") * 1000.0,  # 千元→元
        })
        out["raw_close"] = out["close"].astype("float64")
        return out.sort_values("date").reset_index(drop=True)
