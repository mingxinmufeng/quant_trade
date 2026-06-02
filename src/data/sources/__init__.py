"""
数据源子包（只负责拉取**不复权 OHLCV**）。

公开：
    DataSourceBase / DataFetchError / ProxyConfigError
    MootdxLocalSource / AkshareSource / BaostockSource / TushareSource
    build_source(name, ...) 工厂
    DEFAULT_SOURCES 默认优先级
"""

from __future__ import annotations

from typing import Optional

from .akshare_source import AkshareSource
from .base import (
    DataFetchError,
    DataSourceBase,
    ProxyConfigError,
    _ak_call,
    _is_proxy_error,
    _raise_if_proxy_error,
)
from .baostock_source import BaostockSource
from .mootdx_source import (
    MootdxLocalSource,
    _auto_discover_tdx_path,
    _build_bj_reader,
)
from .tushare_source import TushareSource

#: 默认数据源优先级
DEFAULT_SOURCES = ("mootdx", "akshare", "baostock", "tushare")


def build_source(
    name: str, *, tdx_path: Optional[str] = None, jitter: float = 0.0
) -> DataSourceBase:
    name = name.lower().strip()
    if name == "akshare":
        return AkshareSource(jitter=jitter)
    if name == "baostock":
        return BaostockSource()
    if name == "tushare":
        return TushareSource()
    if name == "mootdx":
        return MootdxLocalSource(tdx_path=tdx_path)
    raise ValueError(f"未知数据源: {name}（支持 mootdx/akshare/baostock/tushare）")


__all__ = [
    "DEFAULT_SOURCES",
    "AkshareSource",
    "BaostockSource",
    "DataFetchError",
    "DataSourceBase",
    "MootdxLocalSource",
    "ProxyConfigError",
    "TushareSource",
    "build_source",
    "_ak_call",
    "_auto_discover_tdx_path",
    "_build_bj_reader",
    "_is_proxy_error",
    "_raise_if_proxy_error",
]
