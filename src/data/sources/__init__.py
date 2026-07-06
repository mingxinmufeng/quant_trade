"""
数据源子包：源通道 + 外部资源拉取
==================================

**架构分层约定（新增数据一律照此）**：

1. 源通道（每个供应商一个文件，只管"怎么调这个源"，零业务语义）：
   ``tushare_pool``（token 池/节流/限频重试/权限归类/翻页）、``base._ak_call``
   （akshare 代理错误处理）。
2. 数据域（每类数据一个模块，拥有 schema/落盘/源优先级与 fallback/合并语义）：
   OHLCV 行情 = 本包各 ``*_source.py`` 适配器 + ``build_source`` 工厂 + 优先级链；
   其余数据域在 ``src/data/`` 下各自成模块（calendar/suspend/factors/gbbq/profile），
   或本包内（``daily_basic``/``tushare_meta``）；源细节写成域内小函数
   （如 ``FactorProvider._from_sina/_from_tushare``），通道一律经第 1 层。
3. 消费方（fetcher/universe/engine）只说数据域的语言，不直接碰任何源。

公开：
    DataSourceBase / DataFetchError / ProxyConfigError
    PytdxLocalSource / AkshareSource / BaostockSource / TushareSource
    build_source(name, ...) 工厂
    DEFAULT_SOURCES 默认优先级
    load_daily_basic / update_daily_basic 每日指标
"""

from __future__ import annotations

from typing import Optional

from .akshare_source import AkshareSource
from .baostock_source import BaostockSource
from .daily_basic import load_daily_basic, update_daily_basic
from .base import (
    DataFetchError,
    DataSourceBase,
    ProxyConfigError,
    _ak_call,
    _is_proxy_error,
    _raise_if_proxy_error,
)
from .pytdx_source import (
    PytdxLocalSource,
    _auto_discover_tdx_path,
)
from .tushare_source import TushareSource

#: 默认数据源优先级（TDX 本地盘 → tushare 官方 → akshare → baostock；
#: tushare 无 token 时构建失败，fetcher 的源链会自动跳到下一源）
DEFAULT_SOURCES = ("pytdx", "tushare", "akshare", "baostock")


def build_source(
    name: str, *, tdx_path: str | None = None, jitter: float = 0.0
) -> DataSourceBase:
    name = name.lower().strip()
    if name == "akshare":
        return AkshareSource(jitter=jitter)
    if name == "baostock":
        return BaostockSource()
    if name == "tushare":
        return TushareSource()
    if name == "pytdx":
        return PytdxLocalSource(tdx_path=tdx_path)
    raise ValueError(f"未知数据源: {name}（支持 pytdx/akshare/baostock/tushare）")


__all__ = [
    "DEFAULT_SOURCES",
    "AkshareSource",
    "BaostockSource",
    "DataFetchError",
    "DataSourceBase",
    "ProxyConfigError",
    "PytdxLocalSource",
    "TushareSource",
    "_ak_call",
    "_auto_discover_tdx_path",
    "_is_proxy_error",
    "_raise_if_proxy_error",
    "build_source",
    "load_daily_basic",
    "update_daily_basic",
]
