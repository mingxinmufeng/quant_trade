"""
数据源基类与公共工具
====================

所有数据源**一律返回不复权 OHLCV**（复权由 ``src.data.adjust`` 在加载时按需处理）：
    日线: date, open, high, low, close, volume, amount[, raw_close, _suspended]
    分钟: datetime, open, high, low, close, volume, amount[, raw_close]
空 DataFrame 表示该区间无数据。

本模块还提供：
- ``DataFetchError`` / ``ProxyConfigError`` 异常；
- 代理错误识别与转换（``_is_proxy_error`` / ``_raise_if_proxy_error``）；
- akshare 统一调用包装 ``_ak_call``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

# ============================================================
# 异常
# ============================================================


class DataFetchError(RuntimeError):
    """所有数据源均失败时抛出。"""


class ProxyConfigError(RuntimeError):
    """检测到系统代理/VPN 导致的网络错误时抛出，提示用户关闭代理。"""


def _is_proxy_error(exc: BaseException) -> bool:
    """粗粒度判断异常是否为代理/连接代理类错误。"""
    text = f"{type(exc).__name__}: {exc}".lower()
    keywords = (
        "proxyerror", "proxy", "cannot connect to proxy",
        "由于目标计算机积极拒绝", "无法连接", "tunnel connection failed",
    )
    return any(k in text for k in keywords)


def _raise_if_proxy_error(exc: BaseException, code: str = "") -> None:
    """若为代理错误则抛 ``ProxyConfigError`` 附带明确处置提示。"""
    if _is_proxy_error(exc):
        raise ProxyConfigError(
            f"{('[' + code + '] ') if code else ''}检测到代理错误（系统代理/VPN 不可达）：{exc}\n"
            "本程序已不再自动绕过系统代理，请执行以下任一操作后重试：\n"
            "  1) 关闭 Windows 系统代理 / VPN / 加速器；\n"
            "  2) 或在运行前设置环境变量 NO_PROXY=* 直连（详见 README）。"
        ) from exc


def _ak_call(func, *args, code: str = "", **kwargs):
    """统一调用 akshare 接口：捕获代理错误并转换为带提示的 ``ProxyConfigError``。"""
    try:
        return func(*args, **kwargs)
    except ProxyConfigError:
        raise
    except Exception as exc:
        _raise_if_proxy_error(exc, code)
        raise


# ============================================================
# 数据源抽象基类
# ============================================================


class DataSourceBase(ABC):
    """数据源抽象基类。子类至少实现 ``fetch_daily``，可选实现 ``fetch_minute``。"""

    name: str = "base"
    supports_minute: bool = False

    @abstractmethod
    def fetch_daily(self, code: str, start: date, end: date) -> pd.DataFrame:
        ...

    def fetch_minute(
        self, code: str, period: str, start: date, end: date
    ) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} 不支持分钟线")
