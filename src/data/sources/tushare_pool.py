"""
tushare 统一通道：多账号 token 池 + 节流 / 限频重试 / 权限归类 / 翻页
======================================================================

本模块是全框架访问 tushare 的**唯一通道层**，只管"怎么调这个源"，不含任何
业务语义（不知道什么是停牌、什么是分红）。数据域模块（daily_basic /
tushare_meta / suspend / factors / trading_calendar ...）一律经由本模块调用。

能力分两级：

1. ``call``：多账号轮换。``.env`` / 环境变量里配置的多个 token
   （``TUSHARE_TOKEN`` / ``TUSHARE_TOKEN1`` / ...）收集成池，某账号被限流
   就自动轮换下一账号重试；其它错误照常上抛；全部限流抛最后一次异常。
2. ``call_throttled``：在 ``call`` 之上加**主动节流**（默认 0.35s/次）、
   **全池限频等待重试**（默认等 65s、最多 3 轮）、**权限错误归类**
   （积分不足 → ``TusharePermissionError``，账号级问题，调用方据此整体降级）。
   ``call_paged`` 再叠加 offset 翻页拼接。

无任何 token 时 ``available=False``，调用方据此回退（返回空/标记不可用）。
"""

from __future__ import annotations

import os
import re
import threading
import time

import pandas as pd
from loguru import logger

#: token 环境变量名模式（TUSHARE_TOKEN / TUSHARE_TOKEN1 / TUSHARE_TOKEN2 ...）
_TOKEN_KEY_RE = re.compile(r"^TUSHARE_TOKEN\d*$")

#: 限流错误的文本特征（tushare 中文报错；命中即轮换下一账号 / 等待重试）
_RATELIMIT_HINTS = ("频率超限", "每分钟", "每小时", "次/分", "次/小时", "抱歉，您")

#: 权限/积分类错误文本特征（命中 → TusharePermissionError，账号级问题）
_PERMISSION_HINTS = ("权限", "积分")

#: call_throttled 默认参数：两次请求最小间隔 / 全池限频后的等待秒数与最大轮数
DEFAULT_THROTTLE_SECONDS = 0.35
DEFAULT_RATELIMIT_SLEEP = 65
DEFAULT_RATELIMIT_ROUNDS = 3


class TusharePermissionError(RuntimeError):
    """tushare 账号无该接口权限（积分不足）——账号级问题，调用方应整体降级而非重试。"""


def get_tushare_tokens() -> list[str]:
    """收集环境中所有 tushare 账号 token，去重保序（主 ``TUSHARE_TOKEN`` 优先，再按编号）。

    依赖 ``config_loader`` 已把 ``.env`` 注入 ``os.environ``（且凭证类已取得权威，见
    token-shadow 修复）。无任何配置时返回空列表。
    """
    keys = sorted(
        (k for k in os.environ if _TOKEN_KEY_RE.match(k)),
        key=lambda k: (len(k), k),  # "TUSHARE_TOKEN"(13) 最短优先，再 TOKEN1/2/...
    )
    out: list[str] = []
    for k in keys:
        v = (os.environ.get(k) or "").strip()
        if v and v not in out:
            out.append(v)
    return out


def is_ratelimit(exc: Exception) -> bool:
    """异常是否为限频类（请求级问题，可等待/轮换重试）。"""
    s = str(exc)
    return any(h in s for h in _RATELIMIT_HINTS)


def is_permission(exc: Exception) -> bool:
    """异常是否为权限/积分类（账号级问题，重试无意义）。"""
    s = str(exc)
    return any(h in s for h in _PERMISSION_HINTS)


_is_ratelimit = is_ratelimit  # 向后兼容别名


class TusharePool:
    """多 token 轮换的 tushare ``pro`` 调用器（线程安全，限流自动切账号）。"""

    def __init__(
        self,
        tokens: list[str] | None = None,
        throttle_seconds: float = DEFAULT_THROTTLE_SECONDS,
        ratelimit_sleep: float = DEFAULT_RATELIMIT_SLEEP,
        ratelimit_rounds: int = DEFAULT_RATELIMIT_ROUNDS,
    ) -> None:
        self._tokens = list(tokens) if tokens is not None else get_tushare_tokens()
        self._pros: dict[str, object] = {}
        self._idx = 0
        self._lock = threading.Lock()
        self._throttle_seconds = max(0.0, float(throttle_seconds))
        self._ratelimit_sleep = max(0.0, float(ratelimit_sleep))
        self._ratelimit_rounds = max(1, int(ratelimit_rounds))
        self._last_request = 0.0

    @property
    def available(self) -> bool:
        """是否配置了至少一个 token。"""
        return bool(self._tokens)

    @property
    def n_tokens(self) -> int:
        return len(self._tokens)

    def _get_pro(self, token: str):
        pro = self._pros.get(token)
        if pro is None:
            import tushare as ts
            pro = ts.pro_api(token)
            self._pros[token] = pro
        return pro

    def call(self, method: str, **kwargs):
        """调用 ``pro.<method>(**kwargs)``；当前账号被限流则轮换下一账号重试。

        Returns:
            该接口返回值（通常为 DataFrame）。

        Raises:
            RuntimeError: 未配置任何 token。
            Exception: 非限流错误立即上抛；所有账号都限流则抛最后一次限流异常。
        """
        if not self._tokens:
            raise RuntimeError("未配置任何 TUSHARE_TOKEN")
        n = len(self._tokens)
        start = self._idx
        last_exc: Exception | None = None
        for i in range(n):
            idx = (start + i) % n
            try:
                result = getattr(self._get_pro(self._tokens[idx]), method)(**kwargs)
            except Exception as exc:
                if _is_ratelimit(exc) and n > 1:
                    last_exc = exc
                    logger.debug(f"tushare token#{idx} 限流({method})，轮换下一账号: {exc}")
                    continue
                raise
            with self._lock:
                self._idx = idx  # 记住可用账号，下次从它起，减少无谓轮换
            return result
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------
    # 数据域模块的标准通道：节流 + 限频等待重试 + 权限归类 + 翻页
    # ------------------------------------------------------------

    def _throttle(self) -> None:
        wait = self._throttle_seconds - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def call_throttled(self, method: str, **kwargs) -> pd.DataFrame:
        """``call`` 的批量取数包装：主动节流；全池限频则等待再重试；权限错误归类。

        Returns:
            该接口返回的 DataFrame（``None`` 归一为空表）。

        Raises:
            TusharePermissionError: 账号无该接口权限（积分不足），调用方整体降级。
            Exception: 非限频/权限错误原样上抛；重试轮数耗尽抛最后一次限频异常。
        """
        last_exc: Exception | None = None
        for round_no in range(self._ratelimit_rounds):
            self._throttle()
            try:
                df = self.call(method, **kwargs)
            except Exception as exc:
                self._last_request = time.monotonic()
                if is_permission(exc):
                    raise TusharePermissionError(str(exc)) from exc
                if not is_ratelimit(exc):
                    raise
                last_exc = exc
                if round_no < self._ratelimit_rounds - 1:
                    logger.warning(
                        f"tushare 全池限频（{method}），等待 {self._ratelimit_sleep:g}s 重试"
                    )
                    time.sleep(self._ratelimit_sleep)
                continue
            self._last_request = time.monotonic()
            return df if df is not None else pd.DataFrame()
        assert last_exc is not None
        raise last_exc

    def call_paged(self, method: str, page_size: int = 5000, **kwargs) -> pd.DataFrame:
        """``call_throttled`` + offset 翻页拼接（接口单次返回有行数上限时用）。"""
        frames: list[pd.DataFrame] = []
        offset = 0
        while True:
            df = self.call_throttled(method, limit=page_size, offset=offset, **kwargs)
            if df is None or df.empty:
                break
            frames.append(df)
            if len(df) < page_size:
                break
            offset += len(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
