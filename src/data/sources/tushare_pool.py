"""
tushare 多账号 token 池（限流自动轮换）
========================================

免费 tushare 账号对 ``stock_basic`` / ``adj_factor`` / ``suspend_d`` 等接口限频很紧
（实测低积分账号 ``stock_basic`` 约 1 次/小时）。本模块把 ``.env`` / 环境变量里配置的
**多个**账号 token（``TUSHARE_TOKEN`` / ``TUSHARE_TOKEN1`` / ``TUSHARE_TOKEN2`` / ...）
收集成一个池，调用接口时**某账号被限流就自动轮换到下一个账号**重试，从而把多账号
的额度叠加利用，缓解限流中断。

仅在识别为"频率超限"类错误时轮换；其它错误（参数错、网络错）照常上抛。所有账号都
限流时抛出最后一次异常。无任何 token 时 ``available=False``，调用方据此回退（返回空/
标记不可用），与单 token 时代行为兼容。
"""

from __future__ import annotations

import os
import re
import threading

from loguru import logger

#: token 环境变量名模式（TUSHARE_TOKEN / TUSHARE_TOKEN1 / TUSHARE_TOKEN2 ...）
_TOKEN_KEY_RE = re.compile(r"^TUSHARE_TOKEN\d*$")

#: 限流错误的文本特征（tushare 中文报错；命中即轮换下一账号）
_RATELIMIT_HINTS = ("频率超限", "每分钟", "每小时", "次/分", "次/小时", "抱歉，您")


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


def _is_ratelimit(exc: Exception) -> bool:
    s = str(exc)
    return any(h in s for h in _RATELIMIT_HINTS)


class TusharePool:
    """多 token 轮换的 tushare ``pro`` 调用器（线程安全，限流自动切账号）。"""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = list(tokens) if tokens is not None else get_tushare_tokens()
        self._pros: dict[str, object] = {}
        self._idx = 0
        self._lock = threading.Lock()

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
