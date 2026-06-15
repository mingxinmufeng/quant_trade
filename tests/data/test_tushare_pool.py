"""tushare 多账号 token 池单元测试 —— 纯逻辑，零网络。"""

from __future__ import annotations

import os

import pytest
from src.data.sources import tushare_pool as tp


def _clear_tokens(monkeypatch):
    for k in list(os.environ):
        if k.startswith("TUSHARE_TOKEN"):
            monkeypatch.delenv(k, raising=False)


def test_get_tokens_order_and_dedup(monkeypatch):
    _clear_tokens(monkeypatch)
    monkeypatch.setenv("TUSHARE_TOKEN", "A")
    monkeypatch.setenv("TUSHARE_TOKEN2", "C")
    monkeypatch.setenv("TUSHARE_TOKEN1", "B")
    monkeypatch.setenv("TUSHARE_TOKEN3", "A")  # 与主 token 重复
    # 主 TUSHARE_TOKEN 优先，再按编号序；去重
    assert tp.get_tushare_tokens() == ["A", "B", "C"]


def test_get_tokens_empty(monkeypatch):
    _clear_tokens(monkeypatch)
    assert tp.get_tushare_tokens() == []


class _FakePro:
    def __init__(self, name: str, fail: bool = False, exc: Exception | None = None):
        self.name = name
        self.fail = fail
        self.exc = exc

    def daily(self, **kwargs):
        if self.fail:
            raise self.exc or Exception("抱歉，您访问接口(daily)频率超限(1次/分钟)")
        return f"ok-{self.name}"


def test_pool_rotates_on_ratelimit(monkeypatch):
    pool = tp.TusharePool(tokens=["t1", "t2"])
    pros = {"t1": _FakePro("t1", fail=True), "t2": _FakePro("t2", fail=False)}
    monkeypatch.setattr(pool, "_get_pro", lambda token: pros[token])
    assert pool.call("daily", x=1) == "ok-t2"  # t1 限流 → 轮换 t2
    assert pool._idx == 1                        # 记住可用账号 t2


def test_pool_non_ratelimit_raises(monkeypatch):
    pool = tp.TusharePool(tokens=["t1", "t2"])
    pros = {"t1": _FakePro("t1", fail=True, exc=ValueError("参数错误")),
            "t2": _FakePro("t2")}
    monkeypatch.setattr(pool, "_get_pro", lambda token: pros[token])
    with pytest.raises(ValueError):   # 非限流错误立即上抛，不轮换
        pool.call("daily")


def test_pool_all_ratelimited_raises(monkeypatch):
    pool = tp.TusharePool(tokens=["t1", "t2"])
    monkeypatch.setattr(pool, "_get_pro", lambda token: _FakePro(token, fail=True))
    with pytest.raises(Exception, match="频率超限"):
        pool.call("daily")


def test_pool_unavailable():
    pool = tp.TusharePool(tokens=[])
    assert not pool.available
    with pytest.raises(RuntimeError):
        pool.call("daily")
