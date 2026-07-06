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


# ------------------------------------------------------------
# call_throttled / call_paged（数据域模块的标准通道）
# ------------------------------------------------------------

def _pool_no_wait():
    """零节流、零限频等待的池（sleep 行为单独用 monkeypatch 断言）。"""
    return tp.TusharePool(["t1"], throttle_seconds=0, ratelimit_sleep=0)


def test_call_throttled_permission_classified(monkeypatch):
    pool = _pool_no_wait()
    monkeypatch.setattr(pool, "call", lambda m, **kw: (_ for _ in ()).throw(
        RuntimeError("抱歉，您没有访问该接口的权限")))
    with pytest.raises(tp.TusharePermissionError):
        pool.call_throttled("daily_basic", trade_date="20240102")


def test_call_throttled_ratelimit_retries_then_succeeds(monkeypatch):
    import pandas as pd

    pool = tp.TusharePool(["t1"], throttle_seconds=0, ratelimit_sleep=65)
    sleeps: list[float] = []
    monkeypatch.setattr(tp.time, "sleep", lambda s: sleeps.append(s))
    attempts = []

    def fake_call(method, **kw):
        attempts.append(method)
        if len(attempts) == 1:
            raise RuntimeError("抱歉，您每分钟最多访问该接口200次")
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(pool, "call", fake_call)
    df = pool.call_throttled("trade_cal")
    assert len(df) == 1 and len(attempts) == 2
    assert 65 in sleeps  # 全池限频后按 ratelimit_sleep 等待


def test_call_throttled_ratelimit_exhausts(monkeypatch):
    pool = tp.TusharePool(["t1"], throttle_seconds=0, ratelimit_sleep=0, ratelimit_rounds=2)
    monkeypatch.setattr(tp.time, "sleep", lambda s: None)
    monkeypatch.setattr(pool, "call", lambda m, **kw: (_ for _ in ()).throw(
        RuntimeError("频率超限")))
    with pytest.raises(RuntimeError, match="频率超限"):
        pool.call_throttled("trade_cal")


def test_call_throttled_other_errors_raise_and_none_normalized(monkeypatch):
    pool = _pool_no_wait()
    monkeypatch.setattr(pool, "call", lambda m, **kw: (_ for _ in ()).throw(
        ValueError("参数错误")))
    with pytest.raises(ValueError):
        pool.call_throttled("trade_cal")

    monkeypatch.setattr(pool, "call", lambda m, **kw: None)
    assert pool.call_throttled("trade_cal").empty  # None 归一为空表


def test_call_paged_concatenates(monkeypatch):
    import pandas as pd

    pool = _pool_no_wait()
    pages = [pd.DataFrame({"i": range(3)}), pd.DataFrame({"i": range(2)})]

    def fake_call(method, **kw):
        assert kw["limit"] == 3
        return pages[kw["offset"] // 3]

    monkeypatch.setattr(pool, "call", fake_call)
    df = pool.call_paged("namechange", page_size=3)
    assert len(df) == 5  # 3 + 2（第二页不足页大小即停）
