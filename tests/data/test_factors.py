"""FactorProvider 缓存行为单元测试（零网络，monkeypatch 源函数）。"""

from __future__ import annotations

import pandas as pd


def test_factor_caches_success(monkeypatch):
    """源成功返回因子 → 缓存，第二次不再调用源。"""
    from src.data.factors import FactorProvider

    fp = FactorProvider(source="sina")
    calls = {"n": 0}

    def fake_ok(code):
        calls["n"] += 1
        return pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "cum_factor": [1.0]})

    monkeypatch.setattr(fp, "_from_sina", fake_ok)
    assert len(fp.get_factor("000001.SZ")) == 1
    assert len(fp.get_factor("000001.SZ")) == 1
    assert calls["n"] == 1


def test_factor_negative_cache_on_empty(monkeypatch):
    """P2-18：源成功但无数据 → 负缓存空表，第二次命中缓存不再重发请求。"""
    from src.data.factors import FactorProvider

    fp = FactorProvider(source="sina")
    calls = {"n": 0}

    def fake_empty(code):
        calls["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(fp, "_from_sina", fake_empty)
    assert fp.get_factor("830799.BJ").empty
    assert fp.get_factor("830799.BJ").empty
    assert calls["n"] == 1, "成功返回空应负缓存，不应重复请求"


def test_factor_no_cache_on_exception(monkeypatch):
    """源异常（网络/限流）→ 不缓存，后续每次重试。"""
    from src.data.factors import FactorProvider

    fp = FactorProvider(source="sina")
    calls = {"n": 0}

    def fake_raise(code):
        calls["n"] += 1
        raise RuntimeError("net fail")

    monkeypatch.setattr(fp, "_from_sina", fake_raise)
    assert fp.get_factor("000001.SZ").empty
    assert fp.get_factor("000001.SZ").empty
    assert calls["n"] == 2, "异常不应缓存，应每次重试"
