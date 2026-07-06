"""tushare 元数据补充源单元测试 —— mock 池，零网络，临时目录 IO。

覆盖：
  - ``dividend_to_events``        实施过滤、同笔去重防翻倍、同日跨方案叠加、税前优先
  - ``namechange_to_profile``     生效日→更名日语义换算、退市股末名封口
  - ``refresh_dividend_supplement``  待补集合、checked 幂等、权限降级
  - ``GbbqStore`` / ``ProfileStore`` 的按股票合并（TDX 优先，无记录用补充表）
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.gbbq import GbbqStore
from src.data.profile import ProfileStore
from src.data.sources import tushare_meta as tm


# ---------------------------------------------------------------------------
# 换算纯函数
# ---------------------------------------------------------------------------

def test_dividend_to_events_dedup_and_stack():
    div = pd.DataFrame({
        # 同一笔（同 end_date+ex_date）两条"实施"（公告日不同）→ 去重防翻倍
        "end_date":     ["20251231", "20251231", "20241231", "20240630", "20230630"],
        "ann_date":     ["20260417", "20260612", "20250315", "20250315", None],
        "div_proc":     ["实施", "实施", "实施", "实施", "预案"],
        "stk_div":      [0.0, 0.0, 0.1, 0.0, 9.9],
        "cash_div":     [28.0, 28.0, 2.0, 1.0, 9.9],
        "cash_div_tax": [28.02423, 28.02423, 2.24, None, 9.9],
        # 后两条同 ex_date（年度+中期同日除权）→ 真叠加；预案行被过滤
        "ex_date":      ["20260626", "20260626", "20250422", "20250422", "20230801"],
    })
    ev = tm.dividend_to_events(div)
    assert len(ev) == 2
    r26 = ev[ev["date"] == "2026-06-26"].iloc[0]
    assert r26["fenhong"] == pytest.approx(280.2423)  # 去重后单笔 ×10
    r25 = ev[ev["date"] == "2025-04-22"].iloc[0]
    assert r25["fenhong"] == pytest.approx(22.4 + 10.0)  # 税前 22.4 + 税前缺失回退税后 10
    assert r25["song"] == pytest.approx(1.0)
    assert (ev["pei"] == 0).all() and (ev["peijia"] == 0).all()


def test_namechange_to_profile_semantics():
    nc = pd.DataFrame({
        # 000418 每条重复两次（tushare 实测行为，公告重发）→ 须先按生效日去重
        "ts_code":    ["000418.SZ"] * 6 + ["600519.SH"],
        "name":       ["小天鹅A", "小天鹅A", "ST天鹅A", "ST天鹅A", "小天鹅A", "小天鹅A", "贵州茅台"],
        "start_date": ["19960528", "19960528", "20030428", "20030428", "20040608", "20040608", "20010827"],
    })
    delist = {"000418.SZ": pd.Timestamp("2019-06-21")}
    prof = tm.namechange_to_profile(nc, delist)

    p418 = prof[prof["code"] == "000418.SZ"].reset_index(drop=True)
    # (小天鹅A, 更名日=ST 生效日), (ST天鹅A, 更名日=复名生效日), 末名以退市日+1 封口
    assert list(p418["name"]) == ["小天鹅A", "ST天鹅A", "小天鹅A"]
    assert list(p418["change_date"]) == [
        pd.Timestamp("2003-04-28"), pd.Timestamp("2004-06-08"), pd.Timestamp("2019-06-22"),
    ]
    # 在市股（无退市日）末名不入表：茅台单条记录 → 无"下一条生效日"→ 零行
    assert prof[prof["code"] == "600519.SH"].empty


# ---------------------------------------------------------------------------
# 刷新流程（mock pool）
# ---------------------------------------------------------------------------

class FakePool:
    available = True

    def __init__(self, dividend_map=None, namechange_df=None, raise_exc=None):
        self.dividend_map = dividend_map or {}
        self.namechange_df = namechange_df
        self.raise_exc = raise_exc
        self.calls: list[tuple] = []

    def call_throttled(self, method, **kw):
        self.calls.append((method, kw))
        if self.raise_exc is not None:
            raise self.raise_exc
        assert method == "dividend"
        return self.dividend_map.get(kw["ts_code"], pd.DataFrame())

    def call_paged(self, method, page_size=5000, **kw):
        self.calls.append((method, kw))
        if self.raise_exc is not None:
            raise self.raise_exc
        assert method == "namechange"
        return self.namechange_df if self.namechange_df is not None else pd.DataFrame()


def _mk_store(tmp_path, daily_codes=(), gbbq_rows=None):
    (tmp_path / "daily").mkdir()
    for c in daily_codes:
        pd.DataFrame({"date": ["2020-01-02"], "close": [1.0]}).to_parquet(
            tmp_path / "daily" / f"{c}.parquet")
    if gbbq_rows is not None:
        gbbq_rows.to_parquet(tmp_path / "gbbq_events.parquet", index=False)
    return tmp_path


def _gbbq_snapshot(rows):
    """rows: [(market, code6, date, category), ...] → 最小可用事件快照。"""
    return pd.DataFrame(
        [
            {
                "market": m, "code": c, "date": pd.Timestamp(d), "category": cat,
                "hongli_panqianliutong": 10.0, "peigujia_qianzongguben": 0.0,
                "songgu_qianzongguben": 0.0, "peigu_houzongguben": 0.0,
            }
            for m, c, d, cat in rows
        ]
    )


def test_refresh_dividend_supplement_targets_gap_codes(tmp_path):
    # 600000.SH 在 gbbq 有 cat1 记录 → 不在待补集合；000418.SZ 缺 → 待补
    store = _mk_store(
        tmp_path, daily_codes=["600000.SH", "000418.SZ"],
        gbbq_rows=_gbbq_snapshot([(1, "600000", "2020-06-01", 1)]),
    )
    div = pd.DataFrame({
        "end_date": ["20181231"], "ann_date": ["20190401"], "div_proc": ["实施"],
        "stk_div": [0.0], "cash_div": [12.0], "cash_div_tax": [12.0], "ex_date": ["20190423"],
    })
    pool = FakePool(dividend_map={"000418.SZ": div})
    new_codes = tm.refresh_dividend_supplement(store, pool)

    assert new_codes == ["000418.SZ"]
    asked = {kw["ts_code"] for m, kw in pool.calls if m == "dividend"}
    assert asked == {"000418.SZ"}
    supp = pd.read_parquet(store / tm.DIVIDEND_SUPPLEMENT_FILE)
    assert supp.iloc[0]["fenhong"] == pytest.approx(120.0)

    # 幂等：已检查代码（含空结果）不再重复请求
    pool2 = FakePool()
    assert tm.refresh_dividend_supplement(store, pool2) == []
    assert pool2.calls == []


def test_recompute_gbbq_factors_after_supplement(tmp_path):
    """补洞后重算：退市股 factors_gbbq 从恒 1.0 变为含除权点。"""
    store = _mk_store(tmp_path, daily_codes=[])
    (store / "daily").rmdir(); (store / "daily").mkdir()
    pd.DataFrame({
        "date": pd.to_datetime(["2019-04-22", "2019-04-23"]),
        "close": [58.0, 46.2],
    }).to_parquet(store / "daily" / "000418.SZ.parquet")
    pd.DataFrame({
        "code": ["000418.SZ"], "date": [pd.Timestamp("2019-04-23")],
        "fenhong": [120.0], "peijia": [0.0], "song": [0.0], "pei": [0.0],
    }).to_parquet(store / tm.DIVIDEND_SUPPLEMENT_FILE, index=False)

    assert tm.recompute_gbbq_factors(store, ["000418.SZ"]) == 1
    f = pd.read_parquet(store / "factors_gbbq" / "000418.SZ.parquet")
    assert len(f) == 2  # 基准行 + 除权变化点
    # 除权日因子 = 前收/除权参考价 = 58 / (58 - 12) ≈ 1.2609
    assert f.iloc[-1]["cum_factor"] == pytest.approx(58.0 / 46.0, rel=1e-6)


def test_refresh_dividend_permission_raises(tmp_path):
    store = _mk_store(tmp_path, daily_codes=["000418.SZ"])
    pool = FakePool(raise_exc=tm.TusharePermissionError("没有访问该接口的权限"))
    with pytest.raises(tm.TusharePermissionError):
        tm.refresh_dividend_supplement(store, pool)


def test_update_tushare_supplements_graceful(tmp_path):
    class NoToken:
        available = False

    assert tm.update_tushare_supplements(tmp_path, pool=NoToken()) is False

    store = _mk_store(tmp_path, daily_codes=["000418.SZ"])
    perm = FakePool(raise_exc=tm.TusharePermissionError("积分不足"))
    assert tm.update_tushare_supplements(store, pool=perm) is False  # 降级不抛


# ---------------------------------------------------------------------------
# Store 合并语义
# ---------------------------------------------------------------------------

def test_gbbq_store_merges_supplement(tmp_path):
    snap = tmp_path / "gbbq_events.parquet"
    _gbbq_snapshot([(1, "600000", "2020-06-01", 1)]).to_parquet(snap, index=False)
    supp = pd.DataFrame({
        "code": ["000418.SZ"], "date": [pd.Timestamp("2019-04-23")],
        "fenhong": [120.0], "peijia": [0.0], "song": [0.0], "pei": [0.0],
    })
    supp_file = tmp_path / "gbbq_events_tushare.parquet"
    supp.to_parquet(supp_file, index=False)

    g = GbbqStore(tdx_path="/nonexistent", snapshot_path=snap, supplement_path=supp_file)
    # TDX 有记录 → 用 TDX（补充表不覆盖）
    ev_tdx = g.events("600000.SH")
    assert len(ev_tdx) == 1 and ev_tdx.iloc[0]["fenhong"] == 10.0
    # TDX 无记录 → 补充表兜底
    ev_supp = g.events("000418.SZ")
    assert len(ev_supp) == 1 and ev_supp.iloc[0]["fenhong"] == 120.0
    # 两边都无 → 空
    assert g.events("300750.SZ").empty


def test_profile_store_merges_supplement(tmp_path):
    snap = tmp_path / "profile_names.parquet"
    pd.DataFrame({
        "code": ["000001.SZ"], "name": ["深发展A"],
        "change_date": [pd.Timestamp("2012-08-02")],
    }).to_parquet(snap, index=False)
    supp_file = tmp_path / "profile_names_tushare.parquet"
    pd.DataFrame({
        "code": ["000418.SZ", "000001.SZ"],
        "name": ["小天鹅A", "不应生效"],  # 000001 在 TDX 有记录 → 补充行被忽略
        "change_date": [pd.Timestamp("2019-06-22"), pd.Timestamp("2000-01-01")],
    }).to_parquet(supp_file, index=False)

    p = ProfileStore(tdx_path="/nonexistent", snapshot_path=snap, supplement_path=supp_file)
    assert p.name_at("000001.SZ", "2010-01-01") == "深发展A"       # TDX 优先
    assert p.name_at("000418.SZ", "2010-01-01") == "小天鹅A"       # 补充表兜底
    assert p.name_at("000418.SZ", "2020-01-01") is None            # 退市封口后回落当前名
    assert "000418.SZ" in p.names_at("2010-01-01")
