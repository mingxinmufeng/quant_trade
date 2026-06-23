"""Universe（动态股票池，防幸存者偏差）单元测试 —— 离线，零网络。"""

from __future__ import annotations

import pandas as pd
from src.data.profile import ProfileStore
from src.data.universe import Universe


def test_norm_index_code_unifies_suffix_forms():
    """P2-18：同一指数的带/不带后缀形式归一到同一纯数字目录键，避免快照目录分裂。"""
    n = Universe._norm_index_code
    assert n("000300") == n("000300.SH") == n("000300.SZ") == n("sh000300") == "000300"
    assert n("399006") == "399006"
    assert n("000905.SH") == "000905"


def _inject_profile(uni: Universe, rows) -> None:
    """给 Universe 注入一个离线 ProfileStore（rows: [(code, name, change_date)]）。"""
    df = pd.DataFrame(rows, columns=["code", "name", "change_date"])
    df["change_date"] = pd.to_datetime(df["change_date"])
    ps = ProfileStore(tdx_path="__nonexistent__")
    ps._loaded = True
    ps._df = df
    uni._profile = ps


def test_load_tushare_delisted_merges_bj_and_drops_dirty(tmp_path):
    """tushare 退市清单：并入北交所/老三板，剔除脏码与在市行。"""
    ts = pd.DataFrame(
        {
            "ts_code": ["833994.BJ", "000003.SZ", "T600018.SH", "600000.SH"],
            "name": ["北交退", "深市退", "脏码", "在市股"],
            "list_status": ["D", "D", "D", "L"],
            "list_date": ["20160101", "19910114", "19900101", "19991110"],
            "delist_date": ["20211103", "20020614", "20200101", ""],
            "industry": ["x", "y", "z", "w"],
        }
    )
    ts.to_parquet(tmp_path / "stock_basic_tushare.parquet", index=False)

    uni = Universe(store_path=tmp_path, auto_load=False)
    d = uni._load_tushare_delisted()
    codes = set(d["code"])

    assert "833994.BJ" in codes        # 北交所退市股并入（akshare 退市接口不覆盖）
    assert "000003.SZ" in codes        # 深市退市股并入
    assert "T600018.SH" not in codes   # 脏码剔除
    assert "600000.SH" not in codes    # 在市（L）不进退市清单
    row = d[d["code"] == "833994.BJ"].iloc[0]
    assert pd.Timestamp(row["delist_date"]) == pd.Timestamp("2021-11-03")


def test_load_tushare_delisted_absent_file(tmp_path):
    """无 tushare 文件时安全返回空表（不抛异常）。"""
    uni = Universe(store_path=tmp_path, auto_load=False)
    assert uni._load_tushare_delisted().empty


def test_tradable_excludes_delisted_after_delist_date(tmp_path):
    """退市股在退市日之后被排除、之前可交易（防幸存者偏差点位过滤）。"""
    uni = Universe(
        store_path=tmp_path, auto_load=False, exclude_st=False, exclude_new_ipo=False
    )
    uni._basic = uni._coerce_basic(
        pd.DataFrame(
            {
                "code": ["000001.SZ", "833994.BJ"],
                "name": ["平安银行", "北交退"],
                "list_date": ["2010-01-01", "2016-01-01"],
                "delist_date": [pd.NaT, "2021-11-03"],
                "exchange": ["SZ", "BJ"],
                "market_type": ["主板", "北交所"],
                "industry": ["", ""],
            }
        )
    )
    before = uni.get_tradable_stocks("2021-01-01")
    assert "833994.BJ" in before and "000001.SZ" in before
    after = uni.get_tradable_stocks("2022-01-01")
    assert "833994.BJ" not in after and "000001.SZ" in after


def _st_universe(tmp_path) -> Universe:
    """构造一个仅含 000004（当前名非 ST）的 Universe，注入其 ST 曾用名史。"""
    uni = Universe(store_path=tmp_path, auto_load=False, exclude_new_ipo=False)
    uni._basic = uni._coerce_basic(
        pd.DataFrame(
            {
                "code": ["000004.SZ"],
                "name": ["国华网安"],  # 当前名非 ST
                "list_date": ["1991-01-14"],
                "delist_date": [pd.NaT],
                "exchange": ["SZ"],
                "market_type": ["主板"],
                "industry": [""],
            }
        )
    )
    _inject_profile(
        uni,
        [
            ("000004.SZ", "ST原野", "2001-03-06"),
            ("000004.SZ", "深中浩A", "2008-01-01"),  # 摘帽（非 ST）
            ("000004.SZ", "*ST国农", "2010-05-31"),
        ],
    )
    return uni


def test_is_st_at_point_in_time(tmp_path):
    """点位 ST：按 as_of 的曾用名判定，当前名兜底。"""
    uni = _st_universe(tmp_path)
    assert uni.is_st_at("000004.SZ", "2000-06-01") is True   # ST原野
    assert uni.is_st_at("000004.SZ", "2005-06-01") is False  # 深中浩A（摘帽）
    assert uni.is_st_at("000004.SZ", "2009-06-01") is True   # *ST国农
    assert uni.is_st_at("000004.SZ", "2024-06-01") is False  # 当前名 国华网安
    assert uni.name_at("000004.SZ", "2000-06-01") == "ST原野"
    assert uni.name_at("000004.SZ", "2024-06-01") == "国华网安"


def test_exclude_st_uses_point_in_time(tmp_path):
    """exclude_st 按点位 ST：ST 时段剔除、摘帽/当前非 ST 时段保留。"""
    uni = _st_universe(tmp_path)  # exclude_st 默认 True
    assert "000004.SZ" not in uni.get_tradable_stocks("2000-06-01")  # ST原野，剔除
    assert "000004.SZ" in uni.get_tradable_stocks("2005-06-01")      # 摘帽，保留
    assert "000004.SZ" not in uni.get_tradable_stocks("2009-06-01")  # *ST国农，剔除
    assert "000004.SZ" in uni.get_tradable_stocks("2024-06-01")      # 当前非 ST，保留


def test_exclude_st_falls_back_to_current_name(tmp_path):
    """profile 不可用（无注入）→ 回退当前名近似：当前非 ST 全程保留。"""
    uni = Universe(
        store_path=tmp_path, auto_load=False, exclude_new_ipo=False, use_profile_st=False
    )
    uni._basic = uni._coerce_basic(
        pd.DataFrame(
            {
                "code": ["000004.SZ"],
                "name": ["国华网安"],
                "list_date": ["1991-01-14"],
                "delist_date": [pd.NaT],
                "exchange": ["SZ"],
                "market_type": ["主板"],
                "industry": [""],
            }
        )
    )
    # 回退口径：用当前名（非 ST），历史时点也判为非 ST
    assert "000004.SZ" in uni.get_tradable_stocks("2000-06-01")
    assert uni.is_st_at("000004.SZ", "2000-06-01") is False
