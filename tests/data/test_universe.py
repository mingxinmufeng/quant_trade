"""Universe（动态股票池，防幸存者偏差）单元测试 —— 离线，零网络。"""

from __future__ import annotations

import pandas as pd
from src.data.universe import Universe


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
