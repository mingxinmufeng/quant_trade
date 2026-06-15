"""ProfileStore（通达信本地更名史 → 点位证券简称）单元测试 —— 离线，零网络。"""

from __future__ import annotations

import struct

import pandas as pd
from src.data.profile import ProfileStore

# ============================================================
# 二进制解析（合成 profile.dat 字节，覆盖 _read_and_normalize）
# ============================================================


def _record(code6: str, name: str, date_int: int) -> bytes:
    """按真实 profile.dat 布局编码一条 64 字节记录。"""
    r = bytearray(64)
    r[1:7] = code6.encode("ascii")
    nb = name.encode("gbk")
    assert len(nb) <= 9, "名称超过 9 字节字段宽度"
    r[8 : 8 + len(nb)] = nb
    struct.pack_into("<I", r, 17, date_int)
    return bytes(r)


def _write_profile(tdx_root, records: list[bytes]) -> None:
    d = tdx_root / "T0002" / "hq_cache"
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.dat").write_bytes(b"".join(records))


def test_parse_real_layout_roundtrip(tmp_path):
    """合成 000001 三段更名，解析后曾用名 / 更名日逐条对上。"""
    _write_profile(
        tmp_path,
        [
            _record("000001", "深发展A", 20061009),
            _record("000001", "S深发展A", 20070620),
            _record("000001", "深发展A", 20120802),
        ],
    )
    ps = ProfileStore(tdx_path=str(tmp_path))
    fn = ps.former_names("000001.SZ")
    assert list(fn["name"]) == ["深发展A", "S深发展A", "深发展A"]
    assert list(fn["change_date"]) == list(
        pd.to_datetime(["2006-10-09", "2007-06-20", "2012-08-02"])
    )


def test_parse_skips_dirty_records(tmp_path):
    """脏记录（非数字代码 / 非法日期 / 空名）被剔除。"""
    _write_profile(
        tmp_path,
        [
            _record("000001", "深发展A", 20061009),  # 合法
            _record("ABCDEF", "脏码", 20061009),      # 非数字代码
            _record("000002", "万科A", 17000101),     # 日期越界
            _record("000003", "", 20061009),          # 空名
        ],
    )
    ps = ProfileStore(tdx_path=str(tmp_path))
    ps._load()
    assert set(ps._df["code"]) == {"000001.SZ"}


# ============================================================
# 点位曾用名查询（注入 _df，绕过文件）
# ============================================================


def _make_store(rows) -> ProfileStore:
    """rows: list of (std_code, name, change_date)。"""
    df = pd.DataFrame(rows, columns=["code", "name", "change_date"])
    df["change_date"] = pd.to_datetime(df["change_date"])
    ps = ProfileStore(tdx_path="__nonexistent__")
    ps._loaded = True
    ps._df = df
    return ps


def test_name_at_interval_semantics():
    """更名日当天起用新名；晚于所有更名日 → None（用当前名）。"""
    ps = _make_store(
        [
            ("000001.SZ", "深发展A", "2006-10-09"),
            ("000001.SZ", "S深发展A", "2007-06-20"),
            ("000001.SZ", "深发展A", "2012-08-02"),
        ]
    )
    assert ps.name_at("000001.SZ", "2006-01-01") == "深发展A"
    assert ps.name_at("000001.SZ", "2006-10-08") == "深发展A"
    assert ps.name_at("000001.SZ", "2006-10-09") == "S深发展A"  # 更名日当天=新名
    assert ps.name_at("000001.SZ", "2012-08-01") == "深发展A"
    assert ps.name_at("000001.SZ", "2012-08-02") is None       # 之后 = 当前名
    assert ps.name_at("000001.SZ", "2025-01-01") is None
    assert ps.name_at("600000.SH", "2010-01-01") is None       # 无更名史


def test_names_at_batch():
    """names_at 只返回当日仍处于曾用名阶段的股票。"""
    ps = _make_store(
        [
            ("000004.SZ", "ST原野", "2001-03-06"),
            ("000004.SZ", "深中浩A", "2008-01-01"),
            ("000007.SZ", "ST万恒", "2007-04-12"),
        ]
    )
    at_2000 = ps.names_at("2000-06-01")
    assert at_2000["000004.SZ"] == "ST原野"
    assert at_2000["000007.SZ"] == "ST万恒"
    # 2005：000004 已进入下一段曾用名 深中浩A（非 ST，但仍是曾用名，由 ST 判定环节区分）
    at_2005 = ps.names_at("2005-06-01")
    assert at_2005["000004.SZ"] == "深中浩A"
    assert at_2005["000007.SZ"] == "ST万恒"
    # 2010：晚于 000004 所有更名日 → 不在结果（用当前名）
    at_2010 = ps.names_at("2010-06-01")
    assert "000004.SZ" not in at_2010


def test_unavailable_without_file_or_data():
    ps = ProfileStore(tdx_path="__nonexistent__")
    assert ps.available is False
