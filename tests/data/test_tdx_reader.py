"""``tdx_reader`` 自洽解析器测试。

分两部分：
- **纯函数**（无需本地通达信）：板块分类、系数表口径；
- **golden**（需本地 vipdoc，自动寻径，找不到则整组 skip）：用真实 .day/.lc 文件断言
  个股 volume 为「股」、指数 volume 不被放大、分钟与日线口径一致、尾部读取与全量一致。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from src.data.sources import tdx_reader
from src.data.sources.pytdx_source import _auto_discover_tdx_path

# ============================================================
# 纯函数：板块分类
# ============================================================


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("sh600519.day", "SH_A_STOCK"),
        ("sh688981.day", "SH_A_STOCK"),   # 科创板：pytdx 原版漏判
        ("sh900901.day", "SH_B_STOCK"),
        ("sh000300.day", "SH_INDEX"),
        ("sh880001.day", "SH_INDEX"),
        ("sh510050.day", "SH_FUND"),
        ("sz000001.day", "SZ_A_STOCK"),
        ("sz300750.day", "SZ_A_STOCK"),
        ("sz200011.day", "SZ_B_STOCK"),
        ("sz399001.day", "SZ_INDEX"),
        ("sz159915.day", "SZ_FUND"),
        ("bj430047.day", "BJ_A_STOCK"),   # 北交所：pytdx 原版不支持
        ("bj830799.lc5", "BJ_A_STOCK"),
    ],
)
def test_classify_security(filename, expected):
    assert tdx_reader.classify_security(filename) == expected


def test_classify_security_unknown():
    assert tdx_reader.classify_security("xx123456.day") is None
    assert tdx_reader.classify_security("sh700000.day") is None


def test_stock_volume_coefficient_is_shares():
    """个股 volume 系数应为 1.0（「股」），而非 pytdx 原版的 0.01（「手」）。"""
    for t in ("SH_A_STOCK", "SZ_A_STOCK", "SH_B_STOCK", "SZ_B_STOCK", "BJ_A_STOCK"):
        assert tdx_reader.SECURITY_COEFFICIENT[t][1] == 1.0
    # 指数 volume 系数沿用 pytdx 的 1.0；价格系数 A 股仍为 0.01（分→元）
    assert tdx_reader.SECURITY_COEFFICIENT["SH_INDEX"][1] == 1.0
    assert tdx_reader.SECURITY_COEFFICIENT["SH_A_STOCK"][0] == 0.01


# ============================================================
# golden：需本地 vipdoc
# ============================================================

_TDX = _auto_discover_tdx_path()
needs_tdx = pytest.mark.skipif(_TDX is None, reason="未发现本地通达信 vipdoc，跳过 golden")


def _lday(market: str, code: str) -> Path:
    return Path(_TDX, "vipdoc", market, "lday", f"{market}{code}.day")


@needs_tdx
@pytest.mark.parametrize(("market", "code"), [("sh", "600519"), ("sz", "000001"), ("sh", "688981")])
def test_stock_daily_volume_is_shares(market, code):
    """个股日线 volume 为「股」：amount ≈ close × volume（不复权口径自洽）。"""
    fp = _lday(market, code)
    if not fp.exists():
        pytest.skip(f"样本文件缺失: {fp}")
    df = tdx_reader.read_day(str(fp), start=date(2024, 1, 1))
    assert not df.empty
    df = df[df["volume"] > 0]
    ratio = df["amount"] / (df["close"] * df["volume"])
    # 成交额 ≈ 均价×成交量(股)，比值应集中在 1 附近（成交均价≈收盘价，允许日内偏离）
    assert ratio.median() == pytest.approx(1.0, abs=0.15)


@needs_tdx
def test_index_daily_volume_not_scaled():
    """指数日线 volume 系数为 1.0，不应被「手→股」放大（与个股区别对待）。"""
    fp = _lday("sh", "000300")
    if not fp.exists():
        pytest.skip(f"样本文件缺失: {fp}")
    df = tdx_reader.read_day(str(fp), start=date(2024, 1, 1))
    assert not df.empty
    # 指数无「amount=close×volume」勾稽，这里只断言能正常读出且为正值
    assert (df["volume"] > 0).any()
    assert (df["close"] > 0).all()


@needs_tdx
def test_minute_volume_matches_daily():
    """同一交易日：分钟线 volume 合计应等于日线 volume（均为「股」）。"""
    day_fp = _lday("sz", "000001")
    min_fp = Path(_TDX, "vipdoc", "sz", "fzline", "sz000001.lc5")
    if not (day_fp.exists() and min_fp.exists()):
        pytest.skip("样本文件缺失")
    daily = tdx_reader.read_day(str(day_fp), start=date(2024, 1, 1))
    last_day = daily.index.max().date()
    d_vol = float(daily.loc[daily.index.max(), "volume"])
    minute = tdx_reader.read_lc(str(min_fp), start=last_day)
    m_vol = float(minute[minute.index.date == last_day]["volume"].sum())
    assert m_vol == pytest.approx(d_vol, rel=1e-6)


@needs_tdx
def test_tail_read_matches_full():
    """尾部增量读取与全量读取在重叠区间逐字段相等。"""
    fp = _lday("sh", "600519")
    if not fp.exists():
        pytest.skip(f"样本文件缺失: {fp}")
    full = tdx_reader.read_day(str(fp), start=None)
    tail = tdx_reader.read_day(str(fp), start=date(2025, 1, 1))
    assert 0 < len(tail) < len(full)
    overlap = full.loc[tail.index]
    assert (overlap.round(6) == tail.round(6)).all().all()


@needs_tdx
def test_bj_star_files_read_without_error():
    """北交所与科创板文件均可正常解析（pytdx 原版会失败/漏判）。"""
    star = _lday("sh", "688981")
    if star.exists():
        assert not tdx_reader.read_day(str(star), start=date(2024, 1, 1)).empty
    bj_dir = Path(_TDX, "vipdoc", "bj", "lday")
    bj_files = sorted(bj_dir.glob("bj*.day")) if bj_dir.exists() else []
    if bj_files:
        df = tdx_reader.read_day(str(bj_files[0]))
        assert tdx_reader.classify_security(str(bj_files[0])) == "BJ_A_STOCK"
        assert not df.empty


# ============================================================
# golden：权息 gbbq（解密）
# ============================================================


def _hq_cache(name: str) -> Path:
    return Path(_TDX, "T0002", "hq_cache", name)


@needs_tdx
def test_read_gbbq_structure():
    """gbbq 解密+解析：列正确、行数等于文件按 29 字节定长推算、类别含除权(1)与股本(5)。"""
    fp = _hq_cache("gbbq")
    if not fp.exists():
        pytest.skip(f"gbbq 文件缺失: {fp}")
    df = tdx_reader.read_gbbq(str(fp))
    assert list(df.columns) == list(tdx_reader.GBBQ_COLUMNS)
    # 文件头 4 字节为记录数，逐条 29 字节
    assert len(df) == (fp.stat().st_size - 4) // 29
    assert df["category"].isin([1, 5]).any()
    assert df["code"].map(lambda c: isinstance(c, str) and len(c) <= 6).all()


@needs_tdx
def test_gbbq_known_event_decrypts_correctly():
    """解密正确性间接验证：000001 应有除权除息事件且最近事件日合理（乱码解密无法成立）。"""
    from src.data.gbbq import GbbqStore
    store = GbbqStore(tdx_path=_TDX)
    if not store.available:
        pytest.skip("gbbq 不可用")
    ev = store.events("000001.SZ")
    assert not ev.empty
    assert ev["date"].is_monotonic_increasing
    # category 5 历史股本：总股本应为正
    sh = store.shares("000001.SZ")
    assert not sh.empty
    assert (sh["total_shares"] > 0).all()


# ============================================================
# golden：更名史 profile.dat
# ============================================================


@needs_tdx
def test_read_profile_structure():
    fp = _hq_cache("profile.dat")
    if not fp.exists():
        pytest.skip(f"profile.dat 缺失: {fp}")
    df = tdx_reader.read_profile(str(fp))
    assert list(df.columns) == list(tdx_reader.PROFILE_COLUMNS)
    assert not df.empty
    assert df["code"].map(lambda c: len(c) == 6 and c.isdigit()).all()
    assert df["name"].map(lambda n: bool(n)).all()


@needs_tdx
def test_profile_known_former_names():
    """000001 的曾用名史应含「深发展A」（profile.dat 明文解码 + 领域归一）。"""
    from src.data.profile import ProfileStore
    store = ProfileStore(tdx_path=_TDX)
    if not store.available:
        pytest.skip("profile.dat 不可用")
    names = store.former_names("000001.SZ")
    assert not names.empty
    assert names["name"].str.contains("深发展").any()
