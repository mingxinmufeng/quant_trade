"""daily_basic 回填/增量/读取单元测试 —— mock 池，零网络，临时目录 IO。

覆盖 ``src.data.sources.daily_basic``：
  - ``run_backfill``        首拉落盘、断点续传跳过、rebuild 重拉、空/残缺数据不标记完成、
                            年度合并层、连续失败中止
  - ``update_daily_basic``  无 token / 权限不足 / 异常时优雅跳过（返回 False 且不上抛）
  - ``load_daily_basic``    日期区间与字段过滤

节流/限频等待重试/权限归类属通道层，见 ``test_tushare_pool.py``。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.sources import daily_basic as db
from src.data.sources.tushare_pool import TusharePermissionError


class FakePool:
    """离线假 TusharePool：可指定交易日、异常日（返回行数）与抛错行为。"""

    available = True

    def __init__(self, dates, bad_dates=None, raise_map=None):
        self.dates = list(dates)
        self.bad_dates = dict(bad_dates or {})   # date -> 返回行数（模拟空/残缺）
        self.raise_map = dict(raise_map or {})   # date -> [异常, ...] 依次抛出后恢复正常
        self.daily_basic_calls: list[str] = []

    def _day_frame(self, d: str, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "ts_code": [f"{i:06d}.SZ" for i in range(n)],
            "trade_date": [d] * n,
            "turnover_rate_f": [1.0] * n,
            "float_share": [1000.0] * n,
        })

    def call_throttled(self, method: str, **kw):
        if method == "trade_cal":
            sel = [d for d in self.dates if kw["start_date"] <= d <= kw["end_date"]]
            return pd.DataFrame({"cal_date": sel})
        assert method == "daily_basic"
        d = kw["trade_date"]
        self.daily_basic_calls.append(d)
        pending = self.raise_map.get(d)
        if pending:
            raise pending.pop(0)
        n = self.bad_dates.get(d, db.MIN_ROWS_OK + 5)
        return self._day_frame(d, n)


DATES = ["20231228", "20231229", "20240102"]


def test_backfill_writes_by_date_and_year_compact(tmp_path):
    pool = FakePool(DATES)
    stats = db.run_backfill(tmp_path, pool, start="20231201", end="20240131")

    assert stats == {"fetched": 3, "skipped": 0, "failed": 0}
    by_date = tmp_path / "daily_basic" / "by_date"
    assert {f.stem for f in by_date.glob("*.parquet")} == set(DATES)
    y23 = pd.read_parquet(tmp_path / "daily_basic" / "2023.parquet")
    y24 = pd.read_parquet(tmp_path / "daily_basic" / "2024.parquet")
    assert sorted(y23["trade_date"].unique()) == ["20231228", "20231229"]
    assert list(y24["trade_date"].unique()) == ["20240102"]
    assert y23["trade_date"].is_monotonic_increasing


def test_backfill_resume_skips_existing(tmp_path):
    db.run_backfill(tmp_path, FakePool(DATES), start="20231201", end="20240131")
    pool2 = FakePool(DATES)
    stats = db.run_backfill(tmp_path, pool2, start="20231201", end="20240131")

    assert pool2.daily_basic_calls == []  # 全部命中断点续传，零请求
    assert stats == {"fetched": 0, "skipped": 3, "failed": 0}


def test_backfill_rebuild_refetches_all(tmp_path):
    db.run_backfill(tmp_path, FakePool(DATES), start="20231201", end="20240131")
    pool2 = FakePool(DATES)
    stats = db.run_backfill(tmp_path, pool2, start="20231201", end="20240131", rebuild=True)

    assert pool2.daily_basic_calls == DATES
    assert stats["fetched"] == 3


def test_bad_day_not_marked_done(tmp_path):
    """空/残缺交易日不落盘（不标记完成），下轮只补那一天。"""
    pool = FakePool(DATES, bad_dates={"20231229": 0})
    stats = db.run_backfill(tmp_path, pool, start="20231201", end="20240131")
    assert stats == {"fetched": 2, "skipped": 0, "failed": 1}
    assert not (tmp_path / "daily_basic" / "by_date" / "20231229.parquet").exists()

    pool2 = FakePool(DATES)
    stats2 = db.run_backfill(tmp_path, pool2, start="20231201", end="20240131")
    assert pool2.daily_basic_calls == ["20231229"]
    assert stats2 == {"fetched": 1, "skipped": 2, "failed": 0}


def test_consecutive_failures_abort(tmp_path):
    """非限频异常连续超阈值 → 中止本轮，不再请求后续交易日。"""
    many = [f"202401{d:02d}" for d in range(2, 20)]
    pool = FakePool(many, raise_map={d: [RuntimeError("boom")] for d in many})
    stats = db.run_backfill(tmp_path, pool, start="20240101", end="20240131")
    assert stats["failed"] == db.MAX_CONSECUTIVE_FAILURES
    assert len(pool.daily_basic_calls) == db.MAX_CONSECUTIVE_FAILURES


def test_permission_error_propagates(tmp_path):
    """通道层归类出的权限错误应穿透 run_backfill（供 update_daily_basic 整体降级）。"""
    pool = FakePool(DATES, raise_map={"20231228": [TusharePermissionError("没有访问该接口的权限")]})
    with pytest.raises(TusharePermissionError):
        db.run_backfill(tmp_path, pool, start="20231201", end="20240131")


def test_update_daily_basic_graceful(tmp_path):
    """增量入口：无 token / 权限不足 / 异常一律 False 且不上抛。"""

    class NoTokenPool:
        available = False

    assert db.update_daily_basic(tmp_path, pool=NoTokenPool()) is False
    assert not (tmp_path / "daily_basic").exists()

    perm = FakePool(DATES, raise_map={"20231228": [TusharePermissionError("积分不足")]})
    assert db.update_daily_basic(tmp_path, start="20231201", end="20240131", pool=perm) is False

    ok = FakePool(DATES)
    assert db.update_daily_basic(tmp_path, start="20231201", end="20240131", pool=ok) is True


def test_load_daily_basic_filters(tmp_path):
    db.run_backfill(tmp_path, FakePool(DATES), start="20231201", end="20240131")

    df = db.load_daily_basic(tmp_path, start="20231229", end="20240131")
    assert sorted(df["trade_date"].unique()) == ["20231229", "20240102"]

    df2 = db.load_daily_basic(tmp_path, fields=["float_share"])
    assert list(df2.columns) == ["ts_code", "trade_date", "float_share"]

    assert db.load_daily_basic(tmp_path, start="20990101").empty
