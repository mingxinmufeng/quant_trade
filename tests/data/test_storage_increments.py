"""DataStore reads frozen history files plus market_fetch's yearly increment files."""
import os
import time

import pandas as pd

from src.data.storage import DataStore


def _bars(times, close, source):
    return pd.DataFrame({
        "datetime": pd.to_datetime(times), "code": "600000.SH", "open": close, "high": close,
        "low": close, "close": close, "volume": 100.0, "amount": 1000.0, "source": source,
    })


def test_read_raw_merges_increments(tmp_path):
    s = DataStore(tmp_path)
    _bars(["2026-09-24 14:59", "2026-09-24 15:00"], 9.0, "pytdx").to_parquet(s.freq_path("600000.SH", "min1"))
    inc = tmp_path / "min1_2026"
    inc.mkdir()
    # 15:00 of 09-24 is re-delivered by the increment (increment wins), 09-25 is new
    _bars(["2026-09-24 15:00", "2026-09-25 09:31"], 9.5, "sync:tdx").to_parquet(inc / "600000.SH.parquet")
    df = s.read_raw("600000.SH", "min1")
    assert list(df["datetime"].astype(str)) == ["2026-09-24 14:59:00", "2026-09-24 15:00:00", "2026-09-25 09:31:00"]
    assert list(df["close"]) == [9.0, 9.5, 9.5]
    assert s.increment_paths("600000.SH", "min1") == [inc / "600000.SH.parquet"]
    assert s.increment_paths("600000.SH", "min5") == []


def test_read_raw_increment_only(tmp_path):
    s = DataStore(tmp_path)
    (tmp_path / "min1_2026").mkdir()
    _bars(["2026-09-25 09:31"], 10.0, "sync:tdx").to_parquet(tmp_path / "min1_2026" / "600000.SH.parquet")
    assert len(s.read_raw("600000.SH", "min1")) == 1
    assert s.read_raw("000001.SZ", "min1") is None


def test_hfq_cache_invalidated_by_increment(tmp_path):
    s = DataStore(tmp_path)
    _bars(["2026-09-24 15:00"], 9.0, "pytdx").to_parquet(s.freq_path("600000.SH", "min1"))
    s.get_hfq("600000.SH", "min1")
    assert s._hfq_cache_valid("600000.SH", "min1")
    (tmp_path / "min1_2026").mkdir()
    p = tmp_path / "min1_2026" / "600000.SH.parquet"
    _bars(["2026-09-25 09:31"], 9.5, "sync:tdx").to_parquet(p)
    later = time.time() + 5
    os.utime(p, (later, later))
    assert not s._hfq_cache_valid("600000.SH", "min1")
    assert len(s.get_hfq("600000.SH", "min1")) == 2
