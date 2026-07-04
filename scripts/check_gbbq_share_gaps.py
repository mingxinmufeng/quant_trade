"""Diagnose gbbq category-5 share snapshot gaps.

This script is intentionally a diagnostic, not a failing data test. TDX gbbq
category 5 is an incomplete share-structure snapshot stream on some stocks, so
the useful output is gap/freshness distribution plus concrete examples.

Usage:
    python scripts/check_gbbq_share_gaps.py
    python scripts/check_gbbq_share_gaps.py --store-path C:/QuantData/data_store
    python scripts/check_gbbq_share_gaps.py --code 600519.SH
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

MARKET_SUFFIX = {0: "SZ", 1: "SH", 2: "BJ"}


def _store_path(arg: str | None) -> Path:
    raw = arg or os.getenv("QUANT_DATA__STORE_PATH") or "data_store"
    return Path(raw).expanduser()


def _load_events(store: Path) -> pd.DataFrame:
    path = store / "gbbq_events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["std_code"] = df["code"].astype(str).str.zfill(6) + "." + df["market"].map(MARKET_SUFFIX)
    return df


def _load_active_codes(store: Path) -> pd.DataFrame:
    path = store / "stock_basic_tushare.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts_code", "list_date", "delist_date"])
    df = pd.read_parquet(path)
    df = df[df["ts_code"].astype(str).str.match(r"^[0-9]{6}\.(SH|SZ|BJ)$", na=False)].copy()
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df["delist_date"] = pd.to_datetime(df["delist_date"], errors="coerce")
    return df[df["list_status"].astype(str).eq("L")]


def _print_sample(df: pd.DataFrame, code: str) -> None:
    sub = df[df["std_code"] == code].sort_values("date")
    if sub.empty:
        print(f"\n{code}: no gbbq records")
        return

    cat5 = sub[sub["category"] == 5][
        [
            "date",
            "hongli_panqianliutong",
            "peigujia_qianzongguben",
            "songgu_qianzongguben",
            "peigu_houzongguben",
        ]
    ]
    print(f"\n{code} category=5 snapshots:")
    print(cat5.to_string(index=False) if not cat5.empty else "(none)")

    cat1 = sub[sub["category"] == 1].copy()
    cat1 = cat1[
        (cat1["songgu_qianzongguben"].abs() > 1e-9)
        | (cat1["peigu_houzongguben"].abs() > 1e-9)
        | (cat1["hongli_panqianliutong"].abs() > 1e-9)
    ][
        [
            "date",
            "hongli_panqianliutong",
            "peigujia_qianzongguben",
            "songgu_qianzongguben",
            "peigu_houzongguben",
        ]
    ]
    print(f"\n{code} category=1 nonzero ex-dividend events:")
    print(cat1.to_string(index=False) if not cat1.empty else "(none)")


def _gap_stats(df: pd.DataFrame, active_codes: set[str], since: pd.Timestamp, max_age_days: int) -> None:
    cat5 = df[(df["category"] == 5) & (df["std_code"].isin(active_codes))].sort_values(["std_code", "date"])
    after = cat5[cat5["date"] >= since]
    gaps = after.groupby("std_code")["date"].apply(lambda s: s.diff().dt.days.max()).dropna()
    if gaps.empty:
        print("\nNo category=5 gap stats available.")
        return

    print(f"\ncategory=5 max inter-snapshot gap since {since.date()} (active listed stocks)")
    print(f"stocks with >=2 snapshots: {len(gaps)}")
    print(f"gap > {max_age_days} days: {int((gaps > max_age_days).sum())} ({(gaps > max_age_days).mean():.2%})")
    print("gap quantiles (days):")
    print(gaps.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(2).to_string())


def _freshness_stats(
    df: pd.DataFrame,
    stocks: pd.DataFrame,
    since: pd.Timestamp,
    max_age_days: int,
) -> None:
    cat5 = df[df["category"] == 5][["std_code", "date"]].sort_values(["std_code", "date"])
    by_code = {c: g["date"].to_numpy(dtype="datetime64[ns]") for c, g in cat5.groupby("std_code")}
    months = pd.date_range(since + pd.offsets.MonthEnd(0), pd.Timestamp.today().normalize(), freq="ME")
    rows = []
    records = stocks[["ts_code", "list_date", "delist_date"]].to_dict("records")
    for month in months:
        denom = 0
        has_snapshot = 0
        fresh = 0
        ages: list[int] = []
        month64 = np.datetime64(month)
        for row in records:
            list_date = row["list_date"]
            delist_date = row["delist_date"]
            if pd.isna(list_date) or list_date > month or (pd.notna(delist_date) and delist_date <= month):
                continue
            denom += 1
            arr = by_code.get(row["ts_code"])
            if arr is None or len(arr) == 0:
                continue
            idx = np.searchsorted(arr, month64, side="right") - 1
            if idx < 0:
                continue
            age = (month - pd.Timestamp(arr[idx])).days
            ages.append(age)
            has_snapshot += 1
            fresh += int(age <= max_age_days)
        rows.append(
            {
                "month": month,
                "active": denom,
                "has_snapshot": has_snapshot,
                "fresh": fresh,
                "fresh_rate": fresh / denom if denom else np.nan,
                "coverage": has_snapshot / denom if denom else np.nan,
                "mean_age": float(np.mean(ages)) if ages else np.nan,
            }
        )

    res = pd.DataFrame(rows)
    if res.empty:
        print("\nNo monthly freshness stats available.")
        return
    print(f"\nmonth-end category=5 freshness <= {max_age_days} days")
    print(res[["fresh_rate", "coverage", "mean_age"]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
    print("\nlast 12 months:")
    print(res.tail(12).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-path", default=None)
    parser.add_argument("--code", default="600519.SH")
    parser.add_argument("--since", default="2014-01-01")
    parser.add_argument("--max-age-days", type=int, default=400)
    args = parser.parse_args()

    store = _store_path(args.store_path)
    events = _load_events(store)
    stocks = _load_active_codes(store)
    active_codes = set(stocks["ts_code"]) if not stocks.empty else set(events["std_code"].unique())
    since = pd.Timestamp(args.since)

    print(f"store: {store}")
    print(f"gbbq events: {len(events)} rows")
    print(events["category"].value_counts().sort_index().to_string())
    print(f"active stock universe: {len(active_codes)}")

    _print_sample(events, args.code)
    _gap_stats(events, active_codes, since, args.max_age_days)
    if not stocks.empty:
        _freshness_stats(events, stocks, since, args.max_age_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
