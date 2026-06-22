#!/usr/bin/env python
"""拉取并保存全市场行业分类映射 ``industry_map``（供 RiskManager 单行业仓位限制用）。

落盘 ``data_store/industry_map.parquet``，列：
    code            标准代码 XXXXXX.SH/SZ/BJ
    name            股票名称
    industry        行业名（建图 {code: industry} 用这一列）
    classification  细分门类原文（baostock 的 industryClassification；tushare 行留空）
    source          行级来源（baostock / tushare）
    fetch_date      拉取日（快照溯源；行业归属会变，回测点位口径需注意前视）

数据源策略（与 docstring 对比一致）：
  - **baostock 为主**：``query_stock_industry()``，免 token、标准证监会行业；**不含北交所**。
  - **tushare 补缺**：仅填 baostock 缺失的代码（北交所 + baostock 空行业），
    东财一级行业；需 TUSHARE_TOKEN。无 token / 调用失败则跳过，仅存 baostock 部分。

用法（务必绕过代理，二者均为境内 API）：
    NO_PROXY=* python scripts/fetch_industry_map.py                 # baostock+tushare
    NO_PROXY=* python scripts/fetch_industry_map.py --source baostock
    NO_PROXY=* python scripts/fetch_industry_map.py --token <tok>

token 取值优先级：--token > .env 的 TUSHARE_TOKEN（绕开被系统环境变量遮蔽的陈旧值）。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# 允许脚本直接运行时导入 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.helpers import format_code  # noqa: E402

OUT = Path("data_store/industry_map.parquet")
COLS = ["code", "name", "industry", "classification", "source", "fetch_date"]


def _bs_code_to_std(bs_code: str) -> str:
    """baostock 'sh.600000' → '600000.SH'（取数字部分交给 format_code 推断）。"""
    parts = str(bs_code).split(".")
    return format_code(parts[1] if len(parts) == 2 else parts[0])


def fetch_baostock(today: str) -> pd.DataFrame:
    """baostock 证监会行业分类（不含北交所）。"""
    import baostock as bs

    lg = bs.login()
    if getattr(lg, "error_code", "0") != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    try:
        rs = bs.query_stock_industry()
        if getattr(rs, "error_code", "0") != "0":
            raise RuntimeError(f"baostock 行业查询失败: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()

    if not rows:
        return pd.DataFrame(columns=COLS)
    raw = pd.DataFrame(rows, columns=rs.fields)
    out = pd.DataFrame({
        "code": raw["code"].map(_bs_code_to_std),
        "name": raw["code_name"].astype("string").str.strip(),
        "industry": raw["industry"].astype("string").str.strip(),
        "classification": raw["industryClassification"].astype("string").str.strip(),
    })
    out["industry"] = out["industry"].replace("", pd.NA)
    out["source"] = "baostock"
    out["fetch_date"] = today
    return out[COLS]


def fetch_tushare(token: str, today: str) -> pd.DataFrame:
    """tushare 东财一级行业（覆盖北交所；一次取全 L）。"""
    import tushare as ts

    pro = ts.pro_api(token)
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,name,industry")
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS)
    out = pd.DataFrame({
        "code": df["ts_code"].map(format_code),
        "name": df["name"].astype("string").str.strip(),
        "industry": df["industry"].astype("string").str.strip(),
        "classification": pd.array([pd.NA] * len(df), dtype="string"),
    })
    out["industry"] = out["industry"].replace("", pd.NA)
    out["source"] = "tushare"
    out["fetch_date"] = today
    return out[COLS]


def _load_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token.strip()
    try:
        from dotenv import dotenv_values
        env_path = Path(__file__).resolve().parent.parent / ".env"
        tok = (dotenv_values(env_path).get("TUSHARE_TOKEN") or "").strip()
    except Exception:
        tok = ""
    return tok or os.environ.get("TUSHARE_TOKEN", "").strip() or None


def main() -> int:
    ap = argparse.ArgumentParser(description="拉取并保存全市场行业分类 industry_map")
    ap.add_argument("--source", choices=("both", "baostock", "tushare"), default="both")
    ap.add_argument("--token", default=None, help="显式 tushare token（覆盖 .env）")
    args = ap.parse_args()

    today = date.today().isoformat()
    frames: list[pd.DataFrame] = []

    if args.source in ("both", "baostock"):
        bs_df = fetch_baostock(today)
        print(f"baostock: {len(bs_df)} 只 | 有行业 {int(bs_df['industry'].notna().sum())} "
              f"| 行业数 {bs_df['industry'].nunique(dropna=True)}（不含北交所）")
        frames.append(bs_df)

    if args.source in ("both", "tushare"):
        token = _load_token(args.token)
        if not token:
            print("跳过 tushare：未找到 TUSHARE_TOKEN（仅靠 baostock，缺北交所）")
        else:
            try:
                ts_df = fetch_tushare(token, today)
                print(f"tushare: {len(ts_df)} 只 | 有行业 {int(ts_df['industry'].notna().sum())} "
                      f"| 行业数 {ts_df['industry'].nunique(dropna=True)}")
                if args.source == "both":
                    # baostock 为主，tushare 只补 baostock 缺失/空行业的代码
                    have = set(frames[0].loc[frames[0]["industry"].notna(), "code"])
                    ts_df = ts_df[~ts_df["code"].isin(have)]
                    print(f"tushare 补缺 {len(ts_df)} 只（北交所 + baostock 缺行业）")
                frames.append(ts_df)
            except Exception as exc:
                print(f"tushare 拉取失败，忽略（仅存 baostock）: {type(exc).__name__}: {exc}")

    if not frames:
        print("无任何数据，未写盘。")
        return 1

    out = pd.concat(frames, ignore_index=True)
    out = out[out["industry"].notna()]  # 只保留能定位行业的行
    out = out.drop_duplicates("code", keep="first").sort_values("code").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\n已保存 {OUT} | 合计 {len(out)} 只 | 来源分布:")
    print(out.groupby("source").size().to_string())
    print(f"北交所(.BJ)覆盖: {int(out['code'].str.endswith('.BJ').sum())} 只")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
