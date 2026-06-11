#!/usr/bin/env python
"""拉取并保存 tushare ``stock_basic`` 全量股票基础信息（含历史退市股，防幸存者偏差）。

tushare ``stock_basic`` 按 ``list_status`` 分三类：
  L=上市  D=退市  P=暂停上市
低积分账号该接口限流约 **1 次/小时**，故本脚本：
  - 只补 ``data_store/stock_basic_tushare.parquet`` 里**尚缺**的状态，已有的跳过；
  - 单次运行最多取一个状态（避免触发小时级冷却），重复运行即可逐步补齐。

用法（务必绕过代理，tushare 为境内 API）：
    NO_PROXY=* python scripts/fetch_tushare_basic.py            # 自动补一个缺失状态
    NO_PROXY=* python scripts/fetch_tushare_basic.py --status D # 指定状态

token 取值优先级：--token > .env 的 TUSHARE_TOKEN（显式读 .env，绕开被系统环境变量遮蔽的陈旧值）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

OUT = Path("data_store/stock_basic_tushare.parquet")
FIELDS = "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
STATUSES = ("L", "D", "P")


def _load_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token.strip()
    # 显式读 .env，避免系统环境里陈旧的 TUSHARE_TOKEN 遮蔽真实值
    env_path = Path(__file__).resolve().parent.parent / ".env"
    tok = (dotenv_values(env_path).get("TUSHARE_TOKEN") or "").strip()
    if not tok:
        tok = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not tok:
        raise SystemExit("未找到 TUSHARE_TOKEN（.env 或环境变量）")
    return tok


def main() -> int:
    ap = argparse.ArgumentParser(description="拉取 tushare stock_basic 全量(含退市)")
    ap.add_argument("--status", choices=STATUSES, default=None,
                    help="指定 list_status；缺省=自动补一个尚缺的状态")
    ap.add_argument("--token", default=None, help="显式 token（覆盖 .env）")
    args = ap.parse_args()

    existing = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    have = set(existing["list_status"].unique()) if "list_status" in existing else set()

    if args.status:
        targets = [args.status]
    else:
        targets = [s for s in STATUSES if s not in have]
        if not targets:
            print(f"已齐全（{sorted(have)}），无需补取。")
            return 0
        targets = targets[:1]  # 单次只取一个，规避 1 次/小时 限流

    import tushare as ts

    pro = ts.pro_api(_load_token(args.token))

    frames = [existing] if not existing.empty else []
    for st in targets:
        df = pro.stock_basic(exchange="", list_status=st, fields=FIELDS)
        df["list_status"] = st
        frames.append(df)
        print(f"{st}: {len(df)} 只"
              + (f" | delist {df['delist_date'].min()}~{df['delist_date'].max()}" if st == "D" else ""))

    out = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"已保存 {OUT} | 合计 {len(out)} | 状态分布:")
    print(out.groupby("list_status").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
