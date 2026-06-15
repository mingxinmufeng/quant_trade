#!/usr/bin/env python
"""用退市股**总清单**回填本地缺失的退市行情（防幸存者偏差）。

总清单 = tushare ``stock_basic`` list_status='D' ∪ akshare 沪深终止上市清单，
去重、去脏码（如 tushare 偶发的 ``T600018``）。逻辑上互补：
  - tushare D 覆盖北交所/老三板，akshare 覆盖深 B 股(200xxx)；
  - 取并集即"最全"退市清单。详见 memory: survivorship-backfill-plan。

回填走项目既有 ``DataFetcher.update`` 管线（源链首选 pytdx 本地盘，零网络命中
退市股），落盘 schema 与在市股完全一致（含停牌 gap / 涨跌停 / 复权因子）。

用法（务必绕代理，因子/停牌/名称表仍走境内网络）：
    NO_PROXY=* python scripts/backfill_delisted.py --dry-run         # 只打印清单与缺失数
    NO_PROXY=* python scripts/backfill_delisted.py --limit 5         # 试跑前 5 只
    NO_PROXY=* python scripts/backfill_delisted.py                   # 全量回填(daily)
    NO_PROXY=* python scripts/backfill_delisted.py --freqs daily,min5,min1
"""

from __future__ import annotations

import argparse
import contextlib
import re
from pathlib import Path

import pandas as pd
from src.main import _build_fetcher, _load_cfg  # 复用 CLI 的 fetcher 构建

TUSHARE_BASIC = Path("data_store/stock_basic_tushare.parquet")
STORE = Path("data_store")
_TS_CODE_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")


def _tushare_delisted() -> set[str]:
    """tushare stock_basic 里 list_status='D' 的合法 ts_code（剔除脏码）。"""
    if not TUSHARE_BASIC.exists():
        return set()
    df = pd.read_parquet(TUSHARE_BASIC)
    d = df[df["list_status"] == "D"]["ts_code"].astype(str)
    return {c for c in d if _TS_CODE_RE.match(c)}


def _akshare_delisted() -> set[str]:
    """akshare 沪深终止上市清单 → 标准 ts_code（含深 B 股 200xxx）。"""
    import akshare as ak
    from src.utils.helpers import format_code

    out: set[str] = set()
    for fn, kwargs in (
        (ak.stock_info_sh_delist, {}),
        (ak.stock_info_sz_delist, {"symbol": "终止上市公司"}),
    ):
        try:
            df = fn(**kwargs)
        except Exception as exc:
            print(f"[warn] akshare {fn.__name__}({kwargs}) 失败: {exc}")
            continue
        for raw in df.iloc[:, 0].astype(str):
            with contextlib.suppress(Exception):
                out.add(format_code(raw))
    return out


def build_universe() -> set[str]:
    ts = _tushare_delisted()
    ak = _akshare_delisted()
    uni = ts | ak
    print(f"tushare D: {len(ts)} | akshare 终止: {len(ak)} | 并集去重: {len(uni)}")
    return uni


def main() -> int:
    ap = argparse.ArgumentParser(description="回填退市股总清单缺失行情")
    ap.add_argument("--freqs", default="daily", help="空格/逗号分隔周期：daily min5 min1")
    ap.add_argument("--limit", type=int, default=0, help="只回填前 N 只（试跑用，0=全部）")
    ap.add_argument("--throttle", type=float, default=0.0, help="每只 sleep 秒（纯 pytdx 本地可 0；分钟会网络 fallback，建议 ≥1 防风控）")
    ap.add_argument("--workers", type=int, default=None, help="并行线程数；缺省=config(4)。分钟网络 fallback 多时建议 2~3 防风控")
    ap.add_argument("--sources", default=None, help="逗号分隔覆盖数据源链，如 'pytdx'。退市股分钟网络源基本无效，建议纯 pytdx")
    ap.add_argument("--dry-run", action="store_true", help="只打印清单与缺失，不落盘")
    ap.add_argument("--config", default=None, help="配置文件路径")
    args = ap.parse_args()
    freqs = tuple(f for f in re.split(r"[\s,;]+", args.freqs.strip()) if f)

    universe = build_universe()
    # 按周期分别 diff 对应 store 子目录（daily/min5/min1），取并集为本次待处理代码。
    missing_set: set[str] = set()
    for fq in freqs:
        have = {p.stem for p in (STORE / fq).glob("*.parquet")}
        miss_fq = universe - have
        missing_set |= miss_fq
        print(f"[{fq}] 已有 {len(have)} | 退市清单中缺 {len(miss_fq)}")
    missing = sorted(missing_set)
    print(f"退市总清单: {len(universe)} | 本次待回填(任一周期缺): {len(missing)}")

    if args.limit:
        missing = missing[: args.limit]
        print(f"[limit] 本次只处理前 {len(missing)} 只")

    if not missing:
        print("无缺失，已齐全。")
        return 0
    print("待回填示例:", missing[:10])

    if args.dry_run:
        print("[dry-run] 不落盘，结束。")
        return 0

    cfg = _load_cfg(args.config)
    if args.sources:
        src_list = [s for s in re.split(r"[\s,;]+", args.sources.strip()) if s]
        cfg.setdefault("data", {})["sources"] = src_list
        print(f"[sources] 覆盖数据源链为: {src_list}")
    fetcher = _build_fetcher(cfg)
    print(f"开始回填 | 股票={len(missing)} | 周期={freqs} | throttle={args.throttle} | workers={args.workers or 'config'}")
    fetcher.update(codes=missing, freqs=freqs, throttle=args.throttle, max_workers=args.workers)

    for fq in freqs:
        have = {p.stem for p in (STORE / fq).glob("*.parquet")}
        still = [c for c in missing if c not in have]
        print(f"[{fq}] 回填后仍缺(本地盘+网络都无): {len(still)}" + (f" 例:{still[:10]}" if still else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
