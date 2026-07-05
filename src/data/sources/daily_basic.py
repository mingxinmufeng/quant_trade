"""
tushare ``daily_basic`` 每日指标：回填 / 增量更新 / 读取
==========================================================

按交易日整市场拉取 tushare ``daily_basic``（逐日 **PIT** 股本 / 换手率 / 估值，
需约 2000 积分），落盘到 ``{store}/daily_basic/``。这是免费源普遍缺失的一块：
gbbq cat5 股本快照存在多年断档（见 ``gbbq.py``），本数据可提供逐日官方股本作
计量基准与互验。

落盘结构（by_date 为事实层，年度文件为派生读取层）::

    daily_basic/by_date/YYYYMMDD.parquet   # 一交易日一文件；文件存在 = 已完成（断点续传依据）
    daily_basic/2015.parquet ...           # 年度合并（研究读取层）；更新时自动重建被触碰年份

退化策略（供 ``fetch`` 命令每次更新后无条件调用，任何情况不影响主流程）：

- 无 ``TUSHARE_TOKEN``            → WARNING 后跳过；
- 权限 / 积分不足（报错文本判定）  → WARNING 后跳过；
- 限频                            → 等待重试（多账号先轮换，见 ``TusharePool``）；
- 连续多日失败                    → 中止本轮（防 token 失效 / 断网时刷屏），下轮续传。

用法::

    python -m src.data.sources.daily_basic                    # 按配置 store，默认起点回填/增量
    python -m src.data.sources.daily_basic --rebuild          # 无视已有全部重拉
    python -m src.data.sources.daily_basic --compact-only     # 只重建年度合并层

    from src.data.sources.daily_basic import load_daily_basic
    df = load_daily_basic(store, start="20200101", fields=["float_share", "turnover_rate_f"])
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from .tushare_pool import TusharePool

#: 默认回填起点（配置键 ``data.daily_basic_start`` 可覆盖）
DEFAULT_START = "20140101"

#: 单交易日行数下限：低于视为脏数据不落盘（全市场 2014 年已 >2000 只；0 行=拉取失败）
MIN_ROWS_OK = 1000

#: 两次请求最小间隔（秒），~170 次/分钟，留出 2000 积分档限频余量
THROTTLE_SECONDS = 0.35

#: 限频时的等待秒数与最大重试轮数（TusharePool 已先做多账号轮换，这里是全池限频后的等待）
RATELIMIT_SLEEP = 65
MAX_RATELIMIT_ROUNDS = 3

#: 连续失败超过该数即中止本轮（token 失效/断网时避免对每个交易日重复报错）
MAX_CONSECUTIVE_FAILURES = 5

#: 限频类错误文本特征（与 tushare_pool 一致；命中→等待重试而非失败）
_RATELIMIT_HINTS = ("频率超限", "每分钟", "每小时", "次/分", "次/小时", "抱歉，您")

#: 权限/积分类错误文本特征（命中→整轮直接放弃，属"账号级"而非"请求级"问题）
_PERMISSION_HINTS = ("权限", "积分")


class DailyBasicPermissionError(RuntimeError):
    """tushare 账号无 daily_basic 权限（积分不足）。"""


def _is_ratelimit(exc: Exception) -> bool:
    s = str(exc)
    return any(h in s for h in _RATELIMIT_HINTS)


def _is_permission(exc: Exception) -> bool:
    s = str(exc)
    return any(h in s for h in _PERMISSION_HINTS)


def _call_with_ratelimit_retry(pool, method: str, **kwargs) -> pd.DataFrame:
    """经 pool 调用接口：限频则等待重试（pool 内部已先轮换账号），权限错误归类上抛。"""
    last_exc: Exception | None = None
    for round_no in range(MAX_RATELIMIT_ROUNDS):
        try:
            df = pool.call(method, **kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:
            if _is_permission(exc):
                raise DailyBasicPermissionError(str(exc)) from exc
            if not _is_ratelimit(exc):
                raise
            last_exc = exc
            if round_no < MAX_RATELIMIT_ROUNDS - 1:
                logger.warning(f"tushare 限频（{method}），等待 {RATELIMIT_SLEEP}s 重试")
                time.sleep(RATELIMIT_SLEEP)
    assert last_exc is not None
    raise last_exc


def trade_dates(pool, start: str, end: str) -> list[str]:
    """交易日清单（YYYYMMDD 升序），来自 tushare trade_cal。"""
    df = _call_with_ratelimit_retry(
        pool, "trade_cal",
        exchange="SSE", is_open="1", start_date=start, end_date=end,
        fields="cal_date",
    )
    return sorted(df["cal_date"].astype(str)) if not df.empty else []


def compact_years(store: Path, years: set[str]) -> None:
    """把 by_date 事实层按年合并为 daily_basic/{YYYY}.parquet（整年重建，幂等）。"""
    by_date = Path(store) / "daily_basic" / "by_date"
    for y in sorted(years):
        files = sorted(by_date.glob(f"{y}*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        out = Path(store) / "daily_basic" / f"{y}.parquet"
        df.to_parquet(out, index=False, compression="snappy")
        logger.info(f"  [daily_basic] 合并 {y}: {len(files)} 天 / {len(df):,} 行")


def run_backfill(
    store: str | Path,
    pool,
    start: str = DEFAULT_START,
    end: str | None = None,
    rebuild: bool = False,
) -> dict[str, int]:
    """逐日回填 daily_basic：已有日期跳过（rebuild=True 除外），完成后重建触碰年份。

    Returns:
        统计 {"fetched": 成功天数, "skipped": 跳过天数, "failed": 失败天数}。

    Raises:
        DailyBasicPermissionError: 账号无权限（积分不足），调用方据此整体降级。
    """
    end = end or date.today().strftime("%Y%m%d")
    by_date = Path(store) / "daily_basic" / "by_date"
    by_date.mkdir(parents=True, exist_ok=True)

    cal = trade_dates(pool, start, end)
    existing = {f.stem for f in by_date.glob("*.parquet")}
    todo = cal if rebuild else [d for d in cal if d not in existing]
    stats = {"fetched": 0, "skipped": len(cal) - len(todo), "failed": 0}
    if not todo:
        logger.debug(f"daily_basic 已是最新（{start}~{end} 共 {len(cal)} 个交易日均已落盘）")
        return stats
    logger.info(f"daily_basic 待拉取 {len(todo)} / {len(cal)} 个交易日（{start}~{end}）")

    touched_years: set[str] = set()
    consecutive_failures = 0
    for i, d in enumerate(todo, 1):
        try:
            df = _call_with_ratelimit_retry(pool, "daily_basic", trade_date=d)
        except DailyBasicPermissionError:
            raise
        except Exception as exc:
            stats["failed"] += 1
            consecutive_failures += 1
            logger.warning(f"  daily_basic {d} 拉取失败: {exc}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"daily_basic 连续失败 {consecutive_failures} 天，"
                             f"中止本轮（token/网络问题？），下轮自动续传")
                break
            continue
        consecutive_failures = 0
        if len(df) < MIN_ROWS_OK:
            # 不落盘 = 不标记完成，下轮重试；避免把空/残缺数据当作已完成
            stats["failed"] += 1
            logger.warning(f"  daily_basic {d} 仅 {len(df)} 行（<{MIN_ROWS_OK}），视为异常不落盘")
            continue
        df.to_parquet(by_date / f"{d}.parquet", index=False, compression="snappy")
        stats["fetched"] += 1
        touched_years.add(d[:4])
        if i % 200 == 0 or i == len(todo):
            logger.info(f"  daily_basic 进度 {i}/{len(todo)}（最新 {d}，{len(df)} 行）")
        time.sleep(THROTTLE_SECONDS)

    if touched_years:
        compact_years(Path(store), touched_years)
    logger.info(f"daily_basic 更新完成：新增 {stats['fetched']} 天，"
                f"跳过 {stats['skipped']}，失败 {stats['failed']}")
    return stats


def update_daily_basic(
    store: str | Path,
    start: str = DEFAULT_START,
    end: str | None = None,
    pool=None,
) -> bool:
    """增量更新入口（``fetch`` 命令每次更新后调用）：**任何失败只告警，不上抛**。

    Returns:
        True = 本轮正常完成；False = 无 token / 无权限 / 异常而跳过。
    """
    pool = pool if pool is not None else TusharePool()
    if not getattr(pool, "available", True):
        logger.warning("未配置 TUSHARE_TOKEN，跳过 daily_basic 增量更新")
        return False
    try:
        run_backfill(store, pool, start=start, end=end)
        return True
    except DailyBasicPermissionError as exc:
        logger.warning(f"tushare 账号无 daily_basic 权限（约需 2000 积分），跳过: {exc}")
        return False
    except Exception as exc:
        logger.error(f"daily_basic 增量更新失败（不影响行情更新结果）: {exc}")
        return False


def load_daily_basic(
    store: str | Path,
    start: str | None = None,
    end: str | None = None,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """读取年度合并层，按 [start, end]（YYYYMMDD 闭区间）与字段过滤。

    ``fields`` 无需包含 ts_code/trade_date，二者总是保留。无数据返回空表。
    """
    root = Path(store) / "daily_basic"
    files = sorted(root.glob("[0-9][0-9][0-9][0-9].parquet"))
    if start:
        files = [f for f in files if f.stem >= start[:4]]
    if end:
        files = [f for f in files if f.stem <= end[:4]]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    td = df["trade_date"].astype(str)
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= td >= start
    if end:
        mask &= td <= end
    df = df[mask]
    if fields:
        keep = ["ts_code", "trade_date"] + [
            c for c in fields if c in df.columns and c not in ("ts_code", "trade_date")
        ]
        df = df[keep]
    return df.reset_index(drop=True)


def main() -> None:
    """独立 CLI：手动回填 / 重建合并层（store 按配置链解析，同框架其它入口）。"""
    import argparse

    from ...utils.config_loader import load_config

    parser = argparse.ArgumentParser(description="tushare daily_basic 回填/增量（断点续传）")
    parser.add_argument("--store-path", default=None, help="数据仓库根目录（缺省按配置链解析）")
    parser.add_argument("--start", default=None, help=f"起始日 YYYYMMDD（默认配置 daily_basic_start 或 {DEFAULT_START}）")
    parser.add_argument("--end", default=None, help="截止日 YYYYMMDD（默认今天）")
    parser.add_argument("--rebuild", action="store_true", help="无视已有 by_date 文件全部重拉")
    parser.add_argument("--compact-only", action="store_true", help="不拉数据，只重建全部年度合并层")
    args = parser.parse_args()

    cfg = load_config()
    store = Path(args.store_path or cfg.data.store_path)
    start = args.start or str(cfg.data.get("daily_basic_start", DEFAULT_START))

    if args.compact_only:
        by_date = store / "daily_basic" / "by_date"
        years = {f.stem[:4] for f in by_date.glob("*.parquet")}
        if not years:
            logger.warning("by_date 层为空，无可合并数据")
            return
        compact_years(store, years)
        return

    pool = TusharePool()
    if not pool.available:
        raise SystemExit("未发现 TUSHARE_TOKEN：请在 .env 或环境变量中配置")
    run_backfill(store, pool, start=start, end=args.end, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
