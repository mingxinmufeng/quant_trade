"""
本地数据完整性检查
==================
扫描 data_store/{daily,min5,min1,factors} 中的 parquet 文件，
逐文件检查字段、类型、空值、价格逻辑、日期连续性等。

运行：
    python tests/test_data_integrity.py
或：
    python -m pytest tests/test_data_integrity.py -q
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.fetcher import (  # noqa: E402
    DAILY_COLUMNS,
    FREQ_DIRS,
    MINUTE_COLUMNS,
)
from src.data.trading_calendar import TradingCalendar  # noqa: E402

# ============================================================
# 常量
# ============================================================

STORE_PATH = _REPO_ROOT / "data_store"
FREQS = list(FREQ_DIRS.keys())  # daily, min5, min1

# 关键数值列：不允许 NaN，且必须 >0（或 >=0）
PRICE_COLS = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume", "amount"]

# 分钟线合法交易时段（A 股）
AM_START, AM_END = (9, 30), (11, 30)
PM_START, PM_END = (13, 0), (15, 0)

# 预计算时间对象，避免逐行重复构造
_AM_START = pd.Timestamp("09:30").time()
_AM_END = pd.Timestamp("11:30").time()
_PM_START = pd.Timestamp("13:00").time()
_PM_END = pd.Timestamp("15:00").time()

# ============================================================
# 辅助函数
# ============================================================

def _load_df(path: Path, freq: str) -> pd.DataFrame | None:
    """读取 parquet；失败返回 None。"""
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.error(f"读取失败 {path}: {exc}")
        return None
    tc = "date" if freq == "daily" else "datetime"
    if tc in df.columns:
        df[tc] = pd.to_datetime(df[tc])
    return df


def _check_columns(df: pd.DataFrame, expected: Dict[str, str], label: str) -> List[str]:
    """检查字段是否齐全、类型是否匹配（允许向下兼容，如 float32 视为 float64）。"""
    errs = []
    missing = [c for c in expected if c not in df.columns]
    if missing:
        errs.append(f"[{label}] 缺失字段: {missing}")
    for col, exp_dtype in expected.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        # 放宽判断：datetime64[ns, UTC] 也视为 datetime64；float32 视为 float
        ok = False
        if "datetime64" in exp_dtype and "datetime64" in actual:
            ok = True
        elif exp_dtype == "float64" and ("float" in actual or actual == "float32"):
            ok = True
        elif exp_dtype == "int64" and ("int" in actual or actual == "int32"):
            ok = True
        elif exp_dtype == "bool" and actual == "bool":
            ok = True
        elif exp_dtype == "string" and (actual == "string" or actual == "object"):
            ok = True
        elif actual == exp_dtype:
            ok = True
        if not ok:
            errs.append(f"[{label}|{col}] 类型不符: 期望 {exp_dtype}, 实际 {actual}")
    return errs


def _check_nulls(df: pd.DataFrame, freq: str, label: str) -> List[str]:
    """检查关键字段是否存在空值。"""
    errs = []
    cols = list(df.columns)
    for col in PRICE_COLS + VOLUME_COLS:
        if col in cols and df[col].isna().any():
            n = df[col].isna().sum()
            errs.append(f"[{label}|{col}] 存在 {n} 个 NaN")
    # adj_factor 允许为空（早期数据可能无因子），但若存在则不应有 0/负数
    if "adj_factor" in cols:
        bad = df["adj_factor"].notna() & (df["adj_factor"] <= 0)
        if bad.any():
            errs.append(f"[{label}|adj_factor] 存在 {bad.sum()} 个 <=0 的值")
    return errs


def _check_price_logic(df: pd.DataFrame, freq: str, label: str) -> List[str]:
    """检查价格逻辑：high >= max(open,close,low), low <= min(open,close,high)。"""
    errs = []
    if not all(c in df.columns for c in PRICE_COLS):
        return errs
    bad_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    bad_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
    if bad_high.any():
        errs.append(f"[{label}] 存在 {bad_high.sum()} 行 high < max(open,close,low)")
    if bad_low.any():
        errs.append(f"[{label}] 存在 {bad_low.sum()} 行 low > min(open,close,high)")
    # volume 应 >= 0（停牌日可能为 0，允许）
    if "volume" in df.columns and (df["volume"] < 0).any():
        errs.append(f"[{label}] 存在 volume < 0")
    if "amount" in df.columns and (df["amount"] < 0).any():
        errs.append(f"[{label}] 存在 amount < 0")
    return errs


_tdays_cache: Dict[Tuple[date, date], List[date]] = {}


def _get_trading_days_cached(calendar: TradingCalendar, start: date, end: date) -> List[date]:
    key = (start, end)
    if key not in _tdays_cache:
        _tdays_cache[key] = calendar.get_trading_days(start, end)
    return _tdays_cache[key]


def _check_daily_continuity(df: pd.DataFrame, code: str, calendar: TradingCalendar, label: str) -> List[str]:
    """日线：检查日期是否为交易日，且是否连续（允许停牌导致的缺失）。"""
    errs = []
    if df.empty:
        return errs
    dates = pd.to_datetime(df["date"]).dt.date.tolist()
    # 所有日期必须是交易日
    bad_days = [d for d in dates if not calendar.is_trading_day(d)]
    if bad_days:
        sample = bad_days[:5]
        errs.append(f"[{label}] 含非交易日: {sample}{'...' if len(bad_days) > 5 else ''}")
    # 检查相邻交易日是否跨度太大（>20 个交易日无数据则报警，提示可能缺数据）
    trading_days = _get_trading_days_cached(calendar, min(dates), max(dates))
    # 把已有日期转 set，看最长连续缺失
    has_set = set(dates)
    max_gap = 0
    gap_start = None
    cur_gap = 0
    for d in trading_days:
        if d in has_set:
            if cur_gap > max_gap:
                max_gap = cur_gap
            cur_gap = 0
        else:
            if cur_gap == 0:
                gap_start = d
            cur_gap += 1
    if cur_gap > max_gap:
        max_gap = cur_gap
    # 允许正常停牌（ST 等），但超过 60 个交易日报警
    if max_gap > 60:
        errs.append(f"[{label}] 最长连续缺失 {max_gap} 个交易日（从 {gap_start} 起）")
    return errs


def _parse_minute_period(freq: str) -> int:
    """从 freq 字符串提取分钟周期数（min1→1, min5→5），无法解析返回 0。"""
    if freq.startswith("min"):
        try:
            return int(freq[3:])
        except ValueError:
            pass
    return 0


def _check_minute_time(df: pd.DataFrame, freq: str, label: str) -> List[str]:
    """分钟线：检查时间是否落在合法交易时段，且严格递增。"""
    errs = []
    if df.empty or "datetime" not in df.columns:
        return errs
    ts = pd.to_datetime(df["datetime"])
    t = ts.dt.time
    am = (t >= _AM_START) & (t <= _AM_END)
    pm = (t >= _PM_START) & (t <= _PM_END)
    bad = ~(am | pm)
    if bad.any():
        sample = ts[bad].head(3).tolist()
        errs.append(f"[{label}] 存在 {bad.sum()} 行不在交易时段（样例 {sample}）")
    # 检查同一天的分钟 bar 是否有重复（按分钟）
    dup = df.duplicated(subset=["datetime"], keep=False).sum()
    if dup:
        errs.append(f"[{label}] 存在 {dup} 行 datetime 重复")
    # 检查全局时间严格递增（不回流、不倒置）
    if not ts.is_monotonic_increasing:
        n_inv = (ts.diff() <= pd.Timedelta(0)).sum()
        errs.append(f"[{label}] datetime 未严格递增，存在 {n_inv} 处逆序或重复")
    return errs


def _check_minute_bar_count(
    df: pd.DataFrame, freq: str, label: str, min_bar_ratio: float = 0.8
) -> List[str]:
    """分钟线：通用化单日 bar 数检查，并对少于 240 根的情况做合理性判断。

    判断逻辑：
    1. 先统计该股票正常交易日（排除停牌）的 bar 数分布，以中位数作为"正常密度"。
    2. 若中位数 ≈ 理论值（如 240）→ 数据源为"固定时间格"（通达信特征）。
       若中位数 < 理论值但 ≥ 200 → 数据源为"有成交才生成"（东财/新浪特征）。
    3. 对每一个 bar 数 < 理论值的交易日，按偏离程度分级：
       - 合理：bar 数 ≥ 中位数 × 0.9（在数据源正常波动范围内）
       - 存疑：bar 数 < 中位数 × 0.9 但 ≥ 中位数 × 0.5（可能半日市/数据源波动）
       - 不合理：bar 数 < 中位数 × 0.5（明显偏离，疑似数据缺失）
    """
    errs = []
    if df.empty or "datetime" not in df.columns:
        return errs

    period = _parse_minute_period(freq)
    if period <= 0:
        return errs

    theory = 240 // period  # A 股 4h = 240 min

    ts = pd.to_datetime(df["datetime"])
    df = df.copy()
    df["_date"] = ts.dt.date

    grouped = df.groupby("_date")
    day_counts = grouped.size().reset_index(name="count")
    day_vol = grouped["volume"].sum().reset_index(name="vol_sum")
    day_info = day_counts.merge(day_vol, on="_date")
    # 停牌：volume 总和为 0 或 bar 数极少
    day_info["is_suspended"] = (day_info["vol_sum"] == 0) | (day_info["count"] < 10)

    active = day_info[~day_info["is_suspended"]]
    if active.empty:
        return errs

    median_cnt = active["count"].median()
    if median_cnt <= 0:
        return errs

    # 先判断数据源特征，作为全局理由
    if median_cnt >= theory * 0.98:
        source_feature = "固定时间格（通达信类）"
    elif median_cnt >= theory * 0.8:
        source_feature = "有成交才生成（东财/新浪类）"
    else:
        source_feature = "数据稀疏（早期数据/特殊源）"

    # 仅关注 bar 数 < 理论值 的交易日
    below_theory = active[active["count"] < theory]
    if below_theory.empty:
        return errs

    # 分级阈值
    ok_threshold = max(theory * min_bar_ratio, median_cnt * 0.9)
    suspect_threshold = max(1, median_cnt * 0.5)

    ok_days = below_theory[below_theory["count"] >= ok_threshold]
    suspect_days = below_theory[
        (below_theory["count"] < ok_threshold) & (below_theory["count"] >= suspect_threshold)
    ]
    bad_days = below_theory[below_theory["count"] < suspect_threshold]

    # 汇总输出：按分级给出判断结果 + 理由
    if not ok_days.empty:
        samples = ok_days.head(3)[["_date", "count"]].values.tolist()
        errs.append(
            f"[{label}] 【判断结果：合理】 {len(ok_days)} 个交易日 bar 数 < {theory} "
            f"但处于该数据源正常波动范围（≥{ok_threshold:.0f} 根）。"
            f"判断理由：该股票数据源特征为「{source_feature}」，"
            f"正常交易日中位数为 {median_cnt:.0f} 根，上述日期 bar 数偏离不大，"
            f"属于数据源正常省略（无成交分钟未生成 bar）。"
            f"样例（日期, 实际根数）: {samples}"
        )

    if not suspect_days.empty:
        samples = suspect_days.head(3)[["_date", "count"]].values.tolist()
        errs.append(
            f"[{label}] 【判断结果：存疑】 {len(suspect_days)} 个交易日 bar 数偏低 "
            f"（{suspect_threshold:.0f} ~ {ok_threshold:.0f} 根之间）。"
            f"判断理由：该股票正常交易日中位数为 {median_cnt:.0f} 根，"
            f"上述日期明显低于正常水平但尚未达到严重缺失标准。"
            f"可能原因：① 半日市（如除夕等特殊交易日）；② 当日某时段无成交被省略；"
            f"③ 数据源接口波动。建议人工复核。"
            f"样例（日期, 实际根数）: {samples}"
        )

    if not bad_days.empty:
        samples = bad_days.head(3)[["_date", "count"]].values.tolist()
        errs.append(
            f"[{label}] 【判断结果：不合理】 {len(bad_days)} 个交易日 bar 数严重偏低 "
            f"（<{suspect_threshold:.0f} 根）。"
            f"判断理由：该股票正常交易日中位数为 {median_cnt:.0f} 根，"
            f"上述日期不足中位数的 50%，已超出任何正常数据源的合理波动范围，"
            f"极大概率存在数据缺失（如某半天未下载、增量更新断流等）。"
            f"建议立即排查该日期区间的数据源或重新拉取。"
            f"样例（日期, 实际根数）: {samples}"
        )

    return errs


def _check_code_consistency(df: pd.DataFrame, code: str, label: str) -> List[str]:
    """检查 code 列是否全部一致。"""
    errs = []
    if "code" not in df.columns:
        return errs
    codes = df["code"].dropna().unique()
    if len(codes) == 0:
        errs.append(f"[{label}] code 列全部为空")
    elif len(codes) > 1:
        errs.append(f"[{label}] code 列不唯一: {list(codes)}")
    elif str(codes[0]).upper() != code.upper():
        errs.append(f"[{label}] code 列与文件名不符: {codes[0]} != {code}")
    return errs


# ============================================================
# 主逻辑
# ============================================================

def _check_one_file(
    args: Tuple[Path, str, Dict[str, str], TradingCalendar]
) -> Tuple[str, List[str]]:
    """单个文件检查（供线程池调用）。"""
    path, freq, expected_cols, calendar = args
    code = path.stem
    label = f"{code}|{freq}"
    df = _load_df(path, freq)
    if df is None:
        return label, [f"[{label}] 文件读取失败"]
    if df.empty:
        return label, [f"[{label}] 文件为空"]

    errs = []
    errs += _check_columns(df, expected_cols, label)
    errs += _check_nulls(df, freq, label)
    errs += _check_price_logic(df, freq, label)
    errs += _check_code_consistency(df, code, label)
    if freq == "daily":
        errs += _check_daily_continuity(df, code, calendar, label)
    else:
        errs += _check_minute_time(df, freq, label)
        errs += _check_minute_bar_count(df, freq, label)
    return label, errs


def scan_freq(
    freq: str, calendar: TradingCalendar, workers: int = 8
) -> Tuple[int, int, List[str]]:
    """扫描单个周期目录，使用线程池并行检查，返回 (文件数, 通过数, 错误列表)。"""
    d = STORE_PATH / FREQ_DIRS[freq]
    if not d.exists():
        logger.warning(f"目录不存在: {d}")
        return 0, 0, []

    files = sorted(d.glob("*.parquet"))
    total = len(files)
    if total == 0:
        return 0, 0, []

    expected_cols = DAILY_COLUMNS if freq == "daily" else MINUTE_COLUMNS
    passed = 0
    all_errs: List[str] = []
    args = [(p, freq, expected_cols, calendar) for p in files]

    # IO 密集型任务，用线程池加速；默认 workers 取 CPU 核心数或 8
    w = min(workers, total) if workers > 0 else 1
    with ThreadPoolExecutor(max_workers=w) as ex:
        futures = {ex.submit(_check_one_file, a): a for a in args}
        for fut in as_completed(futures):
            label, errs = fut.result()
            if errs:
                all_errs.extend(errs)
            else:
                passed += 1

    return total, passed, all_errs


def main(workers: int = 0) -> int:
    """workers=0 表示自动取 min(os.cpu_count() or 1, 16)。"""
    if workers <= 0:
        workers = min((os.cpu_count() or 1) * 2, 16)

    logger.info("=" * 60)
    logger.info(f"本地数据完整性检查开始 | 并行 workers={workers}")
    logger.info("=" * 60)

    calendar = TradingCalendar(store_path=STORE_PATH)
    grand_total = 0
    grand_passed = 0
    grand_errs: List[str] = []

    for freq in FREQS:
        total, passed, errs = scan_freq(freq, calendar, workers=workers)
        grand_total += total
        grand_passed += passed
        grand_errs.extend(errs)
        logger.info(f"{freq:6s} | 文件 {total:4d} | 通过 {passed:4d} | 异常 {total - passed:4d}")

    # factors 目录（只做基本可读性检查）
    factor_dir = STORE_PATH / "factors"
    if factor_dir.exists():
        factor_files = list(factor_dir.glob("*.parquet"))
        logger.info(f"factors | 文件 {len(factor_files):4d} （仅统计，未做内容校验）")

    logger.info("-" * 60)
    if grand_errs:
        logger.warning(f"共发现 {len(grand_errs)} 处异常（{grand_total - grand_passed}/{grand_total} 文件）")
        for e in grand_errs:
            logger.warning(e)
    else:
        logger.success(f"全部通过 ({grand_total} 文件)")

    logger.info("=" * 60)
    return 1 if grand_errs else 0


if __name__ == "__main__":
    sys.exit(main())
