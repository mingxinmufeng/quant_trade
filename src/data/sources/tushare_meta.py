"""
tushare 元数据补充源：除权事件 / 更名历史 的"补洞"刷新
========================================================

**背景（交叉验证实测，见 ``scripts/cross_validate_sources.py`` 报告）**：通达信会
清除退市股的本地记录——退市股在 gbbq 里仅 ~11% 有除权事件（在市股 98%），
profile.dat 更名史覆盖率 0%。而本地日线冻结档保留全部退市股参与回测，导致：

1. gbbq 自算复权因子对有分红史的退市股静默算成"从不除权"（实测偏差可达 2 倍+）；
2. 点位 ST 判定退化为"当前名近似"，退市股当前名带"退"→ 整个历史被误判剔除。

防幸存者偏差的两块基石恰恰对退市股失效。本模块用 tushare 官方接口**按股票补洞**：

- ``refresh_dividend_supplement``   tushare ``dividend`` → ``{store}/gbbq_events_tushare.parquet``
  （schema 同 ``GbbqStore.events()`` 输出 + ``code`` 列，每 10 股口径；**不含配股**——
  dividend 接口无配股数据，退市股的配股事件仍缺，属已知残余误差）
- ``refresh_namechange_supplement`` tushare ``namechange`` → ``{store}/profile_names_tushare.parquet``
  （schema 同 ``PROFILE_COLUMNS``：(code, 原名称, 更名日)；末名以退市日+1 封口，
  使退市股在市期间的点位名完整可查）

``GbbqStore`` / ``ProfileStore`` 通过 ``supplement_path`` 参数按股票合并（TDX 有记录
用 TDX，无记录用补充表）。``update_tushare_supplements`` 供 ``fetch`` 命令挂钩：
无 token / 无权限 / 异常一律告警跳过，不影响主流程。
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from .tushare_pool import TusharePool

#: 补充事件表 / 已检查名单 / 补充更名表 文件名（相对 store 根）
DIVIDEND_SUPPLEMENT_FILE = "gbbq_events_tushare.parquet"
DIVIDEND_CHECKED_FILE = "gbbq_events_tushare_checked.parquet"
NAMECHANGE_SUPPLEMENT_FILE = "profile_names_tushare.parquet"

#: 补充事件表 schema（与 GbbqStore.events() 输出一致 + code；每 10 股口径）
SUPPLEMENT_EVENT_COLUMNS = ("code", "date", "fenhong", "peijia", "song", "pei")

#: 逐次请求间隔（秒）
THROTTLE_SECONDS = 0.35

#: namechange 快照的刷新周期（天）；未过期不重拉
NAMECHANGE_REFRESH_DAYS = 7

#: namechange 全市场分页步长
_NC_PAGE = 5000

#: 权限/积分类错误文本特征（命中→整轮放弃，由 update_tushare_supplements 降级）
_PERMISSION_HINTS = ("权限", "积分")


class TushareMetaPermissionError(RuntimeError):
    """tushare 账号无 dividend/namechange 权限（积分不足）。"""


def _classify(exc: Exception) -> Exception:
    if any(h in str(exc) for h in _PERMISSION_HINTS):
        return TushareMetaPermissionError(str(exc))
    return exc


# ---------------------------------------------------------------------------
# 除权事件补充（dividend）
# ---------------------------------------------------------------------------

def dividend_to_events(div: pd.DataFrame) -> pd.DataFrame:
    """tushare ``dividend`` 原始表 → 每 10 股口径事件表 ``[date, fenhong, peijia, song, pei]``。

    规则（与交叉验证脚本一致）：
    - 仅 ``div_proc == 实施`` 且有 ``ex_date`` 的行；
    - 同一笔分红（同 end_date+ex_date）因公告日不同出现多条时去重防翻倍；
    - 不同 end_date 同 ex_date（年度+特别分红同日除权）真叠加，按 ex_date 累加；
    - 现金分红取税前 ``cash_div_tax``（缺失回退 ``cash_div``）；送转 ``stk_div``；
    - dividend 接口**无配股数据**，pei/peijia 恒为 0。
    """
    empty = pd.DataFrame(columns=["date", "fenhong", "peijia", "song", "pei"])
    if div is None or div.empty:
        return empty
    d = div[(div.get("div_proc") == "实施") & div["ex_date"].notna()].copy()
    if d.empty:
        return empty
    d = d.drop_duplicates(subset=["end_date", "ex_date"], keep="last")
    d["date"] = pd.to_datetime(d["ex_date"].astype(str), errors="coerce")
    cash = pd.to_numeric(d.get("cash_div_tax"), errors="coerce")
    cash = cash.where(cash.notna() & (cash != 0), pd.to_numeric(d.get("cash_div"), errors="coerce"))
    d["fenhong"] = cash.fillna(0.0) * 10.0
    d["song"] = pd.to_numeric(d.get("stk_div"), errors="coerce").fillna(0.0) * 10.0
    out = (
        d.dropna(subset=["date"])
        .groupby("date", as_index=False)[["fenhong", "song"]].sum()
    )
    out["peijia"] = 0.0
    out["pei"] = 0.0
    return out[["date", "fenhong", "peijia", "song", "pei"]].sort_values("date").reset_index(drop=True)


def _codes_needing_dividend(store: Path) -> list[str]:
    """待补洞代码 = 本地日线全部代码 −（gbbq 快照里有除权事件的） −（已检查过的）。"""
    daily_dir = store / "daily"
    codes = {p.stem for p in daily_dir.glob("*.parquet")} if daily_dir.is_dir() else set()
    # 指数代码（000300.SH 等与个股同后缀无法靠代码区分）留在集合中无妨：dividend 返回空，
    # 检查一次后进入 checked 名单不再重复请求
    snap = store / "gbbq_events.parquet"
    if snap.exists():
        try:
            ev = pd.read_parquet(snap, columns=["market", "code", "category"])
            cat1 = ev[ev["category"] == 1]
            suffix = {0: ".SZ", 1: ".SH", 2: ".BJ"}
            covered = {
                f"{str(c).strip()[:6]}{suffix.get(int(m), '')}"
                for m, c in zip(cat1["market"], cat1["code"], strict=True)
            }
            codes -= covered
        except Exception as exc:
            logger.warning(f"读取 gbbq 事件快照失败（按全部代码补洞）: {exc}")
    checked_file = store / DIVIDEND_CHECKED_FILE
    if checked_file.exists():
        try:
            codes -= set(pd.read_parquet(checked_file)["code"].astype(str))
        except Exception:
            pass
    return sorted(codes)


def refresh_dividend_supplement(
    store: str | Path,
    pool,
    codes: list[str] | None = None,
) -> list[str]:
    """对缺除权事件的代码逐只拉取 tushare ``dividend``，并入补充事件表。

    幂等增量：已检查过的代码（含"确无分红"的空结果）记入 checked 名单不再重复请求；
    新退市/新增代码下轮自动进入待补集合。

    Returns: 本轮**新增了事件**的代码列表（调用方据此重算 gbbq 自算因子）。
    Raises: TushareMetaPermissionError（账号无权限，调用方整体降级）。
    """
    store = Path(store)
    todo = codes if codes is not None else _codes_needing_dividend(store)
    if not todo:
        logger.debug("gbbq 补充事件表已是最新（无待补代码）")
        return []
    logger.info(f"gbbq 补充事件：待检查 {len(todo)} 只（TDX 无除权记录的代码）")

    supp_file = store / DIVIDEND_SUPPLEMENT_FILE
    old = pd.read_parquet(supp_file) if supp_file.exists() else pd.DataFrame(
        columns=list(SUPPLEMENT_EVENT_COLUMNS))
    frames, checked, new_codes = [old] if not old.empty else [], [], []
    n_new = 0
    for i, code in enumerate(todo, 1):
        try:
            div = pool.call(
                "dividend", ts_code=code,
                fields="ts_code,end_date,ann_date,div_proc,stk_div,cash_div,cash_div_tax,ex_date",
            )
        except Exception as exc:
            exc = _classify(exc)
            if isinstance(exc, TushareMetaPermissionError):
                raise exc from None
            logger.warning(f"  dividend {code} 拉取失败（下轮重试）: {exc}")
            continue
        ev = dividend_to_events(div)
        checked.append(code)
        if not ev.empty:
            ev.insert(0, "code", code)
            frames.append(ev)
            n_new += len(ev)
            new_codes.append(code)
        if i % 100 == 0 or i == len(todo):
            logger.info(f"  dividend 进度 {i}/{len(todo)}（新增事件 {n_new} 行）")
        time.sleep(THROTTLE_SECONDS)

    if n_new:
        merged = pd.concat(frames, ignore_index=True)
        merged = (
            merged.drop_duplicates(subset=["code", "date"], keep="last")
            .sort_values(["code", "date"])
            .reset_index(drop=True)
        )
        merged.to_parquet(supp_file, index=False, compression="snappy")
        logger.info(f"gbbq 补充事件表已更新: {len(merged)} 行 | {supp_file}")
    if checked:
        checked_file = store / DIVIDEND_CHECKED_FILE
        old_checked = (
            pd.read_parquet(checked_file) if checked_file.exists()
            else pd.DataFrame(columns=["code", "checked_date"])
        )
        add = pd.DataFrame({"code": checked,
                            "checked_date": date.today().strftime("%Y%m%d")})
        pd.concat([old_checked, add], ignore_index=True).drop_duplicates(
            subset=["code"], keep="last"
        ).to_parquet(checked_file, index=False, compression="snappy")
    return new_codes


def recompute_gbbq_factors(store: str | Path, codes: list[str]) -> int:
    """补充事件生效后，重算指定代码的 gbbq 自算因子并覆盖 ``factors_gbbq/``。

    背景：``factors_gbbq/{code}.parquet`` 是落盘并存记录，load 的 gbbq 口径优先用它；
    退市股此前按"无事件"算出恒 1.0 的陈旧因子，补洞后必须重算。文件 mtime 刷新会
    使下游复权缓存按版本戳自动重建。

    Returns: 实际重写的股票数。
    """
    # 局部导入避免 sources 子包与 data 包的环形依赖
    from ..factors import FactorCalculator
    from ..gbbq import GbbqStore
    from ..storage import DataStore

    store = Path(store)
    ds = DataStore(store)
    gbbq = GbbqStore(
        snapshot_path=store / "gbbq_events.parquet",
        supplement_path=store / DIVIDEND_SUPPLEMENT_FILE,
    )
    calc = FactorCalculator(gbbq, raw_close_loader=lambda c: ds.read_raw(c, "daily"))
    n = 0
    for code in codes:
        try:
            factor = calc.get_factor(code)
        except Exception as exc:
            logger.warning(f"  [{code}] gbbq 自算因子重算失败: {exc}")
            continue
        if factor is not None and not factor.empty:
            ds.write_factor(code, factor, gbbq=True)
            n += 1
    if n:
        logger.info(f"已重算 {n} 只补洞股票的 gbbq 自算因子（factors_gbbq/）")
    return n


# ---------------------------------------------------------------------------
# 更名历史补充（namechange）
# ---------------------------------------------------------------------------

def namechange_to_profile(nc: pd.DataFrame, delist_dates: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """tushare ``namechange``（name, start_date=生效日）→ profile 口径（原名称, 更名日）。

    语义换算：第 i 条名称的"更名日"= 第 i+1 条的生效日（该名称生效至更名日**不含**）。
    最后一条（当前名/退市时末名）通常不入表（与 profile.dat 一致，由 stock_basic 兜底）；
    但**退市股**的末名以 ``退市日 + 1 天`` 封口入表——否则其在市期间点位名会退化为
    带"(退)"的当前名，被点位 ST 过滤整段误杀。
    """
    empty = pd.DataFrame(columns=["code", "name", "change_date"])
    if nc is None or nc.empty:
        return empty
    d = nc.dropna(subset=["start_date"]).copy()
    if d.empty:
        return empty
    d["start"] = pd.to_datetime(d["start_date"].astype(str), errors="coerce")
    d = d.dropna(subset=["start"]).sort_values(["ts_code", "start"])
    # namechange 实测同一记录会重复多行（公告重发）；同 (股票, 生效日) 只留一条，
    # 否则相邻配对会把"旧名的更名日"错配成它自己的生效日
    d = d.drop_duplicates(subset=["ts_code", "start"], keep="last")
    rows: list[tuple[str, str, pd.Timestamp]] = []
    for code, grp in d.groupby("ts_code"):
        names = list(zip(grp["name"].astype(str), grp["start"], strict=True))
        for (name, _), (_, nxt_start) in zip(names, names[1:], strict=False):
            rows.append((str(code), name, nxt_start))
        delist = delist_dates.get(str(code))
        if delist is not None and pd.notna(delist):
            last_name = names[-1][0]
            rows.append((str(code), last_name, delist + pd.Timedelta(days=1)))
    if not rows:
        return empty
    out = pd.DataFrame(rows, columns=["code", "name", "change_date"])
    return (
        out.drop_duplicates(subset=["code", "change_date"], keep="last")
        .sort_values(["code", "change_date"])
        .reset_index(drop=True)
    )


def _load_delist_dates(store: Path) -> dict[str, pd.Timestamp]:
    """{ts_code: 退市日} 来自本地 tushare 基础信息表；缺失返回空字典。"""
    f = store / "stock_basic_tushare.parquet"
    if not f.exists():
        return {}
    try:
        sb = pd.read_parquet(f, columns=["ts_code", "delist_date"])
    except Exception:
        return {}
    d = pd.to_datetime(sb["delist_date"].astype("string"), format="%Y%m%d", errors="coerce")
    return {c: t for c, t in zip(sb["ts_code"], d, strict=True) if pd.notna(t)}


def refresh_namechange_supplement(
    store: str | Path,
    pool,
    refresh_days: int = NAMECHANGE_REFRESH_DAYS,
    force: bool = False,
) -> int:
    """全市场拉取 tushare ``namechange`` 并转为 profile 口径补充表（mtime 周期刷新）。

    Returns: 补充表行数（未到刷新周期时返回 -1 表示跳过）。
    Raises: TushareMetaPermissionError（账号无权限，调用方整体降级）。
    """
    store = Path(store)
    out_file = store / NAMECHANGE_SUPPLEMENT_FILE
    if not force and out_file.exists():
        age_days = (time.time() - out_file.stat().st_mtime) / 86400.0
        if age_days <= refresh_days:
            logger.debug(f"profile 补充更名表未过期（{age_days:.1f}d ≤ {refresh_days}d），跳过")
            return -1

    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        try:
            page = pool.call(
                "namechange",
                fields="ts_code,name,start_date",
                limit=_NC_PAGE, offset=offset,
            )
        except Exception as exc:
            raise _classify(exc) from None
        if page is None or page.empty:
            break
        frames.append(page)
        if len(page) < _NC_PAGE:
            break
        offset += len(page)
        time.sleep(THROTTLE_SECONDS)
    nc = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    prof = namechange_to_profile(nc, _load_delist_dates(store))
    if prof.empty:
        logger.warning("tushare namechange 返回空，补充更名表未更新")
        return 0
    prof.to_parquet(out_file, index=False, compression="snappy")
    logger.info(f"profile 补充更名表已更新: {len(prof)} 行（{prof['code'].nunique()} 只）| {out_file}")
    return len(prof)


# ---------------------------------------------------------------------------
# fetch 挂钩入口
# ---------------------------------------------------------------------------

def update_tushare_supplements(store: str | Path, pool=None) -> bool:
    """刷新两张补充表（``fetch`` 命令挂钩）：**任何失败只告警，不上抛**。

    Returns: True = 本轮正常完成；False = 无 token / 无权限 / 异常而跳过。
    """
    pool = pool if pool is not None else TusharePool()
    if not getattr(pool, "available", True):
        logger.warning("未配置 TUSHARE_TOKEN，跳过 gbbq/profile 补充表刷新")
        return False
    try:
        new_codes = refresh_dividend_supplement(store, pool)
        if new_codes:
            recompute_gbbq_factors(store, new_codes)
        refresh_namechange_supplement(store, pool)
        return True
    except TushareMetaPermissionError as exc:
        logger.warning(f"tushare 账号无 dividend/namechange 权限，跳过补充表刷新: {exc}")
        return False
    except Exception as exc:
        logger.error(f"tushare 补充表刷新失败（不影响行情更新结果）: {exc}")
        return False
