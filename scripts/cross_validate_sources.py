#!/usr/bin/env python
"""多数据源交叉验证：用相互独立的源核对同一类数据，输出差异报告。

对比项（每项相互独立的两到三个源）：

  calendar    交易日历      本地缓存 vs tushare trade_cal vs akshare 新浪
  factor      复权因子      gbbq 自算(本地) vs tushare adj_factor vs 外部源缓存(factors/)
                            —— 三者绝对刻度不同，统一归一后比**比值**（hfq 收益等价性）
  exdiv       除权除息事件  通达信 gbbq(本地) vs tushare dividend（注：dividend 无配股，
                            gbbq 独有的配股事件单独计数不算差异）
  rename      更名历史      通达信 profile.dat(本地快照) vs tushare namechange
                            —— 按月末点位名逐点对比（点位 ST 判定的正确性基础）
  suspend     停牌名单      东方财富 stock_tfp_em vs tushare suspend_d（最近一个交易日）
  delist      退市清单      本地 tushare stock_basic(D) vs akshare 沪深退市表

用法（务必绕过代理，均为境内 API）：
    NO_PROXY=* python scripts/cross_validate_sources.py                    # 默认样本股
    NO_PROXY=* python scripts/cross_validate_sources.py --codes "600519.SH 000001.SZ"
    NO_PROXY=* python scripts/cross_validate_sources.py --checks factor,exdiv
    NO_PROXY=* python scripts/cross_validate_sources.py --out reports/xxx.md

token 从 .env / 环境变量取（TUSHARE_TOKEN*，同框架约定）；无 token 时仅跑不需要
tushare 的对比并给出提示。退出码：发现"疑似问题级"差异 → 1，否则 0。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.gbbq import GbbqStore  # noqa: E402
from src.data.profile import ProfileStore  # noqa: E402
from src.data.sources.tushare_pool import TusharePool  # noqa: E402
from src.data.suspend import SuspendProvider  # noqa: E402
from src.data.universe import is_st_name  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.helpers import format_code  # noqa: E402

#: 默认样本：主板/创业板/科创板/北交所、高分红、多次更名、有配股史、已退市
DEFAULT_CODES = [
    "600519.SH",  # 茅台：高分红
    "000001.SZ",  # 平安银行：多次更名（深发展A→…）
    "000002.SZ",  # 万科：早期配股
    "600000.SH",  # 浦发银行：送转+配股
    "600036.SH",  # 招商银行
    "000651.SZ",  # 格力电器：高分红
    "300750.SZ",  # 宁德时代：创业板
    "688981.SH",  # 中芯国际：科创板
    "430047.BJ",  # 诺思兰德：北交所
    "000418.SZ",  # 小天鹅A：已退市（吸收合并）
]

#: 因子比值相对偏差阈值：超过视为"疑似问题"（复权收益会被显著扭曲）
FACTOR_REL_TOL = 5e-3

#: 除权事件数值容差（每 10 股口径的元/股数）
EXDIV_VALUE_TOL = 1e-4

#: 逐次 tushare 调用间隔（秒），防限频
SLEEP = 0.35

ALL_CHECKS = ("calendar", "factor", "exdiv", "rename", "suspend", "delist")


class Report:
    """收集各项结果，输出 console + markdown。issue 级差异使退出码为 1。"""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.issues = 0

    def section(self, title: str) -> None:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        self.lines += [f"\n## {title}\n"]

    def info(self, msg: str) -> None:
        print(msg)
        self.lines.append(msg)

    def issue(self, msg: str) -> None:
        self.issues += 1
        print(f"[疑似问题] {msg}")
        self.lines.append(f"**[疑似问题]** {msg}")

    def save(self, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# 数据源交叉验证报告\n\n生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M} | "
            f"疑似问题: {self.issues}\n"
        )
        out.write_text(header + "\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\n报告已写入 {out}（疑似问题 {self.issues} 项）")


# ---------------------------------------------------------------------------
# 各对比项
# ---------------------------------------------------------------------------

def check_calendar(store: Path, pool: TusharePool | None, rep: Report) -> None:
    rep.section("calendar 交易日历：本地 vs tushare vs akshare")
    today = pd.Timestamp.today().normalize()
    lo = pd.Timestamp("2005-01-01")

    sets: dict[str, set[pd.Timestamp]] = {}
    local_file = store / "calendar.parquet"
    if local_file.exists():
        sets["local"] = set(pd.to_datetime(pd.read_parquet(local_file)["date"]))
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        sets["akshare"] = set(pd.to_datetime(df["trade_date"]))
    except Exception as exc:
        rep.info(f"akshare 日历拉取失败，跳过该源: {exc}")
    if pool is not None:
        df = pool.call(
            "trade_cal", exchange="SSE", is_open="1",
            start_date="19901201", end_date=f"{today.year}1231",
        )
        sets["tushare"] = set(pd.to_datetime(df["cal_date"]))
        time.sleep(SLEEP)

    if len(sets) < 2:
        rep.info("可用源不足两个，无法对比。")
        return
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # 只比历史段（未来日历各源发布进度不同，单独提示）
            sa = {d for d in sets[a] if lo <= d <= today}
            sb = {d for d in sets[b] if lo <= d <= today}
            only_a, only_b = sorted(sa - sb), sorted(sb - sa)
            msg = (f"{a} vs {b}（2005-01-01~今天）: {a} 独有 {len(only_a)}，"
                   f"{b} 独有 {len(only_b)}")
            if only_a or only_b:
                examples = [d.date().isoformat() for d in (only_a + only_b)[:5]]
                rep.issue(f"{msg}，示例 {examples}")
            else:
                rep.info(f"{msg} —— 一致 ✓")
    for name, s in sets.items():
        future = len([d for d in s if d > today])
        rep.info(f"{name} 未来日历天数: {future}")


def _daily_factor_on_grid(factor: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series | None:
    """稀疏/逐日因子表 → 在 grid 上 as-of 前向取值并归一（首格 = 1.0）。"""
    if factor is None or factor.empty:
        return None
    f = factor.copy()
    f["date"] = pd.to_datetime(f["date"])
    s = f.set_index("date")["cum_factor"].sort_index()
    aligned = s.reindex(grid.union(s.index)).ffill().reindex(grid).dropna()
    if aligned.empty or aligned.iloc[0] <= 0:
        return None
    return aligned / aligned.iloc[0]


def check_factor(store: Path, pool: TusharePool | None, codes: list[str], rep: Report) -> None:
    rep.section("factor 复权因子：gbbq 自算 vs tushare adj_factor vs 外部源缓存")
    if pool is None:
        rep.info("无 tushare token，仅对比本地两源（gbbq vs factors/）。")
    worst: tuple[float, str, str] = (0.0, "", "")
    for code in codes:
        tables: dict[str, pd.DataFrame] = {}
        for name, path in (
            ("gbbq", store / "factors_gbbq" / f"{code}.parquet"),
            ("extern", store / "factors" / f"{code}.parquet"),
        ):
            if path.exists():
                tables[name] = pd.read_parquet(path)
        if pool is not None:
            df = pool.call("adj_factor", ts_code=code)
            time.sleep(SLEEP)
            if df is not None and not df.empty:
                tables["tushare"] = pd.DataFrame({
                    "date": pd.to_datetime(df["trade_date"]),
                    "cum_factor": pd.to_numeric(df["adj_factor"], errors="coerce"),
                }).dropna()
        if len(tables) < 2:
            rep.info(f"{code}: 可用因子源 <2（{sorted(tables)}），跳过")
            continue
        # 公共日网格：优先用 tushare 的逐日交易日，否则用两稀疏表日期并集
        if "tushare" in tables:
            grid_src = pd.to_datetime(tables["tushare"]["date"])
        else:
            grid_src = pd.concat([pd.to_datetime(t["date"]) for t in tables.values()])
        names = sorted(tables)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                lo = max(pd.to_datetime(tables[a]["date"]).min(),
                         pd.to_datetime(tables[b]["date"]).min())
                grid = pd.DatetimeIndex(sorted({d for d in grid_src if d >= lo}))
                sa, sb = (_daily_factor_on_grid(tables[x], grid) for x in (a, b))
                if sa is None or sb is None or len(sa) < 2:
                    rep.info(f"{code} {a} vs {b}: 重叠不足，跳过")
                    continue
                common = sa.index.intersection(sb.index)
                rel = (sa[common] / sb[common] - 1).abs()
                mx, mx_d = float(rel.max()), rel.idxmax()
                fin = float(rel.iloc[-1])  # 末日残差：>tol 说明存在未收敛的刻度差（缺事件）
                line = (f"{code} {a} vs {b}: 重叠 {len(common)} 日，"
                        f"最大偏差 {mx:.2e}（@{mx_d.date()}），末日残差 {fin:.2e}")
                if fin > FACTOR_REL_TOL:
                    rep.issue(line + " —— 持续性刻度差（疑似缺除权事件）")
                elif mx > FACTOR_REL_TOL:
                    rep.info(line + " —— 瞬时差（疑似除权日错位），已收敛")
                else:
                    rep.info(line + " ✓")
                if mx > worst[0]:
                    worst = (mx, code, f"{a} vs {b}")
    if worst[1]:
        rep.info(f"\n全部样本最大偏差: {worst[0]:.2e}（{worst[1]}, {worst[2]}）")


def check_exdiv(store: Path, gbbq: GbbqStore, pool: TusharePool | None,
                codes: list[str], rep: Report) -> None:
    rep.section("exdiv 除权除息事件：通达信 gbbq vs tushare dividend")
    if pool is None:
        rep.info("无 tushare token，跳过。")
        return
    if not gbbq.available:
        rep.info("gbbq 不可用（无通达信且无事件快照），跳过。")
        return
    n_match = n_val_diff = n_gbbq_only = n_ts_only = n_rights = 0
    for code in codes:
        ev = gbbq.events(code)
        div = pool.call(
            "dividend", ts_code=code,
            fields="ts_code,end_date,div_proc,stk_div,cash_div,cash_div_tax,ex_date",
        )
        time.sleep(SLEEP)
        if div is None:
            div = pd.DataFrame()
        if not div.empty:
            div = div[(div.get("div_proc") == "实施") & div["ex_date"].notna()]
            # 同一笔分红（同 end_date+ex_date）会因公告日不同出现多条"实施"，去重防翻倍；
            # 不同 end_date 同 ex_date（年度+特别分红同日除权）是真叠加，保留累加
            div = div.drop_duplicates(subset=["end_date", "ex_date"], keep="last")
        ts_map: dict[pd.Timestamp, tuple[float, float]] = {}
        for _, r in div.iterrows():
            d = pd.to_datetime(str(r["ex_date"]))
            cash = pd.to_numeric(r.get("cash_div_tax"), errors="coerce")
            if pd.isna(cash) or cash == 0:
                cash = pd.to_numeric(r.get("cash_div"), errors="coerce")
            song = pd.to_numeric(r.get("stk_div"), errors="coerce")
            prev = ts_map.get(d, (0.0, 0.0))  # 同日多条（罕见）累加
            ts_map[d] = (
                prev[0] + (0.0 if pd.isna(cash) else float(cash) * 10),
                prev[1] + (0.0 if pd.isna(song) else float(song) * 10),
            )
        gb_map: dict[pd.Timestamp, tuple[float, float, float]] = {}
        if ev is not None and not ev.empty:
            for _, r in ev.iterrows():
                d = pd.Timestamp(r["date"]).normalize()
                gb_map[d] = (float(r["fenhong"]), float(r["song"]), float(r["pei"]))
        for d, (fh, sg, pei) in sorted(gb_map.items()):
            if pei > 0 and d not in ts_map:
                n_rights += 1  # 纯配股事件：dividend 接口不覆盖，不算差异
                continue
            if d not in ts_map:
                n_gbbq_only += 1
                rep.info(f"{code} {d.date()}: 仅 gbbq 有（分红{fh} 送转{sg}）")
                continue
            tfh, tsg = ts_map[d]
            if abs(fh - tfh) > EXDIV_VALUE_TOL or abs(sg - tsg) > EXDIV_VALUE_TOL:
                n_val_diff += 1
                rep.issue(f"{code} {d.date()}: 数值不一致 gbbq(分红{fh},送转{sg}) "
                          f"vs tushare(分红{tfh:g},送转{tsg:g})")
            else:
                n_match += 1
        for d in sorted(set(ts_map) - set(gb_map)):
            n_ts_only += 1
            rep.info(f"{code} {d.date()}: 仅 tushare 有（分红{ts_map[d][0]:g} 送转{ts_map[d][1]:g}）")
    rep.info(f"\n汇总: 一致 {n_match} | 数值不一致 {n_val_diff} | 仅 gbbq {n_gbbq_only} "
             f"| 仅 tushare {n_ts_only} | 配股(不参与对比) {n_rights}")
    if n_gbbq_only + n_ts_only > n_match * 0.1:
        rep.issue(f"事件缺失比例偏高（仅单源事件 {n_gbbq_only + n_ts_only} vs 一致 {n_match}）")


def check_rename(store: Path, profile: ProfileStore, pool: TusharePool | None,
                 codes: list[str], rep: Report) -> None:
    rep.section("rename 更名历史：通达信 profile.dat vs tushare namechange（月末点位名）")
    if pool is None:
        rep.info("无 tushare token，跳过。")
        return
    if not profile.available:
        rep.info("profile 不可用（无通达信且无更名快照），跳过。")
        return
    basic_file = store / "stock_basic_tushare.parquet"
    cur_names: dict[str, str] = {}
    if basic_file.exists():
        sb = pd.read_parquet(basic_file)
        cur_names = dict(zip(sb["ts_code"], sb["name"]))
    month_ends = pd.date_range("2000-01-31", pd.Timestamp.today(), freq="ME")
    n_cmp = n_diff = 0
    for code in codes:
        nc = pool.call("namechange", ts_code=code, fields="ts_code,name,start_date")
        time.sleep(SLEEP)
        if nc is None or nc.empty:
            rep.info(f"{code}: tushare namechange 无记录，跳过")
            continue
        nc = nc.dropna(subset=["start_date"]).copy()
        nc["start"] = pd.to_datetime(nc["start_date"].astype(str))
        nc = nc.sort_values("start").reset_index(drop=True)

        def ts_name_at(d: pd.Timestamp) -> str | None:
            rows = nc[nc["start"] <= d]
            return None if rows.empty else str(rows.iloc[-1]["name"])

        # profile.dat 无该股记录时（从未更名或记录被清除，实测退市股覆盖率 0%），
        # 点位名退化为"当前名近似"——继续逐点对比，由 ST 语义分级判断影响
        if profile.former_names(code).empty:
            rep.info(f"{code}: profile.dat 无更名记录，点位名按当前名近似"
                     f"（当前名 {cur_names.get(code)!r}）")

        diffs_st, diffs_cosmetic = [], []
        for d in month_ends:
            t = ts_name_at(d)
            if t is None:
                continue  # tushare 覆盖不到的早期，无法对比
            p = profile.name_at(code, d)
            if p is None:
                p = cur_names.get(code)  # 晚于所有更名日 → 当前名兜底
            if p is None:
                continue
            n_cmp += 1
            # 名称清洗：全角/空格差异不算不一致
            if str(p).replace(" ", "").upper() != str(t).replace(" ", "").upper():
                n_diff += 1
                item = f"{d.date()}: profile={p!r} tushare={t!r}"
                # 只有两源的 ST/退 判定结果不同才是问题级（影响点位 ST 过滤）；
                # G股改前缀、-U/-W 等命名细节差异仅提示
                if is_st_name(p) != is_st_name(t):
                    diffs_st.append(item)
                else:
                    diffs_cosmetic.append(item)
        if diffs_st:
            rep.issue(f"{code}: {len(diffs_st)} 个点位 ST 判定不一致，示例 {diffs_st[:3]}")
        if diffs_cosmetic:
            rep.info(f"{code}: {len(diffs_cosmetic)} 个点位命名细节差异（不影响 ST 判定），"
                     f"示例 {diffs_cosmetic[:2]}")
        if not diffs_st and not diffs_cosmetic:
            rep.info(f"{code}: 点位名全部一致 ✓")
    rep.info(f"\n汇总: 对比 {n_cmp} 个点位，不一致 {n_diff}")


def check_suspend(pool: TusharePool | None, rep: Report) -> None:
    rep.section("suspend 停牌名单：东方财富 vs tushare（最近一个交易日）")
    if pool is None:
        rep.info("无 tushare token，跳过。")
        return
    today = pd.Timestamp.today()
    cal = pool.call("trade_cal", exchange="SSE", is_open="1",
                    start_date=(today - pd.Timedelta(days=15)).strftime("%Y%m%d"),
                    end_date=today.strftime("%Y%m%d"))
    time.sleep(SLEEP)
    d = pd.to_datetime(sorted(cal["cal_date"].astype(str))[-1]).date()
    rep.info(f"对比交易日: {d}")
    sets: dict[str, frozenset[str]] = {}
    with tempfile.TemporaryDirectory() as tmp:  # 独立缓存目录，不污染真仓库
        for src in ("eastmoney", "tushare"):
            provider = SuspendProvider(store_path=Path(tmp) / src, sources=(src,),
                                       lookback_days=30)
            try:
                sets[src] = provider.get_suspended_set(d)
            except Exception as exc:
                rep.info(f"{src} 拉取失败: {exc}")
    if len(sets) < 2:
        rep.info("可用源不足两个,无法对比。")
        return
    em, ts = sets["eastmoney"], sets["tushare"]
    only_em, only_ts = sorted(em - ts), sorted(ts - em)
    rep.info(f"东财 {len(em)} 只 | tushare {len(ts)} 只 | 交集 {len(em & ts)}")
    if only_em or only_ts:
        rep.info(f"仅东财: {only_em[:10]}")
        rep.info(f"仅 tushare: {only_ts[:10]}")
        # 两源口径略有差异（盘中临停 vs 全日停牌），少量差异仅提示不算问题
        if len(only_em) + len(only_ts) > max(5, 0.5 * max(len(em), len(ts))):
            rep.issue("停牌名单两源差异过半，需排查口径")
    else:
        rep.info("完全一致 ✓")


def check_delist(store: Path, rep: Report) -> None:
    rep.section("delist 退市清单：本地 tushare stock_basic(D) vs akshare 沪深退市表")
    basic_file = store / "stock_basic_tushare.parquet"
    if not basic_file.exists():
        rep.info("本地 stock_basic_tushare.parquet 缺失，跳过。")
        return
    sb = pd.read_parquet(basic_file)
    ts_delist = {c for c in sb.loc[sb["list_status"] == "D", "ts_code"]
                 if c.endswith((".SH", ".SZ"))}
    ak_delist: set[str] = set()
    try:
        import akshare as ak
        for fn, kw in (("stock_info_sh_delist", {}), ("stock_info_sz_delist", {})):
            df = getattr(ak, fn)(**kw)
            col = next((c for c in df.columns if "代码" in str(c)), None)
            if col is None:
                continue
            for raw in df[col]:
                try:
                    ak_delist.add(format_code(str(raw).strip()))
                except Exception:
                    pass
    except Exception as exc:
        rep.info(f"akshare 退市表拉取失败，跳过: {exc}")
        return
    only_ts, only_ak = sorted(ts_delist - ak_delist), sorted(ak_delist - ts_delist)
    # B 股（200xxx.SZ / 900xxx.SH）不在 A 股框架范围内，从缺口统计里剔除
    only_ak_a = [c for c in only_ak if not (c.startswith("200") or c.startswith("900"))]
    # 退市整理期股票：tushare 尚标 L（名"退市XX"）、akshare 已入退市表——时点口径差，非缺口
    still_l = set(sb.loc[sb["list_status"] == "L", "ts_code"])
    transitional = [c for c in only_ak_a if c in still_l]
    missing = [c for c in only_ak_a if c not in set(sb["ts_code"])]
    rep.info(f"tushare(沪深) {len(ts_delist)} | akshare {len(ak_delist)} | "
             f"交集 {len(ts_delist & ak_delist)}")
    rep.info(f"仅 tushare: {len(only_ts)}（示例 {only_ts[:5]}）")
    rep.info(f"仅 akshare: {len(only_ak)}，其中 A 股 {len(only_ak_a)}"
             f"（B 股 {len(only_ak) - len(only_ak_a)} 只不在框架范围）")
    if transitional:
        rep.info(f"  退市整理期口径差（tushare 仍标 L）: {len(transitional)} 只 {transitional[:6]}")
    if missing:
        rep.issue(f"本地完全缺失的 A 股退市股 {len(missing)} 只: {missing[:10]}，"
                  f"防幸存者偏差需补齐")
    if not missing:
        rep.info("无真实缺口 ✓")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="多数据源交叉验证")
    ap.add_argument("--codes", default=None, help="空格/逗号分隔样本代码；缺省用内置样本")
    ap.add_argument("--checks", default=",".join(ALL_CHECKS),
                    help=f"要跑的对比项，逗号分隔（默认全部：{','.join(ALL_CHECKS)}）")
    ap.add_argument("--out", default=None, help="markdown 报告输出路径（默认 reports/ 下按日期）")
    args = ap.parse_args()

    cfg = load_config()
    store = Path(cfg.data.store_path)
    codes = ([format_code(c) for c in args.codes.replace(",", " ").split()]
             if args.codes else list(DEFAULT_CODES))
    checks = {c.strip() for c in args.checks.split(",") if c.strip()}
    unknown = checks - set(ALL_CHECKS)
    if unknown:
        ap.error(f"未知对比项: {sorted(unknown)}")

    pool = TusharePool()
    pool_or_none = pool if pool.available else None
    if pool_or_none is None:
        print("警告: 未配置 TUSHARE_TOKEN，只跑不需要 tushare 的对比。")

    tdx_path = str(cfg.data.tdx_path or "")
    gbbq = GbbqStore(tdx_path=tdx_path, snapshot_path=store / "gbbq_events.parquet")
    profile = ProfileStore(tdx_path=tdx_path, snapshot_path=store / "profile_names.parquet")

    rep = Report()
    rep.info(f"store = {store} | 样本 {len(codes)} 只: {' '.join(codes)}")
    runners = {
        "calendar": lambda: check_calendar(store, pool_or_none, rep),
        "factor": lambda: check_factor(store, pool_or_none, codes, rep),
        "exdiv": lambda: check_exdiv(store, gbbq, pool_or_none, codes, rep),
        "rename": lambda: check_rename(store, profile, pool_or_none, codes, rep),
        "suspend": lambda: check_suspend(pool_or_none, rep),
        "delist": lambda: check_delist(store, rep),
    }
    for name in ALL_CHECKS:
        if name not in checks:
            continue
        try:
            runners[name]()
        except Exception as exc:  # 单项失败不中断其余对比
            rep.issue(f"{name} 对比运行异常: {exc!r}")

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "reports"
        / f"cross_validation_{date.today():%Y%m%d}.md")
    rep.save(out)
    return 1 if rep.issues else 0


if __name__ == "__main__":
    sys.exit(main())
