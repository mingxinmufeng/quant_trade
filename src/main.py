"""
命令行入口（main）—— typer CLI
================================

把数据层 / 因子层 / 策略层 / 引擎层 / 风控层装配成可一键运行的命令：

    python -m src.main fetch      [--codes ...] [--freqs daily,min5,min1]  # 增量更新全市场行情
    python -m src.main backtest   --strategy ma_rsi --codes 000001.SZ,600519.SH \
                                  --start 2022-01-01 --end 2023-12-31
    python -m src.main optimize   --strategy ma_rsi --codes ... --trials 50   # Optuna 调参
    python -m src.main calendar   --date 2024-01-02                   # 交易日历查询
    python -m src.main version

通用选项：``--config`` 指定配置文件（默认按 config_loader 规则加载 config.yaml +
config.private.yaml + 环境变量）。私有策略用 ``--strategy-path`` 指向外部目录。
"""

from __future__ import annotations

from datetime import date

import typer
from loguru import logger

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A_Share_Quant_Pro 量化交易框架 CLI",
)


# ============================================================
# 公共装配
# ============================================================


def _load_cfg(config: str | None, store_path: str | None = None):
    from .utils.config_loader import apply_overrides, load_config
    from .utils.helpers import init_logging

    cfg = load_config(config_path=config)
    # --store-path 是最高优先级覆盖（高于环境变量 / config.private.yaml / config.yaml），
    # 供一次性指定本机数据仓库位置；缺省则按配置链解析。
    if store_path:
        cfg = apply_overrides(cfg, {"data.store_path": store_path})
    lg = cfg.get("logging", {}) if hasattr(cfg, "get") else {}
    init_logging(
        log_dir=lg.get("log_dir", "logs"),
        app_name=lg.get("app_name", "quant_pro"),
        level=lg.get("level", "INFO"),
        rotation=lg.get("rotation", "00:00"),
        retention=lg.get("retention", "30 days"),
    )
    return cfg


def _build_fetcher(cfg):
    from .data import DataFetcher

    d = cfg.get("data", {})
    return DataFetcher(
        store_path=d.get("store_path", "data_store"),
        sources=tuple(d.get("sources", ("pytdx", "akshare", "baostock", "tushare"))),
        retry_times=d.get("retry_times", 3),
        retry_delays=tuple(d.get("retry_delays", (1, 2, 4, 8))),
        tdx_path=d.get("tdx_path", "") or None,
        factor_source=d.get("factor_source", "sina"),
        load_factor_source=d.get("load_factor_source", "gbbq"),
        max_workers=d.get("max_workers", 4),
        suspend_sources=tuple(d.get("suspend_sources", ("eastmoney", "tushare"))),
        suspend_lookback_days=d.get("suspend_lookback_days", 30),
        suspend_enabled=d.get("suspend_enabled", True),
    )


def _parse_codes(codes: str | None) -> list[str]:
    """把 ``--codes`` 字符串切成代码列表，分隔符兼容空格 / 逗号 / 分号。"""
    if not codes:
        return []
    import re

    from .utils.helpers import format_code

    return [format_code(raw) for raw in re.split(r"[\s,;]+", codes.strip()) if raw]


def _benchmark_index_codes(cfg) -> list[str]:
    """需随全市场一并采集的指数代码：``backtest.benchmark`` ∪ ``data.index_codes``（去重保序）。"""
    from .utils.helpers import format_code

    raw = list(cfg.get("data", {}).get("index_codes", []) or [])
    bench = cfg.get("backtest", {}).get("benchmark")
    if bench:
        raw.append(bench)
    out: list[str] = []
    for c in raw:
        try:
            std = format_code(str(c))
        except Exception:
            continue
        if std not in out:
            out.append(std)
    return out


def _resolve_codes(cfg, codes: str | None, as_of: date, limit: int) -> list[str]:
    """优先用 --codes；否则用 Universe 在 as_of 时点取可交易股票（防幸存者偏差），截断 limit。"""
    parsed = _parse_codes(codes)
    if parsed:
        return parsed
    from .data import Universe

    logger.info(f"未指定 --codes，使用 Universe 在 {as_of} 的可交易股票（截断 {limit}）")
    uni = Universe.from_config(cfg)
    tradable = uni.get_tradable_stocks(as_of)
    if not tradable:
        raise typer.BadParameter("Universe 返回空且未提供 --codes，请显式指定股票")
    return tradable[:limit]


def _walk_forward_windows(days: list, folds: int) -> list[tuple]:
    """把交易日序列切成 ``folds`` 个 (训练, 测试) 滚动窗口（防过拟合的样本外验证）。

    方案：训练段**锚定起点、逐折扩张**（fold i 训练 [0, i)、测试紧邻的第 i 段）。
    把日序均分为 ``folds+1`` 段：第 1..folds 段轮流作测试段，其之前的全部历史作训练段，
    **训练段严格早于测试段**（无未来泄漏）。

    Returns:
        ``[(train_start, train_end, test_start, test_end), ...]``（元素为各窗口首尾交易日）。

    Raises:
        ValueError: ``folds < 1`` 或交易日不足以切分。
    """
    if folds < 1:
        raise ValueError(f"folds 必须 >= 1，got {folds}")
    n = len(days)
    if n < (folds + 1) * 2:
        raise ValueError(
            f"交易日仅 {n} 天，不足以切 {folds} 折 walk-forward（需 ≥ {(folds + 1) * 2} 天）"
        )
    chunk = n // (folds + 1)
    out: list[tuple] = []
    for i in range(1, folds + 1):
        train = days[: i * chunk]
        test = days[i * chunk:] if i == folds else days[i * chunk: (i + 1) * chunk]
        out.append((train[0], train[-1], test[0], test[-1]))
    return out


# ============================================================
# 命令：数据更新
# ============================================================


@app.command()
def fetch(
    codes: str | None = typer.Option(None, "--codes", help="空格/逗号分隔股票代码（多值含空格时加引号）；缺省=全市场 akshare 清单"),
    freqs: str = typer.Option("daily,min5,min1", "--freqs", help="空格/逗号分隔周期：daily min5 min1"),
    no_bse: bool = typer.Option(False, "--no-bse", help="全市场清单剔除北交所"),
    throttle: float = typer.Option(0.3, "--throttle", help="每只股票处理完 sleep 秒数（防网络源风控；纯 pytdx 本地源可设 0）"),
    config: str | None = typer.Option(None, "--config", help="配置文件路径"),
    store_path: str | None = typer.Option(None, "--store-path", help="数据仓库根目录（最高优先级覆盖 config/环境变量；缺省按配置链解析）"),
):
    """增量更新本地原始行情 + 刷新复权因子（亦即 README 中的 fetch 命令）。

    缺省（不传 --codes）= 更新 akshare 当前全市场清单（已退市股不在其中），
    与 ``python -m src.data.fetcher`` 行为一致；显式 --codes 则只更新指定股票。
    """
    import re

    cfg = _load_cfg(config, store_path)
    fetcher = _build_fetcher(cfg)
    freq_list = tuple(f for f in re.split(r"[\s,;]+", freqs.strip()) if f)

    code_list = _parse_codes(codes)
    if not code_list:
        code_list = fetcher.list_market_codes(include_bse=not no_bse)
        if not code_list:
            raise typer.BadParameter("未能获取全市场代码清单（网络/接口异常），请检查网络或显式指定 --codes")
        logger.info(f"全市场代码数: {len(code_list)}（含北交所={not no_bse}）")
        # 全市场更新时并入需采集的指数（基准 + data.index_codes），供回测离线用基准
        idx = _benchmark_index_codes(cfg)
        if idx:
            code_list = list(dict.fromkeys([*code_list, *idx]))
            logger.info(f"并入指数 {idx}")

    logger.info(f"开始更新 | 股票={len(code_list)} | 周期={freq_list}")
    fetcher.update(codes=code_list, freqs=freq_list, throttle=throttle)
    logger.info("数据更新完成")


# ============================================================
# 命令：回测
# ============================================================


@app.command()
def backtest(
    strategy: str = typer.Option("ma_rsi", "--strategy", help="策略名（类名/文件名/strategy_name）"),
    strategy_path: str | None = typer.Option(None, "--strategy-path", help="私有策略外部目录；缺省用内置示例"),
    start: str = typer.Option(..., "--start", help="开始日期 YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="结束日期 YYYY-MM-DD"),
    codes: str | None = typer.Option(None, "--codes", help="空格/逗号分隔股票代码；缺省用 Universe"),
    limit: int = typer.Option(50, "--limit", help="缺省 codes 时从 Universe 取的股票数上限"),
    adjust: str = typer.Option("hfq", "--adjust", help="复权方式 none/hfq/qfq（回测建议 hfq）"),
    position_size: float | None = typer.Option(None, "--position-size", help="单票目标权重；缺省取 risk.max_single_position"),
    output: str | None = typer.Option(None, "--output", help="将净值曲线写出到该 CSV"),
    config: str | None = typer.Option(None, "--config", help="配置文件路径"),
    store_path: str | None = typer.Option(None, "--store-path", help="数据仓库根目录（最高优先级覆盖 config/环境变量；缺省按配置链解析）"),
):
    """运行回测并打印绩效摘要。"""
    from .engine import Backtester
    from .risk import RiskManager
    from .strategy import load_strategy
    from .utils.helpers import parse_date

    cfg = _load_cfg(config, store_path)
    s, e = parse_date(start), parse_date(end)

    # 策略类（外部优先，回退内置示例）；external_path 缺省取 config.strategy.external_path
    ext = strategy_path or (cfg.get("strategy", {}).get("external_path") or None)
    strat_cls = load_strategy(strategy, external_path=ext)
    strat = strat_cls()
    logger.info(f"策略: {strat}")

    code_list = _resolve_codes(cfg, codes, s, limit)
    fetcher = _build_fetcher(cfg)
    logger.info(f"加载 {len(code_list)} 只股票数据 {s}~{e}（adjust={adjust}）")
    data = fetcher.load_batch(code_list, s, e, adjust=adjust)
    if not data:
        raise typer.Exit(code=1)

    # 基准：指数不在常规股票清单里，缺则按需 update 补全（pytdx 本地源近乎瞬时）
    benchmark = None
    bench_code = cfg.get("backtest", {}).get("benchmark")
    if bench_code:
        try:
            bdf = fetcher.load_daily(bench_code, s, e, adjust="none", auto_fetch=True)
            if len(bdf):
                benchmark = bdf.set_index("date")["close"]
                logger.info(f"基准 {bench_code} 载入 {len(benchmark)} 日")
            else:
                logger.warning(f"基准 {bench_code} 区间内无数据，跳过基准对比")
        except Exception as exc:
            logger.warning(f"基准 {bench_code} 载入失败（{exc}），跳过基准对比")

    risk = RiskManager.from_config(cfg)
    bt = Backtester(config=cfg, calendar=fetcher._calendar, risk_manager=risk, position_size=position_size)
    result = bt.run(strat, s, e, data=data, benchmark=benchmark)

    typer.echo("\n===== 回测绩效 =====")
    typer.echo(result.summary())
    typer.echo(f"净值区间: {result.equity_curve.index.min().date()} ~ {result.equity_curve.index.max().date()}"
               if len(result.equity_curve) else "无净值")
    if result.trades:
        typer.echo("前 5 笔交易:")
        for t in result.trades[:5]:
            typer.echo(f"  {t.code} {t.entry_date}->{t.exit_date} {t.shares}股 "
                       f"盈亏 {t.pnl:+.2f}({t.pnl_pct:+.2%}) 持有{t.holding_days}天")
    if output and len(result.equity_curve):
        result.equity_curve.to_csv(output, header=["equity"])
        logger.info(f"净值曲线已写出: {output}")


# ============================================================
# 命令：超参优化（Optuna）
# ============================================================


@app.command()
def optimize(
    strategy: str = typer.Option("ma_rsi", "--strategy", help="策略名"),
    strategy_path: str | None = typer.Option(None, "--strategy-path", help="私有策略外部目录"),
    start: str = typer.Option(..., "--start", help="开始日期"),
    end: str = typer.Option(..., "--end", help="结束日期"),
    codes: str | None = typer.Option(None, "--codes", help="空格/逗号分隔股票代码；缺省用 Universe"),
    limit: int = typer.Option(50, "--limit", help="缺省 codes 时 Universe 股票数上限"),
    trials: int = typer.Option(30, "--trials", help="每折 Optuna 试验次数"),
    folds: int = typer.Option(3, "--folds", help="walk-forward 滚动折数（训练段锚定起点扩张、测试段紧邻其后）；1=单次五五分样本内外"),
    metric: str = typer.Option("sharpe", "--metric", help="优化目标：sharpe/annual/calmar"),
    adjust: str = typer.Option("hfq", "--adjust"),
    config: str | None = typer.Option(None, "--config", help="配置文件路径"),
    store_path: str | None = typer.Option(None, "--store-path", help="数据仓库根目录（最高优先级覆盖 config/环境变量；缺省按配置链解析）"),
):
    """用 Optuna + **walk-forward 样本外验证**搜索策略超参。

    每折在训练段（锚定起点、逐折扩张）上搜参，再用最优参在紧邻的测试段上验证；汇报各折
    样本外(OOS)指标与均值。OOS 均值远低于训练值即为过拟合信号——它才是对未来表现的现实
    预期，绝不要只看训练段最优值。注意：每个窗口内因子从头预热，窗口应远大于最长指标周期。
    """
    import optuna

    from .engine import Backtester
    from .risk import RiskManager
    from .strategy import load_strategy
    from .utils.helpers import parse_date

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # 降噪：每折 trials 次回测

    cfg = _load_cfg(config, store_path)
    s, e = parse_date(start), parse_date(end)
    ext = strategy_path or (cfg.get("strategy", {}).get("external_path") or None)
    strat_cls = load_strategy(strategy, external_path=ext)

    code_list = _resolve_codes(cfg, codes, s, limit)
    fetcher = _build_fetcher(cfg)
    data = fetcher.load_batch(code_list, s, e, adjust=adjust)
    if not data:
        raise typer.Exit(code=1)

    metric_attr = {"sharpe": "sharpe_ratio", "annual": "annual_return", "calmar": "calmar_ratio"}.get(metric)
    if metric_attr is None:
        raise typer.BadParameter("metric 须为 sharpe/annual/calmar")

    days = fetcher._calendar.get_trading_days(s, e)
    try:
        windows = _walk_forward_windows(days, folds)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    risk = RiskManager.from_config(cfg)
    bt = Backtester(config=cfg, calendar=fetcher._calendar, risk_manager=risk)

    typer.echo(f"\n===== Walk-forward 调参（{folds} 折，样本外验证）=====")
    oos_metrics: list[float] = []
    for k, (tr_s, tr_e, te_s, te_e) in enumerate(windows, start=1):
        # 默认参数绑定避免闭包晚绑定（study.optimize 在本折同步调用 objective）
        def objective(trial: optuna.Trial, _ts=tr_s, _te=tr_e) -> float:
            params = strat_cls().get_param_space(trial)
            res = bt.run(strat_cls(**params), _ts, _te, data=data)
            return float(getattr(res, metric_attr))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)
        best = study.best_params
        test_res = bt.run(strat_cls(**best), te_s, te_e, data=data)
        test_metric = float(getattr(test_res, metric_attr))
        oos_metrics.append(test_metric)
        typer.echo(
            f"[折{k}] 训练 {tr_s}~{tr_e}（{metric}={study.best_value:+.4f}）"
            f" → 测试 {te_s}~{te_e}（{metric}={test_metric:+.4f}，交易 {test_res.total_trades} 笔）"
        )
        typer.echo(f"        最优参数: {best}")
        if test_res.total_trades == 0:
            typer.echo(
                "        ⚠ 测试段 0 交易：窗口可能过短（装不下因子预热）或参数不触发信号，"
                "该折 OOS 指标不可信——请增大数据区间/减少 --folds 或收窄参数周期。"
            )

    mean_oos = sum(oos_metrics) / len(oos_metrics) if oos_metrics else 0.0
    typer.echo(f"\n样本外(OOS) {metric} 各折: {[round(m, 4) for m in oos_metrics]}")
    typer.echo(f"样本外(OOS) {metric} 均值: {mean_oos:+.4f}  ← 对未来表现的现实预期")
    typer.echo("（各折最优参数不同属正常；OOS 均值远低于训练值即为过拟合，勿只信训练段最优）")


# ============================================================
# 命令：交易日历
# ============================================================


@app.command()
def calendar(
    date_str: str = typer.Option(..., "--date", help="查询日期 YYYY-MM-DD"),
    config: str | None = typer.Option(None, "--config", help="配置文件路径"),
    store_path: str | None = typer.Option(None, "--store-path", help="数据仓库根目录（最高优先级覆盖 config/环境变量；缺省按配置链解析）"),
):
    """查询某日是否交易日及相邻交易日。"""
    cfg = _load_cfg(config, store_path)
    from .data import TradingCalendar
    from .utils.helpers import parse_date

    cal = TradingCalendar(store_path=cfg.get("data", {}).get("store_path", "data_store"))
    d = parse_date(date_str)
    is_td = cal.is_trading_day(d)
    typer.echo(f"{d} 是否交易日: {is_td}")
    typer.echo(f"上一交易日: {cal.previous_trading_day(d)}")
    typer.echo(f"下一交易日: {cal.next_trading_day(d)}")


# ============================================================
# 命令：版本
# ============================================================


@app.command()
def version():
    """打印框架与关键依赖版本。"""
    import numpy
    import pandas

    typer.echo("A_Share_Quant_Pro 0.1.0")
    typer.echo(f"pandas {pandas.__version__} | numpy {numpy.__version__}")


def main() -> None:
    """console_scripts 入口。"""
    app()


if __name__ == "__main__":
    app()
