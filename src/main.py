"""
命令行入口（main）—— typer CLI
================================

把数据层 / 因子层 / 策略层 / 引擎层 / 风控层装配成可一键运行的命令：

    python -m src.main fetch      [--codes ...] [--freqs daily]      # 增量更新本地行情
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


def _load_cfg(config: str | None):
    from .utils.config_loader import load_config
    from .utils.helpers import init_logging

    cfg = load_config(config_path=config)
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
        max_workers=d.get("max_workers", 4),
        suspend_sources=tuple(d.get("suspend_sources", ("eastmoney", "tushare"))),
        suspend_lookback_days=d.get("suspend_lookback_days", 30),
        suspend_enabled=d.get("suspend_enabled", True),
    )


def _parse_codes(codes: str | None) -> list[str]:
    if not codes:
        return []
    from .utils.helpers import format_code

    out = []
    for raw in codes.replace(";", ",").split(","):
        raw = raw.strip()
        if raw:
            out.append(format_code(raw))
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


# ============================================================
# 命令：数据更新
# ============================================================


@app.command()
def fetch(
    codes: str | None = typer.Option(None, "--codes", help="逗号分隔股票代码；缺省=全市场"),
    freqs: str = typer.Option("daily", "--freqs", help="逗号分隔周期：daily,min5,min1"),
    config: str | None = typer.Option(None, "--config", help="配置文件路径"),
):
    """增量更新本地原始行情 + 刷新复权因子（亦即 README 中的 fetch 命令）。"""
    cfg = _load_cfg(config)
    fetcher = _build_fetcher(cfg)
    code_list = _parse_codes(codes) or None
    freq_list = tuple(f.strip() for f in freqs.split(",") if f.strip())
    logger.info(f"开始更新 | 股票={'全市场' if code_list is None else len(code_list)} | 周期={freq_list}")
    fetcher.update(codes=code_list, freqs=freq_list)
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
    codes: str | None = typer.Option(None, "--codes", help="逗号分隔股票代码；缺省用 Universe"),
    limit: int = typer.Option(50, "--limit", help="缺省 codes 时从 Universe 取的股票数上限"),
    adjust: str = typer.Option("hfq", "--adjust", help="复权方式 none/hfq/qfq（回测建议 hfq）"),
    position_size: float | None = typer.Option(None, "--position-size", help="单票目标权重；缺省取 risk.max_single_position"),
    output: str | None = typer.Option(None, "--output", help="将净值曲线写出到该 CSV"),
    config: str | None = typer.Option(None, "--config", help="配置文件路径"),
):
    """运行回测并打印绩效摘要。"""
    from .engine import Backtester
    from .risk import RiskManager
    from .strategy import load_strategy
    from .utils.helpers import parse_date

    cfg = _load_cfg(config)
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

    # 基准
    benchmark = None
    bench_code = cfg.get("backtest", {}).get("benchmark")
    if bench_code:
        try:
            bdf = fetcher.load_daily(bench_code, s, e, adjust="none")
            benchmark = bdf.set_index("date")["close"]
            logger.info(f"基准 {bench_code} 载入 {len(benchmark)} 日")
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
    codes: str | None = typer.Option(None, "--codes", help="逗号分隔股票代码；缺省用 Universe"),
    limit: int = typer.Option(50, "--limit", help="缺省 codes 时 Universe 股票数上限"),
    trials: int = typer.Option(30, "--trials", help="Optuna 试验次数"),
    metric: str = typer.Option("sharpe", "--metric", help="优化目标：sharpe/annual/calmar"),
    adjust: str = typer.Option("hfq", "--adjust"),
    config: str | None = typer.Option(None, "--config", help="配置文件路径"),
):
    """用 Optuna 搜索策略 get_param_space 定义的超参，最大化指定指标。"""
    import optuna

    from .engine import Backtester
    from .risk import RiskManager
    from .strategy import load_strategy
    from .utils.helpers import parse_date

    cfg = _load_cfg(config)
    s, e = parse_date(start), parse_date(end)
    ext = strategy_path or (cfg.get("strategy", {}).get("external_path") or None)
    strat_cls = load_strategy(strategy, external_path=ext)

    code_list = _resolve_codes(cfg, codes, s, limit)
    fetcher = _build_fetcher(cfg)
    data = fetcher.load_batch(code_list, s, e, adjust=adjust)
    if not data:
        raise typer.Exit(code=1)

    risk = RiskManager.from_config(cfg)
    bt = Backtester(config=cfg, calendar=fetcher._calendar, risk_manager=risk)

    metric_attr = {"sharpe": "sharpe_ratio", "annual": "annual_return", "calmar": "calmar_ratio"}.get(metric)
    if metric_attr is None:
        raise typer.BadParameter("metric 须为 sharpe/annual/calmar")

    def objective(trial: optuna.Trial) -> float:
        params = strat_cls().get_param_space(trial)
        res = bt.run(strat_cls(**params), s, e, data=data)
        return float(getattr(res, metric_attr))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    typer.echo("\n===== 调参结果 =====")
    typer.echo(f"最优 {metric}={study.best_value:.4f}")
    typer.echo(f"最优参数: {study.best_params}")


# ============================================================
# 命令：交易日历
# ============================================================


@app.command()
def calendar(
    date_str: str = typer.Option(..., "--date", help="查询日期 YYYY-MM-DD"),
    config: str | None = typer.Option(None, "--config", help="配置文件路径"),
):
    """查询某日是否交易日及相邻交易日。"""
    cfg = _load_cfg(config)
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
