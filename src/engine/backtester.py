"""
回测主引擎 + 绩效评估（backtester）
====================================

把 数据 → 策略信号 → 撮合 → 持仓/资金 → 绩效 串成一条**防未来函数**的回测流水线。

时序（核心）
------------
策略在 ``T`` 日收盘后产出信号矩阵；引擎在 **``T+1`` 日开盘**用 :class:`ExecutionEngine`
撮合（默认 ``mode='next_open'``）。实现上：一次性生成全程信号矩阵（因子均为因果计算，
``T`` 日信号只用 ``<=T`` 数据），日循环时**用上一交易日的信号**驱动当日撮合，从而强制
T+1 滞后，杜绝未来函数。

复权口径与分红送股
------------------
默认在 **hfq（后复权）** 价上回测：复权价已把送转/分红连续地折进价格，**无需**再对持仓
股数/现金做除权调整（否则双重计提）。因此引擎默认 ``apply_corporate_actions=False``。
仅当显式传入**不复权(none)**数据时才应开启（用 :meth:`ExecutionEngine.apply_corporate_action`）；
``qfq`` 同样已折除权、**不可**开启，否则双重计提。

仓位规则（示例引擎的默认实现）
------------------------------
信号为方向（BUY/HOLD/SELL）。引擎采用**目标权重**建仓：``BUY`` 且未持仓 → 以
``position_size``（默认取 ``risk.max_single_position``）× 当前总资产 估算目标市值买入；
``SELL`` 且持仓 → 全部卖出；``HOLD`` 不动。每个交易日**先卖后买**以释放资金。

绩效指标
--------
总/年化收益、夏普、最大回撤、Calmar、胜率、盈亏比、平均持仓天数、完整交易笔数，以及
（提供基准时）基准收益、超额年化 alpha、beta。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..strategy.base import BaseStrategy, Signal
from ..utils.helpers import (
    calculate_max_drawdown,
    calculate_sharpe,
    parse_date,
    safe_divide,
    truncate_to_100,
)
from .execution import ExecutionEngine, Order
from .portfolio import Portfolio

__all__ = ["BacktestResult", "Backtester", "Trade"]

#: 年交易日数
TRADING_DAYS_PER_YEAR = 252

#: 撮合 bar 需要的列
_BAR_COLS = ("open", "high", "low", "close", "volume", "limit_up", "limit_down", "is_suspended", "adj_factor")
_PRICE_COLS = ("open", "high", "low", "close", "limit_up", "limit_down")


@dataclass
class Trade:
    """一笔完整交易（一买一卖，均价法）。"""

    code: str
    entry_date: date | None
    exit_date: date | None
    entry_price: float
    exit_price: float
    shares: int
    pnl: float           # 绝对盈亏（元，净额）
    pnl_pct: float       # 收益率
    holding_days: int


@dataclass
class BacktestResult:
    """回测结果与绩效指标。"""

    # 收益指标
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    # 交易统计
    win_rate: float
    profit_loss_ratio: float
    avg_holding_days: float
    total_trades: int
    # 基准对比
    benchmark_return: float
    alpha: float
    beta: float
    # 明细
    equity_curve: pd.Series
    trades: list[Trade]
    daily_positions: pd.DataFrame

    def summary(self) -> str:
        """单行可读摘要。"""
        return (
            f"总收益 {self.total_return:+.2%} | 年化 {self.annual_return:+.2%} | "
            f"夏普 {self.sharpe_ratio:.2f} | 回撤 {self.max_drawdown:.2%} | "
            f"Calmar {self.calmar_ratio:.2f} | 胜率 {self.win_rate:.2%} | "
            f"盈亏比 {self.profit_loss_ratio:.2f} | 交易 {self.total_trades} 笔 | "
            f"超额 {self.alpha:+.2%} beta {self.beta:.2f}"
        )


class Backtester:
    """回测主引擎。

    Args:
        config: 全局配置（读取 ``backtest`` / ``execution`` / ``risk``）；可为 None 用默认。
        execution: 撮合引擎实例；None 时按 config 构造。
        calendar: 交易日历（用于确定交易日序列）；None 时由数据日期推断。
        fetcher: 数据源（需有 ``load_batch``）；当 :meth:`run` 未直接传 data 时使用。
        risk_manager: 风控（传给撮合引擎计算费用；可选）。
        position_size: 单票目标权重；None 时取 ``risk.max_single_position`` 或 0.2。
        apply_corporate_actions: 是否对持仓做除权调整（仅不复权 none 数据下开启；hfq/qfq 已折除权，开启会双重计提）。
    """

    def __init__(
        self,
        config: Any = None,
        execution: ExecutionEngine | None = None,
        calendar: Any = None,
        fetcher: Any = None,
        risk_manager: Any | None = None,
        position_size: float | None = None,
        apply_corporate_actions: bool = False,
    ) -> None:
        self.config = config
        self.calendar = calendar
        self.fetcher = fetcher
        self.risk_manager = risk_manager
        self.execution = execution or ExecutionEngine.from_config(config or {}, risk_manager)
        self.apply_corporate_actions = bool(apply_corporate_actions)

        bt = _section(config, "backtest")
        rk = _section(config, "risk")
        self.initial_capital = float(_get(bt, "initial_capital", 1_000_000))
        self.risk_free_rate = float(_get(bt, "risk_free_rate", 0.025))
        self.benchmark_code = _get(bt, "benchmark", "000300.SH")
        ex = _section(config, "execution")
        self.t1_cash_freeze = bool(_get(ex, "t1_cash_freeze", False))
        self.position_size = float(
            position_size if position_size is not None else _get(rk, "max_single_position", 0.20)
        )

    # ------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------

    def run(
        self,
        strategy: BaseStrategy,
        start: str | date | datetime,
        end: str | date | datetime,
        data: dict[str, pd.DataFrame] | None = None,
        trade_data: dict[str, pd.DataFrame] | None = None,
        codes: Sequence[str] | None = None,
        benchmark: pd.Series | None = None,
        point_in_time_signal_adjust: bool = False,
    ) -> BacktestResult:
        """运行回测。

        Args:
            strategy: 策略实例。
            start/end: 回测区间（含）。
            data: 信号口径 ``{code: 日线df}``（含 date 列，建议 hfq/qfq）；None 时用 fetcher 拉取。
            trade_data: 交易/估值口径 ``{code: 日线df}``，应为不复权 raw；None 时沿用 data 以兼容旧接口。
            codes: data 为 None 时，要回测的股票池。
            benchmark: 基准日收盘 Series（index=date）；None 时基准指标为 0。
            point_in_time_signal_adjust: True 时，策略按每个历史信号日的 cum_factor 对此前价格做前复权，
                避免用未来复权因子生成历史信号。

        Returns:
            :class:`BacktestResult`。
        """
        s, e = parse_date(start), parse_date(end)
        if data is None:
            load_adjust = "none" if point_in_time_signal_adjust or self.apply_corporate_actions else "hfq"
            data = self._load_data(codes, s, e, adjust=load_adjust)
        signal_data = self._prepare_data(data, s, e)
        trade_data = self._prepare_data(trade_data, s, e) if trade_data is not None else signal_data
        if not signal_data or not trade_data:
            logger.warning("回测数据为空，返回空结果")
            return self._empty_result()
        if self.apply_corporate_actions:
            self._validate_corporate_action_inputs(trade_data)

        trading_days = self._trading_days(trade_data, s, e)
        if len(trading_days) < 2:
            logger.warning("可用交易日不足 2 天，返回空结果")
            return self._empty_result()

        # 全程信号矩阵（因果），用上一交易日信号驱动当日撮合（T+1）
        signals = self._generate_signals(strategy, signal_data, trading_days, point_in_time_signal_adjust)
        signals = self._align_signals(signals, trading_days, list(trade_data.keys()))

        # 每只股票按日期建索引，便于 O(1) 取 bar
        indexed = {code: df.set_index("date") for code, df in trade_data.items()}
        # 各股最后一根行情日：用于退市/行情终止后的持仓强制清仓（防幻值）
        last_dt = {code: df.index.max() for code, df in indexed.items() if not df.empty}
        final_day = trading_days[-1]

        pf = Portfolio(self.initial_capital, t1_cash_freeze=self.t1_cash_freeze)
        # 风控初始化（仅在显式注入 risk_manager 时启用止损/熔断/仓位限制）
        if self.risk_manager is not None and hasattr(self.risk_manager, "reset"):
            self.risk_manager.reset(self.initial_capital)
        equity_dates: list[date] = []
        equity_values: list[float] = []
        positions_log: list[dict[str, int]] = []

        for i, day in enumerate(trading_days):
            pf.settle_new_day()  # 解冻 T+1
            # 风控：记录当日起点权益并刷新历史峰值（单日止损 / 累计回撤熔断的基准）
            if self.risk_manager is not None and hasattr(self.risk_manager, "on_new_day"):
                self.risk_manager.on_new_day(pf)

            if self.apply_corporate_actions and i > 0:
                self._handle_corporate_actions(pf, indexed, trading_days[i - 1], day)

            if i > 0:
                signal_day = trading_days[i - 1]      # T：上一交易日信号
                self._execute_day(pf, signals, indexed, signal_day, day)

            # 收盘估值
            pf.mark_to_market(self._close_prices(indexed, day))
            # 退市/行情终止的持仓强制清仓（在估值后、记录权益前，避免幻值持续计入净值）
            self._liquidate_ended_positions(pf, day, last_dt, final_day)
            equity_dates.append(day)
            equity_values.append(pf.total_value)
            positions_log.append(pf.position_shares())

        equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates), dtype="float64")
        equity_curve.index.name = "date"
        equity_curve = equity_curve / self.initial_capital   # 归一，初始=1.0

        trades = [Trade(**t) for t in pf.realized_trades]
        daily_positions = self._positions_frame(equity_dates, positions_log)

        result = self._compute_metrics(equity_curve, trades, daily_positions, benchmark)
        logger.info(f"回测完成 | {strategy.strategy_name} | {result.summary()}")
        return result

    # ------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------

    def _load_data(
        self,
        codes: Sequence[str] | None,
        s: date,
        e: date,
        adjust: str = "hfq",
    ) -> dict[str, pd.DataFrame]:
        if self.fetcher is None or not codes:
            raise ValueError("run() 未提供 data，且 fetcher/codes 不足以加载数据")
        logger.info(f"通过 fetcher 加载 {len(codes)} 只股票 {s}~{e} 数据（{adjust}）")
        return self.fetcher.load_batch(list(codes), s, e, adjust=adjust)

    @staticmethod
    def _prepare_data(data: dict[str, pd.DataFrame], s: date, e: date) -> dict[str, pd.DataFrame]:
        """裁剪区间、规整 date、剔除空表。"""
        out: dict[str, pd.DataFrame] = {}
        ts, te = pd.Timestamp(s), pd.Timestamp(e)
        for code, df in data.items():
            if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
                continue
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"]).dt.normalize()
            d = d[(d["date"] >= ts) & (d["date"] <= te)]
            if d.empty:
                continue
            d = d.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
            if "is_suspended" not in d.columns:
                d["is_suspended"] = False
            out[code] = d
        return out

    def _trading_days(self, data: dict[str, pd.DataFrame], s: date, e: date) -> list[pd.Timestamp]:
        if self.calendar is not None:
            return [pd.Timestamp(d) for d in self.calendar.get_trading_days(s, e)]
        # 退化：用全部数据日期并集
        all_dates = pd.DatetimeIndex(
            np.concatenate([df["date"].to_numpy() for df in data.values()])
        )
        return list(all_dates.normalize().unique().sort_values())

    @staticmethod
    def _align_signals(signals: pd.DataFrame, days: list[pd.Timestamp], codes: list[str]) -> pd.DataFrame:
        if signals is None or signals.empty:
            return pd.DataFrame(int(Signal.HOLD), index=pd.DatetimeIndex(days), columns=codes, dtype="int64")
        sig = signals.copy()
        sig.index = pd.DatetimeIndex(pd.to_datetime(sig.index)).normalize()
        sig = sig.reindex(index=pd.DatetimeIndex(days), columns=codes).fillna(int(Signal.HOLD))
        return sig.astype("int64")

    def _generate_signals(
        self,
        strategy: BaseStrategy,
        data: dict[str, pd.DataFrame],
        days: list[pd.Timestamp],
        point_in_time_adjust: bool,
    ) -> pd.DataFrame:
        if not point_in_time_adjust:
            return strategy.generate_signals(data)
        rows: list[pd.Series] = []
        idx: list[pd.Timestamp] = []
        codes = list(data.keys())
        for day in days:
            visible = self._point_in_time_adjust_data(data, day, time_col="date")
            if not visible:
                continue
            sig = strategy.generate_signals(visible)
            aligned = self._align_signals(sig, [day], codes)
            rows.append(aligned.iloc[0])
            idx.append(day)
        if not rows:
            return pd.DataFrame(int(Signal.HOLD), index=pd.DatetimeIndex(days), columns=codes, dtype="int64")
        out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=codes)
        out.index.name = "date"
        return out

    @staticmethod
    def _point_in_time_adjust_data(
        data: dict[str, pd.DataFrame],
        as_of: pd.Timestamp,
        *,
        time_col: str,
    ) -> dict[str, pd.DataFrame]:
        """以 as_of 当日累计因子为锚点，对可见历史价格做前复权。"""
        out: dict[str, pd.DataFrame] = {}
        as_of = pd.Timestamp(as_of)
        for code, df in data.items():
            if df is None or df.empty or time_col not in df.columns:
                continue
            d = df.copy()
            d[time_col] = pd.to_datetime(d[time_col])
            d = d[d[time_col] <= as_of].copy()
            if d.empty:
                continue
            if "cum_factor" not in d.columns:
                out[code] = d
                continue
            cum = pd.to_numeric(d["cum_factor"], errors="coerce").ffill().bfill()
            if cum.empty or not cum.notna().any():
                out[code] = d
                continue
            anchor = float(cum.iloc[-1])
            if anchor > 0:
                mult = (cum / anchor).to_numpy(dtype="float64")
                for col in _PRICE_COLS:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype="float64") * mult
                d["adj_factor"] = mult
            out[code] = d.reset_index(drop=True)
        return out

    # ------------------------------------------------------------
    # 日内撮合
    # ------------------------------------------------------------

    def _execute_day(self, pf, signals, indexed, signal_day, exec_day) -> None:
        try:
            row = signals.loc[signal_day]
        except KeyError:
            return
        rm = self.risk_manager
        sells = [c for c in signals.columns if int(row.get(c, 0)) == int(Signal.SELL) and pf.has_position(c)]
        buys = [c for c in signals.columns if int(row.get(c, 0)) == int(Signal.BUY)]

        # 先卖后买，释放资金
        for code in sells:
            bar = self._bar(indexed, code, exec_day)
            if bar is None:
                continue
            pos = pf.get_position(code)
            order = Order(code=code, direction=int(Signal.SELL), shares=pos.shares, trade_date=exec_day.date())
            # 风控：累计回撤熔断时全面停盘（含卖出）；单日止损不拦截卖出
            if rm is not None and not rm.check_order(order, pf):
                continue
            self.execution.match(order, bar, pf)

        for code in buys:
            if pf.has_position(code):
                continue  # 已持仓不加仓（示例规则）
            bar = self._bar(indexed, code, exec_day)
            if bar is None:
                continue
            base = bar.get("open")
            if base is None or float(base) != float(base) or float(base) <= 0:
                continue
            target_value = pf.total_value * self.position_size
            shares = truncate_to_100(int(target_value / float(base)))
            if shares < 100:
                continue
            order = Order(code=code, direction=int(Signal.BUY), shares=shares, trade_date=exec_day.date())
            # 风控：熔断/单日止损拦截新开仓；并按单票(/行业)上限裁减股数
            if rm is not None:
                if not rm.check_order(order, pf):
                    continue
                rm.clip_position_size(order, pf, price=float(base))
                if order.shares < 100:
                    continue
            self.execution.match(order, bar, pf)

    @staticmethod
    def _bar(indexed: dict[str, pd.DataFrame], code: str, day: pd.Timestamp) -> dict[str, Any] | None:
        df = indexed.get(code)
        if df is None or day not in df.index:
            return None
        row = df.loc[day]
        if isinstance(row, pd.DataFrame):  # 重复索引兜底
            row = row.iloc[-1]
        bar: dict[str, Any] = {"date": day.date()}
        for c in _BAR_COLS:
            bar[c] = row[c] if c in row.index else (False if c == "is_suspended" else np.nan)
        return bar

    @staticmethod
    def _close_prices(indexed: dict[str, pd.DataFrame], day: pd.Timestamp) -> dict[str, float]:
        out: dict[str, float] = {}
        for code, df in indexed.items():
            if day in df.index:
                row = df.loc[day]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                out[code] = row["close"]
        return out

    @staticmethod
    def _validate_corporate_action_inputs(data: dict[str, pd.DataFrame]) -> None:
        """Guard against applying split/dividend adjustments to adjusted-price data."""
        for code, df in data.items():
            if df is None or df.empty or "cum_factor" in df.columns or "adj_factor" not in df.columns:
                continue
            adj = pd.to_numeric(df["adj_factor"], errors="coerce").dropna()
            if not adj.empty and not np.isclose(adj.to_numpy(dtype="float64"), 1.0).all():
                raise ValueError(
                    "apply_corporate_actions=True 仅支持不复权 raw 数据；"
                    f"{code} 缺少 cum_factor 且 adj_factor 非 1.0，疑似 hfq/qfq 数据，"
                    "继续会造成除权双重计提"
                )

    def _handle_corporate_actions(self, pf, indexed, prev_day, day) -> None:
        """不复权数据下，按累计复权因子变化率对持仓做除权调整。"""
        for code in list(pf.positions.keys()):
            df = indexed.get(code)
            if df is None or day not in df.index or prev_day not in df.index:
                continue
            factor_col = "cum_factor" if "cum_factor" in df.columns else "adj_factor"
            if factor_col not in df.columns:
                continue
            ratio = ExecutionEngine.detect_ex_factor_ratio(df.loc[prev_day, factor_col], df.loc[day, factor_col])
            cash_div = self._cash_dividend_per_share(df, day)
            if ratio > 1.001 or cash_div > 0:
                self.execution.apply_corporate_action(
                    pf,
                    code,
                    ratio,
                    cash_dividend_per_share_gross=cash_div,
                    action_date=day.date(),
                )

    @staticmethod
    def _cash_dividend_per_share(df: pd.DataFrame, day: pd.Timestamp) -> float:
        row = df.loc[day]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        for col in ("cash_dividend_per_share_gross", "cash_dividend", "fenhong_per_share"):
            if col in row.index:
                try:
                    val = float(row[col])
                except (TypeError, ValueError):
                    return 0.0
                return val if val == val and val > 0 else 0.0
        if "fenhong" in row.index:
            try:
                val = float(row["fenhong"]) / 10.0
            except (TypeError, ValueError):
                return 0.0
            return val if val == val and val > 0 else 0.0
        return 0.0

    def _liquidate_ended_positions(self, pf, day, last_dt: dict, final_day) -> None:
        """对"行情已终止"（最后一根数据早于回测末日）的持仓按最后市价强制清仓。

        退市股在退市日后即无行情：若仍被持有，``mark_to_market`` 会永久按最后已知价
        计入净值（幻值），且该笔持仓永远不会被卖出 → 已实现盈亏 / 交易统计漏记。本方法
        在某股**最后一根数据之后**的首个交易日把其持仓按最后市价强制离场（退市离场不受
        T+1 约束），使盈亏正确入账、净值不再挂幻值。

        注意：仅处理"行情早于回测末日就终止"的股票；正常持有到回测末日的仓位按惯例保留
        （末日按市值估值，不强平）。中途停牌但后续复牌的股票其 ``last_dt`` 为真实末日，
        不会被误清。
        """
        for code in list(pf.positions.keys()):
            pos = pf.get_position(code)
            if pos is None or pos.shares <= 0:
                continue
            ld = last_dt.get(code)
            if ld is None or ld >= final_day or day <= ld:
                continue
            shares = pos.shares
            px = pos.last_price if pos.last_price > 0 else pos.avg_cost
            pos.frozen = 0  # 退市 / 行情终止强制离场，不受 T+1 冻结约束
            fee = self.execution.commission(shares * px, is_buy=False)
            pf.sell(code, shares, px, fee, day.date())
            logger.info(
                f"[{code}] 行情于 {ld.date()} 终止（退市/停更），按最后市价 {px:.4f} 强制清仓 {shares} 股"
            )

    # ------------------------------------------------------------
    # 结果组装
    # ------------------------------------------------------------

    @staticmethod
    def _positions_frame(dates: list[date], logs: list[dict[str, int]]) -> pd.DataFrame:
        all_codes = sorted({c for d in logs for c in d})
        frame = pd.DataFrame(0, index=pd.DatetimeIndex(dates), columns=all_codes, dtype="int64")
        frame.index.name = "date"
        for i, _d in enumerate(dates):
            for code, sh in logs[i].items():
                frame.iat[i, frame.columns.get_loc(code)] = sh
        return frame

    def _compute_metrics(
        self,
        equity_curve: pd.Series,
        trades: list[Trade],
        daily_positions: pd.DataFrame,
        benchmark: pd.Series | None,
    ) -> BacktestResult:
        eq = equity_curve.to_numpy(dtype="float64")
        # 保留 DatetimeIndex（供基准按日期对齐，避免位置错配）；sharpe 用 .tolist() 不受影响
        returns = equity_curve.pct_change().dropna()
        n_periods = max(1, len(eq) - 1)

        total_return = float(eq[-1] - 1.0)
        annual_return = float((eq[-1]) ** (TRADING_DAYS_PER_YEAR / n_periods) - 1.0) if eq[-1] > 0 else -1.0
        sharpe = calculate_sharpe(returns.tolist(), self.risk_free_rate, TRADING_DAYS_PER_YEAR)
        mdd = calculate_max_drawdown(eq.tolist())
        calmar = safe_divide(annual_return, abs(mdd), 0.0)

        # 交易统计
        total_trades = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        win_rate = safe_divide(len(wins), total_trades, 0.0)
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        profit_loss_ratio = safe_divide(avg_win, abs(avg_loss), 0.0)
        avg_holding_days = float(np.mean([t.holding_days for t in trades])) if trades else 0.0

        # 基准
        benchmark_return, alpha, beta = self._benchmark_metrics(equity_curve, returns, annual_return, benchmark)

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            avg_holding_days=avg_holding_days,
            total_trades=total_trades,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            equity_curve=equity_curve,
            trades=trades,
            daily_positions=daily_positions,
        )

    def _benchmark_metrics(self, equity_curve, strat_returns, annual_return, benchmark):
        if benchmark is None or len(benchmark) < 2:
            return 0.0, 0.0, 0.0
        b = benchmark.copy()
        b.index = pd.DatetimeIndex(pd.to_datetime(b.index)).normalize()
        b = b.reindex(equity_curve.index).ffill().dropna()
        if len(b) < 2:
            return 0.0, 0.0, 0.0
        bench_return = float(b.iloc[-1] / b.iloc[0] - 1.0)
        n = max(1, len(b) - 1)
        bench_annual = float((1 + bench_return) ** (TRADING_DAYS_PER_YEAR / n) - 1.0)
        bench_ret = b.pct_change().dropna()
        # 按**日期**对齐两条日收益（inner join）——绝不按位置拼接：基准不覆盖回测起点时
        # 两条序列各自 dropna 后长度/起点不同，按位置会把错位日期的收益配对，算出错误 beta。
        aligned = pd.concat([strat_returns, bench_ret], axis=1, join="inner").dropna()
        beta = 0.0
        if len(aligned) >= 2:
            sr, br = aligned.iloc[:, 0].to_numpy(), aligned.iloc[:, 1].to_numpy()
            var_b = float(np.var(br, ddof=1))
            beta = float(np.cov(sr, br, ddof=1)[0, 1] / var_b) if var_b > 0 else 0.0
        alpha = float(annual_return - bench_annual)   # 超额年化
        return bench_return, alpha, beta

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            total_return=0.0, annual_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            calmar_ratio=0.0, win_rate=0.0, profit_loss_ratio=0.0, avg_holding_days=0.0,
            total_trades=0, benchmark_return=0.0, alpha=0.0, beta=0.0,
            equity_curve=pd.Series(dtype="float64"), trades=[],
            daily_positions=pd.DataFrame(),
        )


# ============================================================
# 配置辅助
# ============================================================


def _section(cfg: Any, key: str) -> Any:
    if cfg is None:
        return {}
    if hasattr(cfg, "get"):
        sub = cfg.get(key)
        if sub is not None:
            return sub
    return cfg


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        val = cfg.get(key, default)
        return default if val is None else val
    return getattr(cfg, key, default)


# ============================================================
# 模块自测  python -m src.engine.backtester
# ============================================================

if __name__ == "__main__":
    from ..strategy.examples.ma_rsi import MaRsiStrategy
    from ..utils.helpers import init_logging

    init_logging(level="INFO")
    rng = np.random.RandomState(1)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    def _mk(seed_shift):
        trend = np.concatenate([np.linspace(20, 14, n // 2), np.linspace(14, 28, n - n // 2)]) + seed_shift
        close = pd.Series(trend + rng.randn(n) * 0.3).clip(lower=1.0)
        return pd.DataFrame({
            "date": dates, "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.03, "low": close * 0.97, "close": close,
            "volume": pd.Series(rng.randint(int(5e6), int(1e7), n)).astype("float64"),
            "limit_up": close * 1.1, "limit_down": close * 0.9, "is_suspended": False,
        })

    data = {"000001.SZ": _mk(0.0), "600519.SH": _mk(2.0)}
    bench = pd.Series(np.linspace(1.0, 1.1, n) * 3000, index=dates)

    bt = Backtester(config={"backtest": {"initial_capital": 1_000_000}}, position_size=0.4)
    res = bt.run(MaRsiStrategy(fast_period=5, slow_period=20), "2024-01-01", "2024-08-01", data=data, benchmark=bench)
    logger.info(res.summary())
    logger.info(f"净值点数={len(res.equity_curve)} 交易={res.total_trades} 持仓表 shape={res.daily_positions.shape}")
