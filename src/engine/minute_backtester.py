"""
分钟级回测引擎（MinuteBacktester）
===================================

把 日内分钟数据 → 策略信号 → 撮合 → 持仓/资金 → 绩效 串成一条**防未来函数**的分钟回测流水线。
与日级 :class:`~src.engine.backtester.Backtester` 共享底层（:class:`Portfolio` 账本 +
:class:`ExecutionEngine` 撮合），仅驱动循环按 **bar** 而非按交易日推进。

时序（核心，防未来函数）
------------------------
策略在 bar ``t`` 收盘后产出信号（只用 ``<=t`` 数据）；引擎在 **bar ``t+1``** 撮合。实现上：
一次性生成全程信号矩阵，逐 bar 循环时**用上一根 bar 的信号**驱动当前 bar 撮合，强制一个 bar
的滞后，杜绝未来函数。

A 股制度
--------
- **T+1 仍按自然日**：当日买入的股票当日不可卖；``Portfolio.settle_new_day`` 只在**跨交易日**
  （``ts.date()`` 变化）时调用一次解冻，绝不每根 bar 调。
- **日内涨跌停**：``ExecutionEngine(intraday=True)``，把成交价夹进当日涨跌停区间，完全封板
  （一字）的分钟直接拒单。每根分钟 bar 注入**当日 hfq 涨跌停价**（与 hfq 分钟价同尺度）。

复权口径
--------
支持与日级一致的双数据口径：策略信号可用历史时点前复权数据，撮合/资金/估值使用
``none`` raw 分钟真实价。``apply_corporate_actions=True`` 时，跨交易日按 raw 数据上的
``cum_factor`` 变化调整持仓，并可按税前现金分红递延扣红利税。

年化
----
夏普的年化因子 ``N``（每年 bar 数）**从数据实测、按自然年取中位数**（见
:meth:`_annualization_factor`），不沿用日级的 252。
"""

from __future__ import annotations

from collections.abc import Sequence
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
from .backtester import (
    TRADING_DAYS_PER_YEAR,
    Backtester,
    BacktestResult,
    Trade,
    _get,
    _section,
)
from .execution import ExecutionEngine, Order
from .portfolio import Portfolio

__all__ = ["MinuteBacktester"]

#: 撮合 bar 需要的列（与日级一致；分钟 bar 的涨跌停/停牌由日级 hfq 注入）
_BAR_COLS = ("open", "high", "low", "close", "volume", "limit_up", "limit_down", "is_suspended", "adj_factor")

#: 注入到分钟 bar 的日级列（按 date 左连接）
_DAILY_INJECT_COLS = ("limit_up", "limit_down", "is_suspended")


class MinuteBacktester:
    """分钟级回测主引擎。

    Args:
        config: 全局配置（读取 ``backtest`` / ``execution`` / ``risk``）；可为 None 用默认。
        execution: 撮合引擎；None 时按 config 构造并**强制 ``intraday=True``**。
        store: :class:`~src.data.storage.DataStore`；``run`` 未直接传 data 时用于加载分钟数据。
        freq: 分钟周期（``min1`` / ``min5``）。
        position_size: 单票目标权重；None 时取 ``risk.max_single_position`` 或 0.2。
        apply_corporate_actions: 是否在分钟 raw 交易数据上按日级 cum_factor 变化调整持仓。
    """

    def __init__(
        self,
        config: Any = None,
        execution: ExecutionEngine | None = None,
        store: Any = None,
        freq: str = "min1",
        risk_manager: Any | None = None,
        position_size: float | None = None,
        apply_corporate_actions: bool = False,
    ) -> None:
        self.config = config
        self.store = store
        self.freq = str(freq)
        self.risk_manager = risk_manager
        if execution is not None:
            self.execution = execution
        else:
            self.execution = ExecutionEngine.from_config(config or {}, risk_manager)
            self.execution.intraday = True          # 分钟级强制日内涨跌停夹价
        self.apply_corporate_actions = bool(apply_corporate_actions)

        bt = _section(config, "backtest")
        rk = _section(config, "risk")
        ex = _section(config, "execution")
        self.initial_capital = float(_get(bt, "initial_capital", 1_000_000))
        self.risk_free_rate = float(_get(bt, "risk_free_rate", 0.025))
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
        """运行分钟级回测。

        Args:
            strategy: 策略实例（信号矩阵 index 为分钟时间戳）。
            start/end: 回测区间（含）。
            data: 信号口径 ``{code: 分钟 df}``；qfq 历史时点前复权模式下应为 raw + cum_factor。
            trade_data: 交易/估值口径分钟数据，应为 none raw；None 时沿用 data 兼容旧接口。
            codes: data 为 None 时要回测的股票池。
            benchmark: 暂未支持分钟基准对比，alpha/beta 记 0。

        Returns:
            :class:`~src.engine.backtester.BacktestResult`。
        """
        s, e = parse_date(start), parse_date(end)
        auto_loaded_raw_signal = False
        if data is None:
            if self.apply_corporate_actions or point_in_time_signal_adjust:
                data = self.load_trade_data(codes, s, e)
                if trade_data is None:
                    trade_data = data
                auto_loaded_raw_signal = self.apply_corporate_actions and not point_in_time_signal_adjust
            else:
                data = self._load_data(codes, s, e)
        signal_data = self._prepare_data(data, s, e)
        trade_data = self._prepare_data(trade_data, s, e) if trade_data is not None else signal_data
        if not signal_data or not trade_data:
            logger.warning("分钟回测数据为空，返回空结果")
            return self._empty_result()
        if self.apply_corporate_actions:
            Backtester._validate_corporate_action_inputs(trade_data)

        # 全 code 并集的有序分钟时间轴
        timeline = self._timeline(trade_data)
        if len(timeline) < 2:
            logger.warning("可用 bar 不足 2 根，返回空结果")
            return self._empty_result()
        if auto_loaded_raw_signal:
            logger.warning(
                "MinuteBacktester 自动加载了 none raw 分钟数据作为信号口径，但 "
                "point_in_time_signal_adjust=False；策略信号可能受除权跳空影响。"
            )

        # 全程信号矩阵（因果），对齐到时间轴与 code（用上一根 bar 信号撮合 → 强制一 bar 滞后）
        signals = self._generate_signals(strategy, signal_data, timeline, point_in_time_signal_adjust)
        signals = self._align_signals(signals, timeline, list(trade_data.keys()))

        indexed = {code: df.set_index("datetime") for code, df in trade_data.items()}

        pf = Portfolio(self.initial_capital, t1_cash_freeze=self.t1_cash_freeze)
        # 风控初始化（仅在显式注入 risk_manager 时启用止损/熔断/仓位限制；与日级一致）
        if self.risk_manager is not None and hasattr(self.risk_manager, "reset"):
            self.risk_manager.reset(self.initial_capital)
        equity_ts: list[pd.Timestamp] = []
        equity_values: list[float] = []
        positions_log: list[dict[str, int]] = []

        prev_ts: pd.Timestamp | None = None     # 上一根 bar（其信号驱动当前 bar）
        prev_day: date | None = None            # 上一根 bar 所属交易日（跨日检测）

        for ts in timeline:
            cur_day = ts.date()
            if cur_day != prev_day:
                pf.settle_new_day()              # 跨交易日：解冻 T+1（每日仅一次）
                if self.apply_corporate_actions and prev_ts is not None:
                    self._handle_corporate_actions(pf, indexed, prev_ts, ts)
                # 风控：跨交易日刷新当日起点权益与历史峰值（单日止损 / 累计回撤熔断基准）
                if self.risk_manager is not None and hasattr(self.risk_manager, "on_new_day"):
                    self.risk_manager.on_new_day(pf)
                prev_day = cur_day

            if prev_ts is not None:
                self._execute_bar(pf, signals, indexed, prev_ts, ts)

            pf.mark_to_market(self._bar_closes(indexed, ts))
            equity_ts.append(ts)
            equity_values.append(pf.total_value)
            positions_log.append(pf.position_shares())
            prev_ts = ts

        equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_ts), dtype="float64")
        equity_curve.index.name = "datetime"
        equity_curve = equity_curve / self.initial_capital   # 归一，初始=1.0

        trades = [Trade(**t) for t in pf.realized_trades]
        result = self._compute_metrics(equity_curve, trades)
        logger.info(f"分钟回测完成 | {strategy.strategy_name} | {self.freq} | {result.summary()}")
        return result

    # ------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------

    def _load_data(self, codes: Sequence[str] | None, s: date, e: date) -> dict[str, pd.DataFrame]:
        """用 DataStore 加载 hfq 分钟数据，并注入当日 hfq 涨跌停/停牌列。"""
        if self.store is None or not codes:
            raise ValueError("run() 未提供 data，且 store/codes 不足以加载分钟数据")
        out: dict[str, pd.DataFrame] = {}
        for code in codes:
            mdf = self.store.load(code, self.freq, adjust="hfq")
            if mdf is None or mdf.empty:
                logger.warning(f"[{code}] 无 {self.freq} 分钟数据，跳过")
                continue
            daily = self.store.load(code, "daily", adjust="hfq")
            out[code] = self._inject_daily_limits(mdf, daily)
        logger.info(f"通过 store 加载 {len(out)} 只股票 {s}~{e} 的 {self.freq} 数据（hfq）")
        return out

    def load_trade_data(self, codes: Sequence[str] | None, s: date, e: date) -> dict[str, pd.DataFrame]:
        """用 DataStore 加载 none raw 分钟交易数据，并注入 raw 日级涨跌停/停牌列。"""
        if self.store is None or not codes:
            raise ValueError("store/codes 不足以加载分钟交易数据")
        out: dict[str, pd.DataFrame] = {}
        for code in codes:
            mdf = self.store.load(code, self.freq, adjust="none")
            if mdf is None or mdf.empty:
                continue
            daily = self.store.load(code, "daily", adjust="none")
            out[code] = self._inject_daily_limits(mdf, daily)
        logger.info(f"通过 store 加载 {len(out)} 只股票 {s}~{e} 的 {self.freq} raw 交易数据")
        return out

    @staticmethod
    def _inject_daily_limits(mdf: pd.DataFrame, daily: pd.DataFrame | None) -> pd.DataFrame:
        """把日级 hfq 涨跌停/停牌按 date 左连接进每根分钟 bar。"""
        m = mdf.copy()
        m["datetime"] = pd.to_datetime(m["datetime"])
        if daily is None or daily.empty:
            return m
        d = daily.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        cols = [c for c in _DAILY_INJECT_COLS if c in d.columns]
        if not cols:
            return m
        m["date"] = m["datetime"].dt.normalize()
        m = m.merge(d[["date", *cols]], on="date", how="left").drop(columns=["date"])
        return m

    @staticmethod
    def _prepare_data(data: dict[str, pd.DataFrame], s: date, e: date) -> dict[str, pd.DataFrame]:
        """裁剪区间、规整 datetime、剔除空表。"""
        out: dict[str, pd.DataFrame] = {}
        ts, te = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)  # 含 end 当天全部分钟
        for code, df in data.items():
            if df is None or df.empty or "datetime" not in df.columns or "close" not in df.columns:
                continue
            d = df.copy()
            d["datetime"] = pd.to_datetime(d["datetime"])
            d = d[(d["datetime"] >= ts) & (d["datetime"] < te)]
            if d.empty:
                continue
            d = d.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime").reset_index(drop=True)
            if "is_suspended" not in d.columns:
                d["is_suspended"] = False
            out[code] = d
        return out

    @staticmethod
    def _timeline(data: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
        """全 code 并集的有序唯一分钟时间轴。"""
        all_ts = pd.DatetimeIndex(
            np.concatenate([df["datetime"].to_numpy() for df in data.values()])
        )
        return list(all_ts.unique().sort_values())

    @staticmethod
    def _align_signals(signals: pd.DataFrame, timeline: list[pd.Timestamp], codes: list[str]) -> pd.DataFrame:
        idx = pd.DatetimeIndex(timeline)
        if signals is None or signals.empty:
            return pd.DataFrame(int(Signal.HOLD), index=idx, columns=codes, dtype="int64")
        sig = signals.copy()
        sig.index = pd.DatetimeIndex(pd.to_datetime(sig.index))
        sig = sig.reindex(index=idx, columns=codes).fillna(int(Signal.HOLD))
        return sig.astype("int64")

    def _generate_signals(
        self,
        strategy: BaseStrategy,
        data: dict[str, pd.DataFrame],
        timeline: list[pd.Timestamp],
        point_in_time_adjust: bool,
    ) -> pd.DataFrame:
        if not point_in_time_adjust:
            return strategy.generate_signals(data)
        codes = list(data.keys())
        seg_ends = Backtester._pit_segment_end_marks(data, timeline, time_col="datetime")
        if len(seg_ends) > 500:
            logger.warning(
                f"point_in_time_signal_adjust=True 检测到 {len(seg_ends)} 个除权分段，"
                "仍需按段重算策略信号，除权密集的大股票池/长区间分钟回测可能变慢。"
            )
        frames: list[pd.DataFrame] = []
        start = 0
        ts_index = {t: i for i, t in enumerate(timeline)}
        for seg_end in seg_ends:
            seg_marks = timeline[start : ts_index[seg_end] + 1]
            start = ts_index[seg_end] + 1
            if not seg_marks:
                continue
            visible = Backtester._point_in_time_adjust_data(data, seg_end, time_col="datetime")
            if not visible:
                continue
            sig = strategy.generate_signals(visible)
            frames.append(self._align_signals(sig, seg_marks, codes))
        if not frames:
            return pd.DataFrame(int(Signal.HOLD), index=pd.DatetimeIndex(timeline), columns=codes, dtype="int64")
        out = pd.concat(frames)
        out.index.name = "datetime"
        return out

    # ------------------------------------------------------------
    # 逐 bar 撮合（先卖后买）
    # ------------------------------------------------------------

    def _execute_bar(self, pf, signals, indexed, signal_ts, exec_ts) -> None:
        try:
            row = signals.loc[signal_ts]
        except KeyError:
            return
        rm = self.risk_manager
        sells = [c for c in signals.columns if int(row.get(c, 0)) == int(Signal.SELL) and pf.has_position(c)]
        buys = [c for c in signals.columns if int(row.get(c, 0)) == int(Signal.BUY)]

        for code in sells:
            bar = self._bar(indexed, code, exec_ts)
            if bar is None:
                continue
            pos = pf.get_position(code)
            order = Order(code=code, direction=int(Signal.SELL), shares=pos.shares, trade_date=exec_ts)
            # 风控：累计回撤熔断时全面停盘（含卖出）；单日止损不拦截卖出（与日级一致）
            if rm is not None and not rm.check_order(order, pf):
                continue
            self.execution.match(order, bar, pf)

        for code in buys:
            if pf.has_position(code):
                continue  # 已持仓不加仓（示例规则，与日级一致）
            bar = self._bar(indexed, code, exec_ts)
            if bar is None:
                continue
            base = bar.get("open")
            if base is None or float(base) != float(base) or float(base) <= 0:
                continue
            target_value = pf.total_value * self.position_size
            shares = truncate_to_100(int(target_value / float(base)))
            if shares < 100:
                continue
            order = Order(code=code, direction=int(Signal.BUY), shares=shares, trade_date=exec_ts)
            # 风控：熔断/单日止损拦截新开仓；并按单票(/行业)上限裁减股数（与日级一致）
            if rm is not None:
                if not rm.check_order(order, pf):
                    continue
                rm.clip_position_size(order, pf, price=float(base))
                if order.shares < 100:
                    continue
            self.execution.match(order, bar, pf)

    @staticmethod
    def _bar(indexed: dict[str, pd.DataFrame], code: str, ts: pd.Timestamp) -> dict[str, Any] | None:
        df = indexed.get(code)
        if df is None or ts not in df.index:
            return None
        row = df.loc[ts]
        if isinstance(row, pd.DataFrame):  # 重复索引兜底
            row = row.iloc[-1]
        bar: dict[str, Any] = {"date": ts}
        for c in _BAR_COLS:
            bar[c] = row[c] if c in row.index else (False if c == "is_suspended" else np.nan)
        return bar

    @staticmethod
    def _bar_closes(indexed: dict[str, pd.DataFrame], ts: pd.Timestamp) -> dict[str, float]:
        out: dict[str, float] = {}
        for code, df in indexed.items():
            if ts in df.index:
                row = df.loc[ts]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                out[code] = row["close"]
        return out

    def _handle_corporate_actions(self, pf, indexed, prev_ts, ts) -> None:
        for code in list(pf.positions.keys()):
            df = indexed.get(code)
            if df is None or ts not in df.index or prev_ts not in df.index:
                continue
            factor_col = "cum_factor" if "cum_factor" in df.columns else "adj_factor"
            if factor_col not in df.columns:
                continue
            ratio = ExecutionEngine.detect_ex_factor_ratio(df.loc[prev_ts, factor_col], df.loc[ts, factor_col])
            cash_div = Backtester._cash_dividend_per_share(df, ts)
            if ratio > 1.001 or cash_div > 0:
                self.execution.apply_corporate_action(
                    pf,
                    code,
                    ratio,
                    cash_dividend_per_share_gross=cash_div,
                    action_date=ts,
                )

    # ------------------------------------------------------------
    # 年化因子 / 绩效
    # ------------------------------------------------------------

    @staticmethod
    def _annualization_factor(ts_index: pd.DatetimeIndex) -> float:
        """每年 bar 数 N（夏普 ×√N 用）：按自然年取 bar 数中位数。

        - 覆盖 ≥1 整年：丢掉明显不完整的边界年（bar 数 < 最大年 50%），取剩余各年中位数；
        - 不足一整年：退回 ``中位每日 bar 数 × 252``（fallback）。
        """
        idx = pd.DatetimeIndex(ts_index)
        if len(idx) < 2:
            return float(TRADING_DAYS_PER_YEAR)
        span_days = (idx[-1] - idx[0]).days
        by_year = pd.Series(1, index=idx).groupby(idx.year).count()
        if span_days >= 365 and len(by_year) >= 1:
            full = by_year[by_year >= 0.5 * by_year.max()]
            if len(full) >= 1:
                return max(float(full.median()), 1.0)
        # fallback：中位每日 bar 数 × 252
        by_day = pd.Series(1, index=idx).groupby(idx.normalize()).count()
        return max(float(by_day.median()) * TRADING_DAYS_PER_YEAR, 1.0)

    def _compute_metrics(self, equity_curve: pd.Series, trades: list[Trade]) -> BacktestResult:
        eq = equity_curve.to_numpy(dtype="float64")
        returns = pd.Series(eq).pct_change().dropna()
        n_periods = max(1, len(eq) - 1)
        ann = self._annualization_factor(equity_curve.index)  # 每年 bar 数

        total_return = float(eq[-1] - 1.0)
        annual_return = float((eq[-1]) ** (ann / n_periods) - 1.0) if eq[-1] > 0 else -1.0
        sharpe = calculate_sharpe(returns.tolist(), self.risk_free_rate, round(ann))
        mdd = calculate_max_drawdown(eq.tolist())
        calmar = safe_divide(annual_return, abs(mdd), 0.0)

        total_trades = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        win_rate = safe_divide(len(wins), total_trades, 0.0)
        avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
        profit_loss_ratio = safe_divide(avg_win, abs(avg_loss), 0.0)
        avg_holding_days = float(np.mean([t.holding_days for t in trades])) if trades else 0.0

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
            benchmark_return=0.0,
            alpha=0.0,
            beta=0.0,
            equity_curve=equity_curve,
            trades=trades,
            daily_positions=pd.DataFrame(),
        )

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            total_return=0.0, annual_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            calmar_ratio=0.0, win_rate=0.0, profit_loss_ratio=0.0, avg_holding_days=0.0,
            total_trades=0, benchmark_return=0.0, alpha=0.0, beta=0.0,
            equity_curve=pd.Series(dtype="float64"), trades=[],
            daily_positions=pd.DataFrame(),
        )


# ============================================================
# 模块自测  python -m src.engine.minute_backtester
# ============================================================

if __name__ == "__main__":
    from ..utils.helpers import init_logging

    init_logging(level="INFO")

    # 构造跨 2 个交易日的合成 1 分钟 bar（每天 4 根，便于肉眼核对 T+1）
    def _day_bars(day: str, base: float) -> pd.DataFrame:
        ts = pd.to_datetime([f"{day} 09:31", f"{day} 09:32", f"{day} 14:59", f"{day} 15:00"])
        px = base + np.arange(4) * 0.10
        return pd.DataFrame({
            "datetime": ts, "code": "000001.SZ",
            "open": px, "high": px + 0.05, "low": px - 0.05, "close": px,
            "volume": 1_000_000.0, "limit_up": base + 5, "limit_down": base - 5,
            "is_suspended": False, "adj_factor": 1.0,
        })

    df = pd.concat([_day_bars("2024-01-02", 10.0), _day_bars("2024-01-03", 11.0)], ignore_index=True)

    class _FirstBarBuy(BaseStrategy):
        """第 1 根 bar 发买入、最后 1 根 bar 发卖出，验证 T+1 与撮合滞后。"""
        strategy_name = "first_bar_buy"

        def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
            d = data["000001.SZ"]
            sig = self.empty_signals(pd.to_datetime(d["datetime"]), ["000001.SZ"])
            sig.iloc[0] = int(Signal.BUY)    # bar0 信号 → bar1 撮合（当日，T+1 冻结）
            sig.iloc[-2] = int(Signal.SELL)  # 倒数第2 bar 信号 → 末 bar 撮合（次日，已解冻）
            return self.validate_signals(sig)

    bt = MinuteBacktester(position_size=0.5)
    bt.execution.intraday = True
    res = bt.run(_FirstBarBuy(), "2024-01-02", "2024-01-03", data={"000001.SZ": df})
    logger.info(f"成交笔数={res.total_trades} 总收益={res.total_return:+.4%}")
    for t in res.trades:
        logger.info(f"  trade entry={t.entry_date} exit={t.exit_date} shares={t.shares} pnl={t.pnl:.2f}")
    assert res.total_trades >= 0
    logger.info(f"年化因子(每年bar数)实测={bt._annualization_factor(res.equity_curve.index):.0f}")
