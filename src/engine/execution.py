"""
撮合引擎（execution）—— A 股交易制度建模
==========================================

把策略信号在**次日开盘**撮合为真实成交，严格建模 A 股交易制度，并防未来函数。

撮合时序（防未来函数）
----------------------
``T`` 日收盘后 ``strategy.generate_signals(data<=T)`` → 信号；引擎在 ``T+1`` 日用 **T+1
开盘价**撮合（见 ``mode``）。本模块只处理"给定某根 ``T+1`` 日 bar + 一个订单 → 成交结果"，
日循环由 :mod:`backtester` 驱动。

建模要点
--------
1. **T+1 持仓/资金冻结**：由 :class:`~src.engine.portfolio.Portfolio` 承载（买入冻结股、
   卖出冻结资金）；本引擎只负责把可成交量算对。
2. **涨跌停一字板**：``high==low`` 且贴着 ``limit_up`` → 买入失败；贴着 ``limit_down`` →
   卖出失败。失败订单**直接丢弃**（不累积到 T+2）。
3. **滑点**：``fixed`` / ``percent``(默认) / ``open_gap``；买入向上、卖出向下，并按
   ``tick_size`` 归到合法报价档（买上取整、卖下取整，对己方不利），``min_ticks`` 保底。
4. **成交量上限**：单笔 ≤ 当日成交量 × ``volume_pct_limit``；超出按比例裁减（PARTIAL）。
   买入整手（100 股）向下取整；卖出可清掉不足整手的零股（全部卖出时）。
5. **最小下单金额**：低于 ``min_order_amount`` 直接忽略（IGNORED）。
6. **分红送股**：除权日由 :meth:`detect_ex_factor_ratio` 探测（累计复权因子日变化率
   >1.001），:meth:`apply_corporate_action` 把送转折算为持仓股数调整、现金分红计入现金
   （现金红利明细由数据层提供）。

费用
----
优先调用注入的 ``risk_manager.apply_commission(amount, is_buy)``；缺省用内置规则
（佣金 + 卖出印花税 + 最低佣金），便于在 ``risk_manager`` 尚未接入时独立运行。

成交量口径假设：``volume`` 视为**股数**。若数据源以"手"计，请在数据层换算或调整
``volume_pct_limit``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from loguru import logger

from ..strategy.base import Signal
from ..utils.helpers import truncate_to_100
from .portfolio import Portfolio

__all__ = ["ExecutionEngine", "Order", "OrderStatus"]

_EPS = 1e-9


class OrderStatus(StrEnum):
    """订单状态。"""

    PENDING = "pending"     # 待撮合
    FILLED = "filled"       # 全部成交
    PARTIAL = "partial"     # 部分成交（受成交量/资金限制裁减）
    FAILED = "failed"       # 撮合失败（停牌/一字板/无持仓）—— 丢弃，不顺延
    IGNORED = "ignored"     # 主动忽略（金额过小/不足 1 手）


@dataclass
class Order:
    """一笔订单（撮合前的意图 + 撮合后的结果）。"""

    code: str
    direction: int                      # Signal.BUY(1) / Signal.SELL(-1)
    shares: int                         # 期望股数（买入为目标股数；卖出为期望卖出股数）
    trade_date: date | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_shares: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    pnl: float = 0.0                    # 仅卖出有意义（已实现盈亏）
    reason: str = ""                    # 失败/忽略原因

    @property
    def is_buy(self) -> bool:
        return int(self.direction) > 0

    @property
    def amount(self) -> float:
        return self.filled_shares * self.fill_price


class ExecutionEngine:
    """撮合引擎。

    可用 :meth:`from_config` 从全局配置构造，或直接传参。
    """

    def __init__(
        self,
        *,
        slippage_type: str = "percent",
        fixed_amount: float = 0.01,
        percent_rate: float = 0.001,
        tick_size: float = 0.01,
        min_ticks: int = 1,
        volume_pct_limit: float = 0.10,
        min_order_amount: float = 1000.0,
        mode: str = "next_open",
        commission_rate: float = 0.00025,
        stamp_duty: float = 0.0005,
        min_commission: float = 5.0,
        intraday: bool = False,
        risk_manager: Any | None = None,
    ) -> None:
        self.slippage_type = str(slippage_type).lower().strip()
        if self.slippage_type not in ("fixed", "percent", "open_gap"):
            raise ValueError(f"未知滑点模式 {slippage_type!r}（fixed/percent/open_gap）")
        self.fixed_amount = float(fixed_amount)
        self.percent_rate = float(percent_rate)
        self.tick_size = float(tick_size)
        self.min_ticks = int(min_ticks)
        self.volume_pct_limit = float(volume_pct_limit)
        self.min_order_amount = float(min_order_amount)
        self.mode = str(mode).lower().strip()
        self.commission_rate = float(commission_rate)
        self.stamp_duty = float(stamp_duty)
        self.min_commission = float(min_commission)
        #: 日内（分钟级）模式：把成交价夹进当日涨跌停区间；日级（=False）行为不变
        self.intraday = bool(intraday)
        self.risk_manager = risk_manager

    @classmethod
    def from_config(cls, config: Any, risk_manager: Any | None = None) -> ExecutionEngine:
        """从全局配置（含 ``execution`` 与 ``risk`` 段）构造。"""
        ex = _section(config, "execution")
        rk = _section(config, "risk")
        slip = _section(ex, "slippage")
        return cls(
            slippage_type=_get(slip, "type", "percent"),
            fixed_amount=_get(slip, "fixed_amount", 0.01),
            percent_rate=_get(slip, "percent_rate", 0.001),
            tick_size=_get(slip, "tick_size", 0.01),
            min_ticks=_get(slip, "min_ticks", 1),
            volume_pct_limit=_get(ex, "volume_pct_limit", 0.10),
            min_order_amount=_get(ex, "min_order_amount", 1000.0),
            mode=_get(ex, "mode", "next_open"),
            commission_rate=_get(rk, "commission_rate", 0.00025),
            stamp_duty=_get(rk, "stamp_duty", 0.0005),
            min_commission=_get(rk, "min_commission", 5.0),
            intraday=_get(ex, "intraday", False),
            risk_manager=risk_manager,
        )

    # ------------------------------------------------------------
    # 价格：执行价（成交价）
    # ------------------------------------------------------------

    def execution_price(self, bar: dict[str, float], is_buy: bool) -> float:
        """根据 ``mode``/滑点/tick 计算成交价。

        ``mode='close'`` 用当日收盘价为基准，否则用开盘价（次日开盘撮合，默认）。
        """
        base_col = "close" if self.mode == "close" else "open"
        base = float(bar[base_col])
        t = self.tick_size

        if self.slippage_type == "open_gap":
            px = base  # 直接用实际开盘/收盘价，不另加滑点（已是合法报价档）
        else:
            slip = self.fixed_amount if self.slippage_type == "fixed" else base * self.percent_rate
            slip = max(slip, self.min_ticks * t)        # 保底至少 min_ticks 个 tick
            px = base + slip if is_buy else base - slip
            px = max(px, t)                              # 不为非正
            px = self._round_to_tick(px, is_buy)
        if self.intraday:
            px = self._clamp_to_limit(px, bar, is_buy)   # 日内：成交价不越当日涨跌停
        return px

    def _clamp_to_limit(self, px: float, bar: dict[str, float], is_buy: bool) -> float:
        """日内模式：把成交价夹进当日涨跌停区间（买不超 ``limit_up``、卖不低 ``limit_down``）。

        滑点叠加后买价可能越过 ``limit_up``、卖价越过 ``limit_down``——现实中不可能成交在停板外，
        故夹到停板价。``limit_up/limit_down`` 缺失（NaN）时不夹。完全封板（一字）由
        :meth:`is_one_word_up`/:meth:`is_one_word_down` 在 :meth:`match` 处直接拒单。
        """
        if is_buy:
            lu = bar.get("limit_up")
            if not self._is_nan(lu):
                px = min(px, float(lu))
        else:
            ld = bar.get("limit_down")
            if not self._is_nan(ld):
                px = max(px, float(ld))
        return px

    def _round_to_tick(self, px: float, is_buy: bool) -> float:
        """归到合法报价档：买入向上取整、卖出向下取整（对己方不利，保守）。"""
        t = self.tick_size
        if t <= 0:
            return round(px, 4)
        n = px / t
        if is_buy:
            return round(math.ceil(n - 1e-9) * t, 4)
        return round(math.floor(n + 1e-9) * t, 4)

    # ------------------------------------------------------------
    # 涨跌停一字板 / 停牌判定
    # ------------------------------------------------------------

    @staticmethod
    def _is_nan(x: Any) -> bool:
        try:
            return x is None or float(x) != float(x)
        except (TypeError, ValueError):
            return True

    def _is_tradable(self, bar: dict[str, float]) -> bool:
        """停牌或无有效开盘价 → 不可成交。"""
        if bool(bar.get("is_suspended", False)):
            return False
        base_col = "close" if self.mode == "close" else "open"
        px = bar.get(base_col)
        return not self._is_nan(px) and float(px) > 0

    def _tol(self, ref: float) -> float:
        """一字板比较容差：tick 与相对 1‰ 取大。"""
        return max(self.tick_size * 0.5, abs(ref) * 1e-3)

    def is_one_word_up(self, bar: dict[str, float]) -> bool:
        """涨停一字板：``high==low`` 且贴着 ``limit_up``。"""
        hi, lo, lu = bar.get("high"), bar.get("low"), bar.get("limit_up")
        if any(self._is_nan(v) for v in (hi, lo, lu)):
            return False
        hi, lo, lu = float(hi), float(lo), float(lu)
        return abs(hi - lo) <= self._tol(hi) and abs(hi - lu) <= self._tol(lu)

    def is_one_word_down(self, bar: dict[str, float]) -> bool:
        """跌停一字板：``high==low`` 且贴着 ``limit_down``。"""
        hi, lo, ld = bar.get("high"), bar.get("low"), bar.get("limit_down")
        if any(self._is_nan(v) for v in (hi, lo, ld)):
            return False
        hi, lo, ld = float(hi), float(lo), float(ld)
        return abs(hi - lo) <= self._tol(hi) and abs(lo - ld) <= self._tol(ld)

    # ------------------------------------------------------------
    # 费用
    # ------------------------------------------------------------

    def commission(self, amount: float, is_buy: bool) -> float:
        """成交费用：优先用注入的 risk_manager，否则内置规则。"""
        if self.risk_manager is not None and hasattr(self.risk_manager, "apply_commission"):
            return float(self.risk_manager.apply_commission(amount, is_buy))
        fee = amount * self.commission_rate
        if not is_buy:
            fee += amount * self.stamp_duty           # 印花税仅卖出
        return max(fee, self.min_commission)

    # ------------------------------------------------------------
    # 成交量上限
    # ------------------------------------------------------------

    def _volume_cap(self, bar: dict[str, float]) -> int | None:
        """单笔可成交股数上限；无量信息返回 None（不限制）。"""
        vol = bar.get("volume")
        if self._is_nan(vol) or float(vol) <= 0 or self.volume_pct_limit <= 0:
            return None
        return int(float(vol) * self.volume_pct_limit)

    # ------------------------------------------------------------
    # 撮合主入口
    # ------------------------------------------------------------

    def match(self, order: Order, bar: dict[str, float], portfolio: Portfolio) -> Order:
        """对单根 bar 撮合订单，更新 ``order`` 与 ``portfolio``，返回 ``order``。"""
        order.trade_date = order.trade_date or _to_date(bar.get("date"))
        if not self._is_tradable(bar):
            return self._set(order, OrderStatus.FAILED, "停牌/无有效成交价")

        if order.is_buy:
            if self.is_one_word_up(bar):
                return self._set(order, OrderStatus.FAILED, "涨停一字板，买入无法成交")
            return self._match_buy(order, bar, portfolio)
        else:
            if self.is_one_word_down(bar):
                return self._set(order, OrderStatus.FAILED, "跌停一字板，卖出无法成交")
            return self._match_sell(order, bar, portfolio)

    def _match_buy(self, order: Order, bar: dict[str, float], pf: Portfolio) -> Order:
        px = self.execution_price(bar, is_buy=True)
        target = int(order.shares)
        cap = self._volume_cap(bar)
        if cap is not None:
            target = min(target, cap)
        target = truncate_to_100(target)
        if target < 100:
            return self._set(order, OrderStatus.IGNORED, "成交量上限/不足 1 手")

        # 资金可负担性（含手续费）：不足则按手回退
        while target >= 100:
            amount = target * px
            fee = self.commission(amount, is_buy=True)
            if amount + fee <= pf.available_cash + _EPS:
                break
            target -= 100
        if target < 100:
            return self._set(order, OrderStatus.IGNORED, "可用资金不足 1 手")

        amount = target * px
        if amount < self.min_order_amount:
            return self._set(order, OrderStatus.IGNORED,
                             f"下单金额 {amount:.0f} < 最小 {self.min_order_amount:.0f}")
        fee = self.commission(amount, is_buy=True)
        pf.buy(order.code, target, px, fee, order.trade_date)
        return self._fill(order, target, px, fee, partial=target < int(order.shares))

    def _match_sell(self, order: Order, bar: dict[str, float], pf: Portfolio) -> Order:
        pos = pf.get_position(order.code)
        if pos is None or pos.available <= 0:
            return self._set(order, OrderStatus.FAILED, "无可卖持仓（或 T+1 冻结）")
        px = self.execution_price(bar, is_buy=False)
        target = min(int(order.shares), pos.available)
        cap = self._volume_cap(bar)
        if cap is not None:
            target = min(target, cap)
        # 非清仓则整手；清仓允许零股
        if target < pos.available:
            target = truncate_to_100(target)
        if target <= 0:
            return self._set(order, OrderStatus.IGNORED, "成交量上限导致可卖为 0")

        amount = target * px
        # 金额过小：若非整仓清出则忽略；整仓清出允许（零股清仓）
        if amount < self.min_order_amount and target < pos.available:
            return self._set(order, OrderStatus.IGNORED,
                             f"卖出金额 {amount:.0f} < 最小 {self.min_order_amount:.0f}")
        fee = self.commission(amount, is_buy=False)
        pnl = pf.sell(order.code, target, px, fee, order.trade_date)
        order.pnl = pnl
        return self._fill(order, target, px, fee, partial=target < int(order.shares))

    # ------------------------------------------------------------
    # 分红送股（除权日由 backtester 在日循环中触发）
    # ------------------------------------------------------------

    @staticmethod
    def detect_ex_factor_ratio(prev_adj_factor: float, cur_adj_factor: float) -> float:
        """累计复权因子日变化率；>1.001 视为除权除息日（送转/分红）。

        返回 ``cur/prev``（无效输入返回 1.0，表示非除权日）。
        """
        try:
            p, c = float(prev_adj_factor), float(cur_adj_factor)
        except (TypeError, ValueError):
            return 1.0
        if p <= 0 or c != c or p != p:
            return 1.0
        ratio = c / p
        return ratio if ratio > 1.001 else 1.0

    def apply_corporate_action(
        self,
        portfolio: Portfolio,
        code: str,
        factor_ratio: float,
        cash_dividend_per_share_after_tax: float = 0.0,
        cash_dividend_per_share_gross: float = 0.0,
        action_date: date | datetime | None = None,
    ) -> None:
        """在除权日对持仓做送转/分红调整（除权日前一交易日收盘后触发）。

        Args:
            factor_ratio: 复权因子变化率（>1 表示发生送转/分红，用于送股折算股数）。
            cash_dividend_per_share_after_tax: 税后每股现金红利（由数据层提供；>0 时计入现金）。
            cash_dividend_per_share_gross: 税前每股现金红利（gbbq 口径）；先全额入账，卖出时扣税。
            action_date: 除权除息日，用于递延红利税记录。
        """
        if not portfolio.has_position(code):
            return
        if cash_dividend_per_share_gross > 0 and hasattr(portfolio, "add_taxable_cash_dividend"):
            portfolio.add_taxable_cash_dividend(
                code, cash_dividend_per_share_gross, action_date or date.today()
            )
        elif cash_dividend_per_share_after_tax > 0:
            portfolio.add_cash_dividend(code, cash_dividend_per_share_after_tax)
        if factor_ratio > 1.001:
            # 不整手：A 股送转产生的零股可保留并一次性卖出（execution._match_sell 已支持
            # 清仓卖零股）。整手取整会凭空蒸发零股、低估持仓与收益。
            portfolio.apply_split(code, factor_ratio, round_lot=False)

    # ------------------------------------------------------------
    # 内部：状态置位
    # ------------------------------------------------------------

    @staticmethod
    def _set(order: Order, status: OrderStatus, reason: str) -> Order:
        order.status = status
        order.reason = reason
        order.filled_shares = 0
        logger.debug(f"[{order.code}] 订单 {status.value}: {reason}")
        return order

    @staticmethod
    def _fill(order: Order, shares: int, price: float, fee: float, partial: bool) -> Order:
        order.filled_shares = int(shares)
        order.fill_price = float(price)
        order.commission = float(fee)
        order.status = OrderStatus.PARTIAL if partial else OrderStatus.FILLED
        return order


# ============================================================
# 配置/日期辅助
# ============================================================


def _section(cfg: Any, key: str) -> Any:
    """取配置子段，兼容 DotDict / dict / 已是子段本身。"""
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


def _to_date(d: str | date | datetime | None) -> date | None:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    import pandas as pd

    return pd.Timestamp(d).date()


# ============================================================
# 模块自测  python -m src.engine.execution
# ============================================================

if __name__ == "__main__":
    from datetime import date as _d

    from ..utils.helpers import init_logging

    init_logging(level="INFO")
    eng = ExecutionEngine()
    pf = Portfolio(1_000_000)

    bar = {"date": _d(2024, 1, 3), "open": 10.0, "high": 10.6, "low": 9.8,
           "close": 10.4, "volume": 1_000_000, "limit_up": 11.0, "limit_down": 9.0,
           "is_suspended": False}

    o = Order(code="000001.SZ", direction=int(Signal.BUY), shares=5000)
    eng.match(o, bar, pf)
    logger.info(f"买单 {o.status.value} 成交 {o.filled_shares}@{o.fill_price} 费 {o.commission:.2f}")

    pf.settle_new_day()
    bar2 = dict(bar, date=_d(2024, 1, 4), open=11.0)
    o2 = Order(code="000001.SZ", direction=int(Signal.SELL), shares=5000)
    eng.match(o2, bar2, pf)
    logger.info(f"卖单 {o2.status.value} 成交 {o2.filled_shares}@{o2.fill_price} 盈亏 {o2.pnl:.2f}")

    # 涨停一字板买入失败
    board = dict(bar, open=11.0, high=11.0, low=11.0, close=11.0)
    o3 = Order(code="000001.SZ", direction=int(Signal.BUY), shares=1000)
    eng.match(o3, board, pf)
    logger.info(f"一字板买单 {o3.status.value}: {o3.reason}")
