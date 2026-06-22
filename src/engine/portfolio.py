"""
持仓与资金管理（portfolio）
============================

回测引擎的账户状态中枢：维护现金、持仓、T+1 冻结、逐日估值与已实现盈亏，并产出可供
:mod:`backtester` 组装净值曲线 / 交易明细的快照与成交记录。

A 股资金 / 持仓规则
-------------------
- **T+1 持仓冻结**：买入当日新增股数 ``frozen``，**当日不可卖**；下一交易日开盘前
  （:meth:`settle_new_day`）解冻为可卖。
- **T+1 资金（默认不冻结，符合 A 股真实规则）**：A 股卖出所得资金**当日即可继续买入**
  股票（仅不可当日转出银行，回测不模拟出金），故默认 ``t1_cash_freeze=False``。设 ``True``
  则把卖出所得计入 ``frozen_cash`` 当日不可再买——一种**更保守、非真实**的口径，仅按需启用。
  注意"卖出当日买入的股票"始终受 T+1 约束（由 ``Position.frozen`` 管），与本资金开关无关。
- **可用资金**：``available_cash = cash - frozen_cash``（``cash`` 为含冻结的总现金）。
- **总资产**：``total_value = cash + market_value``。

成本与盈亏口径
--------------
- ``avg_cost`` 为**含买入手续费**的每股成本（成本基）；
- 卖出已实现盈亏 = 卖出净额（已扣卖出手续费/印花税）− 卖出股数 × ``avg_cost``，
  即同时扣除买卖两端费用，口径干净；
- 每笔卖出产出一条 round-trip 成交记录（``entry=建仓日/成本价``，``exit=卖出日/价``），
  供 backtester 统计胜率 / 盈亏比 / 持仓天数（"一买一卖为一笔"，均价法）。

分红送股（除权除息）
--------------------
由 :mod:`execution` 在除权日触发，调用 :meth:`apply_split`（送转，股数×比例，默认保留零股）与
:meth:`add_cash_dividend`（现金分红，税后每股红利×持股数计入现金，并记为当期已实现收益，不冲减成本）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from loguru import logger

from ..utils.helpers import truncate_to_100

__all__ = ["Portfolio", "Position"]

#: 浮点比较容差（资金校验用）
_EPS = 1e-6


@dataclass
class Position:
    """单只股票持仓。"""

    code: str
    shares: int = 0                 # 总持股
    frozen: int = 0                 # 当日买入冻结（T+1 不可卖）
    avg_cost: float = 0.0           # 含买入手续费的每股成本（成本基）
    open_date: date | datetime | None = None  # 建仓时点（清仓后重置）：日级为 date，分钟级为 datetime
    last_price: float = 0.0         # 最新市价（mark-to-market）

    @property
    def available(self) -> int:
        """当日可卖股数（已解冻部分）。"""
        return self.shares - self.frozen

    @property
    def cost_value(self) -> float:
        """持仓成本市值（avg_cost × shares）。"""
        return self.avg_cost * self.shares

    @property
    def market_value(self) -> float:
        """持仓市值（last_price × shares）。"""
        return self.last_price * self.shares

    @property
    def unrealized_pnl(self) -> float:
        """浮动盈亏。"""
        return (self.last_price - self.avg_cost) * self.shares


class Portfolio:
    """账户：现金 + 持仓 + T+1 冻结 + 估值 + 已实现盈亏。

    Args:
        initial_cash: 初始资金（元）。
        t1_cash_freeze: 卖出所得是否 T+1 冻结而当日不可再买。默认 ``False``（符合 A 股：
            回款当日可买）；``True`` 为更保守的非真实口径。
    """

    def __init__(self, initial_cash: float, t1_cash_freeze: bool = False) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)         # 含冻结的总现金
        self.frozen_cash = 0.0                  # 当日卖出冻结资金（T+1）
        self.t1_cash_freeze = bool(t1_cash_freeze)
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        #: round-trip 成交记录（dict 字段与 backtester.Trade 对齐）
        self.realized_trades: list[dict] = []

    # ------------------------------------------------------------
    # 资金 / 估值
    # ------------------------------------------------------------

    @property
    def available_cash(self) -> float:
        """可用资金 = 总现金 − 冻结现金。"""
        return self.cash - self.frozen_cash

    @property
    def market_value(self) -> float:
        """全部持仓市值。"""
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        """总资产 = 现金 + 持仓市值。"""
        return self.cash + self.market_value

    def has_position(self, code: str) -> bool:
        """是否持有该股（shares>0）。"""
        pos = self.positions.get(code)
        return pos is not None and pos.shares > 0

    def get_position(self, code: str) -> Position | None:
        return self.positions.get(code)

    def position_shares(self) -> dict[str, int]:
        """``{code: shares}``（仅含 shares>0）。"""
        return {c: p.shares for c, p in self.positions.items() if p.shares > 0}

    # ------------------------------------------------------------
    # 交易日切换：解冻 T+1
    # ------------------------------------------------------------

    def settle_new_day(self) -> None:
        """新交易日开盘前结算：解冻 T+1 持仓与资金。每个交易日开始时调用一次。"""
        self.frozen_cash = 0.0
        for pos in self.positions.values():
            pos.frozen = 0

    # ------------------------------------------------------------
    # 撮合后记账：买 / 卖
    # ------------------------------------------------------------

    def buy(self, code: str, shares: int, price: float, commission: float, trade_date) -> None:
        """买入记账（成交价 ``price``、手续费 ``commission`` 由 execution 计算后传入）。

        扣减总现金，更新加权成本，新增股数计入 T+1 冻结。
        """
        shares = int(shares)
        if shares <= 0:
            return
        cost = shares * float(price) + float(commission)
        if cost > self.available_cash + _EPS:
            raise ValueError(
                f"[{code}] 买入资金不足：需 {cost:.2f}，可用 {self.available_cash:.2f}"
            )
        self.cash -= cost
        pos = self.positions.get(code)
        if pos is None:
            pos = Position(code=code)
            self.positions[code] = pos
        new_cost_value = pos.cost_value + cost          # 含本次手续费
        pos.shares += shares
        pos.frozen += shares                             # T+1：当日买入不可卖
        pos.avg_cost = new_cost_value / pos.shares
        pos.last_price = float(price)
        if pos.open_date is None:
            pos.open_date = _to_dt(trade_date)

    def sell(self, code: str, shares: int, price: float, commission: float, trade_date) -> float:
        """卖出记账，返回本笔已实现盈亏（净额，已扣买卖两端费用）。

        校验可卖股数（T+1），更新现金（``t1_cash_freeze`` 时冻结当日所得），并产出一条
        round-trip 成交记录。
        """
        shares = int(shares)
        if shares <= 0:
            return 0.0
        pos = self.positions.get(code)
        if pos is None or pos.shares <= 0:
            raise ValueError(f"[{code}] 无持仓可卖")
        if shares > pos.available:
            raise ValueError(
                f"[{code}] 可卖不足：拟卖 {shares}，可卖 {pos.available}（持有 {pos.shares}，冻结 {pos.frozen}）"
            )
        price = float(price)
        proceeds = shares * price - float(commission)
        cost_basis = pos.avg_cost * shares
        pnl = proceeds - cost_basis
        self.realized_pnl += pnl

        entry_date = pos.open_date
        td = _to_dt(trade_date)
        self.realized_trades.append(
            {
                "code": code,
                "entry_date": entry_date,
                "exit_date": td,
                "entry_price": pos.avg_cost,
                "exit_price": price,
                "shares": shares,
                "pnl": pnl,
                "pnl_pct": (pnl / cost_basis) if cost_basis > 0 else 0.0,
                "holding_days": (td - entry_date).days if entry_date else 0,
            }
        )

        pos.shares -= shares
        self.cash += proceeds
        if self.t1_cash_freeze:
            self.frozen_cash += proceeds                 # 卖出所得当日冻结
        if pos.shares == 0:
            # 清仓：重置成本/建仓日；移除空仓位
            del self.positions[code]
        return pnl

    # ------------------------------------------------------------
    # 估值
    # ------------------------------------------------------------

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """用当日价更新持仓市价；缺价/停牌（NaN）的股票保留上一市价。"""
        for code, pos in self.positions.items():
            px = prices.get(code)
            if px is None:
                continue
            try:
                pxf = float(px)
            except (TypeError, ValueError):
                continue
            if pxf == pxf and pxf > 0:  # 非 NaN 且为正
                pos.last_price = pxf

    # ------------------------------------------------------------
    # 分红送股（除权日由 execution 触发）
    # ------------------------------------------------------------

    def apply_split(self, code: str, ratio: float, round_lot: bool = False) -> int:
        """送股/转增：持仓股数 ×= ratio。

        默认 ``round_lot=False``——A 股送转产生的**零股可保留**（且可一次性卖出），整手取整
        会凭空蒸发零股、低估持仓与收益；仅在确需整手时显式传 ``round_lot=True``。
        总成本不变 → ``avg_cost`` 等比下调。返回调整后的股数。``ratio<=1`` 不处理。
        """
        pos = self.positions.get(code)
        if pos is None or pos.shares <= 0 or ratio <= 1.0 + _EPS:
            return pos.shares if pos else 0
        old_cost_value = pos.cost_value
        new_shares = pos.shares * ratio
        new_shares = truncate_to_100(int(new_shares)) if round_lot else int(new_shares)
        if new_shares <= 0:
            return pos.shares
        # 冻结股数等比放大（仍受 T+1 约束）
        pos.frozen = min(new_shares, int(pos.frozen * ratio))
        pos.shares = new_shares
        pos.avg_cost = old_cost_value / new_shares
        logger.debug(f"[{code}] 送转调整 ratio={ratio:.4f} → shares={new_shares} avg_cost={pos.avg_cost:.4f}")
        return new_shares

    def add_cash_dividend(self, code: str, per_share_after_tax: float) -> float:
        """现金分红：现金 += 持股数 × 税后每股红利，并计入已实现收益。返回入账现金（无持仓返回 0）。

        分红视为**当期现金收入**记入 ``realized_pnl``，**不冲减** ``avg_cost``：成本基始终
        是真实买入成本，既不会被压成负数，也不会扭曲后续卖出的已实现盈亏（旧实现用
        ``max(0, cost-amount)`` 冲减成本，超额分红会被静默吞掉，导致卖出 pnl 虚高）。
        """
        pos = self.positions.get(code)
        if pos is None or pos.shares <= 0 or per_share_after_tax <= 0:
            return 0.0
        amount = pos.shares * float(per_share_after_tax)
        self.cash += amount
        self.realized_pnl += amount          # 分红为当期现金收益，成本基不变
        logger.debug(f"[{code}] 现金分红 {amount:.2f}（每股税后 {per_share_after_tax:.4f}）")
        return amount

    # ------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------

    def snapshot(self, trade_date) -> dict:
        """返回当日账户快照（供 backtester 组装净值曲线 / 持仓表）。"""
        return {
            "date": _to_dt(trade_date),
            "cash": self.cash,
            "frozen_cash": self.frozen_cash,
            "available_cash": self.available_cash,
            "market_value": self.market_value,
            "total_value": self.total_value,
            "n_positions": len([p for p in self.positions.values() if p.shares > 0]),
        }

    def __repr__(self) -> str:
        return (
            f"<Portfolio total={self.total_value:,.2f} cash={self.cash:,.2f} "
            f"frozen={self.frozen_cash:,.2f} positions={len(self.positions)} "
            f"realized_pnl={self.realized_pnl:,.2f}>"
        )


def _to_dt(d: str | date | datetime) -> date | datetime:
    """归一时间戳但**不截断到日**：``datetime``/``date`` 原样返回，``str`` → ``pd.Timestamp``。

    日级回测传入 ``date`` → 仍得 ``date``（行为与旧 ``_to_date`` 一致）；分钟级传入
    ``datetime`` → 保留时分秒，使 ``open_date`` 与成交记录的 entry/exit 具备日内精度。
    """
    if isinstance(d, (datetime, date)):  # datetime 是 date 的子类，二者均原样返回
        return d
    import pandas as pd  # 局部导入，避免无谓依赖

    return pd.Timestamp(d)


# ============================================================
# 模块自测  python -m src.engine.portfolio
# ============================================================

if __name__ == "__main__":
    from datetime import date as _d

    from ..utils.helpers import init_logging

    init_logging(level="INFO")
    pf = Portfolio(initial_cash=1_000_000, t1_cash_freeze=True)
    logger.info(f"初始: {pf}")

    # T 日买入 1000 股 @10，手续费 5
    pf.buy("000001.SZ", 1000, 10.0, 5.0, _d(2024, 1, 2))
    logger.info(f"买入后 可用={pf.available_cash:.2f} 持仓可卖={pf.get_position('000001.SZ').available}")
    assert pf.get_position("000001.SZ").available == 0  # T+1 冻结

    # 次日解冻 + 标记市价 11
    pf.settle_new_day()
    pf.mark_to_market({"000001.SZ": 11.0})
    logger.info(f"次日 可卖={pf.get_position('000001.SZ').available} 总资产={pf.total_value:,.2f}")

    # 卖出 1000 @11，手续费 8（含印花）
    pnl = pf.sell("000001.SZ", 1000, 11.0, 8.0, _d(2024, 1, 3))
    logger.info(f"卖出 已实现盈亏={pnl:.2f} 冻结现金={pf.frozen_cash:.2f}")
    logger.info(f"成交记录: {pf.realized_trades[-1]}")
    logger.info(f"最终: {pf}")
