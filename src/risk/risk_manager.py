"""
风控管理器（risk_manager）
===========================

集中实现 A 股交易的费用、仓位、止损规则，供撮合引擎与回测/实盘循环调用。

费用（``apply_commission``）
---------------------------
- 买入：``成交金额 × commission_rate``（默认万 2.5 = 0.00025）；
- 卖出：``成交金额 ×（commission_rate + stamp_duty）``（印花税默认万 5，仅卖出）；
- 每笔最低佣金 ``min_commission``（默认 5 元）。

仓位控制（``clip_position_size``）
----------------------------------
- 单票最大仓位：``总资产 × max_single_position``（默认 20%）；
- 单一行业最大仓位：``总资产 × max_industry_position``（默认 30%，需传 ``industry_map``）；
- **超限自动裁减**到限额（整手向下取整），而非直接拒单。

止损（``check_order`` / ``check_drawdown_stop``，有状态）
--------------------------------------------------------
- **单日止损**：当日浮亏 ≥ ``daily_stop_loss``（默认 5%）→ 当日**暂停新开仓**
  （已有持仓不动、卖出放行）；
- **累计回撤熔断**：相对历史峰值回撤 ≥ ``total_drawdown_stop``（默认 15%）→ **暂停所有
  交易**并写 CRITICAL 日志；一旦熔断保持停盘（不自动恢复）。

状态管理：每个交易日开盘调用 :meth:`on_new_day` 记录当日起点权益并刷新峰值；
:meth:`reset` 在回测开始时初始化。无状态初始化时止损判定安全返回不触发。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..utils.helpers import format_code, truncate_to_100

if TYPE_CHECKING:  # 仅类型提示，避免任何潜在导入环
    from ..engine.execution import Order
    from ..engine.portfolio import Portfolio

__all__ = ["RiskManager", "load_industry_map"]

_EPS = 1e-9


def load_industry_map(
    path: str | Path = "data_store/industry_map.parquet",
) -> dict[str, str]:
    """读取 ``scripts/fetch_industry_map.py`` 落盘的行业表为 ``{code: 行业}``。

    供 :meth:`RiskManager.clip_position_size` 的 ``industry_map`` 参数使用。
    代码统一过 :func:`format_code` 规范化；文件缺失或读失败返回空字典（退化为仅单票限仓）。

    注意：落盘为**拉取日快照**，长跨度回测直接用会有行业漂移/前视，需自行权衡。
    """
    p = Path(path)
    if not p.exists():
        logger.warning(f"行业映射文件不存在: {p}（先跑 scripts/fetch_industry_map.py）")
        return {}
    try:
        import pandas as pd

        df = pd.read_parquet(p)
        df = df.dropna(subset=["code", "industry"])
        return {format_code(str(c)): str(ind) for c, ind in zip(df["code"], df["industry"])}
    except Exception as exc:
        logger.warning(f"读取行业映射 {p} 失败: {exc}")
        return {}


class RiskManager:
    """费用 / 仓位 / 止损 风控管理器。"""

    def __init__(
        self,
        commission_rate: float = 0.00025,
        stamp_duty: float = 0.0005,
        min_commission: float = 5.0,
        max_single_position: float = 0.20,
        max_industry_position: float = 0.30,
        daily_stop_loss: float = 0.05,
        total_drawdown_stop: float = 0.15,
    ) -> None:
        self.commission_rate = float(commission_rate)
        self.stamp_duty = float(stamp_duty)
        self.min_commission = float(min_commission)
        self.max_single_position = float(max_single_position)
        self.max_industry_position = float(max_industry_position)
        self.daily_stop_loss = float(daily_stop_loss)
        self.total_drawdown_stop = float(total_drawdown_stop)

        # 有状态字段
        self._peak_equity: float | None = None
        self._day_start_equity: float | None = None
        self._halted: bool = False   # 累计回撤熔断后置位

    @classmethod
    def from_config(cls, config: Any) -> RiskManager:
        """从全局配置的 ``risk`` 段构造。"""
        rk = config.get("risk", config) if hasattr(config, "get") else (config or {})

        def g(k, d):
            v = rk.get(k, d) if hasattr(rk, "get") else getattr(rk, k, d)
            return d if v is None else v

        return cls(
            commission_rate=g("commission_rate", 0.00025),
            stamp_duty=g("stamp_duty", 0.0005),
            min_commission=g("min_commission", 5.0),
            max_single_position=g("max_single_position", 0.20),
            max_industry_position=g("max_industry_position", 0.30),
            daily_stop_loss=g("daily_stop_loss", 0.05),
            total_drawdown_stop=g("total_drawdown_stop", 0.15),
        )

    # ------------------------------------------------------------
    # 费用
    # ------------------------------------------------------------

    def apply_commission(self, amount: float, is_buy: bool) -> float:
        """计算单笔成交费用（含最低佣金；卖出含印花税）。"""
        amount = abs(float(amount))
        fee = amount * self.commission_rate
        if not is_buy:
            fee += amount * self.stamp_duty
        return max(fee, self.min_commission)

    # ------------------------------------------------------------
    # 状态：交易日切换 / 初始化
    # ------------------------------------------------------------

    def reset(self, initial_equity: float) -> None:
        """回测开始时初始化峰值与当日起点。"""
        eq = float(initial_equity)
        self._peak_equity = eq
        self._day_start_equity = eq
        self._halted = False

    def on_new_day(self, portfolio: Portfolio) -> None:
        """每个交易日开盘调用：记录当日起点权益并刷新历史峰值。"""
        eq = float(portfolio.total_value)
        self._day_start_equity = eq
        self._peak_equity = eq if self._peak_equity is None else max(self._peak_equity, eq)

    @property
    def halted(self) -> bool:
        """是否已触发累计回撤熔断。"""
        return self._halted

    # ------------------------------------------------------------
    # 止损 / 熔断
    # ------------------------------------------------------------

    def current_drawdown(self, portfolio: Portfolio) -> float:
        """相对历史峰值的回撤（负数）；峰值未知时返回 0。"""
        eq = float(portfolio.total_value)
        peak = self._peak_equity if self._peak_equity is not None else eq
        peak = max(peak, eq)
        self._peak_equity = peak
        return (eq - peak) / peak if peak > 0 else 0.0

    def daily_loss(self, portfolio: Portfolio) -> float:
        """当日相对开盘起点的盈亏（负数为亏）；起点未知时返回 0。"""
        if self._day_start_equity is None or self._day_start_equity <= 0:
            return 0.0
        return (float(portfolio.total_value) - self._day_start_equity) / self._day_start_equity

    def check_drawdown_stop(self, portfolio: Portfolio) -> bool:
        """累计回撤是否达到熔断线。触发则置位 ``_halted`` 并写 CRITICAL 日志。"""
        dd = self.current_drawdown(portfolio)
        if dd <= -self.total_drawdown_stop + _EPS:
            if not self._halted:
                logger.critical(
                    f"累计回撤 {dd:.2%} ≥ 熔断线 {self.total_drawdown_stop:.0%}，暂停所有交易！"
                    f"（当前权益 {portfolio.total_value:,.2f}，峰值 {self._peak_equity:,.2f}）"
                )
            self._halted = True
            return True
        return False

    def daily_stop_triggered(self, portfolio: Portfolio) -> bool:
        """当日浮亏是否达到单日止损线（暂停新开仓）。"""
        loss = self.daily_loss(portfolio)
        triggered = loss <= -self.daily_stop_loss + _EPS
        if triggered:
            logger.warning(f"单日浮亏 {loss:.2%} ≥ 止损线 {self.daily_stop_loss:.0%}，当日暂停新开仓")
        return triggered

    # ------------------------------------------------------------
    # 订单准入
    # ------------------------------------------------------------

    def check_order(self, order: Order, portfolio: Portfolio) -> bool:
        """订单是否放行。

        - 累计回撤熔断 → 一律拒绝（含卖出，全面停盘）；
        - 单日止损触发 → 拒绝**新开仓（买入）**，放行卖出（允许减仓/止损离场）;
        - 其余放行（具体可成交量/资金由撮合引擎裁定）。
        """
        if self._halted or self.check_drawdown_stop(portfolio):
            logger.debug(f"[{order.code}] 风控拒单：已触发累计回撤熔断")
            return False
        if order.is_buy and self.daily_stop_triggered(portfolio):
            logger.debug(f"[{order.code}] 风控拒单：当日止损，暂停新开仓")
            return False
        return True

    # ------------------------------------------------------------
    # 仓位裁减
    # ------------------------------------------------------------

    def clip_position_size(
        self,
        order: Order,
        portfolio: Portfolio,
        price: float | None = None,
        industry_map: dict[str, str] | None = None,
    ) -> Order:
        """将买入订单裁减到单票/单行业仓位上限内（整手向下取整）。卖出原样返回。

        Args:
            order: 待裁减订单（``order.shares`` 为期望买入股数）。
            portfolio: 当前组合（用于总资产与现有持仓市值）。
            price: 估算成交价；None 时用 ``order.fill_price``，仍为 0 则无法按市值裁减，原样返回。
            industry_map: ``{code: 行业}``；提供时施加单行业上限，否则仅单票上限。

        Returns:
            裁减后的 ``order``（``shares`` 可能减少；超限到 0 时 ``shares=0``）。
        """
        if not order.is_buy:
            return order
        px = float(price) if price is not None else float(order.fill_price or 0.0)
        if px <= 0:
            logger.debug(f"[{order.code}] clip_position_size 无有效价格，跳过市值裁减")
            return order

        total = float(portfolio.total_value)
        if total <= 0:
            order.shares = 0
            return order

        pos = portfolio.get_position(order.code)
        held_value = pos.market_value if pos is not None else 0.0

        # 单票上限剩余可买市值
        single_cap = total * self.max_single_position
        room = single_cap - held_value

        # 单行业上限剩余可买市值
        if industry_map is not None:
            ind = industry_map.get(order.code)
            if ind:
                ind_cap = total * self.max_industry_position
                ind_value = sum(
                    p.market_value
                    for c, p in portfolio.positions.items()
                    if industry_map.get(c) == ind
                )
                room = min(room, ind_cap - ind_value)

        if room <= px * 100:  # 连一手都放不下
            logger.debug(f"[{order.code}] 仓位已达上限，裁减为 0（room={room:.2f}）")
            order.shares = 0
            return order

        max_shares = truncate_to_100(int(room / px))
        if order.shares > max_shares:
            logger.debug(f"[{order.code}] 仓位裁减 {order.shares} → {max_shares}（上限市值 {room:.2f}）")
            order.shares = max_shares
        return order

    def __repr__(self) -> str:
        return (
            f"<RiskManager comm={self.commission_rate} stamp={self.stamp_duty} "
            f"single={self.max_single_position:.0%} industry={self.max_industry_position:.0%} "
            f"daily_stop={self.daily_stop_loss:.0%} dd_stop={self.total_drawdown_stop:.0%} "
            f"halted={self._halted}>"
        )


# ============================================================
# 模块自测  python -m src.risk.risk_manager
# ============================================================

if __name__ == "__main__":
    from ..engine.execution import Order
    from ..engine.portfolio import Portfolio
    from ..strategy.base import Signal
    from ..utils.helpers import init_logging

    init_logging(level="INFO")
    rm = RiskManager()

    # 费用
    logger.info(f"买入 10万 费用: {rm.apply_commission(100_000, True):.2f}")
    logger.info(f"卖出 10万 费用: {rm.apply_commission(100_000, False):.2f}")
    logger.info(f"小额 1000 费用(最低佣金): {rm.apply_commission(1000, True):.2f}")

    # 仓位裁减：100万总资产，单票上限 20% = 20万 → @10 元最多 2万股
    pf = Portfolio(1_000_000)
    rm.reset(1_000_000)
    o = Order(code="000001.SZ", direction=int(Signal.BUY), shares=50_000)
    rm.clip_position_size(o, pf, price=10.0)
    logger.info(f"裁减后股数: {o.shares}（期望 20000）")

    # 止损：手动改 portfolio 价值模拟亏损
    rm.on_new_day(pf)
    pf.cash = 940_000  # 当日 -6%
    logger.info(f"当日止损触发: {rm.daily_stop_triggered(pf)} | 买单放行: {rm.check_order(o, pf)}")
    sell = Order(code="000001.SZ", direction=int(Signal.SELL), shares=100)
    logger.info(f"止损日卖单放行: {rm.check_order(sell, pf)}")

    # 熔断
    pf.cash = 800_000  # -20%
    logger.info(f"回撤熔断: {rm.check_drawdown_stop(pf)} | 熔断后卖单放行: {rm.check_order(sell, pf)}")
