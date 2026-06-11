"""
公共工具函数模块
================
本模块提供全局工具函数，供 data/factor/strategy/engine/risk 五大模块共同调用。

主要功能：
1. 日志初始化（init_logging）
2. 重试装饰器（retry）
3. 日期解析（parse_date）
4. 股票代码格式化（format_code / to_akshare_code / to_baostock_code）
5. A股涨跌停判断（get_limit_pct）
6. 绩效指标计算（calculate_sharpe / calculate_max_drawdown）

设计原则：
- 纯函数为主，无状态
- 不依赖业务模块（避免循环引用）
- 所有函数附带类型注解
"""

import sys
import time
from collections.abc import Callable
from datetime import date, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from loguru import logger

# ============================================================
# 1. 日志系统初始化
# ============================================================

def init_logging(
    log_dir: str = "logs",
    app_name: str = "quant_pro",
    level: str = "INFO",
    rotation: str = "00:00",
    retention: str = "30 days"
) -> None:
    """
    初始化 loguru 日志系统

    特性：
    - 同时输出到控制台（彩色）和文件
    - 按日期自动分割日志文件
    - 自动清理过期日志（默认保留30天）

    Args:
        log_dir: 日志存储目录
        app_name: 日志文件名前缀
        level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        rotation: 日志轮转时间（"00:00"=每天0点切割新文件）
        retention: 日志保留时长

    Example:
        >>> init_logging(level="DEBUG")
        >>> logger.info("系统启动")
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 移除 loguru 默认的 stderr 处理器
    logger.remove()

    # 控制台输出（带颜色，便于开发调试）
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    )

    # 文件输出（无颜色码，便于 grep 和分析）
    log_file = log_path / f"{app_name}_{{time:YYYY-MM-DD}}.log"
    logger.add(
        str(log_file),
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    logger.info(f"日志系统初始化完成 | 目录: {log_path.absolute()} | 级别: {level}")


# ============================================================
# 2. 重试装饰器（指数退避）
# ============================================================

def retry(
    max_attempts: int = 3,
    delays: list[int | float] | None = None,
    exceptions: tuple = (Exception,),
    on_failure: Callable | None = None
) -> Callable:
    """
    指数退避重试装饰器

    用于网络请求、数据库连接等可能失败的操作。

    Args:
        max_attempts: 最大尝试次数（含首次）
        delays: 每次重试前的等待秒数列表，默认 [1, 2, 4]
                超出列表长度时使用最后一个值
        exceptions: 触发重试的异常类型，默认捕获所有 Exception
        on_failure: 最终失败的回调函数（接收最后一个异常对象）

    Example:
        >>> @retry(max_attempts=3, delays=[1, 2, 4])
        ... def fetch_data(code):
        ...     return akshare.stock_zh_a_hist(symbol=code)

    Note:
        - 等待策略：第1次失败等1秒，第2次等2秒，第3次等4秒
        - 最后一次失败不再等待，直接 raise
    """
    if delays is None:
        delays = [1, 2, 4]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception: BaseException | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        # 取对应索引的等待时间，超出时用最后一个
                        wait_time = delays[min(attempt, len(delays) - 1)]
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt + 1}/{max_attempts} 次失败: "
                            f"{type(e).__name__}: {e} | {wait_time}秒后重试..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"函数 {func.__name__} 已重试 {max_attempts} 次，最终失败: "
                            f"{type(e).__name__}: {e}"
                        )

            # 执行失败回调
            if on_failure is not None and last_exception is not None:
                on_failure(last_exception)

            # 最终抛出原始异常
            assert last_exception is not None
            raise last_exception

        return wrapper
    return decorator


# ============================================================
# 3. 路径与文件工具
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    """
    确保目录存在，不存在则递归创建

    Args:
        path: 目录路径

    Returns:
        Path 对象
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================
# 4. 日期解析工具
# ============================================================

def parse_date(date_input: str | date | datetime) -> date:
    """
    统一解析多种日期输入为 date 对象

    支持格式：
    - date 对象 → 直接返回
    - datetime 对象 → 转 date
    - "2024-01-15" / "20240115" / "2024/01/15" / "15-01-2024"

    Args:
        date_input: 日期输入（字符串/date/datetime）

    Returns:
        date 对象

    Raises:
        ValueError: 无法解析的格式
    """
    # 注意：datetime 是 date 的子类，必须先检查 datetime
    if isinstance(date_input, datetime):
        return date_input.date()
    if isinstance(date_input, date):
        return date_input
    if isinstance(date_input, str):
        formats = ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%d-%m-%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_input, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"无法解析日期格式: {date_input}")

    raise ValueError(f"不支持的日期类型: {type(date_input)}")


# ============================================================
# 5. A股股票代码格式化
# ============================================================

def format_code(code: str) -> str:
    """
    统一股票代码为标准格式：XXXXXX.Exchange

    支持输入：
    - "000001.SZ"  → "000001.SZ"  （已标准）
    - "000001"     → "000001.SZ"  （根据前缀推断交易所）
    - "sh600000"   → "600000.SH"  （akshare 格式）
    - "SZ.000001"  → 不支持，请用 to_baostock_code 反向转换

    交易所判断规则（按代码前缀，截至 2024 年最新）：
    - 上交所（.SH）：600/601/603/605 主板、688 科创板、689 科创板CDR、900 B股
    - 深交所（.SZ）：000/001/003 主板、002 原中小板（已并入主板）、
                      300/301 创业板、200 B股
    - 北交所（.BJ）：43/83/87/88/92

    Args:
        code: 任意格式的股票代码

    Returns:
        标准格式代码，如 "000001.SZ"
    """
    code = code.strip().upper()

    # 已是标准格式
    if code.endswith(".SH") or code.endswith(".SZ") or code.endswith(".BJ"):
        return code

    # 处理 sh/sz/bj 前缀（如 sh600000）
    if code.startswith("SH"):
        return code[2:] + ".SH"
    if code.startswith("SZ"):
        return code[2:] + ".SZ"
    if code.startswith("BJ"):
        return code[2:] + ".BJ"

    # 纯数字：根据前缀推断
    if code.isdigit():
        # 上交所
        if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
            return code + ".SH"
        # 北交所（必须在 "8"/"4" 兜底之前判断）
        if code.startswith(("43", "83", "87", "88", "92")):
            return code + ".BJ"
        # 深交所
        if code.startswith(("000", "001", "002", "003", "300", "301", "302","200")):
            return code + ".SZ"
        # 其它 4/8 开头：保守归北交所（原新三板大多已退市/转板）
        if code.startswith(("4", "8")):
            return code + ".BJ"
        # 兜底
        return code + ".SH"

    return code


def to_akshare_code(code: str) -> str:
    """
    转换为 akshare 新浪系接口格式
    000001.SZ → sz000001
    830799.BJ → bj830799

    适用于：stock_zh_a_daily / stock_zh_a_spot 等新浪系 API。
    注意：stock_zh_a_hist（东财接口）需要纯 6 位代码，不能用本函数输出，
          应直接使用 format_code(code).split(".")[0]。
    """
    code = format_code(code)
    parts = code.split(".")
    if len(parts) == 2:
        return parts[1].lower() + parts[0]
    return code


def to_baostock_code(code: str) -> str:
    """
    转换为 baostock 格式
    000001.SZ → sz.000001
    600519.SH → sh.600519

    注意：baostock 数据源不覆盖北交所（.BJ），传入北交所代码虽会返回 "bj.xxxxxx"
          格式字符串，但查询接口实际会返回空。
    """
    code = format_code(code)
    parts = code.split(".")
    if len(parts) == 2:
        return parts[1].lower() + "." + parts[0]
    return code


# ============================================================
# 6. A股涨跌停判断
# ============================================================

def get_limit_pct(
    code: str,
    is_st: bool = False,
    trading_days_since_list: int | None = None,
) -> float:
    """
    获取股票涨跌停限制百分比

    A股涨跌停规则（2023-02-17 全面注册制后）：
    - 主板（60x/00x）：±10%
    - 主板 ST/*ST 股票：±5%
    - 创业板（300/301）：±20%
    - 科创板（688/689）：±20%
    - 北交所（43/83/87/88/92）：±30%
    - 新股上市规则（仅在传入 trading_days_since_list 时生效）：
        * 沪深主板 / 创业板 / 科创板：上市后前 5 个交易日不设涨跌幅限制，
          第 6 交易日起恢复为各板块常规限制
        * 北交所：上市首日不设涨跌幅限制，次交易日起 ±30%

    Args:
        code: 股票代码（任意格式）
        is_st: 是否为 ST 股票（影响主板限制）
        trading_days_since_list: 距上市首日的"已交易天数"（含上市当日，从 1 计数）。
            - None（默认）：忽略新股规则，按板块常规限制返回。
            - 1：上市首日。
            - 2~5：上市后第 2~5 个交易日。
            - >=6：恢复常规限制。
            注意：必须传入"交易日"差值而非自然日；交易日历计算应由调用方
            （如数据/回测引擎中的 trading_calendar）完成，避免本工具函数引入额外依赖。

    Returns:
        涨跌停限制（小数形式，如 0.10 表示 ±10%）。
        新股不设限期间返回 float('inf')，调用方需据此跳过涨跌停过滤。
    """
    code = format_code(code)
    suffix = code.split(".")[-1] if "." in code else ""
    base = code.split(".")[0]
    is_bj = suffix == "BJ"

    # 新股上市无涨跌幅限制窗口
    if trading_days_since_list is not None and trading_days_since_list >= 1:
        if is_bj and trading_days_since_list <= 1:
            # 北交所：仅上市首日不设限
            return float("inf")
        if (not is_bj) and trading_days_since_list <= 5:
            # 沪深主板 / 创业板 / 科创板：前 5 个交易日不设限
            return float("inf")

    # 北交所（按后缀判断更可靠，覆盖 43/83/87/88/92 及其它 4/8 开头）
    if is_bj:
        return 0.30
    # 创业板
    if base.startswith(("300", "301")):
        return 0.20
    # 科创板
    if base.startswith(("688", "689")):
        return 0.20
    # 主板 ST 股
    if is_st:
        return 0.05
    # 主板普通股
    return 0.10


# ============================================================
# 7. A股最小交易单位（100股）
# ============================================================

def truncate_to_100(quantity: int) -> int:
    """
    将股数向下取整到100的倍数（A股最小交易单位为1手=100股）

    Example:
        >>> truncate_to_100(1250) → 1200
        >>> truncate_to_100(99)   → 0
    """
    return (quantity // 100) * 100


# ============================================================
# 8. 列表分块工具
# ============================================================

def chunk_list(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    """
    将列表分块（用于批量请求时控制并发量）

    Example:
        >>> chunk_list([1,2,3,4,5,6,7], 3)
        [[1,2,3], [4,5,6], [7]]
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ============================================================
# 9. 计时器上下文管理器
# ============================================================

class Timer:
    """
    简易计时器，用于性能分析

    Example:
        >>> with Timer("数据加载"):
        ...     load_data()
        # 输出: 数据加载 耗时: 2.34秒
    """

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.start_time is not None
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.name} 耗时: {self.elapsed:.2f}秒")


# ============================================================
# 10. 数值工具
# ============================================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，避免除以零"""
    if b == 0 or b is None:
        return default
    return a / b


def annualize_return(daily_return: float, trading_days: int = 252) -> float:
    """
    日收益率年化
    公式：(1 + r_daily)^252 - 1
    """
    return (1 + daily_return) ** trading_days - 1


def annualize_volatility(daily_vol: float, trading_days: int = 252) -> float:
    """
    日波动率年化
    公式：σ_daily × √252
    """
    return daily_vol * (trading_days ** 0.5)


# ============================================================
# 11. 绩效指标计算
# ============================================================

def calculate_sharpe(
    returns: list[float],
    risk_free_rate: float = 0.025,
    trading_days: int = 252
) -> float:
    """
    计算年化夏普比率

    公式：
        Sharpe = (mean(超额日收益) / std(超额日收益)) × √252
        其中超额日收益 = 日收益 - 无风险日利率

    Args:
        returns: 每日收益率列表（小数形式，如 0.01 表示 1%）
        risk_free_rate: 年化无风险利率（默认 2.5%）
        trading_days: 年交易日数（默认 252）

    Returns:
        夏普比率（越高越好，>1 为良好，>2 为优秀）
    """
    if not returns or len(returns) < 2:
        return 0.0

    import numpy as np

    returns_arr = np.array(returns, dtype=np.float64)
    daily_rf = risk_free_rate / trading_days
    excess_returns = returns_arr - daily_rf

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)  # 样本标准差

    if std_excess == 0:
        return 0.0

    sharpe_daily = mean_excess / std_excess
    return float(sharpe_daily * (trading_days ** 0.5))


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """
    计算最大回撤

    公式：
        Drawdown_t = (Equity_t - max(Equity_0..t)) / max(Equity_0..t)
        MaxDrawdown = min(Drawdown_t)

    Args:
        equity_curve: 净值曲线（如 [1.0, 1.05, 0.98, 1.10]）

    Returns:
        最大回撤（负数，如 -0.15 表示 -15%）
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    import numpy as np

    equity_arr = np.array(equity_curve, dtype=np.float64)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / running_max

    return float(np.min(drawdown))


# ============================================================
# 模块自测（python helpers.py 直接运行）
# ============================================================

if __name__ == "__main__":
    init_logging()

    # 测试重试装饰器
    @retry(max_attempts=2, delays=[0.1])
    def fail_func():
        raise ValueError("模拟失败")

    try:
        fail_func()
    except ValueError:
        logger.info("✓ 重试装饰器测试通过")

    # 测试代码格式化
    assert format_code("000001") == "000001.SZ"
    assert format_code("600519") == "600519.SH"
    assert format_code("sh600000") == "600000.SH"
    assert format_code("830799") == "830799.BJ", "北交所 83 前缀应为 .BJ"
    assert format_code("920001") == "920001.BJ", "北交所 92 前缀应为 .BJ"
    assert format_code("688981") == "688981.SH"
    assert format_code("301236") == "301236.SZ"
    assert to_akshare_code("000001.SZ") == "sz000001"
    assert to_baostock_code("600519.SH") == "sh.600519"
    logger.info("✓ 股票代码格式化测试通过")

    # 测试涨跌停限制
    assert get_limit_pct("000001.SZ") == 0.10
    assert get_limit_pct("300750.SZ") == 0.20
    assert get_limit_pct("688981.SH") == 0.20
    assert get_limit_pct("600000.SH", is_st=True) == 0.05
    assert get_limit_pct("830799") == 0.30, "北交所应为 30%"
    assert get_limit_pct("920001") == 0.30, "北交所 92 前缀应为 30%"
    # 新股上市窗口：沪深主板/创业板/科创板前 5 个交易日不设限
    assert get_limit_pct("301236", trading_days_since_list=1) == float("inf"), "创业板新股首日无限制"
    assert get_limit_pct("688981", trading_days_since_list=5) == float("inf"), "科创板新股第 5 日无限制"
    assert get_limit_pct("301236", trading_days_since_list=6) == 0.20, "创业板第 6 日恢复 ±20%"
    # 新股上市窗口：北交所仅首日不设限
    assert get_limit_pct("830799", trading_days_since_list=1) == float("inf"), "北交所新股首日无限制"
    assert get_limit_pct("830799", trading_days_since_list=2) == 0.30, "北交所第 2 日恢复 ±30%"
    logger.info("✓ 涨跌停限制测试通过")

    # 测试绩效计算
    returns = [0.01, -0.005, 0.02, -0.01, 0.015]
    sharpe = calculate_sharpe(returns)
    logger.info(f"✓ 夏普比率: {sharpe:.4f}")

    equity = [1.0, 1.05, 0.98, 1.02, 0.95, 1.10]
    mdd = calculate_max_drawdown(equity)
    logger.info(f"✓ 最大回撤: {mdd:.4f}")

    logger.success("所有测试通过！")
