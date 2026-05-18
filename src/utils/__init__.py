"""
公共工具子包

对外导出：
    init_logging              初始化 loguru 日志系统
    retry                     指数退避重试装饰器
    parse_date                统一日期解析
    format_code               股票代码标准化
    to_akshare_code           转 akshare 格式
    to_baostock_code          转 baostock 格式
    get_limit_pct             涨跌停限制百分比
    truncate_to_100           股数取整到 100
    chunk_list                列表分块
    Timer                     计时器上下文管理器
    safe_divide               安全除法
    annualize_return          收益率年化
    annualize_volatility      波动率年化
    calculate_sharpe          年化夏普比率
    calculate_max_drawdown    最大回撤
    ensure_dir                确保目录存在
    load_config               加载并合并多源配置
    apply_overrides           对已加载配置施加覆盖
    DotDict                   支持点访问的字典
    ConfigError               配置异常
"""

from .config_loader import ConfigError, DotDict, apply_overrides, load_config
from .helpers import (
    Timer,
    annualize_return,
    annualize_volatility,
    calculate_max_drawdown,
    calculate_sharpe,
    chunk_list,
    ensure_dir,
    format_code,
    get_limit_pct,
    init_logging,
    parse_date,
    retry,
    safe_divide,
    to_akshare_code,
    to_baostock_code,
    truncate_to_100,
)

__all__ = [
    "ConfigError",
    "DotDict",
    "Timer",
    "annualize_return",
    "annualize_volatility",
    "apply_overrides",
    "calculate_max_drawdown",
    "calculate_sharpe",
    "chunk_list",
    "ensure_dir",
    "format_code",
    "get_limit_pct",
    "init_logging",
    "load_config",
    "parse_date",
    "retry",
    "safe_divide",
    "to_akshare_code",
    "to_baostock_code",
    "truncate_to_100",
]
