"""
配置加载器
==========
统一加载与合并多源配置，对外暴露单一入口 `load_config()`。

加载优先级（高 > 低，后者覆盖前者）：
    1. CLI 显式覆盖（apply_overrides() 调用方传入）
    2. 环境变量（QUANT_<SECTION>__<KEY>=value）
    3. 私有覆盖文件 config.private.yaml（已加入 .gitignore）
    4. 公开模板 config.yaml
    5. 代码内置默认值（DEFAULT_CONFIG）

设计要点：
- 深度合并：嵌套字典逐层合并，而非顶层覆盖
- 类型保持：YAML 中数值/布尔保持原类型；环境变量自动推断类型
- 双访问语法：cfg["risk"]["commission_rate"] 与 cfg.risk.commission_rate 均可
- .env 自动加载：项目根目录存在 .env 时通过 python-dotenv 注入到 os.environ
- 不依赖业务模块（仅依赖 pyyaml、python-dotenv、loguru），避免循环引用

使用示例：
    >>> from a_share_quant_pro.utils.config_loader import load_config
    >>> cfg = load_config()
    >>> cfg.risk.commission_rate
    0.00025
    >>> cfg["risk"]["commission_rate"]
    0.00025

    # CLI 覆盖：
    >>> cfg = load_config(overrides={"backtest.initial_capital": 500_000})
    >>> cfg.backtest.initial_capital
    500000
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


# ============================================================
# 常量
# ============================================================

#: 环境变量前缀（QUANT_DATA__STORE_PATH=... → cfg.data.store_path）
ENV_PREFIX = "QUANT_"

#: 嵌套键分隔符（用于环境变量展开和 CLI 覆盖路径）
ENV_NESTED_SEP = "__"

#: 点路径分隔符（用于 CLI 覆盖：apply_overrides({"risk.commission_rate": 0.0001})）
DOT_SEP = "."

#: 内置默认配置（最低优先级，保证 config.yaml 缺字段时仍可运行）
#: 与 config.yaml 公开模板字段一一对应，注释见 config.yaml
DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "store_path": "data_store",
        "sources": ["pytdx", "akshare", "baostock", "tushare"],
        "factor_source": "sina",
        "factor_selfheal_window": 30,
        "suspend_sources": ["eastmoney", "tushare"],
        "suspend_lookback_days": 30,
        "suspend_enabled": True,
        "tdx_path": "",
        "request_timeout": 15,
        "stock_timeout": 60,
        "retry_times": 3,
        "retry_delays": [1, 2, 4, 8],
        "calendar_refresh_days": 7,
        "max_workers": 4,
    },
    "universe": {
        "min_list_days": 60,
        "exclude_st": True,
        "exclude_new_ipo": True,
    },
    "strategy": {
        "external_path": "",
        "strategy_name": "ma_rsi",
    },
    "execution": {
        "mode": "next_open",
        "volume_pct_limit": 0.10,
        "min_order_amount": 1000,
        "t1_cash_freeze": False,
        "slippage": {
            "type": "percent",
            "fixed_amount": 0.01,
            "percent_rate": 0.001,
            "tick_size": 0.01,
            "min_ticks": 1,
        },
    },
    "risk": {
        "commission_rate": 0.00025,
        "stamp_duty": 0.0005,
        "min_commission": 5.0,
        "max_single_position": 0.20,
        "max_industry_position": 0.30,
        "daily_stop_loss": 0.05,
        "total_drawdown_stop": 0.15,
    },
    "backtest": {
        "initial_capital": 1_000_000,
        "benchmark": "000300.SH",
        "risk_free_rate": 0.025,
    },
    "logging": {
        "log_dir": "logs",
        "app_name": "quant_pro",
        "level": "INFO",
        "rotation": "00:00",
        "retention": "30 days",
    },
}


# ============================================================
# 异常
# ============================================================


class ConfigError(Exception):
    """配置加载或校验异常"""


# ============================================================
# 点访问字典（支持 cfg.risk.commission_rate 与 cfg["risk"]["commission_rate"]）
# ============================================================


class DotDict(dict):
    """
    支持点号访问的字典，递归把嵌套 dict 也包成 DotDict。

    特性：
    - 既可 dict 风格 `d["a"]` 也可属性风格 `d.a`
    - 写入时自动包装嵌套 dict
    - 拷贝/序列化时仍是普通 dict（通过 to_dict()）
    """

    def __init__(self, mapping: Mapping[str, Any] | None = None):
        super().__init__()
        if mapping:
            for key, value in mapping.items():
                self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    # ---- 属性访问 ----
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"配置中不存在键: {key}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(f"配置中不存在键: {key}") from exc

    # ---- 类型转换 ----
    def to_dict(self) -> dict[str, Any]:
        """递归还原为普通 dict（用于序列化、日志打印）"""
        result: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, DotDict) else v for v in value
                ]
            else:
                result[key] = value
        return result


# ============================================================
# 工具函数
# ============================================================


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    深度合并两个字典：override 覆盖 base，嵌套 dict 逐层合并。

    规则：
    - 同键且都是 dict → 递归合并
    - 同键但类型不同（或非 dict） → override 直接替换
    - list 整体替换（不做元素级合并，避免歧义）

    返回新字典，不修改输入。
    """
    result = deepcopy(base)
    for key, override_value in override.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, Mapping):
            result[key] = _deep_merge(base_value, override_value)
        else:
            result[key] = deepcopy(override_value)
    return result


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """
    安全加载 YAML 文件为 dict。空文件返回 {}，文件不存在抛 FileNotFoundError。
    """
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件根节点必须是 mapping，但 {path} 是 {type(data).__name__}")
    return data


def _coerce_env_value(raw: str) -> Any:
    """
    将环境变量字符串转换为合理的 Python 类型。

    转换规则（按优先级）：
    1. 'true'/'false'/'yes'/'no'/'on'/'off' → bool
    2. 'null'/'none'/'~'/'' → None
    3. 纯整数 → int
    4. 浮点数 → float
    5. JSON-like 列表/对象（以 [ 或 { 开头）→ yaml.safe_load 解析
    6. 其他 → str（保留原样）
    """
    s = raw.strip()
    lowered = s.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~", ""):
        return None
    # 整数（保留前导零的纯数字串为字符串：股票代码/版本号/邮编等标识符，如 "000001"，
    # 否则 int("000001")=1 会丢前导零，A 股代码场景直接出错）
    digits = s.lstrip("+-")
    if digits.isdigit():
        if len(digits) > 1 and digits[0] == "0":
            return s
        return int(s)
    # 浮点
    try:
        if any(ch in s for ch in (".", "e", "E")):
            return float(s)
    except ValueError:
        pass
    # 列表/对象（YAML 兼容 JSON）
    if s.startswith(("[", "{")):
        try:
            return yaml.safe_load(s)
        except yaml.YAMLError:
            return s
    return s


def _set_nested(target: dict[str, Any], path_parts: Iterable[str], value: Any) -> None:
    """
    按路径在嵌套字典中设置值，中间缺失节点自动用 dict 补齐。

    Example:
        _set_nested({}, ["risk", "commission_rate"], 0.0001)
        → {"risk": {"commission_rate": 0.0001}}
    """
    parts = list(path_parts)
    if not parts:
        raise ValueError("路径不能为空")
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _collect_env_overrides(prefix: str = ENV_PREFIX) -> dict[str, Any]:
    """
    扫描 os.environ，收集所有 QUANT_ 前缀变量并按 __ 拆分为嵌套覆盖。

    Example:
        QUANT_RISK__COMMISSION_RATE=0.0001
        QUANT_DATA__SOURCES=["akshare"]
        →
        {"risk": {"commission_rate": 0.0001}, "data": {"sources": ["akshare"]}}
    """
    overrides: dict[str, Any] = {}
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(prefix):
            continue
        stripped = raw_key[len(prefix):]
        if not stripped:
            continue
        parts = [p.lower() for p in stripped.split(ENV_NESTED_SEP) if p]
        if not parts:
            continue
        _set_nested(overrides, parts, _coerce_env_value(raw_value))
        logger.debug(f"环境变量覆盖配置: {'.'.join(parts)} = {overrides}")
    return overrides


def _expand_dot_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """
    把 {"risk.commission_rate": 0.0001} 展开为 {"risk": {"commission_rate": 0.0001}}。
    若键不含点号，则视为顶层键。
    """
    nested: dict[str, Any] = {}
    for dotted_key, value in overrides.items():
        parts = [p for p in dotted_key.split(DOT_SEP) if p]
        if not parts:
            raise ConfigError(f"覆盖键不能为空: {dotted_key!r}")
        _set_nested(nested, parts, value)
    return nested


# ============================================================
# 主入口
# ============================================================


def load_config(
    config_path: str | Path | None = None,
    private_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    load_env: bool = True,
    env_file: str | Path | None = None,
) -> DotDict:
    """
    加载并合并多源配置，返回 DotDict。

    Args:
        config_path: 公开模板路径，默认查找项目根 config.yaml
        private_path: 私有覆盖文件路径，默认查找项目根 config.private.yaml；
                      不存在时静默跳过
        overrides: CLI 显式覆盖，键为点号路径（如 "risk.commission_rate"），
                   优先级最高
        load_env: 是否扫描环境变量做覆盖（QUANT_ 前缀，__ 分隔嵌套）
        env_file: .env 文件路径，默认项目根 .env；用于把 .env 注入 os.environ
                  之后再扫描 QUANT_ 变量

    Returns:
        DotDict：支持 cfg.risk.commission_rate 与 cfg["risk"]["commission_rate"]

    Raises:
        FileNotFoundError: config_path 显式传入但文件不存在
        ConfigError: 配置文件结构非法

    合并优先级（后者覆盖前者）：
        DEFAULT_CONFIG  <  config.yaml  <  config.private.yaml
                        <  环境变量      <  CLI overrides
    """
    project_root = _detect_project_root()

    # 1. 加载 .env（注入到 os.environ，后续 _collect_env_overrides 会读到）
    if load_env and _DOTENV_AVAILABLE:
        target_env = Path(env_file) if env_file else project_root / ".env"
        if target_env.exists():
            load_dotenv(target_env, override=False)
            logger.debug(f".env 已加载: {target_env}")
            # 凭证类变量（非 QUANT_ 前缀，如 TUSHARE_TOKEN*）：让 .env 取得权威，
            # 避免被系统中陈旧的同名环境变量遮蔽（见 token-shadow 历史问题）。
            # QUANT_ 前缀是配置覆盖通道，保留"系统环境 > .env"语义，不强制覆盖。
            from dotenv import dotenv_values
            for _k, _v in dotenv_values(target_env).items():
                if _v is not None and not _k.startswith(ENV_PREFIX):
                    os.environ[_k] = _v

    # 2. 起始配置 = 内置默认
    merged: dict[str, Any] = deepcopy(DEFAULT_CONFIG)

    # 3. 合并公开 config.yaml（默认查找项目根；不存在不报错，仅警告）
    yaml_path = Path(config_path) if config_path else project_root / "config.yaml"
    if yaml_path.exists():
        try:
            merged = _deep_merge(merged, _load_yaml_file(yaml_path))
            logger.debug(f"已加载公开配置: {yaml_path}")
        except (yaml.YAMLError, ConfigError) as exc:
            raise ConfigError(f"解析 {yaml_path} 失败: {exc}") from exc
    elif config_path is not None:
        # 显式传入路径但不存在 → 抛错
        raise FileNotFoundError(f"指定的配置文件不存在: {yaml_path}")
    else:
        logger.warning(f"未找到 {yaml_path}，使用内置默认配置")

    # 4. 合并私有 config.private.yaml（不存在静默跳过）
    private_yaml = (
        Path(private_path)
        if private_path
        else project_root / "config.private.yaml"
    )
    if private_yaml.exists():
        try:
            merged = _deep_merge(merged, _load_yaml_file(private_yaml))
            logger.debug(f"已加载私有配置: {private_yaml}")
        except (yaml.YAMLError, ConfigError) as exc:
            raise ConfigError(f"解析 {private_yaml} 失败: {exc}") from exc
    elif private_path is not None:
        raise FileNotFoundError(f"指定的私有配置文件不存在: {private_yaml}")

    # 5. 合并环境变量
    if load_env:
        env_overrides = _collect_env_overrides()
        if env_overrides:
            merged = _deep_merge(merged, env_overrides)
            logger.info(f"应用 {sum(1 for _ in _flatten(env_overrides))} 项环境变量覆盖")

    # 6. 合并 CLI overrides（最高优先级）
    if overrides:
        merged = _deep_merge(merged, _expand_dot_overrides(overrides))
        logger.info(f"应用 {len(overrides)} 项 CLI 覆盖")

    return DotDict(merged)


def apply_overrides(cfg: DotDict, overrides: Mapping[str, Any]) -> DotDict:
    """
    在已加载的 cfg 上再次施加覆盖，返回新的 DotDict（不修改原对象）。

    用于场景：CLI 解析完成后追加覆盖、回测前临时调参等。

    Example:
        >>> cfg = load_config()
        >>> cfg2 = apply_overrides(cfg, {"backtest.initial_capital": 500_000})
    """
    merged = _deep_merge(cfg.to_dict(), _expand_dot_overrides(overrides))
    return DotDict(merged)


# ============================================================
# 辅助：定位项目根
# ============================================================


def _detect_project_root() -> Path:
    """
    定位项目根目录。

    策略：从本文件向上查找含 pyproject.toml 的目录；
    若全部找不到（罕见，例如在 site-packages 中），退化为 CWD。
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _flatten(d: Mapping[str, Any], prefix: str = "") -> Iterable[str]:
    """递归展平嵌套 dict 的键路径，仅用于日志/调试统计"""
    for key, value in d.items():
        path = f"{prefix}{DOT_SEP}{key}" if prefix else key
        if isinstance(value, Mapping):
            yield from _flatten(value, path)
        else:
            yield path


# ============================================================
# 模块自测（python -m a_share_quant_pro.utils.config_loader）
# ============================================================


if __name__ == "__main__":
    from .helpers import init_logging

    init_logging(level="DEBUG")

    cfg = load_config()
    logger.info("=== 配置加载成功 ===")
    logger.info(f"data.store_path           = {cfg.data.store_path}")
    logger.info(f"universe.min_list_days    = {cfg.universe.min_list_days}")
    logger.info(f"strategy.strategy_name    = {cfg.strategy.strategy_name}")
    logger.info(f"execution.mode            = {cfg.execution.mode}")
    logger.info(f"execution.slippage.type   = {cfg.execution.slippage.type}")
    logger.info(f"risk.commission_rate      = {cfg.risk.commission_rate}")
    logger.info(f"backtest.initial_capital  = {cfg.backtest.initial_capital}")
    logger.info(f"logging.level             = {cfg.logging.level}")

    # 测试 dict 风格访问
    assert cfg["risk"]["commission_rate"] == cfg.risk.commission_rate

    # 测试 CLI 覆盖
    cfg2 = apply_overrides(cfg, {"backtest.initial_capital": 500_000})
    assert cfg2.backtest.initial_capital == 500_000
    assert cfg.backtest.initial_capital == 1_000_000  # 原对象未被修改
    logger.success("✓ 双访问语法 + CLI 覆盖 测试通过")

    # 测试环境变量解析（手动注入）
    os.environ["QUANT_RISK__COMMISSION_RATE"] = "0.0001"
    os.environ["QUANT_DATA__SOURCES"] = '["akshare"]'
    cfg3 = load_config()
    assert cfg3.risk.commission_rate == 0.0001
    assert cfg3.data.sources == ["akshare"]
    del os.environ["QUANT_RISK__COMMISSION_RATE"]
    del os.environ["QUANT_DATA__SOURCES"]
    logger.success("✓ 环境变量覆盖 测试通过")

    logger.success("所有测试通过！")
