"""
外部私有策略加载器（loader）
=============================

把"公开框架"与"用户私有策略"解耦：私有策略物理上**不进入本仓库**，运行时按需从
外部目录（或内置示例包）动态加载为 :class:`~src.strategy.base.BaseStrategy` 子类。

两种接入方式（见 config.yaml / CLI）
------------------------------------
1. **外部路径**（本地开发推荐）::

       strategy:
         external_path: "D:/A_Share_Quant_Private/strategies"
         strategy_name: "my_alpha_001"

2. **CLI 直传**（服务器推荐）::

       python main.py backtest --strategy-path /path/to/strategies --strategy my_alpha_001

``external_path`` 为空 → 回退到内置示例包 ``src.strategy.examples``。

健壮的策略类发现
----------------
A 股私有策略常见"文件名 snake_case、类名 PascalCase"不一致（如 ``ma_rsi.py`` →
``MaRsiStrategy``）。本加载器不只做 ``getattr(module, name)``，而是按以下优先级在模块内
定位**唯一**的 BaseStrategy 子类：

1. 模块内名为 ``strategy_name`` 的类；
2. 类属性 ``strategy_name == 请求名`` 的类；
3. 类名规整后（去下划线、小写）与请求名一致的类；
4. 模块内**恰好只有一个** BaseStrategy 子类时直接采用；
5. 否则抛出带候选清单的明确异常。

只考虑"在该模块内定义"的类（``cls.__module__ == module``），避免误选 import 进来的基类。

安全提示
--------
加载外部 ``.py`` 会**执行**该文件代码，请仅加载可信来源的策略文件。
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from loguru import logger

from .base import BaseStrategy

__all__ = ["StrategyLoadError", "create_strategy", "list_examples", "load_strategy"]


class StrategyLoadError(Exception):
    """策略加载失败（文件缺失 / 无合法策略类 / 多义）。"""


# ============================================================
# 公开接口
# ============================================================


def load_strategy(strategy_name: str, external_path: str | None = None) -> type[BaseStrategy]:
    """加载策略类。

    Args:
        strategy_name: 策略标识（类名 / 文件名 / ``BaseStrategy.strategy_name`` 之一）。
        external_path: 外部策略目录或 ``.py`` 文件路径；为空则从内置示例包加载。

    Returns:
        :class:`BaseStrategy` 的子类（**类对象，非实例**）。

    Raises:
        StrategyLoadError: 文件不存在、模块内无合法策略类或存在多义无法确定。
    """
    name = str(strategy_name).strip()
    if not name:
        raise StrategyLoadError("strategy_name 不能为空")

    if external_path and str(external_path).strip():
        module = _load_module_from_path(name, str(external_path).strip())
    else:
        module = _load_example_module(name)

    cls = _find_strategy_class(module, name)
    logger.info(f"已加载策略 {cls.__name__}（strategy_name={getattr(cls, 'strategy_name', '?')!r}）来自 {module.__name__}")
    return cls


def create_strategy(
    strategy_name: str, external_path: str | None = None, **params
) -> BaseStrategy:
    """加载策略类并用 ``params`` 实例化（便捷封装）。"""
    cls = load_strategy(strategy_name, external_path)
    return cls(**params)


def list_examples() -> list[str]:
    """列出内置示例包 ``examples`` 中可加载的策略模块名（文件名，不含 .py）。"""
    try:
        examples = importlib.import_module(f"{__package__}.examples")
    except ImportError:
        return []
    pkg_dir = Path(examples.__file__).parent  # type: ignore[arg-type]
    return sorted(
        p.stem for p in pkg_dir.glob("*.py") if p.stem != "__init__"
    )


# ============================================================
# 内部：模块加载
# ============================================================


def _load_module_from_path(name: str, external_path: str):
    """从外部目录 / 文件加载 Python 模块。"""
    p = Path(external_path).expanduser()
    if p.suffix == ".py":
        file_path = p
        search_dir = p.parent
    else:
        file_path = p / f"{name}.py"
        search_dir = p

    if not file_path.exists():
        raise StrategyLoadError(
            f"策略文件不存在: {file_path}\n"
            f"请检查 external_path={external_path!r} 与 strategy_name={name!r} 是否匹配"
        )

    # 唯一模块名，避免与已加载模块冲突
    mod_name = f"_ext_strategy_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"无法为 {file_path} 创建模块 spec")
    module = importlib.util.module_from_spec(spec)

    # 临时把策略目录加入 sys.path，支持私有策略 import 同目录的兄弟模块
    added = False
    search_str = str(search_dir)
    if search_str not in sys.path:
        sys.path.insert(0, search_str)
        added = True
    sys.modules[mod_name] = module  # 注册，便于 dataclass / pickling
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        raise StrategyLoadError(f"执行策略文件 {file_path} 失败: {type(exc).__name__}: {exc}") from exc
    finally:
        if added:
            with contextlib.suppress(ValueError):
                sys.path.remove(search_str)
    return module


def _load_example_module(name: str):
    """从内置示例包 ``src.strategy.examples`` 加载子模块。"""
    target = f"{__package__}.examples.{name}"
    try:
        return importlib.import_module(target)
    except ImportError as exc:
        available = list_examples()
        raise StrategyLoadError(
            f"内置示例中未找到策略模块 {name!r}（导入 {target} 失败: {exc}）。"
            f"可用示例: {available or '（examples 包为空或未生成）'}；"
            f"如需加载私有策略，请提供 external_path。"
        ) from exc


# ============================================================
# 内部：策略类发现
# ============================================================


def _strategy_candidates(module) -> list[type[BaseStrategy]]:
    """模块内"自身定义"的 BaseStrategy 子类（排除 import 进来的基类）。"""
    out: list[type[BaseStrategy]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, BaseStrategy)
            and obj is not BaseStrategy
            and obj.__module__ == module.__name__
        ):
            out.append(obj)
    return out


def _norm(s: str) -> str:
    return str(s).replace("_", "").replace("-", "").lower()


def _find_strategy_class(module, name: str) -> type[BaseStrategy]:
    """在模块内按优先级定位唯一的 BaseStrategy 子类。"""
    # 1. 精确同名类
    direct = getattr(module, name, None)
    if inspect.isclass(direct) and issubclass(direct, BaseStrategy) and direct is not BaseStrategy:
        return direct

    candidates = _strategy_candidates(module)
    if not candidates:
        raise StrategyLoadError(
            f"模块 {module.__name__} 中未找到任何 BaseStrategy 子类（请确认策略已继承 BaseStrategy）"
        )

    # 2. strategy_name 类属性匹配
    by_attr = [c for c in candidates if getattr(c, "strategy_name", None) == name]
    if len(by_attr) == 1:
        return by_attr[0]

    # 3. 类名规整匹配（去下划线/小写）
    by_norm = [c for c in candidates if _norm(c.__name__) == _norm(name)]
    if len(by_norm) == 1:
        return by_norm[0]

    # 4. 模块内唯一策略类
    if len(candidates) == 1:
        return candidates[0]

    # 5. 多义 / 无匹配
    names = [c.__name__ for c in candidates]
    raise StrategyLoadError(
        f"在模块 {module.__name__} 中无法唯一确定策略 {name!r}；"
        f"候选: {names}。请使 strategy_name 与类名或类属性 strategy_name 一致。"
    )


# ============================================================
# 模块自测  python -m src.strategy.loader
# ============================================================

if __name__ == "__main__":
    import tempfile

    from ..utils.helpers import init_logging

    init_logging(level="INFO")

    # 造一个临时私有策略文件，验证"文件名 snake、类名 Pascal"的健壮加载
    demo = (
        "import pandas as pd\n"
        "from src.strategy.base import BaseStrategy, Signal\n\n"
        "class MyAlpha001(BaseStrategy):\n"
        "    strategy_name = 'my_alpha_001'\n"
        "    def generate_signals(self, data):\n"
        "        codes = list(data.keys())\n"
        "        dates = pd.to_datetime(next(iter(data.values()))['date'])\n"
        "        sig = self.empty_signals(dates, codes)\n"
        "        sig.iloc[0] = int(Signal.BUY)\n"
        "        return self.validate_signals(sig)\n"
    )
    with tempfile.TemporaryDirectory() as d:
        Path(d, "my_alpha_001.py").write_text(demo, encoding="utf-8")
        cls = load_strategy("my_alpha_001", external_path=d)
        logger.info(f"加载成功: {cls.__name__} / strategy_name={cls.strategy_name}")
        inst = create_strategy("my_alpha_001", external_path=d)
        logger.info(f"实例化: {inst!r}")
    logger.info(f"内置示例: {list_examples()}")
