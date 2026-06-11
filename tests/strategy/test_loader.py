"""
外部策略加载器单元测试。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_DEMO = (
    "import pandas as pd\n"
    "from src.strategy.base import BaseStrategy, Signal\n\n"
    "class MaRsiStrategy(BaseStrategy):\n"
    "    strategy_name = 'ma_rsi'\n"
    "    default_params = {'fast': 10}\n"
    "    def generate_signals(self, data):\n"
    "        codes = list(data.keys())\n"
    "        dates = pd.to_datetime(next(iter(data.values()))['date'])\n"
    "        sig = self.empty_signals(dates, codes)\n"
    "        sig.iloc[0] = int(Signal.BUY)\n"
    "        return self.validate_signals(sig)\n"
)

_MULTI = (
    "from src.strategy.base import BaseStrategy\n"
    "class A(BaseStrategy):\n"
    "    strategy_name='a'\n"
    "    def generate_signals(self, data): return self.empty_signals([], [])\n"
    "class B(BaseStrategy):\n"
    "    strategy_name='b'\n"
    "    def generate_signals(self, data): return self.empty_signals([], [])\n"
)


def test_load_by_filename_with_class_mismatch(tmp_path):
    from src.strategy import BaseStrategy
    from src.strategy.loader import load_strategy

    Path(tmp_path, "ma_rsi.py").write_text(_DEMO, encoding="utf-8")
    cls = load_strategy("ma_rsi", external_path=str(tmp_path))
    assert issubclass(cls, BaseStrategy) and cls.__name__ == "MaRsiStrategy"


def test_create_strategy_with_params(tmp_path):
    from src.strategy.loader import create_strategy

    Path(tmp_path, "ma_rsi.py").write_text(_DEMO, encoding="utf-8")
    inst = create_strategy("ma_rsi", external_path=str(tmp_path), fast=20)
    assert inst.fast == 20
    sig = inst.generate_signals(
        {"X": pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="B"), "close": [1, 2, 3]})}
    )
    assert (sig.iloc[0] == 1).all()


def test_direct_py_path_and_class_addressing(tmp_path):
    from src.strategy.loader import load_strategy

    Path(tmp_path, "multi.py").write_text(_MULTI, encoding="utf-8")
    mp = str(Path(tmp_path, "multi.py"))
    assert load_strategy("A", external_path=mp).__name__ == "A"
    assert load_strategy("b", external_path=mp).__name__ == "B"  # by strategy_name attr


def test_missing_file_raises(tmp_path):
    from src.strategy.loader import StrategyLoadError, load_strategy

    with pytest.raises(StrategyLoadError):
        load_strategy("nope", external_path=str(tmp_path))


def test_ambiguous_raises(tmp_path):
    from src.strategy.loader import StrategyLoadError, load_strategy

    Path(tmp_path, "multi.py").write_text(_MULTI, encoding="utf-8")
    with pytest.raises(StrategyLoadError):
        load_strategy("multi", external_path=str(tmp_path))


def test_builtin_example_loadable():
    from src.strategy.loader import list_examples, load_strategy

    assert "ma_rsi" in list_examples()
    assert load_strategy("ma_rsi").__name__ == "MaRsiStrategy"


def test_empty_name_raises():
    from src.strategy.loader import StrategyLoadError, load_strategy

    with pytest.raises(StrategyLoadError):
        load_strategy("")
