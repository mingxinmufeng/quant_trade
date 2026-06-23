"""技术因子单元测试（零网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_sma_handles_unsorted_input():
    """P2-11：输入行序被打乱时，compute 入口排序后结果与有序输入一致（不按错序贴值）。"""
    from src.factor.technical import SMAFactor

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({"date": dates, "close": np.arange(10, 20, dtype="float64")})
    f = SMAFactor(3)

    ordered = f.compute(df)
    shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
    out = f.compute(shuffled)

    # 两者都以交易日为索引；排序后应逐日相等（乱序未导致指标算错）
    pd.testing.assert_series_equal(out.sort_index(), ordered.sort_index())
    # 尾值 = 最后 3 日 close 均值 = mean(17,18,19) = 18
    assert np.isclose(out.sort_index().iloc[-1], 18.0)
