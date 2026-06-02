#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 tushare stock_st 接口（每日 ST 快照）"""

from __future__ import annotations

import akshare as ak


def main() -> None:
    # 测试新浪/akshare 分红送转接口对 BJ 的支持
    codes = [
        ("600848.SH", "600012"),   # 上海
        ("000001.SZ", "000001"),   # 深圳
        ("830799.BJ", "830799"),   # 北交所
        ("835305.BJ", "835305"),   # 北交所
    ]

    for code, symbol in codes:
        print(f"=== {code} (akshare stock_history_dividend_detail) ===")
        try:
            # indicator="分红" 查分红送转明细
            df = ak.stock_history_dividend_detail(indicator="分红", stock=symbol, date="")
            print(f"  返回行数: {len(df) if df is not None else 'None'}")
            if df is not None and not df.empty:
                print(f"  列名: {list(df.columns)}")
                print(df.head(3).to_string(index=False))
            else:
                print("  返回空")
        except Exception as exc:
            print(f"  异常: {exc}")
        print()


if __name__ == "__main__":
    main()
