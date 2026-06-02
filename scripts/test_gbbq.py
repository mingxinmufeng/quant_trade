#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 pytdx GbbqReader 读取通达信本地除权数据（含 BJ 股票）"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fetcher import _auto_discover_tdx_path


def load_bj_code_map(tdx_path: str) -> dict[str, str]:
    """读取 addedcode_bj.cfg，返回 {老代码: 新代码} 映射。"""
    cfg = Path(tdx_path) / "T0002" / "hq_cache" / "addedcode_bj.cfg"
    mapping: dict[str, str] = {}
    if not cfg.exists():
        return mapping
    with open(cfg, "r", encoding="gbk", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("000000"):
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                # 44|老代码|新代码|名称|日期
                mapping[parts[1]] = parts[2]
    return mapping


def main() -> None:
    tdx_path = _auto_discover_tdx_path()
    if not tdx_path:
        print("未找到通达信安装路径（未找到 vipdoc 目录）")
        return

    gbbq_file = Path(tdx_path) / "T0002" / "hq_cache" / "gbbq"
    if not gbbq_file.exists():
        print(f"gbbq 文件不存在: {gbbq_file}")
        return

    print(f"通达信路径: {tdx_path}")
    print(f"gbbq 文件: {gbbq_file}")
    print()

    # 读取代码映射表
    code_map = load_bj_code_map(tdx_path)
    print(f"BJ 代码映射表加载: {len(code_map)} 条")
    print()

    try:
        from pytdx.reader import GbbqReader
    except ImportError:
        print("未安装 pytdx，请执行: venv\\Scripts\\pip install pytdx")
        return

    df = GbbqReader().get_df(str(gbbq_file))
    print(f"gbbq 总行数: {len(df)}")
    print()

    # 直接搜索：当前代码 920799
    new_code = "920799"
    matched_new = df[(df["market"] == 2) & (df["code"].astype(str) == new_code)]
    print(f"=== gbbq 中直接搜 {new_code} ===")
    print(f"记录数: {len(matched_new)}")
    if not matched_new.empty:
        print(matched_new.to_string(index=False))
    else:
        print("未找到")
    print()

    # 直接搜索：老代码 830799
    old_code = "920799"
    matched_old = df[(df["market"] == 2) & (df["code"].astype(str) == old_code)]
    print(f"=== gbbq 中直接搜 {old_code} ===")
    print(f"记录数: {len(matched_old)}")
    if not matched_old.empty:
        print(matched_old.head(10).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
