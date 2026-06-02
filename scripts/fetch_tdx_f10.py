#!/usr/bin/env python3
"""通过 pytdx 直连通达信主站抓取 F10 资料并保存为文本。"""

import json
import re
from pathlib import Path
from collections import OrderedDict

from pytdx.hq import TdxHq_API

# 通达信主站配置（从 D:\new_tdx\connect.cfg 提取）
HOSTS = [
    ("110.41.147.114", 7709),
    ("124.223.163.242", 7709),
    ("101.33.225.16", 7709),
]

# 市场代码: 0=深圳, 1=上海, 2=北京
MARKET_MAP = {"sz": 0, "sh": 1, "bj": 2}


def _pick_host():
    """简单轮询可用主站。"""
    api = TdxHq_API()
    for ip, port in HOSTS:
        try:
            if api.connect(ip, port):
                api.disconnect()
                return ip, port
        except Exception:
            continue
    raise ConnectionError("无可用通达信主站")


def fetch_f10_categories(market: int, code: str) -> list:
    """获取某只股票的 F10 分类列表。"""
    api = TdxHq_API()
    ip, port = _pick_host()
    with api.connect(ip, port):
        cats = api.get_company_info_category(market, code)
        # 转为普通 dict 方便序列化
        return [dict(c) for c in cats]


def fetch_f10_content(market: int, code: str, category: dict) -> str:
    """根据分类信息抓取具体 F10 文本内容。"""
    api = TdxHq_API()
    ip, port = _pick_host()
    with api.connect(ip, port):
        return api.get_company_info_content(
            market,
            code,
            category["filename"],
            category["start"],
            category["length"],
        )


def save_f10(market: int, code: str, out_dir: Path) -> Path:
    """抓取并保存某股票的完整 F10，返回输出目录。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = fetch_f10_categories(market, code)
    if not cats:
        print(f"[{code}] 无 F10 分类数据")
        return out_dir

    # 保存原始分类元信息
    meta_path = out_dir / f"{code}_f10_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(cats, f, ensure_ascii=False, indent=2)

    # 逐个抓取并保存为 txt
    for cat in cats:
        name = cat["name"]
        # 文件名合法化
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", name)
        txt_path = out_dir / f"{code}_{safe_name}.txt"
        try:
            content = fetch_f10_content(market, code, cat)
            if content is None:
                print(f"  [{code}] {name}: 服务器返回空，已跳过")
                continue
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  [{code}] {name}: 已保存 ({len(content)} 字符)")
        except Exception as e:
            print(f"  [{code}] {name}: 抓取失败 - {e}")

    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="抓取通达信 F10 资料")
    parser.add_argument("code", help="股票代码，如 000001")
    parser.add_argument("--market", choices=["sz", "sh", "bj"], default="sz",
                        help="市场，默认 sz")
    parser.add_argument("--out", default=r"D:\new_tdx\f10_export",
                        help="输出目录")
    args = parser.parse_args()

    market = MARKET_MAP[args.market]
    out = save_f10(market, args.code, Path(args.out))
    print(f"\n完成，输出目录: {out}")
