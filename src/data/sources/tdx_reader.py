"""
通达信本地行情文件读取器（自洽实现，零 pytdx 依赖）。

读取通达信本地二进制文件并返回规范化数据。本模块收拢**全部**通达信本地文件解析：

- 行情（不复权 OHLCV）：日线 ``lday/*.day``、5 分钟 ``fzline/*.lc5``、1 分钟
  ``minline/*.lc1``（均 32 字节/条）；
- 权息 ``T0002/hq_cache/gbbq``（加密；解密算法 + 密钥表自带，见 ``read_gbbq``）；
- 更名史 ``T0002/hq_cache/profile.dat``（64 字节/条定长，见 ``read_profile``）。

设计要点
--------
1. **自包含**：只依赖 ``numpy`` / ``pandas`` / 标准库，不 import pytdx，也不 import 本仓库
   其它模块，便于将来原样抽成独立 PyPI 包复用。
2. **向量化解析**：``numpy.frombuffer`` 一次性解包定长记录，比逐行 Python 循环快很多。
3. **尾部增量**：文件按时间升序定长存储，可二分定位首条 ``date >= start`` 的记录偏移，
   只解析尾部（首拉时 ``start`` 极早 → 定位到偏移 0，等价全量）。
4. **板块分类自管**：``classify_security`` 按文件名前缀判板块，**修正 pytdx 两处缺陷**——
   科创板 ``sh68xxxx`` 与北交所 ``bj`` 均能正确识别（pytdx 原版对二者分别返回 ``None`` /
   抛 ``NotImplementedError``）。
5. **成交量统一为「股」**（关键）：通达信 ``.day`` 个股 volume 原始值即「股」，pytdx 原版
   却对其乘 0.01（得到「手」），与本系统及 akshare/tushare/baostock（均为「股」）、以及
   同源分钟线（股）矛盾。本模块对个股 volume 系数取 **1.0**（即「股」），指数/基金/债券
   沿用 pytdx 原系数。价格系数（分→元）与 pytdx 一致。详见 ``SECURITY_COEFFICIENT``。

公开接口::

    classify_security(filename) -> str | None      # 板块类型（未知返回 None）
    read_day(path, start=None) -> pd.DataFrame      # 日线，index=date
    read_lc(path, start=None) -> pd.DataFrame       # 分钟，index=datetime
    UnknownSecurityType                              # read_* 遇未知板块时抛出

返回 DataFrame 列为 ``open/high/low/close/amount/volume``（价=元、量=股、额=元），
``read_day`` 以 ``date`` 命名的 ``DatetimeIndex`` 为索引，``read_lc`` 以 ``datetime`` 命名。
"""

from __future__ import annotations

import os
import struct
from datetime import date

import numpy as np
import pandas as pd

__all__ = [
    "GBBQ_COLUMNS",
    "PROFILE_COLUMNS",
    "SECURITY_COEFFICIENT",
    "UnknownSecurityType",
    "classify_security",
    "read_day",
    "read_gbbq",
    "read_lc",
    "read_profile",
]

#: 通达信定长记录字节数（日线 .day / 分钟 .lc1/.lc5 均为 32 字节/条）
RECORD_SIZE = 32

#: 日线 .day 记录布局（与 pytdx ``'<IIIIIfII'`` 等价）：
#: date(YYYYMMDD,u4), open/high/low/close(u4), amount(f4), volume(u4), reserved(u4)。
#: OHLC 需乘价格系数 coef[0]、volume 乘量系数 coef[1]（见 SECURITY_COEFFICIENT）。
_DAILY_DT = np.dtype([
    ("date", "<u4"), ("open", "<u4"), ("high", "<u4"), ("low", "<u4"),
    ("close", "<u4"), ("amount", "<f4"), ("volume", "<u4"), ("_rsv", "<u4"),
])

#: 分钟 .lc1/.lc5 记录布局（与 pytdx ``'<HHfffffII'`` 等价）：
#: date(packed,u2), time(packed,u2), open/high/low/close/amount(f4), volume(u4), reserved(u4)。
#: 价格为真实「元」、volume 为真实「股」，均无系数。
_MIN_DT = np.dtype([
    ("date", "<u2"), ("time", "<u2"), ("open", "<f4"), ("high", "<f4"),
    ("low", "<f4"), ("close", "<f4"), ("amount", "<f4"), ("volume", "<u4"),
    ("_rsv", "<u4"),
])

_OUT_COLS = ["open", "high", "low", "close", "amount", "volume"]

#: 个股板块（A/B 股 + 北交所）：volume 原始即「股」，量系数取 1.0（不像 pytdx 乘 0.01 成「手」）
_STOCK_TYPES = frozenset(
    {"SH_A_STOCK", "SZ_A_STOCK", "SH_B_STOCK", "SZ_B_STOCK", "BJ_A_STOCK"}
)

#: pytdx 原版各板块 (价格系数, 量系数)，作为非个股板块的权威来源
_PYTDX_COEF: dict[str, tuple[float, float]] = {
    "SH_A_STOCK": (0.01, 0.01), "SH_B_STOCK": (0.001, 0.01),
    "SH_INDEX": (0.01, 1.0), "SH_FUND": (0.001, 1.0), "SH_BOND": (0.001, 1.0),
    "SZ_A_STOCK": (0.01, 0.01), "SZ_B_STOCK": (0.01, 0.01),
    "SZ_INDEX": (0.01, 1.0), "SZ_FUND": (0.001, 0.01), "SZ_BOND": (0.001, 0.01),
    "BJ_A_STOCK": (0.01, 0.01),  # pytdx 不支持北交所；价格系数与 A 股一致
}

#: 生效系数表：个股 volume 系数统一改为 1.0（→「股」），其余沿用 pytdx。价格系数始终不变。
SECURITY_COEFFICIENT: dict[str, tuple[float, float]] = {
    t: (price, 1.0 if t in _STOCK_TYPES else vol)
    for t, (price, vol) in _PYTDX_COEF.items()
}


class UnknownSecurityType(Exception):
    """文件名无法归入已知板块（``classify_security`` 返回 ``None``）。"""


def classify_security(filename: str) -> str | None:
    """按通达信文件名前缀判定板块类型；未知返回 ``None``。

    文件名形如 ``sh600519.day`` / ``sz000001.lc5`` / ``bj430047.day``：前 2 位为市场
    （sh/sz/bj），随后 2 位为代码段首。规则在 pytdx 原版基础上补两处：
    - ``sh68``（科创板）→ ``SH_A_STOCK``（pytdx 原版无此分支、返回 ``None``）；
    - ``bj``（北交所）→ ``BJ_A_STOCK``（pytdx 原版不识别 bj、抛 ``NotImplementedError``）。
    """
    base = os.path.basename(filename)
    exchange = base[:2].lower()
    head = base[2:4]
    if exchange == "sz":
        if head in ("00", "30"):
            return "SZ_A_STOCK"
        if head == "20":
            return "SZ_B_STOCK"
        if head == "39":
            return "SZ_INDEX"
        if head in ("15", "16"):
            return "SZ_FUND"
        if head in ("10", "11", "12", "13", "14"):
            return "SZ_BOND"
    elif exchange == "sh":
        if head in ("60", "68"):  # 68=科创板（pytdx 原版漏判）
            return "SH_A_STOCK"
        if head == "90":
            return "SH_B_STOCK"
        if head in ("00", "88", "99"):
            return "SH_INDEX"
        if head in ("50", "51"):
            return "SH_FUND"
        if head in ("01", "10", "11", "12", "13", "14"):
            return "SH_BOND"
    elif exchange == "bj":  # 北交所（pytdx 原版不支持）
        return "BJ_A_STOCK"
    return None


# ============================================================
# 尾部增量定位（二分）
# ============================================================


def _start_key(start: date, kind: str) -> int:
    """构造与文件首字段（日期）同口径的二分比较键。

    日线：``YYYYMMDD`` 整数；分钟：通达信打包日期 ``(年-2004)*2048+月*100+日``。
    早于数据起点的 ``start`` 得到偏小的键，使二分定位到偏移 0（等价全量）。
    """
    if kind == "daily":
        return start.year * 10000 + start.month * 100 + start.day
    return (start.year - 2004) * 2048 + start.month * 100 + start.day


def _lower_bound(f, n: int, target: int, kind: str) -> int:
    """二分查找首条 ``date >= target`` 的记录序号（记录按时间升序、定长）。"""
    fmt, size = ("<I", 4) if kind == "daily" else ("<H", 2)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        f.seek(mid * RECORD_SIZE)
        key = struct.unpack(fmt, f.read(size))[0]
        if key < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _read_buffer(path: str, start: date | None, kind: str) -> bytes:
    """读取文件中 ``date >= start`` 的尾部字节（``start=None`` → 整文件）。"""
    n = os.path.getsize(path) // RECORD_SIZE
    if n == 0:
        return b""
    if start is None:
        with open(path, "rb") as f:
            return f.read()
    target = _start_key(start, kind)
    with open(path, "rb") as f:
        first = _lower_bound(f, n, target, kind)
        if first >= n:
            return b""
        f.seek(first * RECORD_SIZE)
        return f.read((n - first) * RECORD_SIZE)


# ============================================================
# 解析
# ============================================================


def read_day(path: str, start: date | None = None) -> pd.DataFrame:
    """读取日线 ``.day``，返回不复权 OHLCV（价=元、量=股、额=元），index=date。

    ``start`` 给定时只解析 ``date >= start`` 的尾部记录。未知板块抛 ``UnknownSecurityType``。
    """
    sec = classify_security(path)
    if sec is None:
        raise UnknownSecurityType(path)
    price_coef, vol_coef = SECURITY_COEFFICIENT[sec]
    buf = _read_buffer(path, start, "daily")
    if not buf:
        return pd.DataFrame(columns=_OUT_COLS)
    arr = np.frombuffer(buf, dtype=_DAILY_DT)
    idx = pd.to_datetime(arr["date"].astype("int64").astype(str), format="%Y%m%d")
    df = pd.DataFrame({
        "open": arr["open"].astype("float64") * price_coef,
        "high": arr["high"].astype("float64") * price_coef,
        "low": arr["low"].astype("float64") * price_coef,
        "close": arr["close"].astype("float64") * price_coef,
        "amount": arr["amount"].astype("float64"),
        "volume": arr["volume"].astype("float64") * vol_coef,
    }, index=pd.DatetimeIndex(idx, name="date"))
    return df[_OUT_COLS]


def read_lc(path: str, start: date | None = None) -> pd.DataFrame:
    """读取分钟 ``.lc1``/``.lc5``，返回不复权 OHLCV（价=元、量=股、额=元），index=datetime。

    分钟记录价格/量均为真实值（无系数），不区分板块。``start`` 给定时只解析当天及之后。
    """
    buf = _read_buffer(path, start, "min")
    if not buf:
        return pd.DataFrame(columns=_OUT_COLS)
    arr = np.frombuffer(buf, dtype=_MIN_DT)
    d = arr["date"].astype("int64")
    t = arr["time"].astype("int64")
    idx = pd.to_datetime(pd.DataFrame({
        "year": d // 2048 + 2004,
        "month": (d % 2048) // 100,
        "day": (d % 2048) % 100,
        "hour": t // 60,
        "minute": t % 60,
    }))
    df = pd.DataFrame({
        "open": arr["open"].astype("float64"),
        "high": arr["high"].astype("float64"),
        "low": arr["low"].astype("float64"),
        "close": arr["close"].astype("float64"),
        "amount": arr["amount"].astype("float64"),
        "volume": arr["volume"].astype("float64"),
    }, index=pd.DatetimeIndex(idx, name="datetime"))
    return df[_OUT_COLS]


# ============================================================
# 权息 gbbq（加密；解密 + 解析）
# ============================================================

#: gbbq 解析结果列（与 pytdx ``GbbqReader.get_df`` 完全一致，便于平替）。
#: 四个浮点字段名沿用 ``category==1`` 语义，不同 category 下含义不同（见 ``src.data.gbbq``）。
GBBQ_COLUMNS = (
    "market", "code", "datetime", "category",
    "hongli_panqianliutong", "peigujia_qianzongguben",
    "songgu_qianzongguben", "peigu_houzongguben",
)

#: 单条 gbbq 记录：24 字节密文（3 个 8 字节块）+ 5 字节明文尾 = 29 字节
_GBBQ_RECORD_SIZE = 29
_U32 = 0xFFFFFFFF


def _gbbq_key_u32() -> list[int]:
    """gbbq 密钥表（4176 字节）按小端 u32 切片为整型列表；所有密钥访问均 4 字节对齐。"""
    from ._gbbq_key import GBBQ_KEY_HEX
    key = bytes.fromhex(GBBQ_KEY_HEX)
    return list(struct.unpack(f"<{len(key) // 4}I", key))


def read_gbbq(path: str) -> pd.DataFrame:
    """解密并解析通达信权息文件 ``gbbq``，返回 ``GBBQ_COLUMNS`` 列的 DataFrame。

    通达信权息文件按记录分组加密（TEA 类分组密码 + 固定密钥表）。本函数移植自
    ``pytdx.reader.GbbqReader`` 的解密 + 解包逻辑（密钥表固化在 ``_gbbq_key``，运行时
    不依赖 pytdx），逐条解密 24 字节密文、拼 5 字节明文尾后按 ``<B7sIBffff`` 解包。

    文件头 4 字节为记录数；逐条 29 字节。空/损坏文件返回空表（列齐全）。
    """
    with open(path, "rb") as f:
        content = f.read()
    if len(content) < 4:
        return pd.DataFrame(columns=list(GBBQ_COLUMNS))
    (count,) = struct.unpack("<I", content[:4])
    k = _gbbq_key_u32()
    k0, k44 = k[0], k[0x44 >> 2]
    # 四张子表在 u32 列表中的基址（字节偏移 // 4）
    b48, b448, b848, bc48 = 0x48 >> 2, 0x448 >> 2, 0x848 >> 2, 0xC48 >> 2

    rows: list[tuple] = []
    off = 4
    for _ in range(count):
        clear = bytearray()
        for _blk in range(3):
            ebx = int.from_bytes(content[off:off + 4], "little")
            num = (k44 ^ ebx) & _U32
            numold = int.from_bytes(content[off + 4:off + 8], "little")
            for j in range(0x40, 0, -4):  # 0x40,0x3C,…,4，共 16 轮
                eax = k[b448 + ((num >> 16) & 0xFF)]
                eax = (eax + k[b48 + ((num >> 24) & 0xFF)]) & _U32
                eax = (eax ^ k[b848 + ((num >> 8) & 0xFF)]) & _U32
                eax = (eax + k[bc48 + (num & 0xFF)]) & _U32
                eax = (eax ^ k[j >> 2]) & _U32
                prev = num
                num = (numold ^ eax) & _U32
                numold = prev
            numold = (numold ^ k0) & _U32
            clear += struct.pack("<II", numold, num)
            off += 8
        clear += content[off:off + 5]
        off += 5
        v1, v2, v3, v4, v5, v6, v7, v8 = struct.unpack("<B7sIBffff", clear)
        rows.append((
            v1, v2.rstrip(b"\x00").decode("utf-8", "ignore"), v3, v4, v5, v6, v7, v8,
        ))
    return pd.DataFrame(rows, columns=list(GBBQ_COLUMNS))


# ============================================================
# 更名史 profile.dat（64 字节定长）
# ============================================================

#: read_profile 原始解码列（code 为 6 位字符串、change_date 为 YYYYMMDD 整数）
PROFILE_COLUMNS = ("code", "name", "change_date")

#: profile.dat 定长记录字节数
_PROFILE_RECORD_SIZE = 64


def read_profile(path: str) -> pd.DataFrame:
    """解析通达信更名文件 ``profile.dat``，返回 ``PROFILE_COLUMNS`` 列的原始解码表。

    每条 64 字节定长（小端）：``[0]`` 标志、``[1:7]`` 6 位代码(ASCII)、``[8:17]`` 原名称
    (GBK,0x00 补齐)、``[17:21]`` 更名日 ``uint32``(YYYYMMDD)。仅做**结构性**校验（代码为
    6 位数字、名称非空），不含交易所归一/日期范围过滤（交由 ``src.data.profile`` 领域层）。
    ``code`` 为 6 位字符串，``change_date`` 为整数 YYYYMMDD。
    """
    with open(path, "rb") as f:
        data = f.read()
    n_full = len(data) - (len(data) % _PROFILE_RECORD_SIZE)
    recs: list[tuple[str, str, int]] = []
    for i in range(0, n_full, _PROFILE_RECORD_SIZE):
        r = data[i:i + _PROFILE_RECORD_SIZE]
        code6 = r[1:7].decode("ascii", "ignore").strip("\x00").strip()
        if len(code6) != 6 or not code6.isdigit():
            continue
        name = r[8:17].split(b"\x00")[0].decode("gbk", "ignore").strip()
        if not name:
            continue
        change_date = struct.unpack("<I", r[17:21])[0]
        recs.append((code6, name, change_date))
    return pd.DataFrame(recs, columns=list(PROFILE_COLUMNS))
