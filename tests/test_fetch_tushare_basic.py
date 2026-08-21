"""``scripts/fetch_tushare_basic.py`` 的离线用例（零网络、不碰正式数据仓）。

覆盖四组互不替代的事：

- :func:`check_no_regression` 纯函数的判别力（T1~T3d）；
- 刷新模式的离线 preflight —— 三参齐给、input 存在且成形、out 全新、父目录可写，
  且**全部发生在任何取数调用之前**（P1~P10）；
- 取数响应的规范化与空返回分级（R1~R5）；
- 护栏**接线**：checker 确实在写文件前被调用、且实参基准正确（W1/W2）；脏码过滤
  确实挂在 ``main`` 路径上（D1）；三段链端到端（C1）；promote 的成败两向（PR1~PR5）。

**为什么 T1 不走 refresh()**：真实合并路径是
``pd.concat([existing, fetched]).drop_duplicates("ts_code", keep="last")``，
它只在重复码中选行、**从不整码删除**，故"伪造一个更短的 fetched 帧"永远触发不了
丢码断言。要测出判别力只能**直接给纯函数传一个缺码 candidate**。刷新流程里这道闸
是给将来的合并语义改动兜底的；它**当下**带电的地方在 :func:`promote`（PR2）。

tushare 全程由 :func:`fake_tushare` 顶替（``refresh()`` 内是 ``import tushare as ts``，
故插进 ``sys.modules`` 即可拦截），任何用例都不发起网络请求。可写性用 monkeypatch
注入失败，**不依赖 chmod** —— root 下权限位不代表可写性，chmod 用例会假绿。
"""

from __future__ import annotations

import os
import stat
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from scripts import fetch_tushare_basic as mod


def _frame(codes: list[str], status: str = "L") -> pd.DataFrame:
    """构造最小 stock_basic 帧：列必须齐 :data:`FIELDS`，否则会被响应校验判为口径变更。"""
    n = len(codes)
    return pd.DataFrame(
        {
            "ts_code": codes,
            "symbol": [c.split(".")[0] for c in codes],
            "name": [f"股票{i}" for i in range(n)],
            "area": ["深圳"] * n,
            "industry": ["银行"] * n,
            "market": ["主板"] * n,
            "exchange": ["SZSE"] * n,
            "list_status": status,
            "list_date": "20200101",
            "delist_date": None,
        }
    )


def _write(path: Path, *frames: pd.DataFrame) -> Path:
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


def _codes(path: Path) -> set[str]:
    return set(pd.read_parquet(path)["ts_code"])


@pytest.fixture
def fake_tushare(monkeypatch):
    """把 ``tushare`` 顶替成可编程的假模块，返回一个记录调用的容器。"""
    calls: list[dict] = []
    payloads: dict[str, pd.DataFrame] = {}

    class _Pro:
        def stock_basic(self, exchange="", list_status="L", fields=""):
            calls.append({"list_status": list_status, "fields": fields})
            # 未预置的状态返回空帧，模拟"该状态无数据"
            return payloads.get(list_status, _frame([], list_status)).copy()

    fake = types.ModuleType("tushare")
    fake.pro_api = lambda token: _Pro()
    monkeypatch.setitem(sys.modules, "tushare", fake)

    return types.SimpleNamespace(calls=calls, payloads=payloads)


@pytest.fixture
def run_main(monkeypatch):
    """以给定 argv 调用 ``mod.main()``。"""

    def _run(*argv: str) -> int:
        monkeypatch.setattr(sys, "argv", ["prog", *argv])
        return mod.main()

    return _run


# --------------------------------------------------------------------------
# T1~T3d：check_no_regression 纯函数
# --------------------------------------------------------------------------

def test_t1_rejects_candidate_missing_existing_code():
    """T1：candidate 丢掉 existing 的干净码 → 必须报错。

    这是本护栏**唯一**真正带电的构造。用真实合并结果去测它恒绿，说明不了任何事。
    """
    existing = _frame(["000001.SZ", "000002.SZ", "600000.SH"])
    candidate = _frame(["000001.SZ", "600000.SH"])  # 少了 000002.SZ

    with pytest.raises(SystemExit) as excinfo:
        mod.check_no_regression(existing, candidate)

    msg = str(excinfo.value)
    assert "丢失 1 个既有代码" in msg, msg
    assert "000002.SZ" in msg, "报错须点名丢失的码，否则无法定位"


def test_t2_accepts_real_merge_result():
    """T2：真实合并语义产出的 candidate 必须通过（不得假红）。"""
    existing = _frame(["000001.SZ", "000002.SZ"], status="L")
    fetched = _frame(["000002.SZ", "600000.SH"], status="D")  # 000002 由 L 转 D
    candidate = pd.concat([existing, fetched], ignore_index=True).drop_duplicates(
        "ts_code", keep="last"
    )

    mod.check_no_regression(existing, candidate)  # 不抛即通过

    got = dict(zip(candidate.ts_code, candidate.list_status, strict=True))
    assert got["000002.SZ"] == "D", "keep='last' 应让新状态覆盖旧状态"
    assert set(candidate.ts_code) == {"000001.SZ", "000002.SZ", "600000.SH"}


def test_t3_exempts_dirty_codes_cleaned_away():
    """T3：existing 里的脏码被合法清洗掉，**不得**被判成回归。"""
    existing = _frame(["000001.SZ", "T600018.SH", "600000.SH"])
    assert not mod._clean_mask(existing["ts_code"]).all(), "夹具须真的含脏码，否则本例空转"

    candidate = _frame(["000001.SZ", "600000.SH"])  # 脏码已被清洗

    mod.check_no_regression(existing, candidate)  # 不抛即通过


def test_t3b_rejects_duplicate_codes():
    """T3b：candidate 含重复 ts_code → 报错（合并忘了去重的兜底）。"""
    with pytest.raises(SystemExit, match="重复 ts_code"):
        mod.check_no_regression(_frame(["000001.SZ"]), _frame(["000001.SZ", "000001.SZ"]))


def test_t3c_dup_guard_is_unconditional_when_existing_empty():
    """T3c：``existing`` 为空（首次引导）时，重复码检查**仍须生效**。

    旧实现把 dup 检查挂在"existing 非空"的早退之后，于是引导路径上重复码直接放行。
    重复与否和有没有基准毫无关系。
    """
    dup = _frame(["000001.SZ", "000001.SZ"])
    with pytest.raises(SystemExit, match="重复 ts_code"):
        mod.check_no_regression(pd.DataFrame(), dup)


def test_t3d_rejects_candidate_without_ts_code():
    """T3d：candidate 缺 ts_code 列 → 报错，且同样不依赖 existing 非空。"""
    with pytest.raises(SystemExit, match="缺 ts_code 列"):
        mod.check_no_regression(pd.DataFrame(), pd.DataFrame({"name": ["x"]}))


# --------------------------------------------------------------------------
# P1~P10：刷新模式的离线 preflight
# --------------------------------------------------------------------------

def test_p1_out_only_is_rejected(tmp_path, fake_tushare, run_main, monkeypatch):
    """P1：只给 --out → 必须三参齐给。

    旧实现把"给了任一路径"当成显式模式却仍允许缺参，是 MAJ-2 的入口。
    """
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "cand.parquet"

    with pytest.raises(SystemExit, match="必须同时给出"):
        run_main("--out", str(out), "--token", "x")

    assert not out.exists()
    assert fake_tushare.calls == [], "必须在任何取数调用之前就拒绝"


def test_p2_input_only_is_rejected(tmp_path, fake_tushare, run_main, monkeypatch):
    """P2：只给 --input → 必须三参齐给。"""
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"]))

    with pytest.raises(SystemExit, match="必须同时给出"):
        run_main("--input", str(src), "--token", "x")

    assert fake_tushare.calls == []


def test_p3_missing_status_is_rejected(tmp_path, fake_tushare, run_main, monkeypatch):
    """P3：给了 --input/--out 但缺 --status → 拒绝。

    这是 MAJ-2 的正面复现：旧实现会走"自动补缺失状态"，输入已齐三状态时打印
    "已齐全"、return 0，而 --out 从未落盘 —— 调用者按退出码 0 去提升，提升的是
    上一轮的陈旧候选。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet",
                 _frame(["000001.SZ"], "L"), _frame(["000003.SZ"], "D"),
                 _frame(["600001.SH"], "P"))
    out = tmp_path / "cand.parquet"

    with pytest.raises(SystemExit, match="必须同时给出"):
        run_main("--input", str(src), "--out", str(out), "--token", "x")

    assert not out.exists()
    assert fake_tushare.calls == []


def test_p4_existing_out_is_rejected(tmp_path, fake_tushare, run_main, monkeypatch):
    """P4：--out 已存在 → 拒绝，且既有内容一字节不动。

    这是 MAJ-1 的闭环点：候选恒为全新文件，"写反 --out 把正式文件夷平"这条路径
    根本不存在，而不是靠事后断言去追。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"]))
    out = _write(tmp_path / "cand.parquet", _frame([f"{i:06d}.SZ" for i in range(1000, 1100)]))
    before = out.read_bytes()

    with pytest.raises(SystemExit, match="输出文件已存在"):
        run_main("--input", str(src), "--out", str(out), "--status", "D", "--token", "x")

    assert out.read_bytes() == before, "被拒绝的运行不得改动既有文件"
    assert fake_tushare.calls == []


def test_p5_same_path_after_resolve_is_rejected(tmp_path, fake_tushare, run_main, monkeypatch):
    """P5：--input 与 --out 解析后同一 → 拒绝，且报的必须是"同一个文件"。

    用**相对路径 vs 绝对路径**构造：字符串上并不相等，只有 ``resolve()`` 之后才看得
    出是同一份数据。断言具体文案而非只断言 ``SystemExit`` —— 若实现漏了 ``resolve()``，
    它会退到"输出文件已存在"那条分支，文案不同，本例即变红。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"]))

    with pytest.raises(SystemExit, match="解析后是同一个文件"):
        run_main("--input", str(src), "--out", "./real.parquet",
                 "--status", "D", "--token", "x")

    assert fake_tushare.calls == []


def test_p5b_symlinked_out_is_rejected(tmp_path, fake_tushare, run_main, monkeypatch):
    """P5b：--out 是指向 --input 的符号链接 → 同样按"同一个文件"拒绝。"""
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"]))
    link = tmp_path / "link.parquet"
    link.symlink_to(src)

    with pytest.raises(SystemExit, match="解析后是同一个文件"):
        run_main("--input", str(src), "--out", str(link), "--status", "D", "--token", "x")

    assert fake_tushare.calls == []


def test_p6_missing_out_parent_is_rejected(tmp_path, fake_tushare, run_main, monkeypatch):
    """P6：--out 父目录不存在 → 拒绝（脚本不代建候选目录），且不烧取数额度。"""
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"]))
    out = tmp_path / "nope" / "cand.parquet"

    with pytest.raises(SystemExit, match="输出目录不存在"):
        run_main("--input", str(src), "--out", str(out), "--status", "D", "--token", "x")

    assert not out.exists()
    assert fake_tushare.calls == []


def test_p7_unwritable_out_parent_fails_before_any_fetch(
    tmp_path, fake_tushare, run_main, monkeypatch
):
    """P7：--out 父目录存在但不可写 → 在**任何取数之前**失败。

    "父目录存在"挡不住"父目录不可写"：后者会一路走到 ``to_parquet`` 才抛
    ``PermissionError``，那时额度已经烧掉了。

    失败由 monkeypatch 注入而非 chmod：root 下权限位不代表可写性，chmod 用例会假绿。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"]))
    out = tmp_path / "cand.parquet"

    def _boom(directory):
        raise PermissionError(f"injected: {directory}")

    monkeypatch.setattr(mod, "_probe_writable", _boom)

    with pytest.raises(SystemExit, match="输出目录不可写"):
        run_main("--input", str(src), "--out", str(out), "--status", "D", "--token", "x")

    assert not out.exists()
    assert fake_tushare.calls == [], "配额必须在写路径校验之后才可能被消耗"


def test_p8_writability_probe_leaves_no_residue(tmp_path, fake_tushare, run_main, monkeypatch):
    """P8：可写性探针必须自清理，不得在候选目录留下 .wtest_ 残留。"""
    monkeypatch.chdir(tmp_path)
    outdir = tmp_path / "cands"
    outdir.mkdir()
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"]))
    fake_tushare.payloads["D"] = _frame(["000003.SZ"], "D")

    assert run_main("--input", str(src), "--out", str(outdir / "cand.parquet"),
                    "--status", "D", "--token", "x") == 0

    assert [p.name for p in outdir.iterdir()] == ["cand.parquet"]


def test_p9_explicit_mode_invariant_no_silent_zero_exit(
    tmp_path, fake_tushare, run_main, monkeypatch
):
    """P9：显式模式不变式 —— 要么非零退出，要么 --out 存在，绝不"成功但无产物"。

    参数化跑遍各失败分支。MAJ-2 的本质不是"缺 status"这一种写法，而是"退出码 0 却
    没有候选"这个状态本身；本例锁的是那个状态。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    exists = _write(tmp_path / "taken.parquet", _frame(["000009.SZ"], "L"))
    bad_schema = tmp_path / "bad.parquet"
    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(bad_schema, index=False)

    cases = [
        ["--out", str(tmp_path / "a.parquet")],
        ["--input", str(src)],
        ["--input", str(src), "--out", str(tmp_path / "b.parquet")],
        ["--input", str(src), "--out", str(exists), "--status", "D"],
        ["--input", str(tmp_path / "missing.parquet"), "--out", str(tmp_path / "c.parquet"),
         "--status", "D"],
        ["--input", str(bad_schema), "--out", str(tmp_path / "d.parquet"), "--status", "D"],
        ["--input", str(src), "--out", str(tmp_path / "gone" / "e.parquet"), "--status", "D"],
    ]

    for argv in cases:
        fake_tushare.calls.clear()
        with pytest.raises(SystemExit) as excinfo:
            run_main(*argv, "--token", "x")
        assert excinfo.value.code != 0, argv
        assert fake_tushare.calls == [], f"{argv} 必须在取数之前失败"

    produced = {p.name for p in tmp_path.iterdir() if p.suffix == ".parquet"}
    assert produced == {"real.parquet", "taken.parquet", "bad.parquet"}, \
        "失败的运行不得留下任何候选文件"
    assert _codes(exists) == {"000009.SZ"}, "既有文件不得被改动"


def test_p10_input_schema_is_validated(tmp_path, fake_tushare, run_main, monkeypatch):
    """P10：--input 缺 ts_code/list_status → fail-closed，且不烧额度。

    schema 不对的 input 会让 :func:`check_no_regression` 的丢码判据静默变成空操作 ——
    而显式模式的全部安全性都建立在"基准是 input"之上，所以这里必须拦死。
    """
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(bad, index=False)  # 缺 list_status
    out = tmp_path / "cand.parquet"

    with pytest.raises(SystemExit, match="缺列"):
        run_main("--input", str(bad), "--out", str(out), "--status", "D", "--token", "x")

    assert not out.exists()
    assert fake_tushare.calls == []


def test_t4_explicit_missing_input_fails_closed(tmp_path, fake_tushare, run_main, monkeypatch):
    """T4：显式 --input 指向不存在的文件 → fail-closed，且不落盘任何东西。

    旧实现会把它当成"首次引导"，落盘一份只含单个 list_status 的残缺文件并 return 0。
    """
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out.parquet"

    with pytest.raises(SystemExit, match="输入文件不存在"):
        run_main("--input", str(tmp_path / "not_there.parquet"), "--out", str(out),
                 "--status", "D", "--token", "x")

    assert not out.exists(), "fail-closed 必须在落盘之前发生"
    assert fake_tushare.calls == [], "必须在任何取数调用之前就拒绝"


def test_t5_input_and_out_are_separate_paths(tmp_path, fake_tushare, run_main, monkeypatch):
    """T5：``--input A --out B`` 读 A 写 B，且 **A 字节不变**。"""
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ", "000002.SZ"], "L"))
    dst = tmp_path / "cand.parquet"
    before = src.read_bytes()

    fake_tushare.payloads["D"] = _frame(["000002.SZ", "600000.SH"], "D")

    assert run_main("--input", str(src), "--out", str(dst), "--status", "D", "--token", "x") == 0

    assert src.read_bytes() == before, "输入文件必须一个字节都没动"
    got = pd.read_parquet(dst)
    assert set(got.ts_code) == {"000001.SZ", "000002.SZ", "600000.SH"}
    status = dict(zip(got.ts_code, got.list_status, strict=True))
    assert status["000002.SZ"] == "D", "L→D 迁移应被新 D 行覆盖"
    assert status["000001.SZ"] == "L", "未迁移的既有行应原样保留"
    assert [c["list_status"] for c in fake_tushare.calls] == ["D"], "只应取被指定的那个状态"


def test_t6_no_path_args_keeps_legacy_default(tmp_path, fake_tushare, run_main, monkeypatch):
    """T6：不给路径参数时行为与引入本参数前完全一致（相对 OUT、自动补缺失状态）。"""
    assert Path("data_store/stock_basic_tushare.parquet") == mod.OUT, "默认常量不得改动"

    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / mod.OUT
    legacy.parent.mkdir(parents=True, exist_ok=True)
    _write(legacy, _frame(["000001.SZ"], "L"), _frame(["000003.SZ"], "D"))

    fake_tushare.payloads["P"] = _frame(["600001.SH"], "P")

    assert run_main("--token", "x") == 0

    # 自动补缺失状态：已有 {L, D}，故本次应且只应取 P
    assert [c["list_status"] for c in fake_tushare.calls] == ["P"]
    got = pd.read_parquet(legacy)
    assert set(got.ts_code) == {"000001.SZ", "000003.SZ", "600001.SH"}
    assert dict(zip(got.ts_code, got.list_status, strict=True))["600001.SH"] == "P"


def test_t6b_no_path_args_bootstrap_still_allowed(tmp_path, fake_tushare, run_main, monkeypatch):
    """T6b：无参且文件尚不存在 → 仍走首次引导，不 fail-closed（含自动建仓目录）。"""
    monkeypatch.chdir(tmp_path)
    fake_tushare.payloads["L"] = _frame(["000001.SZ"], "L")

    assert run_main("--token", "x") == 0

    assert (tmp_path / mod.OUT).exists()
    assert [c["list_status"] for c in fake_tushare.calls] == ["L"], "空文件时应从第一个状态补起"


def test_t6c_legacy_complete_store_is_noop(tmp_path, fake_tushare, run_main, monkeypatch):
    """T6c：旧式模式下三状态已齐 → 打印"已齐全"并 return 0，不取数。

    这条 return-0-without-writing 在旧式模式里是正确的（in 与 out 本就同一个文件），
    保留它；显式模式则由 P3 堵死。
    """
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / mod.OUT
    legacy.parent.mkdir(parents=True, exist_ok=True)
    _write(legacy, _frame(["000001.SZ"], "L"), _frame(["000003.SZ"], "D"),
           _frame(["600001.SH"], "P"))
    before = legacy.read_bytes()

    assert run_main("--token", "x") == 0
    assert fake_tushare.calls == []
    assert legacy.read_bytes() == before


# --------------------------------------------------------------------------
# R1~R5：响应规范化与空返回分级
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["D", "L"])
def test_r1_empty_response_for_d_or_l_fails_closed(
    status, tmp_path, fake_tushare, run_main, monkeypatch
):
    """R1：D/L 返回 0 行 → fail-closed，且**不产出任何文件**。

    旧实现照常落盘并打印"已保存 … 合计 N"、return 0：对自动化调用者而言，"刷新成功"
    与"这次啥也没取到"完全不可区分。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    out = tmp_path / "cand.parquet"

    fake_tushare.payloads[status] = _frame([], status)  # 0 行

    with pytest.raises(SystemExit, match="返回 0 行"):
        run_main("--input", str(src), "--out", str(out), "--status", status, "--token", "x")

    assert not out.exists(), "取数失败不得留下候选"


def test_r2_empty_response_for_p_is_a_valid_probe(tmp_path, fake_tushare, run_main, monkeypatch, capsys):
    """R2：P 返回 0 行是合法探针结果 → 成功产出与 input 等价的**全新**候选。

    P=暂停上市，市场上确实可能一只都没有。把它一并 fail-closed 会堵死正常流程。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ", "000002.SZ"], "L"))
    out = tmp_path / "cand_P.parquet"

    fake_tushare.payloads["P"] = _frame([], "P")

    assert run_main("--input", str(src), "--out", str(out), "--status", "P", "--token", "x") == 0

    assert _codes(out) == _codes(src), "P 空时候选应与输入等价"
    assert "探针状态允许为空" in capsys.readouterr().out, "空探针必须在输出里说清楚"


def test_r3_columnless_empty_response_does_not_crash(tmp_path, fake_tushare, run_main, monkeypatch):
    """R3：tushare 返回连列都没有的空帧 → 规范化后正常走完，不再裸抛 KeyError。"""
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    out = tmp_path / "cand.parquet"

    fake_tushare.payloads["P"] = pd.DataFrame()  # 无列空帧

    assert run_main("--input", str(src), "--out", str(out), "--status", "P", "--token", "x") == 0
    assert _codes(out) == {"000001.SZ"}


def test_r4_nonempty_response_missing_field_fails_closed(
    tmp_path, fake_tushare, run_main, monkeypatch
):
    """R4：非空响应缺 FIELDS 任一列 → fail-closed。

    列缺失意味着接口口径变了。静默补空会把"这一列没取到"伪装成"这一列全是空值"，
    一路落盘无人察觉。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    out = tmp_path / "cand.parquet"

    fake_tushare.payloads["D"] = _frame(["000003.SZ"], "D").drop(columns=["area"])

    with pytest.raises(SystemExit, match="缺列"):
        run_main("--input", str(src), "--out", str(out), "--status", "D", "--token", "x")

    assert not out.exists()


def test_r5_empty_input_and_empty_probe_still_produces_a_candidate(
    tmp_path, fake_tushare, run_main, monkeypatch
):
    """R5：输入零行 + P 探针为空 → **rc=0 且产出 0 行标准 schema 候选**，不是 fail-closed。

    P 空是合法探针结果，与输入有几行无关；输入零行时"与输入等价的候选"就是零行候选。
    把这一格写成 fail-closed 会堵死正常流程，而且写反方向的断言比没有断言更糟 ——
    它让一次偏离以全绿的形式固化下来。真正的丢码闸在 promote：拿零行候选去替换非空
    正式文件会被 PR2 那条判据拦住。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame([], "L"))
    out = tmp_path / "cand.parquet"

    fake_tushare.payloads["P"] = pd.DataFrame()  # 无列空帧

    assert run_main("--input", str(src), "--out", str(out),
                    "--status", "P", "--token", "x") == 0

    got = pd.read_parquet(out)
    assert len(got) == 0, "应产出 0 行候选"
    assert list(got.columns) == list(mod.FIELD_LIST), "0 行候选的列顺序须严格等于 FIELD_LIST"


def test_r5b_zero_row_candidate_is_still_gated_by_promote(tmp_path, no_network, run_main):
    """R5b：R5 放行的零行候选，拿去替换非空正式文件时必须被 promote 拦下。

    这是 R5 改向后的配套证明 —— 放宽刷新侧不等于放宽落地侧，闸只是挪到了真正会
    覆盖正式数据的那一步。
    """
    tgt = _write(tmp_path / "real.parquet", _frame(["000001.SZ", "000002.SZ"], "L"))
    empty_cand = _write(tmp_path / "cand.parquet", _frame([], "L"))
    before = tgt.read_bytes()

    with pytest.raises(SystemExit, match="丢失 2 个既有代码"):
        run_main("--promote", str(empty_cand), "--target", str(tgt))

    assert tgt.read_bytes() == before


# --------------------------------------------------------------------------
# W1/W2/D1/C1：护栏接线与端到端
# --------------------------------------------------------------------------

class _Sentinel(Exception):
    """只用于证明某个调用确实发生过。"""


def test_w1_checker_is_wired_before_write(tmp_path, fake_tushare, run_main, monkeypatch):
    """W1：``check_no_regression`` 必须在写文件**之前**被真正调用。

    纯函数测得再全，也测不出"调用点被删掉了"。删掉 ``refresh()`` 里那一行，T1~T3d
    全绿 —— 本例是唯一会红的。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    out = tmp_path / "cand.parquet"
    fake_tushare.payloads["D"] = _frame(["000003.SZ"], "D")

    def _explode(existing, candidate):
        raise _Sentinel

    monkeypatch.setattr(mod, "check_no_regression", _explode)

    with pytest.raises(_Sentinel):
        run_main("--input", str(src), "--out", str(out), "--status", "D", "--token", "x")

    assert not out.exists(), "checker 必须在 to_parquet 之前被调用"


def test_w2_checker_receives_input_as_baseline(tmp_path, fake_tushare, run_main, monkeypatch):
    """W2：checker 的 ``existing`` 实参必须是 **input 的内容**，不是候选自己。

    只断言"被调用过"不够：把调用写成 ``check_no_regression(out, out)``，护栏变成拿
    候选跟自己比、判据恒真，而一个只验"调用发生"的哨兵仍会全绿。MAJ-1 的本质就是
    基准对错了，接线测试必须验基准。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ", "000002.SZ"], "L"))
    out = tmp_path / "cand.parquet"
    fake_tushare.payloads["D"] = _frame(["000003.SZ"], "D")

    seen: list[tuple[set[str], set[str]]] = []

    def _record(existing, candidate):
        seen.append((set(existing["ts_code"]), set(candidate["ts_code"])))

    monkeypatch.setattr(mod, "check_no_regression", _record)

    assert run_main("--input", str(src), "--out", str(out), "--status", "D", "--token", "x") == 0

    assert len(seen) == 1, "应恰好调用一次"
    existing_codes, candidate_codes = seen[0]
    assert existing_codes == {"000001.SZ", "000002.SZ"}, "基准必须是 input 的码集"
    assert candidate_codes == {"000001.SZ", "000002.SZ", "000003.SZ"}
    assert existing_codes != candidate_codes, "拿候选当基准等于没校验"


def test_d1_dirty_codes_are_filtered_on_the_main_path(
    tmp_path, fake_tushare, run_main, monkeypatch, capsys
):
    """D1：脏码过滤确实挂在刷新路径上（不只是纯函数里有个豁免规则）。

    停用 ``refresh()`` 里的过滤，T3 照样绿 —— 因为 T3 只测纯函数。本例是唯一会红的。
    """
    monkeypatch.chdir(tmp_path)
    src = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    out = tmp_path / "cand.parquet"
    fake_tushare.payloads["D"] = _frame(["T600018.SH", "000003.SZ"], "D")

    assert run_main("--input", str(src), "--out", str(out), "--status", "D", "--token", "x") == 0

    assert _codes(out) == {"000001.SZ", "000003.SZ"}, "脏码必须在落盘前被清掉"
    assert "过滤脏码 1 条" in capsys.readouterr().out


def test_c1_three_stage_chain_keeps_the_real_input_untouched(
    tmp_path, fake_tushare, run_main, monkeypatch
):
    """C1：正式输入 → D 候选 → L 候选 → P 候选 三段链端到端。

    这是真实操作流程本身：每段读上一段的产物、写一个**全新**路径，正式输入全程只读，
    且每段码集是前一段的超集。链式传递没有测试覆盖的话，单段绿说明不了流程是对的。
    """
    monkeypatch.chdir(tmp_path)
    real = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    real_bytes = real.read_bytes()

    fake_tushare.payloads["D"] = _frame(["000003.SZ"], "D")
    fake_tushare.payloads["L"] = _frame(["000001.SZ", "600000.SH"], "L")
    fake_tushare.payloads["P"] = _frame(["600001.SH"], "P")

    stages = [("D", real, tmp_path / "cand_D.parquet"),
              ("L", tmp_path / "cand_D.parquet", tmp_path / "cand_L.parquet"),
              ("P", tmp_path / "cand_L.parquet", tmp_path / "cand_P.parquet")]

    for status, src, dst in stages:
        assert run_main("--input", str(src), "--out", str(dst),
                        "--status", status, "--token", "x") == 0
        assert _codes(dst) >= _codes(src), f"{status} 段不得丢码"

    assert real.read_bytes() == real_bytes, "正式输入全程一个字节都不能动"
    assert _codes(tmp_path / "cand_P.parquet") == {
        "000001.SZ", "000003.SZ", "600000.SH", "600001.SH"}
    assert [c["list_status"] for c in fake_tushare.calls] == ["D", "L", "P"]


# --------------------------------------------------------------------------
# PR1~PR5：promote 模式
# --------------------------------------------------------------------------

@pytest.fixture
def no_network(monkeypatch):
    """让 promote 一旦碰 token 或 tushare 就立刻炸。"""
    def _no_token(cli_token):
        raise AssertionError("promote 不得读取 token")

    class _Pro:
        def stock_basic(self, *a, **kw):
            raise AssertionError("promote 不得调用 tushare")

    fake = types.ModuleType("tushare")
    fake.pro_api = lambda token: _Pro()
    monkeypatch.setattr(mod, "_load_token", _no_token)
    monkeypatch.setitem(sys.modules, "tushare", fake)


def test_pr1_promote_replaces_target_atomically(tmp_path, no_network, run_main, capsys):
    """PR1：候选是 target 的超集 → 原子替换成功，target 内容变成候选，**权限位不变**。

    这一步是整条流程里唯一真正覆盖正式数据的动作，此前完全无防护、无脚本、无测试。

    **权限位断言不是凑数**：staged 由 ``tempfile.mkstemp`` 建出，权限恒为 ``0o600``；
    ``os.replace`` 搬的是 staged 那个 inode，所以 :func:`promote` 里的
    ``shutil.copymode`` 是承重的。删掉它，正式文件会从 0o664 静默降级成 0o600 ——
    其他账号/服务再读数据仓就是 ``PermissionError``，而 rc 仍是 0、日志仍打印"已替换"。
    没有这条断言，那行被当成多余清理掉时不会有任何用例变红。

    夹具显式 ``chmod`` 到 0o664（正式文件的真实权限位），断言的是**替换前后一致**，
    因此不受 umask 影响。
    """
    tgt = _write(tmp_path / "real.parquet", _frame(["000001.SZ", "000002.SZ"], "L"))
    cand = _write(tmp_path / "cand.parquet",
                  _frame(["000001.SZ", "000002.SZ"], "L"), _frame(["000003.SZ"], "D"))
    os.chmod(tgt, 0o664)
    mode_before = stat.S_IMODE(tgt.stat().st_mode)
    assert mode_before == 0o664, "夹具须先把权限位钉死，否则断言随 umask 漂移"

    assert run_main("--promote", str(cand), "--target", str(tgt)) == 0

    assert _codes(tgt) == {"000001.SZ", "000002.SZ", "000003.SZ"}
    assert cand.exists(), "promote 不应消耗掉候选文件"
    assert "已替换" in capsys.readouterr().out
    assert {p.name for p in tmp_path.iterdir()} == {"real.parquet", "cand.parquet"}, \
        "落地用的临时文件必须已被 os.replace 消费掉，不留 .promote_ 残留"
    assert stat.S_IMODE(tgt.stat().st_mode) == mode_before, \
        "promote 必须保留正式文件权限位（丢掉 copymode 会退化成 mkstemp 的 0o600）"


def test_pr2_promote_refuses_to_lose_codes_and_leaves_target_intact(
    tmp_path, no_network, run_main
):
    """PR2：候选丢了 target 的既有码 → 拒绝替换，target **一个字节都不动**。

    这正是我在审核里复现过的 5000 行 → 3 行、退出码 0、护栏沉默那一幕；基准换成
    "被覆盖的那一份"之后，它变成一次带电的拦截。
    """
    tgt = _write(tmp_path / "real.parquet",
                 _frame([f"{i:06d}.SZ" for i in range(1000, 6000)], "L"))
    cand = _write(tmp_path / "cand.parquet", _frame(["001000.SZ", "001001.SZ"], "L"))
    before = tgt.read_bytes()

    with pytest.raises(SystemExit) as excinfo:
        run_main("--promote", str(cand), "--target", str(tgt))

    assert "丢失 4998 个既有代码" in str(excinfo.value)
    assert tgt.read_bytes() == before, "校验失败时正式文件必须原封不动"
    assert len(pd.read_parquet(tgt)) == 5000


def test_pr6_validation_reads_staged_bytes_not_the_candidate_path(
    tmp_path, no_network, run_main, monkeypatch
):
    """PR6：校验对象必须是 staged 的**实际字节**，而不是候选路径上"当时读到"的内容。

    注入方式等价于"候选在校验与复制之间被并发改写"（另一个终端重跑刷新阶段、cron、
    ``/tmp`` 清理都够）：让 ``copyfile`` 往 staged 写进一份丢码帧。

    - 旧顺序（先读 cand 校验、再 copyfile）：校验看到的是完好的 cand → 放行 →
      未经校验的字节落进正式文件，且成功日志还打印着第一次读到的行数。本例会红。
    - 新顺序（先复制、再读 staged 校验）：校验看到的就是要写进去的那一份 → 拦下。
    """
    tgt = _write(tmp_path / "real.parquet", _frame(["000001.SZ", "000002.SZ"], "L"))
    cand = _write(tmp_path / "cand.parquet", _frame(["000001.SZ", "000002.SZ"], "L"))
    before = tgt.read_bytes()

    def _copy_something_else(src, dst):
        _frame(["000001.SZ"], "L").to_parquet(dst, index=False)  # 丢了 000002.SZ

    monkeypatch.setattr(mod.shutil, "copyfile", _copy_something_else)

    with pytest.raises(SystemExit, match="丢失 1 个既有代码"):
        run_main("--promote", str(cand), "--target", str(tgt))

    assert tgt.read_bytes() == before, "校验失败时正式文件必须原封不动"
    assert {p.name for p in tmp_path.iterdir()} == {"real.parquet", "cand.parquet"}, \
        "失败路径也必须清掉 staged 临时文件"


def test_pr7_staged_is_cleaned_up_when_replace_fails(tmp_path, no_network, run_main, monkeypatch):
    """PR7：``os.replace`` 阶段抛 OSError → 包装成 SystemExit，target 不变、staged 清干净。

    与 PR6 分开：那条走的是校验失败路径，这条走的是 I/O 失败路径，两者的清理分支不同。
    """
    tgt = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    cand = _write(tmp_path / "cand.parquet", _frame(["000001.SZ", "000002.SZ"], "L"))
    before = tgt.read_bytes()

    def _boom(src, dst):
        raise OSError("injected replace failure")

    monkeypatch.setattr(mod.os, "replace", _boom)

    with pytest.raises(SystemExit, match="替换失败"):
        run_main("--promote", str(cand), "--target", str(tgt))

    assert tgt.read_bytes() == before
    assert {p.name for p in tmp_path.iterdir()} == {"real.parquet", "cand.parquet"}


@pytest.mark.parametrize("dropped", list(mod.FIELD_LIST))
def test_pr8_promote_requires_every_standard_column(
    dropped, tmp_path, no_network, run_main
):
    """PR8：候选缺 FIELD_LIST 任一列 → 拒绝，target 字节不变、staged 无残留。

    候选在其余方面完全合法（ts_code 是 target 的超集、无重复），失败只能归因于缺列。
    只查 ts_code/list_status 两列的话，一份丢了 list_date/delist_date 的候选会通过
    ``check_no_regression`` 并**永久**覆盖正式文件 —— promote 是全流程唯一不可逆的
    动作，且下游对缺列是静默降级（``if "name" in d.columns else ""``），两头都不报错。
    """
    tgt = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    cand = _write(tmp_path / "cand.parquet", _frame(["000001.SZ", "000002.SZ"], "L"))
    pd.read_parquet(cand).drop(columns=[dropped]).to_parquet(cand, index=False)
    before = tgt.read_bytes()

    with pytest.raises(SystemExit, match="缺列"):
        run_main("--promote", str(cand), "--target", str(tgt))

    assert tgt.read_bytes() == before, f"缺 {dropped} 时正式文件必须原封不动"
    assert {p.name for p in tmp_path.iterdir()} == {"real.parquet", "cand.parquet"}, \
        "失败路径必须清掉 staged"


def test_pr9_promote_accepts_real_world_column_order(tmp_path, no_network, run_main):
    """PR9：正式文件盘上的**真实列序**必须能通过（area/industry 在末尾）。

    这条钉死"成员完整性、不是列序相等"。写成 ``list(columns) == list(FIELD_LIST)``
    的话，对真实正式文件的下一次 promote 会当场失败 —— 盘上的列序与 FIELD_LIST 的
    书写顺序本就不同。
    """
    real_order = ["ts_code", "symbol", "name", "market", "exchange",
                  "list_status", "list_date", "delist_date", "area", "industry"]
    assert sorted(real_order) == sorted(mod.FIELD_LIST), "夹具须是 FIELD_LIST 的重排"
    assert real_order != list(mod.FIELD_LIST), "夹具列序须真的不同，否则本例空转"

    tgt = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L")[real_order])
    cand = _write(tmp_path / "cand.parquet",
                  _frame(["000001.SZ", "000002.SZ"], "L")[real_order])

    assert run_main("--promote", str(cand), "--target", str(tgt)) == 0
    assert _codes(tgt) == {"000001.SZ", "000002.SZ"}


def test_pr10_promote_accepts_extra_columns(tmp_path, no_network, run_main):
    """PR10：候选含额外列 → 放行（要求的是"不缺标准列"，不是"只有标准列"）。"""
    tgt = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    extra = _frame(["000001.SZ", "000002.SZ"], "L")
    extra["curr_type"] = "CNY"  # tushare 将来多返回一列不应把流程卡死
    cand = _write(tmp_path / "cand.parquet", extra)

    assert run_main("--promote", str(cand), "--target", str(tgt)) == 0
    assert "curr_type" in pd.read_parquet(tgt).columns


def test_pr3_promote_rejects_duplicate_and_malformed_candidate(tmp_path, no_network, run_main):
    """PR3：候选含重复码 / 缺列 → 拒绝，target 不动。"""
    tgt = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    before = tgt.read_bytes()

    dup = _write(tmp_path / "dup.parquet", _frame(["000001.SZ", "000001.SZ"], "L"))
    with pytest.raises(SystemExit, match="重复 ts_code"):
        run_main("--promote", str(dup), "--target", str(tgt))

    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(bad, index=False)
    with pytest.raises(SystemExit, match="缺列"):
        run_main("--promote", str(bad), "--target", str(tgt))

    assert tgt.read_bytes() == before


def test_pr4_promote_argument_errors(tmp_path, no_network, run_main):
    """PR4：promote 的参数错误全部 fail-closed。"""
    tgt = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    cand = _write(tmp_path / "cand.parquet", _frame(["000001.SZ"], "L"))

    with pytest.raises(SystemExit, match="必须同时给出"):
        run_main("--promote", str(cand))
    with pytest.raises(SystemExit, match="必须同时给出"):
        run_main("--target", str(tgt))
    with pytest.raises(SystemExit, match="解析后是同一个文件"):
        run_main("--promote", str(cand), "--target", str(cand))
    with pytest.raises(SystemExit, match="候选文件不存在"):
        run_main("--promote", str(tmp_path / "nope.parquet"), "--target", str(tgt))
    with pytest.raises(SystemExit, match="正式文件不存在"):
        run_main("--promote", str(cand), "--target", str(tmp_path / "nope.parquet"))


def test_pr5_promote_rejects_refresh_arguments(tmp_path, no_network, run_main):
    """PR5：promote 与刷新参数混用 → 拒绝（两种模式语义不同，混用必是误用）。"""
    tgt = _write(tmp_path / "real.parquet", _frame(["000001.SZ"], "L"))
    cand = _write(tmp_path / "cand.parquet", _frame(["000001.SZ"], "L"))

    for extra in (["--status", "D"], ["--token", "x"], ["--input", str(cand)],
                  ["--out", str(tmp_path / "z.parquet")]):
        with pytest.raises(SystemExit, match="不接受刷新参数"):
            run_main("--promote", str(cand), "--target", str(tgt), *extra)
