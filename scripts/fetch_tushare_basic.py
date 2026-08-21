#!/usr/bin/env python
"""拉取并保存 tushare ``stock_basic`` 全量股票基础信息（含历史退市股，防幸存者偏差）。

tushare ``stock_basic`` 按 ``list_status`` 分类，本脚本目前处理三类：
  L=上市  D=退市  P=暂停上市
接口调用频率**以账户当期权限为准**（不同积分档位不同，官方也会调整），故本脚本默认
单次运行只取一个状态、重复运行逐步补齐；不要把某个具体频次写进代码或注释。

两种模式
========

**刷新模式** —— 产出候选，正式文件全程只读：

    # 旧式就地模式：**仅为兼容保留，不要用于本部署的正式仓**。它读写的是 :data:`OUT`，
    # 一个**相对当前工作目录**的 data_store/ 路径；本部署的正式仓在
    # /opt/data/quant_trade/，仓库根下并没有 data_store/。在仓库根跑它只会新建一个无关
    # 目录、从零引导一份与正式仓无关的文件，而且 rc=0、日志一切正常，看不出走错了地方。
    # 正式刷新一律走下面的显式分阶段模式 + promote。
    # 路径与默认行为与本次改动前保持兼容，但新增的 fail-closed（响应缺列、D/L 空返回、
    # 非回归）在这条路径上同样生效 —— 旧实现会照常落盘 rc=0 的场景现在会拒绝。
    NO_PROXY=* python scripts/fetch_tushare_basic.py            # 自动补一个缺失状态
    NO_PROXY=* python scripts/fetch_tushare_basic.py --status D # 指定状态

    # 显式分阶段模式：--input/--out/--status **三者必须齐给**，--out 必须是全新路径
    NO_PROXY=* python scripts/fetch_tushare_basic.py \
        --input /real/stock_basic_tushare.parquet --out /tmp/cand_D.parquet --status D
    NO_PROXY=* python scripts/fetch_tushare_basic.py \
        --input /tmp/cand_D.parquet --out /tmp/cand_L.parquet --status L
    NO_PROXY=* python scripts/fetch_tushare_basic.py \
        --input /tmp/cand_L.parquet --out /tmp/cand_P.parquet --status P

**promote 模式** —— 把验收通过的候选原子替换到正式文件（**不联网、不读 token**）：

    python scripts/fetch_tushare_basic.py --promote /tmp/cand_P.parquet \
        --target /real/stock_basic_tushare.parquet

设计要点
========
- 显式刷新模式**禁止覆盖已存在的 --out**：候选恒为全新文件。真实误用 ——"写反 --out
  把正式文件就地夷平"—— 在 preflight 当场被拦下（那时正式文件已经存在），而不是靠事后
  断言去追。注意这是 check-then-act 判定而非排他创建，残留窗口见下方登记。
- **显式刷新模式的**所有路径 / schema / 可写性校验都在 :func:`_load_token` 与
  ``import tushare`` **之前**跑完：配置错误必须在消耗账户调用额度之前暴露。
- promote 的非回归基准是**被覆盖的那一份**（``--target``），不是候选的来源文件 ——
  真正会丢数据的动作是替换，校验就必须对准它。
- 校验不通过时 ``--target`` 一个字节都不会被动。

登记延期（本轮不实现）
======================
- ``list_status=G``（转让）语义未纳入 :data:`STATUSES`，口径需单独确认后再接。
- 数据仓定位未迁到仓库既有的 ``--store-path`` / ``QUANT_DATA__STORE_PATH`` 约定，
  :data:`OUT` 仍是硬编码相对路径。
- :data:`CODE_RE` 在仓库内另有 4 份重复副本，跨文件去重待统一处理。
- **两处 check-then-act 残留窗口**（同一类问题，都只有文件锁关得掉）：

  1. :func:`promote` 的 target 侧：基准 ``tgt_df`` 读取与 ``os.replace`` 之间，若有
     **第三方并发写入 target**，其新增的码不在基准里，替换仍会丢掉它们。候选侧的同类
     窗口已由"先复制后校验 staged"关闭，但 target 侧靠重排顺序关不掉。
  2. 显式刷新的 ``--out`` 侧：:func:`preflight_refresh` 判定"该路径不存在"与
     ``to_parquet`` 实际落盘之间，若第三方在窗口内创建了该路径，会被静默覆盖
     （``to_parquet`` 不走 ``O_EXCL`` 排他创建）。

  当前刷新是人工单写者，两处都不阻断；引入 flock 属扩大范围，本轮不做。
- **列存在 ≠ 列有值**：:func:`promote` 的 :data:`FIELD_LIST` 检查查的是列名。显式刷新
  的 ``--input`` 最低要求仍是 ``ts_code``/``list_status`` 两列，若真拿一份两列的 input
  去 concat 一份 10 列的取数帧，候选的列是齐的、但 input 那批行的其余 8 列全是 NaN，
  成员检查放行。本次流程不触发（正式输入 10 列齐全，取数帧经 :func:`normalize_response`
  强制 10 列）。要堵需要非空率判据（如既有行 ``list_date`` 非空率不得较 target 下降），
  属另一个口径决策，待单独裁决，本轮不实现。
- :func:`promote` **不清洗脏码**：落盘前的 :data:`CODE_RE` 清洗只在 :func:`refresh` 里。
  外部手工构造的候选可以把脏码原样送进正式文件（刷新产出的候选恒洁净，正常流程不触发）。
  下游 ``src/data/universe.py`` 自己会再过滤一次，且下一轮刷新会清掉，故不阻断。
- :func:`promote` **不校验 ``list_status`` 的值域**：只查列在不在，不查值是否属于
  :data:`STATUSES`、是否非空。外部构造的候选可带非法状态或 NaN 进正式仓；全部下游都是
  等值判定（``== "D"`` / ``== "L"``），非法值会同时落在两边之外 —— 静默缩集，不报错。
  受控 D→L→P 链引入不了：``--status`` 受 argparse ``choices`` 约束，取数帧的状态又被
  :func:`refresh` 无条件覆写成 ``st``。另注：``groupby("list_status")`` 默认丢弃 NaN，
  故 NaN 状态在"状态分布"里**看不见**（非法字符串则看得见）。加值域闸会把
  ``list_status=G`` 的未决口径提前绑死，故与上一条一并待裁决。
- :func:`promote` **只保留权限位，不保留 owner/group**：``shutil.copymode`` 的实现只有
  ``chmod(S_IMODE)``，从不 ``chown``；staged 由 ``mkstemp`` 建出，uid/gid 取运行进程的
  euid/egid（数据仓目录无 setgid 位，gid 不继承目录）。以 root/sudo 替换一份 ``xll:xll``
  的正式文件，替换后变 ``root:root``，原属主降为 other、写权限静默丢失，而 rc=0、日志
  照打"已替换" —— 与 PR1 守的 ``copymode`` 同一失效类，但 owner 侧既无保护也无用例。
  **运行前提**：promote 必须由 xll 执行，且事前核对 target 的 uid/gid 与执行进程的
  euid/egid 一致（本部署即 ``xll:xll``）；满足时本条不触发。只核对 owner 不够 —— target
  若是 ``xll:users``，由 xll 执行仍会把 group 静默翻成 ``xll``。补 ``os.chown`` 只堵得住
  root 场景（非 root 无权改 uid，只能转成 fail-closed 报错），属另一个部署口径决策。
- :func:`_probe_writable` 的**真实失败行为无用例**：P7 把它整个 monkeypatch 掉（测的是
  :func:`_require_writable_dir` 的接线），P8 只验无残留。探针本身是好的，但将来若退化成
  ``os.access``（即 :func:`_probe_writable` 注释所警告的写法），测试不会变红。补用例需
  ``skipif(os.getuid() == 0)``，否则 root 下必假绿。
- **文档比测试严一档**：上方"所有路径 / schema / 可写性校验都在 :func:`_load_token` 与
  ``import tushare`` 之前跑完"是实现事实，但用例只钉到"任何**取数调用**之前"——
  ``fake_tushare.calls`` 只记 ``stock_basic``，不记 ``pro_api`` 与 import。真正要护的是
  账户额度，那一层由"先取数、后 preflight"这类改动会被 C1 抓住来保证，故不阻断；要与
  文档完全对齐，需给假模块的 ``pro_api`` 也加计数。
- **parquet 读失败裸抛**：``pd.read_parquet`` 对损坏文件抛的是 ``ArrowInvalid``，不是本
  文件其余失败路径统一的带说明 ``SystemExit``。安全性不受影响（target 零字节改动、staged
  已清、读 ``--input`` 发生在 ``import tushare`` 之前不烧额度、退出码非 0），仅呈现不一致。
- :func:`promote` 要求 **target 自身**列齐 :data:`FIELD_LIST`。正式文件一旦损坏成缺列，
  这条唯一带校验的落地通道就锁死了，只能手工绕过。属刻意 fail-closed，登记其后果。
  **且无用例**：删掉这行全套用例仍全绿（已变异验证）。它不只是"锁死损坏 target"——
  还兼作非回归闸的前置条件：target 缺 ``ts_code`` 时 :func:`check_no_regression` 会早退，
  丢码判据静默退化成 no-op，promote 照常覆盖。补用例需构造缺列 target，本轮不做。
- :func:`load_existing` 的 ``required=True`` 分支**不可达**：唯一调用点在 :func:`refresh`
  的旧式分支且恒传 ``required=False``；显式模式的"输入文件不存在"实际由
  :func:`preflight_refresh` 抛出。两处文案相同，故 T4 命中的是 preflight 那一处，测试
  区分不出。删死分支属扩大范围，本轮不做。
- **旧式无参模式对空 P 探针空转**（HEAD 既存行为，本轮审核确认后一并登记）：自动补状态
  用 ``have``（既有行里出现过的 ``list_status``）反推缺哪个。P 返回空不产生任何行，
  ``have`` 永远补不齐 P，于是每次无参运行都会**再取一次 P** 并原样重写 :data:`OUT`
  （内容不变、mtime 递进），每次白烧一次账户调用额度。已实测：连跑三次 → 取数
  ``['P','P','P']``、``have`` 恒 ``['D','L']``、rc 恒 0。显式模式不受影响
  （``--status`` 强制给，不走自动补）。只影响已声明"不要用于本部署"的旧式路径，故不
  阻断；治它需把"已探测过 P"记进状态而非从数据行反推，属另一个口径决策。
- 旧式模式的 ``--status`` 用法（见上方用法示例）**无用例**；行为经手工复核成立。
- 测试文件顶部 docstring 的**用例编号范围过时**：写的是"（PR1~PR5）""（R1~R5）"，实际
  已扩到 PR10 与 R5b；"（P1~P10）"那一节还容纳了 T4/T5/T6/T6b/T6c。陈述本身无误，只是
  范围没跟上后加的用例。
- 测试文件里 **PR6~PR10 排在 PR3~PR5 之前**，编号顺序与文件顺序不一致，纯可读性问题。
  上面两项均不影响下一阶段；为避免项目中途对纯可读性问题反复调整，留待项目完成后集中
  整理。

token 取值优先级：--token > .env 的 TUSHARE_TOKEN（显式读 .env，绕开被系统环境变量遮蔽的陈旧值）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

OUT = Path("data_store/stock_basic_tushare.parquet")
FIELDS = "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
FIELD_LIST = tuple(FIELDS.split(","))
STATUSES = ("L", "D", "P")
#: 允许取数返回空的状态。P=暂停上市是探针性质，市场上可以一只都没有；D/L 返回空只
#: 可能是接口异常，必须 fail-closed，否则"没取到"会被当成"取到了空"落盘。
EMPTY_OK = ("P",)

#: 合法 ts_code 形态。**本文件内唯一定义处**：落盘前的脏码清洗与
#: :func:`check_no_regression` 的脏码豁免共用它。两处各写一份正则，迟早会漂移成
#: "清洗掉的码被判成丢失"的假红。（仓库内另有 4 份副本，跨文件去重已登记延期。）
CODE_RE = r"^\d{6}\.(SH|SZ|BJ)$"


def _clean_mask(codes: pd.Series) -> pd.Series:
    """``codes`` 中符合 :data:`CODE_RE` 的位置为 True。"""
    return codes.astype(str).str.match(CODE_RE)


def _resolved(path: str | Path) -> Path:
    """展开 ``~`` 并解析符号链接/相对路径，供路径同一性判定使用。

    同一性必须比较解析后的路径：``./a.parquet`` 与 ``/abs/a.parquet``、以及指向同一
    文件的符号链接，字符串上都不相等，但它们是同一份数据。
    """
    return Path(path).expanduser().resolve()


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    """``df`` 缺 ``columns`` 任一列即 fail-closed。"""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{what} 缺列：{missing}（实际列：{list(df.columns)}）")


def _probe_writable(directory: Path) -> None:
    """在 ``directory`` 下实际试写一个临时文件并立即删除；不可写则抛 ``OSError``。

    用**真写**探测而不是解读权限位：root 下权限位不代表可写性，``os.access`` 同样会
    在这一点上撒谎。探针无论成败都不留残留。
    """
    fd, name = tempfile.mkstemp(prefix=".wtest_", dir=directory)
    os.close(fd)
    os.unlink(name)


def _require_writable_dir(directory: Path, what: str) -> None:
    """``directory`` 必须已存在且确实可写。"""
    if not directory.is_dir():
        raise SystemExit(f"{what}目录不存在：{directory}（请先手工建好；脚本不代建）")
    try:
        _probe_writable(directory)
    except OSError as exc:
        raise SystemExit(f"{what}目录不可写：{directory}（{exc}）") from exc


def normalize_response(df: pd.DataFrame | None, status: str) -> pd.DataFrame:
    """规范化 tushare 响应帧，使下游不必再区分"有列/无列/空帧"。

    - **非空响应缺 :data:`FIELDS` 任一列 → fail-closed**：列缺失意味着接口口径变了，
      静默补空会把"这一列没取到"伪装成"这一列全是空值"，一路落盘无人察觉。
    - **空响应统一补成 FIELDS 空帧**：tushare 偶尔返回连列都没有的空 DataFrame，
      不规范化的话下游 ``out["ts_code"]`` 会裸抛 ``KeyError``。
    """
    if df is None:
        df = pd.DataFrame()
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in FIELD_LIST})
    _require_columns(df, FIELD_LIST, f"tushare {status} 响应")
    return df


def load_existing(path: Path, *, required: bool) -> pd.DataFrame:
    """读取既有 parquet；``required`` 时缺文件即 fail-closed。

    ``required`` 由"调用者是否显式给了路径"决定，不是由"文件在不在"决定：显式给
    路径意味着执行者**明确知道**要读哪一份既有数据，此时文件不存在是配置错误。
    旧实现在这里静默回退空表，后果是落盘一份只含单个 list_status 的残缺文件、
    退出码仍为 0 —— 没有任何断言拦得住，只能靠人眼看分布输出。
    """
    if path.exists():
        return pd.read_parquet(path)
    if required:
        raise SystemExit(
            f"输入文件不存在：{path}\n"
            "显式指定 --input/--out 时不按空表处理（那会落盘残缺文件）。"
            "首次引导请用不带路径参数的调用。"
        )
    return pd.DataFrame()


def check_no_regression(existing: pd.DataFrame, candidate: pd.DataFrame) -> None:
    """落盘/替换前的非回归校验：``candidate`` 不得丢失 ``existing`` 的任何**干净**代码。

    分两段，**前一段无条件执行**：

    1. ``candidate`` 自身必须成形（有 ``ts_code``、无重复码）。这与 ``existing`` 空不空
       毫无关系 —— 旧实现把它挂在 ``existing`` 非空的前提下，首次引导路径上重复码
       直接放行。
    2. 有基准（``existing`` 非空且成形）时才谈得上"丢码"。

    **脏码豁免**：``existing`` 里不符合 :data:`CODE_RE` 的代码会被落盘前的清洗步骤
    合法剔除，故比较只在干净码上做，否则一次正当清洗就会被判成回归。

    **本函数测不出"输入根本没读进来"** —— 那条路径上 ``existing`` 为空、前件为空，
    任何超集判据都恒真；治它的是 :func:`preflight_refresh`（``--input`` 不存在即
    fail-closed）。本函数在刷新流程里防的是**将来**有人改动合并语义时的静默丢码；
    在 :func:`promote` 里则是**当下**带电的那道闸 —— 那里 ``existing`` 是即将被覆盖
    的正式文件，判据随时可能真的不成立。
    """
    if "ts_code" not in candidate.columns:
        raise SystemExit("非回归校验失败：candidate 缺 ts_code 列")
    dup = int(candidate["ts_code"].astype(str).duplicated().sum())
    if dup:
        raise SystemExit(f"非回归校验失败：candidate 含 {dup} 个重复 ts_code")

    if existing is None or existing.empty or "ts_code" not in existing.columns:
        return

    kept = set(existing.loc[_clean_mask(existing["ts_code"]), "ts_code"].astype(str))
    have = set(candidate["ts_code"].astype(str))
    missing = sorted(kept - have)
    if missing:
        raise SystemExit(
            f"非回归校验失败：candidate 丢失 {len(missing)} 个既有代码"
            f"（前 10：{missing[:10]}）"
        )


def preflight_refresh(
    input_arg: str | None, out_arg: str | None, status: str | None
) -> tuple[Path, Path]:
    """显式刷新模式的全部**离线**前置校验，返回解析后的 ``(输入, 输出)``。

    必须在 :func:`_load_token` / ``import tushare`` / 任何取数调用**之前**跑完：每次
    取数都消耗账户额度，配置错误应当在花掉额度之前暴露，而不是在 ``to_parquet``
    那一刻才炸。
    """
    if not (input_arg and out_arg and status):
        raise SystemExit(
            "显式刷新模式必须同时给出 --input、--out、--status（本次给出："
            f"--input={input_arg!r} --out={out_arg!r} --status={status!r}）。\n"
            "不带任何路径参数则走旧式就地模式。"
        )

    in_path = _resolved(input_arg)
    out_path = _resolved(out_arg)

    if not in_path.is_file():
        raise SystemExit(f"输入文件不存在：{in_path}")
    # 同一性判定放在"已存在"之前：否则同一路径永远只报"输出已存在"，解析逻辑失守也
    # 看不出来。
    if out_path == in_path:
        raise SystemExit(
            f"--out 与 --input 解析后是同一个文件：{out_path}\n"
            "候选必须是全新路径，正式输入全程只读。"
        )
    if out_path.exists():
        raise SystemExit(
            f"输出文件已存在：{out_path}\n"
            "候选必须是全新路径（禁止覆盖既有文件）；如需重跑该阶段请先手工删除它。"
        )
    _require_writable_dir(out_path.parent, "输出")
    return in_path, out_path


def promote(candidate_arg: str | None, target_arg: str | None) -> int:
    """把验收通过的候选**原子替换**到正式文件。

    这是整条流程里唯一真正会覆盖正式数据的动作，所以校验基准是 ``--target`` 自己：
    比较"即将被覆盖的那一份"与"要写进去的那一份"。拿候选的来源文件当基准是对不准
    的 —— 来源文件正常，不代表 target 不会被夷平。

    **全程离线**：不读 token、不 ``import tushare``。

    schema 要求是 :data:`FIELD_LIST` 的**成员完整性**：允许额外列、允许任意列序，但不
    允许缺任一标准列 —— 只查 ``ts_code``/``list_status`` 两列的话，一份丢了
    ``list_date``/``delist_date`` 的候选会照常通过并永久覆盖正式文件，而下游
    （``src/data/universe.py`` 用的是 ``if "name" in d.columns else ""`` 这类写法）
    会把缺列静默降级成空值，两头都不报错。**不要改成列序相等判定**：正式文件盘上的
    列序是 area/industry 在末尾，与 :data:`FIELD_LIST` 的书写顺序并不一致。

    落地顺序是**先复制、再校验、后替换**：候选先落到 ``target`` 同目录的 staged 临时
    文件，随后**只读 staged 的实际字节**做校验，通过才 :func:`os.replace`。

    顺序不能反。先校验 ``cand`` 再复制的话，两次读取之间候选一旦被改写（另一个终端重
    跑刷新阶段、cron、``/tmp`` 清理都够了），落进正式文件的就是一份从未被校验的字节，
    而成功日志还在打印第一次读到的行数 —— 事后连异常都看不出来。校验 staged 才能保证
    "校验的"和"写进去的"是同一份内容。

    多一次复制同时保证 replace 两端同文件系统 —— 跨盘的 ``os.replace`` 会直接抛
    ``OSError``，而候选在 ``/tmp``、正式文件在数据盘恰恰是常态。

    任何失败（校验不过或 I/O 出错）都清理 staged，``target`` 一个字节都不动。
    """
    if not (candidate_arg and target_arg):
        raise SystemExit(
            "promote 模式必须同时给出 --promote（候选）与 --target（正式文件）。"
        )

    cand = _resolved(candidate_arg)
    tgt = _resolved(target_arg)
    if cand == tgt:
        raise SystemExit(f"--promote 与 --target 解析后是同一个文件：{tgt}")
    if not cand.is_file():
        raise SystemExit(f"候选文件不存在：{cand}")
    if not tgt.is_file():
        raise SystemExit(
            f"正式文件不存在：{tgt}\n"
            "promote 只做替换；首次引导请走刷新模式产出正式文件。"
        )

    tgt_df = pd.read_parquet(tgt)
    _require_columns(tgt_df, FIELD_LIST, f"正式文件 {tgt}")

    staged: Path | None = None
    try:
        try:
            fd, staged_name = tempfile.mkstemp(prefix=".promote_", dir=tgt.parent)
            os.close(fd)
            staged = Path(staged_name)
            shutil.copyfile(cand, staged)
        except OSError as exc:
            raise SystemExit(f"候选落地失败（{tgt} 未被改动）：{exc}") from exc

        # 只读 staged 的实际字节：校验对象必须就是即将成为正式文件的那一份。
        cand_df = pd.read_parquet(staged)
        _require_columns(cand_df, FIELD_LIST, f"候选 {cand}")
        check_no_regression(tgt_df, cand_df)

        try:
            shutil.copymode(tgt, staged)
            os.replace(staged, tgt)
        except OSError as exc:
            raise SystemExit(f"替换失败（{tgt} 未被改动）：{exc}") from exc
        staged = None
    finally:
        # 校验不过抛的 SystemExit 原样穿出（两个 except 只各自兜住自己那段 I/O），
        # 但 staged 无论走哪条失败路径都要清掉。
        if staged is not None:
            staged.unlink(missing_ok=True)

    print(f"已替换 {tgt} | 候选 {cand} | 合计 {len(cand_df)} | 状态分布:")
    print(cand_df.groupby("list_status").size().to_string())
    return 0


def _load_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token.strip()
    # 显式读 .env，避免系统环境里陈旧的 TUSHARE_TOKEN 遮蔽真实值
    env_path = Path(__file__).resolve().parent.parent / ".env"
    tok = (dotenv_values(env_path).get("TUSHARE_TOKEN") or "").strip()
    if not tok:
        tok = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not tok:
        raise SystemExit("未找到 TUSHARE_TOKEN（.env 或环境变量）")
    return tok


def refresh(args: argparse.Namespace) -> int:
    """刷新模式：取一个状态并产出候选（显式）或就地更新（旧式）。"""
    explicit = args.out is not None or args.input_path is not None
    if explicit:
        in_path, out_path = preflight_refresh(args.input_path, args.out, args.status)
        existing = pd.read_parquet(in_path)
        _require_columns(existing, ("ts_code", "list_status"), f"输入文件 {in_path}")
    else:
        in_path = out_path = OUT
        existing = load_existing(in_path, required=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)  # 旧式模式保持自动建仓

    have = set(existing["list_status"].unique()) if "list_status" in existing else set()

    if args.status:
        targets = [args.status]
    else:
        # 只有旧式模式走得到这里：显式模式的 preflight 已强制给 --status
        targets = [s for s in STATUSES if s not in have]
        if not targets:
            print(f"已齐全（{sorted(have)}），无需补取。")
            return 0
        targets = targets[:1]  # 单次只取一个状态；频率限制以账户当期权限为准

    import tushare as ts

    pro = ts.pro_api(_load_token(args.token))

    frames = [existing] if not existing.empty else []
    for st in targets:
        df = normalize_response(pro.stock_basic(exchange="", list_status=st, fields=FIELDS), st)
        if df.empty:
            if st not in EMPTY_OK:
                raise SystemExit(
                    f"tushare 返回 0 行（list_status={st}）：该状态不可能为空，判为取数"
                    f"失败，不产出任何文件。"
                )
            print(f"{st}: 0 只（探针状态允许为空，候选与输入等价）")
            continue
        df["list_status"] = st
        frames.append(df)
        print(f"{st}: {len(df)} 只"
              + (f" | delist {df['delist_date'].min()}~{df['delist_date'].max()}" if st == "D" else ""))

    # 输入零行 + 探针状态返回空 → 候选就是一份 0 行的标准 schema parquet，rc=0：这是
    # P 探针的合法结果，不是失败。``pd.concat([])`` 会抛 ValueError，故显式兜住。
    out = (pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="last")
           if frames else normalize_response(None, args.status))
    # 过滤脏码（tushare 偶发非标准 ts_code，如 T600018.SH）；统一为 XXXXXX.SH/SZ/BJ
    bad = ~_clean_mask(out["ts_code"])
    if bad.any():
        print(f"过滤脏码 {int(bad.sum())} 条: {out.loc[bad, 'ts_code'].tolist()}")
        out = out[~bad].reset_index(drop=True)
    check_no_regression(existing, out)
    out.to_parquet(out_path, index=False)
    print(f"已保存 {out_path} | 合计 {len(out)} | 状态分布:")
    print(out.groupby("list_status").size().to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="拉取 tushare stock_basic 全量(含退市)；或把候选 promote 到正式文件")
    ap.add_argument("--status", choices=STATUSES, default=None,
                    help="指定 list_status；旧式模式缺省=自动补一个尚缺的状态，显式模式必给")
    ap.add_argument("--token", default=None, help="显式 token（覆盖 .env）")
    ap.add_argument("--out", default=None,
                    help=f"输出候选 parquet 路径（必须不存在）；缺省 {OUT} 就地模式")
    ap.add_argument("--input", dest="input_path", default=None,
                    help="输入 parquet 路径（必须存在）；与 --out/--status 三者齐给")
    ap.add_argument("--promote", metavar="CANDIDATE", default=None,
                    help="promote 模式：把该候选原子替换到 --target（不联网、不读 token）")
    ap.add_argument("--target", default=None,
                    help="promote 模式的正式文件（必须已存在），校验通过才被替换")
    args = ap.parse_args()

    if args.promote is not None or args.target is not None:
        mixed = [name for name, val in (("--input", args.input_path), ("--out", args.out),
                                        ("--status", args.status), ("--token", args.token))
                 if val is not None]
        if mixed:
            raise SystemExit(f"promote 模式不接受刷新参数：{mixed}")
        return promote(args.promote, args.target)

    return refresh(args)


if __name__ == "__main__":
    raise SystemExit(main())
