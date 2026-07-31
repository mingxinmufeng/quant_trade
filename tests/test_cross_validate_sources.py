"""cross_validate_sources 的逐代码退市合同。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cross_validate_sources.py"
SPEC = importlib.util.spec_from_file_location("cross_validate_sources", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _install_akshare(monkeypatch, *, sh=(), sz=()):
    tables = {
        "stock_info_sh_delist": pd.DataFrame({"公司代码": list(sh)}),
        "stock_info_sz_delist": pd.DataFrame({"公司代码": list(sz)}),
    }
    fake = SimpleNamespace(
        stock_info_sh_delist=lambda: tables["stock_info_sh_delist"],
        stock_info_sz_delist=lambda: tables["stock_info_sz_delist"],
    )
    monkeypatch.setitem(__import__("sys").modules, "akshare", fake)


def test_check_delist_is_per_code_and_ignores_unrelated_market_gap(
    tmp_path, monkeypatch
):
    pd.DataFrame(
        {
            "ts_code": ["000418.SZ", "600999.SH"],
            "list_status": ["D", "D"],
        }
    ).to_parquet(tmp_path / "stock_basic_tushare.parquet", index=False)
    _install_akshare(monkeypatch, sz=("000418",))
    report = MODULE.Report()

    MODULE.check_delist(tmp_path, report, ["000418.SZ"])

    assert report.issues == 0
    assert report.unavailable_count == 0
    assert any(
        "code=000418.SZ status=pass" in line for line in report.lines
    )


def test_check_delist_marks_empty_external_source_unavailable(
    tmp_path, monkeypatch
):
    pd.DataFrame(
        {"ts_code": ["000418.SZ"], "list_status": ["D"]}
    ).to_parquet(tmp_path / "stock_basic_tushare.parquet", index=False)
    _install_akshare(monkeypatch)
    report = MODULE.Report()

    MODULE.check_delist(tmp_path, report, ["000418.SZ"])

    assert report.issues == 0
    assert report.unavailable_count == 1


def test_check_delist_fails_when_requested_code_lacks_second_source(
    tmp_path, monkeypatch
):
    pd.DataFrame(
        {"ts_code": ["000418.SZ"], "list_status": ["D"]}
    ).to_parquet(tmp_path / "stock_basic_tushare.parquet", index=False)
    _install_akshare(monkeypatch, sh=("600001",))
    report = MODULE.Report()

    MODULE.check_delist(tmp_path, report, ["000418.SZ"])

    assert report.issues == 1
    assert report.unavailable_count == 0
    assert any(
        "code=000418.SZ status=fail" in line for line in report.lines
    )


def test_check_delist_marks_missing_local_input_unavailable(tmp_path):
    report = MODULE.Report()

    MODULE.check_delist(tmp_path, report, ["000418.SZ"])

    assert report.issues == 0
    assert report.unavailable_count == 1


def test_check_delist_marks_unsupported_exchange_unavailable(tmp_path):
    pd.DataFrame(
        {"ts_code": ["430047.BJ"], "list_status": ["D"]}
    ).to_parquet(tmp_path / "stock_basic_tushare.parquet", index=False)
    report = MODULE.Report()

    MODULE.check_delist(tmp_path, report, ["430047.BJ"])

    assert report.issues == 0
    assert report.unavailable_count == 1
    assert any(
        "code=430047.BJ status=unavailable" in line
        for line in report.lines
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("pass", 0), ("fail", 1), ("unavailable", 2)],
)
def test_main_wires_explicit_code_store_and_exit_contract(
    tmp_path, monkeypatch, mode, expected
):
    store = tmp_path / f"basic-{mode}"
    store.mkdir()
    if mode != "unavailable":
        pd.DataFrame(
            {"ts_code": ["000418.SZ"], "list_status": ["D"]}
        ).to_parquet(store / "stock_basic_tushare.parquet", index=False)
    _install_akshare(
        monkeypatch,
        sz=("000418",) if mode == "pass" else (),
        sh=("600001",) if mode == "fail" else (),
    )
    monkeypatch.setattr(
        MODULE,
        "load_config",
        lambda: SimpleNamespace(
            data=SimpleNamespace(
                store_path=tmp_path / "wrong",
                tdx_path="",
            )
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "TusharePool",
        lambda: SimpleNamespace(available=False),
    )
    monkeypatch.setattr(MODULE, "GbbqStore", lambda **_kwargs: object())
    monkeypatch.setattr(MODULE, "ProfileStore", lambda **_kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_validate_sources.py",
            "--codes",
            "000418.SZ",
            "--checks",
            "delist",
            "--basic-info-store",
            str(store),
            "--out",
            str(tmp_path / f"{mode}.md"),
        ],
    )

    assert MODULE.main() == expected
