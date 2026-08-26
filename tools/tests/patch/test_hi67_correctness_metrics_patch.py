"""HI67 slice 2b: patches/1223_hi67_machine_readable_correctness_metrics.py
applies cleanly (stacked on top of 1222, its REQUIRES dependency) and
idempotently to the real vendored test-backend-ops.cpp."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "patches" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P1222 = _load("hi67_p1222", "1222_hi67_deterministic_test_backend_ops_seed.py")
_P1223 = _load("hi67_p1223", "1223_hi67_machine_readable_correctness_metrics.py")


def _apply_to_copy(tmp_path: Path) -> Path:
    vendor = ROOT / "vendor" / "llama.cpp"
    target = tmp_path / "tests" / "test-backend-ops.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vendor / "tests" / "test-backend-ops.cpp", target)
    return target


def test_declares_requires_on_1222():
    # 1223 uses bigcherry_deterministic_mode(), defined by 1222's edits --
    # a real dependency, not just apply-order convenience. patchset.py's
    # resolver enforces this via the module-level REQUIRES constant.
    assert _P1223.REQUIRES == ("1222_hi67_deterministic_test_backend_ops_seed",)


def test_applies_cleanly_stacked_on_1222(tmp_path):
    target = _apply_to_copy(tmp_path)
    results = apply_all([_P1222.PATCH, _P1223.PATCH], tmp_path)
    assert all(result.ok for result in results), [
        (r.edit_id, r.status, r.detail) for result in results for r in result.results
    ]
    text = target.read_text(encoding="utf-8")
    assert "BIGCHERRY_CORRECTNESS_METRIC" in text
    # Placed after err is computed, before the existing pass/fail check --
    # both must still be present and in that order.
    err_idx = text.index("double err = ud->tc->err(")
    metric_idx = text.index("BIGCHERRY_CORRECTNESS_METRIC")
    threshold_check_idx = text.index('printf("[%s] ERR = %.9f > %.9f "')
    assert err_idx < metric_idx < threshold_check_idx


def test_apply_is_idempotent(tmp_path):
    _apply_to_copy(tmp_path)
    first = apply_all([_P1222.PATCH, _P1223.PATCH], tmp_path)
    assert all(result.ok for result in first)
    second = apply_all([_P1222.PATCH, _P1223.PATCH], tmp_path)
    assert all(result.ok for result in second)
    assert not any(result.changed for result in second)


def test_metric_line_gated_behind_deterministic_mode():
    for edit in _P1223.PATCH.edits:
        if edit.id == "hi67-correctness-metric-line":
            assert "if (bigcherry_deterministic_mode())" in edit.text


def test_metric_line_reports_nmse_max_abs_and_threshold():
    for edit in _P1223.PATCH.edits:
        if edit.id == "hi67-correctness-metric-line":
            assert "err=%.17g" in edit.text
            assert "max_abs=%.17g" in edit.text
            assert "threshold=%.17g" in edit.text
            assert "ud->tc->max_err(ud->backend1)" in edit.text
