"""HI67 slice 2b: engines/llamacpp/patches/1223_hi67_machine_readable_correctness_metrics/patch.py
applies cleanly (stacked on top of 1222, its REQUIRES dependency) and
idempotently to the real vendored test-backend-ops.cpp."""

import importlib.util
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "engines" / "llamacpp" / "patches" / Path(filename).stem / "patch.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P1222 = _load("hi67_p1222", "1222_hi67_deterministic_test_backend_ops_seed.py")
_P1223 = _load("hi67_p1223", "1223_hi67_machine_readable_correctness_metrics.py")


def _apply_to_copy(tmp_path: Path) -> Path:
    vendor = paths.llama_root()
    target = tmp_path / "tests" / "test-backend-ops.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Pristine pinned bytes from git: the working vendor tree may already carry applied patches.
    target.write_bytes(subprocess.check_output(["git", "-C", str(vendor), "show", "HEAD:tests/test-backend-ops.cpp"]))
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
    # Emitted after the existing pass/fail block (5992f183), so the err line and
    # the check stay untouched for patches anchoring on them; all three present in order.
    err_idx = text.index("double err = ud->tc->err(")
    metric_idx = text.index("BIGCHERRY_CORRECTNESS_METRIC")
    threshold_check_idx = text.index('printf("[%s] ERR = %.9f > %.9f "')
    assert err_idx < threshold_check_idx < metric_idx


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
