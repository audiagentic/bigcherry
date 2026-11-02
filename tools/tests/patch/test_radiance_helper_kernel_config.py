"""Managed application and drift checks for RSC06's kernel configuration ABI."""
import importlib.util
from pathlib import Path

from bigcherry.patch.apply import apply_all

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "engines/radiance/patches/rad0010_helper_kernel_config/patch.py"


def load():
    spec = importlib.util.spec_from_file_location("helper_kernel_config", PACKAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def seed(root):
    libr4d = root / "libr4d"
    libr4d.mkdir()
    (libr4d / "r4d_rows.cpp").write_text('#include "r4d_plugin.h"\n')
    (libr4d / "r4d_args.h").write_text(
        '#include "rad_plugin.h"\n'
        'static inline int r4d_trace_on(void) {\n'
        '    static int on = -1;\n'
        '    return on;\n'
        '}\n'
    )


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_helper_config_patch_is_idempotent(tmp_path):
    seed(tmp_path)
    first = apply_all(load(), tmp_path)
    assert all(r.ok for r in first), first
    assert all(e.status == "applied" for r in first for e in r.results)
    before = snapshot(tmp_path)
    second = apply_all(load(), tmp_path)
    assert all(r.ok for r in second), second
    assert all(e.status == "already-applied" for r in second for e in r.results)
    assert snapshot(tmp_path) == before


def test_helper_config_patch_refuses_drift_atomically(tmp_path):
    seed(tmp_path)
    target = tmp_path / "libr4d/r4d_args.h"
    target.write_text(target.read_text().replace("r4d_trace_on(void)", "r4d_trace_on(int changed)"))
    before = snapshot(tmp_path)
    results = apply_all(load(), tmp_path)
    assert any(not r.ok for r in results)
    assert snapshot(tmp_path) == before
