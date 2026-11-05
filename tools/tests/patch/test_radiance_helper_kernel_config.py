"""Managed application and drift checks for RSC06's kernel configuration ABI."""
import importlib.util
from pathlib import Path

from bigcherry.patch.apply import apply_all

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "engines/radiance/patches/rad0012_helper_kernel_config/patch.py"


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


def _shared_library(tmp_path, name, source):
    """Build a shared library from `source` against the patched tree, refusing undefined strong symbols (-z defs)."""
    import shutil
    import subprocess

    import pytest

    cxx = shutil.which("g++-14") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    src = tmp_path / f"{name}.cpp"
    src.write_text(source)
    out = tmp_path / f"lib{name}.so"
    subprocess.run([cxx, "-shared", "-fPIC", "-Wl,-z,defs", f"-I{tmp_path / 'libr4d'}", str(src), "-o", str(out)],
                   check=True, capture_output=True, text=True)
    return out


def test_library_without_the_row_table_loads_and_keeps_engine_behaviour(tmp_path):
    """The defect a real build hit: spechub includes r4d_args.h but not r4d_rows.cpp, so a strong reference to
    bc_r4d_helper_config left it with an undefined symbol and the engine refused to load the plugin."""
    import ctypes

    seed(tmp_path)
    (tmp_path / "libr4d" / "rad_plugin.h").write_text("")
    assert all(r.ok for r in apply_all(load(), tmp_path))
    lib = _shared_library(tmp_path, "nodef", '#include "r4d_args.h"\nextern "C" int trace(void) { return r4d_trace_on(); }\n')
    assert ctypes.CDLL(str(lib)).trace() == -1   # unconfigured: the engine's own value


def test_library_with_the_row_table_uses_the_installed_config(tmp_path):
    import ctypes

    seed(tmp_path)
    (tmp_path / "libr4d" / "rad_plugin.h").write_text("")
    assert all(r.ok for r in apply_all(load(), tmp_path))
    lib = ctypes.CDLL(str(_shared_library(tmp_path, "withdef", (
        '#define BC_R4D_HELPER_CONFIG_IMPLEMENTATION\n#include "bc_helper_config.h"\n'
        '#undef BC_R4D_HELPER_CONFIG_IMPLEMENTATION\n#include "r4d_args.h"\n'
        'extern "C" int trace(void) { return r4d_trace_on(); }\n'
        'extern "C" int install(unsigned t) { BcR4dHelperConfig c{sizeof c, 1, 1, 1, 0, t}; return bc_r4d_configure_helper(&c); }\n'))))
    assert lib.trace() == -1
    assert lib.install(1) == 0
    assert lib.trace() == 1
