"""C01 diagnostic native LTX builder patch: applicability and fail-closed tests.

A real-source run requires an explicit upstream source path. Only copies the
ltx2 translation unit into a separate work-root directory, never the source.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from bigcherry.patch.apply import apply_all  # noqa: E402

PKG = ROOT / "engines/radiance/patches/rad0060_ltx2_tp3_linear_declare"
CPP = Path("arch/ltx2/ltx2.cpp")
HDR = Path("arch/ltx2/bc_ltx_tp3_c01.h")


def package():
    spec = importlib.util.spec_from_file_location("ltx_c01_patch", PKG / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_upstream() -> str:
    # Exact anchor contract; only the tested functions are present.
    return """#include <arch/rad_fp8.h>
struct Lin {
    int64_t    N = 0, K = 0;
    int declare(int group, const Act& in, rad_buf y, int64_t max_rows, bool small = false) {
        N = n_out;
        K = n_in;
        const char* wn = nm.f("%s.weight", base);
            bias = decl_w(b, bn, RAD_BF16, { N }, access, RAD_SHARD_NONE, grp_layer(group));
            if (!bias) return RAD_E_INVAL;
        RAD_ARCH_TRY(l.declare(b, nm, g, base, N, K, RAD_SHARD_NONE, grp_layer(group), wsrc, in.a, y,
                               max_rows, parts));
    }
};
"""


def make_stage(root: Path, text: str) -> None:
    path = root / CPP
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def patch_copy(root: Path) -> dict[str, int]:
    patched = apply_all(list(package().PATCHES), root)
    errors = [(r.path, x.edit_id, x.detail) for r in patched for x in r.failed]
    if errors:
        raise RuntimeError("C01 anchored patch did not apply: " + repr(errors))
    assert (root / HDR).is_file()
    src = (root / CPP).read_text()
    for marker in (
        "BC_LTX3_C01_INCLUDE", "BC_LTX3_C01_ARGUMENT",
        "BC_LTX3_C01_DEFERRED_STATE",
        "BC_LTX3_C01_PREFLIGHT", "BC_LTX3_C01_BIAS", "BC_LTX3_C01_LINEAR",
    ):
        assert src.count(marker) == 1, marker
    assert 'if (ctx->world_size > 1)' in src or "struct Lin {" in src
    again = apply_all(list(package().PATCHES), root)
    assert all(not r.changed for r in again), "patch is not idempotent"
    return {
        "applied": sum(x.status == "applied" for r in patched for x in r.results),
        "created_headers": int((root / HDR).is_file()),
        "idempotent": 1,
    }


def test_c01_patch_fixture_applies_twice(tmp_path: Path) -> None:
    make_stage(tmp_path, fake_upstream())
    result = patch_copy(tmp_path)
    assert result == {"applied": 7, "created_headers": 1, "idempotent": 1}
    src = (tmp_path / CPP).read_text()
    assert "RAD_SHARD_ROW" in src and "RAD_SHARD_COL" in src
    assert "bias-once reduction not yet implemented" in src
    assert "RAD_SHARD_NONE" in src  # default TP1 branch retained


def test_c01_patch_fixture_refuses_wrong_builder(tmp_path: Path) -> None:
    original = fake_upstream().replace("max_rows, parts));", "max_rows, replaced));")
    make_stage(tmp_path, original)
    results = apply_all(list(package().PATCHES), tmp_path)
    assert any(not r.ok for r in results)
    assert (tmp_path / CPP).read_text() == original
    assert not (tmp_path / HDR).exists()


def test_c01_real_source_opt_in(tmp_path: Path) -> None:
    import pytest
    source = os.getenv("BC_LTX2_FROZEN_SOURCE")
    if not source:
        pytest.skip("requires frozen candidate BC_LTX2_FROZEN_SOURCE")
    src = Path(source) / CPP
    assert src.is_file()
    make_stage(tmp_path, src.read_text())
    assert patch_copy(tmp_path)["applied"] == 7


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_ltx_tp3_builder_patch.py /upstream/source /staged/output")
    upstream, dest = Path(sys.argv[1]), Path(sys.argv[2])
    if not (upstream / CPP).is_file():
        raise SystemExit("missing upstream arch/ltx2/ltx2.cpp")
    target = dest / CPP
    if target.exists() or (dest / HDR).exists():
        raise SystemExit("stage destination must not already contain LTX patch targets")
    dest.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(upstream / CPP, target)
    print(json.dumps(patch_copy(dest), sort_keys=True))
