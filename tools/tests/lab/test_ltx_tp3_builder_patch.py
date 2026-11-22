"""C01 diagnostic native LTX builder patch: applicability and fail-closed tests.

A real-source run requires an explicit upstream source path. Only copies the
ltx2 translation unit into a separate work-root directory, never the source.
"""
from __future__ import annotations

import importlib.util
from dataclasses import replace
import json
import os
import re
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
    /* Input rows [ri, ri + rows) to output rows [ro, ro + rows). */
    void step() {
        if (op_b) RAD_ISSUE_N(c, op_b, rows, out, RAD_W(bias), out);
    }
};
/* C05-C06 helpers insert before the existing architecture attention. */
struct Attn {
};
/* C06 original self-attention source anchors live only in real Radiance.
 * This synthetic fixture is for C01-C05 patch applicability; it cannot
 * stand in for the real attention builder. */
"""


def make_stage(root: Path, text: str) -> None:
    path = root / CPP
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def patches(*, full: bool):
    definitions = list(package().PATCHES)
    if not full:
        # The tiny fixture exercises C01-C05's generic linear anchors.
        # C06 anchors the actual LTX attention source; copying it here
        # would create a second fake attention architecture.
        definitions[1] = replace(
            definitions[1],
            edits=tuple(e for e in definitions[1].edits
                        if not e.id.startswith(("ltx-c06-", "ltx03-"))),
        )
    return definitions


def patch_copy(root: Path, *, full: bool = True) -> dict[str, int]:
    selected = patches(full=full)
    patched = apply_all(selected, root)
    errors = [(r.path, x.edit_id, x.detail) for r in patched for x in r.failed]
    if errors:
        raise RuntimeError("C01 anchored patch did not apply: " + repr(errors))
    assert (root / HDR).is_file()
    src = (root / CPP).read_text()
    for marker in (
        "BC_LTX3_C01_INCLUDE", "BC_LTX3_C01_ARGUMENT",
        "BC_LTX3_C01_DEFERRED_STATE", "BC_LTX3_C04_REDUCE_STATE",
        "BC_LTX3_C04_FINALIZE", "BC_LTX3_C04_ISSUE",
        "BC_LTX3_C05_GLOBAL_RMS",
        "BC_LTX3_C01_PREFLIGHT", "BC_LTX3_C01_BIAS", "BC_LTX3_C01_LINEAR",
    ):
        assert len(re.findall(r"\b" + re.escape(marker) + r"\b", src)) == 1, marker
    assert 'if (ctx->world_size > 1)' in src or "struct Lin {" in src
    if full:
        for marker in (
            "BC_LTX3_C06_ATTN_STATE", "BC_LTX3_C06_ATTN_GEOMETRY",
            "BC_LTX3_C06_QKV", "BC_LTX3_C06_ATTN_OUTPUT",
            "BC_LTX3_C06_ATTN_NORM", "BC_LTX3_C06_ATTN_ROPE",
            "BC_LTX3_C06_GATE", "BC_LTX3_C06_RETURN",
            "BC_LTX3_C06_ATTN_CORE", "BC_LTX3_LTX03_CROSS_KV", "BC_LTX3_C06_GATE_ISSUE",
            "BC_LTX3_C06_STG_GATE",
        ):
            assert len(re.findall(r"\b" + re.escape(marker) + r"\b", src)) == 1, marker
    again = apply_all(selected, root)
    assert all(not r.changed for r in again), "patch is not idempotent"
    return {
        "applied": sum(x.status == "applied" for r in patched for x in r.results),
        "created_headers": int((root / HDR).is_file()),
        "idempotent": 1,
    }


def test_c01_patch_fixture_applies_twice(tmp_path: Path) -> None:
    make_stage(tmp_path, fake_upstream())
    result = patch_copy(tmp_path, full=False)
    assert result == {"applied": 10, "created_headers": 1, "idempotent": 1}
    src = (tmp_path / CPP).read_text()
    assert "RAD_SHARD_ROW" in src and "RAD_SHARD_COL" in src
    assert "explicit reduction ownership" in src
    assert "RAD_SHARD_NONE" in src  # default TP1 branch retained


def test_c01_patch_fixture_refuses_wrong_builder(tmp_path: Path) -> None:
    original = fake_upstream().replace("max_rows, parts));", "max_rows, replaced));")
    make_stage(tmp_path, original)
    results = apply_all(patches(full=False), tmp_path)
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
    assert patch_copy(tmp_path)["applied"] == 23


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


def test_c06_coordinate_allocation_unwind_contract() -> None:
    """Static fault-path guard: never orphan a successful GPU allocation before the next may fail."""
    source = (ROOT / "tools/lab/radiance/ltx-tp3/native-c01-gate/ltx_c02_lin_declare_test.cpp").read_text()
    contract = source.split("static void verify_c06_load_contract", 1)[1].split(
        "static void run_self_attn", 1)[0]
    assert "issued->owned_device.reserve(program.weights.size()+2)" in contract
    allocation = source.split("r.device_cu=rad_dev_alloc", 1)[1].split(
        "need(rad_memcpy_async(r.device_cu", 1)[0]

    def owns_every_completed_allocation(block: str) -> bool:
        locations = [block.find(part) for part in (
            "need(r.device_cu,", "r.owned_device.push_back(r.device_cu);",
            "r.device_pos=rad_dev_alloc", "need(r.device_pos,",
            "r.owned_device.push_back(r.device_pos);",
        )]
        return all(at >= 0 for at in locations) and locations == sorted(locations)

    assert owns_every_completed_allocation(allocation)
    # These mutations used to leak the cu-seqlens buffer if position allocation refused.
    assert not owns_every_completed_allocation(
        allocation.replace("r.owned_device.push_back(r.device_cu);", ""))
    assert not owns_every_completed_allocation(
        allocation.replace("r.owned_device.push_back(r.device_pos);", ""))
    assert not owns_every_completed_allocation(
        allocation.replace("r.device_pos=rad_dev_alloc", "r.owned_device.push_back(r.device_pos);"))
