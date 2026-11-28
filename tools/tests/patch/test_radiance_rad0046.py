"""Card-free replay check for the retained E4M3 rows."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

from unittest import SkipTest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

from bigcherry.patcher import apply_all

PACKAGE = REPO / "engines/radiance/patches/rad0046_retained_moe_rows/patch.py"
RECIPE = REPO / "engines/radiance/recipes.toml"


def _patches():
    spec = importlib.util.spec_from_file_location("rad0046_retained_moe_rows", PACKAGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PATCHES


def test_retained_rows_apply_and_replay_on_reviewed_abi43_source():
    checkout = os.environ.get("RADIANCE_SRC")
    if not checkout or not Path(checkout).is_dir():
        raise SkipTest("set RADIANCE_SRC to the pinned upstream source checkout")

    candidate_commit = tomllib.loads(RECIPE.read_text(encoding="utf-8"))["pinned"]
    resolved = subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "--verify", f"{candidate_commit}^{{commit}}"], text=True,
    ).strip()
    assert resolved == candidate_commit

    patches = _patches()
    files = [edit.path for edit in patches]
    created = {patch.path for patch in patches if patch.create}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temp:
        root = Path(temp)
        for rel in files:
            if rel in created:
                continue
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(
                ["git", "-C", checkout, "show", f"{candidate_commit}:{rel}"],
            ))

        unpatched_rows = (root / "libr4d/r4d_rows.cpp").read_text()
        assert '"had_quant_act_fp8", "had_quant_act_fp8", "quant"' not in unpatched_rows
        assert '"moe_gemm_fp8a8", "moe_gemm_q", "moe"' not in unpatched_rows

        first = apply_all(patches, root)
        assert first and all(result.ok for result in first), first
        assert all(edit.status == "applied" for result in first for edit in result.results), first

        rows = (root / "libr4d/r4d_rows.cpp").read_text()
        moe = (root / "libr4d/r4d_moe.hip").read_text()
        assert '"had_quant_act_fp8", "had_quant_act_fp8", "quant"' in rows
        assert '"moe_gemm_fp8a8", "moe_gemm_q", "moe"' in rows
        assert "r4d_moe_block_fp8::wide_n<W4, I8>(N, parts, Nc, ncls, kPrefillCols)" in moe

        cxx = shutil.which("c++")
        assert cxx is not None
        fixture = root / "rad0046_geometry_test.cpp"
        fixture.write_text(r'''#include "r4d_moe_block_fp8_geometry.h"
int main() {
    constexpr long long aligned[] = {512, 256};
    constexpr long long split_at_192[] = {512, 384};
    if (!r4d_moe_block_fp8::wide_n<0, false>(512, 2, aligned, 2, 128)) return 1;
    if (r4d_moe_block_fp8::wide_n<0, false>(512, 2, split_at_192, 2, 128)) return 2;
    // Non-block-E4M3 formats preserve the existing max-width-only decision.
    if (!r4d_moe_block_fp8::wide_n<0, true>(512, 2, split_at_192, 2, 128)) return 3;
    if (!r4d_moe_block_fp8::wide_n<1, false>(512, 2, split_at_192, 2, 128)) return 4;
    return 0;
}
''', encoding="utf-8")
        executable = root / "rad0046_geometry_test"
        subprocess.run(
            [cxx, "-std=c++17", "-I", str(root / "libr4d"), str(fixture), "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)

        snapshot = {rel: (root / rel).read_bytes() for rel in files}
        second = apply_all(patches, root)
        assert second and all(result.ok for result in second), second
        assert all(edit.status == "already-applied" for result in second for edit in result.results), second
        assert snapshot == {rel: (root / rel).read_bytes() for rel in files}
