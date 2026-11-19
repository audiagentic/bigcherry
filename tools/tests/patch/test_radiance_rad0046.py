"""Card-free replay check for the retained ABI43 E4M3 rows."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

from bigcherry.patcher import apply_all

PACKAGE = REPO / "engines/radiance/patches/rad0046_retained_moe_rows/patch.py"
CANDIDATE_COMMIT = "3deabdaeca17b58bad90d332f4b204cf7bb8a10e"


def _patches():
    spec = importlib.util.spec_from_file_location("rad0046_retained_moe_rows", PACKAGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PATCHES


def test_retained_rows_apply_and_replay_on_reviewed_abi43_source():
    checkout = os.environ.get("RADIANCE_SRC")
    if not checkout or not Path(checkout).is_dir():
        pytest.skip("set RADIANCE_SRC to the reviewed ABI43 source checkout")

    resolved = subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "--verify", f"{CANDIDATE_COMMIT}^{{commit}}"], text=True,
    ).strip()
    assert resolved == CANDIDATE_COMMIT

    patches = _patches()
    files = [edit.path for edit in patches]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temp:
        root = Path(temp)
        for rel in files:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(
                ["git", "-C", checkout, "show", f"{CANDIDATE_COMMIT}:{rel}"],
            ))

        first = apply_all(patches, root)
        assert first and all(result.ok for result in first), first
        assert all(edit.status == "applied" for result in first for edit in result.results), first

        rows = (root / "libr4d/r4d_rows.cpp").read_text()
        moe = (root / "libr4d/r4d_moe.hip").read_text()
        assert '"had_quant_act_fp8", "had_quant_act_fp8", "quant"' in rows
        assert '"moe_gemm_fp8a8", "moe_gemm_q", "moe"' in rows
        assert "if constexpr (W4 == 0 && !I8)" in moe
        assert "Nc[q] % parts == 0 && (Nc[q] / parts) % kPrefillCols == 0" in moe

        snapshot = {rel: (root / rel).read_bytes() for rel in files}
        second = apply_all(patches, root)
        assert second and all(result.ok for result in second), second
        assert all(edit.status == "already-applied" for result in second for edit in result.results), second
        assert snapshot == {rel: (root / rel).read_bytes() for rel in files}
