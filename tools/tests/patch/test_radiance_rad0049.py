"""Exact-source replay and host registry checks for Flash-Next native E4M3 KV."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcherry.patch import patchset
from bigcherry.patcher import apply_all
from _vtp_promotion_support import PATCHES, assert_promoted_recipe, load_patch_files, patched_pinned_source
from test_vtp14_n1_rank_code_selection import radiance_checkout

PACKAGE = "rad0049_qwen4exp_e4m3_kv"
ABI44 = "rad0048_abi44_kv_contract"
PREFIX = "rad0036_prefill_governor"


class Qwen4expNativeFp8Tests(unittest.TestCase):
    def test_promoted_package_depends_on_abi44(self):
        composition = patchset.expand_composition("radiance", (PACKAGE,), directory=PATCHES)
        resolved = patchset.resolve_exact("radiance", composition.expanded, directory=PATCHES)
        self.assertEqual([patch.patch_id for patch in resolved.modules[-2:]], [ABI44, PACKAGE])
        assert_promoted_recipe(self, ABI44, PACKAGE)

    def test_exact_candidate_replay_and_registry_at_both_geometries(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("set RADIANCE_SRC to a checkout with the frozen ABI43 commit")

        with patched_pinned_source(checkout, PREFIX, full_checkout=True) as root:
            first = apply_all(load_patch_files(ABI44), root)
            self.assertTrue(first and all(result.ok for result in first), first)

            files = load_patch_files(PACKAGE)
            first = apply_all(files, root)
            self.assertTrue(first and all(result.ok for result in first), first)
            self.assertTrue(all(edit.status == "applied" for result in first for edit in result.results), first)
            snapshot = {patch.path: (root / patch.path).read_bytes() for patch in files}
            second = apply_all(files, root)
            self.assertTrue(second and all(result.ok for result in second), second)
            self.assertTrue(
                all(edit.status == "already-applied" for result in second for edit in result.results), second,
            )
            self.assertEqual(snapshot, {patch.path: (root / patch.path).read_bytes() for patch in files})

            cxx = shutil.which("c++")
            if cxx is None:
                self.skipTest("a host C++ compiler is required to compile the row registry")
            rows = root / "libr4d/r4d_rows.cpp"
            args = [cxx, "-std=c++17", "-I", str(root / "abi"), str(rows)]
            subprocess.run([*args, "-fsyntax-only"], check=True)
            expanded = subprocess.check_output([*args, "-E", "-P"], text=True)
            for tag, gqa, width in (("H256G12", 12, 5), ("H256G6Bs4", 6, 10)):
                for row in ("Prefill", "Decode", "DecodeGq"):
                    start = expanded.index(f"cAttn{row}{tag}Fp8[]")
                    constraints = expanded[start:expanded.index("};", start)]
                    self.assertIn(f'("gqa"), RAD_C_EQ, (long long)({gqa})', constraints)
                    self.assertIn('("block_size"), RAD_C_EQ, (long long)(4)', constraints)
                    self.assertIn('("kv_dtype"), RAD_C_IN, 0, ("fp8_e4m3")', constraints)
                    if row != "Prefill":
                        self.assertIn(f'("q_len"), RAD_C_LE, (long long)({width})', constraints)
                for stage in ("prefill", "decode", "decode_gq", "decode_ghq"):
                    # One declaration and at least one registry binding for the new symbol.
                    symbol = f"r4d_rad_attn_{stage}_h256_gqa{gqa}_fp8"
                    self.assertGreaterEqual(len(re.findall(rf"\b{symbol}\b", expanded)), 2, symbol)


if __name__ == "__main__":
    unittest.main()
