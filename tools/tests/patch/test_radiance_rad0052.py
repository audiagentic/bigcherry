"""The remaining packed B16 geometries and independent H128 native drafter row."""
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
from test_radiance_rad0051 import PACKAGE as BASE_PACKAGE, PREREQUISITES
from test_vtp14_n1_rank_code_selection import radiance_checkout

PACKAGE = "rad0052_b16_kv_geometries"


class B16GeometryTests(unittest.TestCase):
    def test_promoted_geometry_requires_the_paired_b16_kernels(self):
        closure = patchset.expand_composition("radiance", (PACKAGE,), directory=PATCHES)
        resolved = patchset.resolve_exact("radiance", closure.expanded, directory=PATCHES)
        self.assertEqual([module.patch_id for module in resolved.modules[-2:]], [BASE_PACKAGE, PACKAGE])
        assert_promoted_recipe(self, "rad0048_abi44_kv_contract", "rad0049_qwen4exp_e4m3_kv", "rad0050_b16_kv_runtime_contract", "rad0051_b16_kv_kernels", PACKAGE)

    def test_exact_source_replay_and_remaining_row_bindings(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("set RADIANCE_SRC to the checkout containing the frozen commit")
        with patched_pinned_source(checkout, "rad0036_prefill_governor", full_checkout=True) as root:
            for name in (*PREREQUISITES, BASE_PACKAGE):
                results = apply_all(load_patch_files(name), root)
                self.assertTrue(results and all(result.ok for result in results), results)
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
                self.skipTest("a host compiler is required for the registry check")
            args = [cxx, "-std=c++17", "-I", str(root / "abi"), str(root / "libr4d/r4d_rows.cpp")]
            subprocess.run([*args, "-fsyntax-only"], check=True)
            expanded = subprocess.check_output([*args, "-E", "-P"], text=True)

            def constraints(name):
                start = expanded.index(f"{name}[]")
                return expanded[start:expanded.index("};", start)]

            for fmt, dtype in (("b16f8", "bf16_fp8"), ("b16i8", "bf16_i8")):
                for page, geo, sym in ((4, "h256_gqa6_bs4", "h256_gqa6"),
                                       (16, "h256_gqa6", "h256_gqa6_page16")):
                    stages = ("prefill", "decode", "decode_gq", "decode_ghq") if page == 4 else (
                        "prefill", "decode",
                    )
                    for stage in stages:
                        row = f"attn_{stage}_{geo}_{fmt}"
                        match = constraints(f"c_{row}")
                        self.assertIn(f'("block_size"), RAD_C_EQ, (long long)({page})', match)
                        self.assertIn('("gqa"), RAD_C_EQ, (long long)(6)', match)
                        self.assertIn(f'("kv_dtype"), RAD_C_IN, 0, ("{dtype}")', match)
                        if stage != "prefill":
                            self.assertIn('("q_len"), RAD_C_LE, (long long)(10)', match)
                        symbol = f"r4d_rad_attn_{stage}_{sym}_{fmt}"
                        self.assertGreaterEqual(len(re.findall(rf"\b{symbol}\b", expanded)), 2)
                for store in ("kv_store", "qk_norm_rope_gate_kv_store"):
                    match = constraints(f"c_{store}_page16_{fmt}")
                    self.assertIn('("block_size"), RAD_C_EQ, (long long)(16)', match)
                    self.assertIn(f'("kv_dtype"), RAD_C_IN, 0, ("{dtype}")', match)
            drafter = constraints("cDflashH128Gqa4Fp8")
            self.assertIn('("head_dim"), RAD_C_EQ, (long long)(128)', drafter)
            self.assertIn('("gqa"), RAD_C_EQ, (long long)(4)', drafter)
            self.assertIn('("block_size"), RAD_C_EQ, (long long)(16)', drafter)
            self.assertIn('("kv_dtype"), RAD_C_IN, 0, ("fp8_e4m3")', drafter)
            self.assertIn('("q_len"), RAD_C_LE, (long long)(16)', drafter)


if __name__ == "__main__":
    unittest.main()
