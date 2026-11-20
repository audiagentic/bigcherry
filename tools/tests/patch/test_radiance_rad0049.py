"""Exact-source replay and host registry checks for Flash-Next native E4M3 KV."""
from __future__ import annotations

import re
import os
import json
import tempfile
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
            launchers = (root / "libr4d/r4d_attn_paged_gqa8.hip").read_text()
            self.assertEqual(launchers.count("!g8_e4m3_cache(a)"), 4)
            snapshot = {patch.path: (root / patch.path).read_bytes() for patch in files}
            second = apply_all(files, root)
            self.assertTrue(second and all(result.ok for result in second), second)
            self.assertTrue(
                all(edit.status == "already-applied" for result in second for edit in result.results), second,
            )
            self.assertEqual(snapshot, {patch.path: (root / patch.path).read_bytes() for patch in files})

            # RUM11: exercise the actual source-generated page16 row and its card selftest.
            rows = (root / "libr4d/r4d_rows.cpp").read_text()
            self.assertIn('cQkNormRopeGateKvH256G6Page16Fp8[]', rows)
            self.assertIn('C_EQ("n_head", 6), C_EQ("n_head_kv", 1), C_EQ("rot", 64)', rows)
            self.assertIn('C_EQ("block_size", 16), C_IN("kv_dtype", "fp8_e4m3")', rows)
            self.assertIn('ROW(cQkNormRopeGateKvH256G6Page16Fp8)', rows)
            self.assertIn('ROW(cQkNormRopeGateKvQ4Fp8)', rows)
            device_test = (root / "libr4d/r4d_selftest.cpp").read_text()
            self.assertIn('bool page16_e4m3 = false', device_test)
            self.assertIn('page16_e4m3 ? 6 : 12, nkv = page16_e4m3 ? 1 : 2', device_test)
            self.assertIn('qk_kv_fold_case(pl, false, 0, true)', device_test)
            self.assertIn('is byte-identical to the prologue + kv_store', device_test)

            cxx = shutil.which("c++")
            if cxx is None:
                self.skipTest("a host C++ compiler is required to compile the row registry")
            rows = root / "libr4d/r4d_rows.cpp"
            args = [cxx, "-std=c++17", "-I", str(root / "abi"), str(rows)]
            subprocess.run([*args, "-fsyntax-only"], check=True)
            expanded = subprocess.check_output([*args, "-E", "-P"], text=True)
            def row_constraint(name):
                start = expanded.index(f"{name}[]")
                return expanded[start:expanded.index("};", start)]

            page16 = row_constraint("cQkNormRopeGateKvH256G6Page16Fp8")
            for key, value in (("head_dim", 256), ("n_head", 6), ("n_head_kv", 1),
                               ("rot", 64), ("block_size", 16)):
                self.assertIn(f'("{key}"), RAD_C_EQ, (long long)({value})', page16)
            self.assertIn('("q_dtype"), RAD_C_IN, 0, ("bf16")', page16)
            self.assertIn('("kv_dtype"), RAD_C_IN, 0, ("fp8_e4m3")', page16)
            page4 = row_constraint("cQkNormRopeGateKvQ4Fp8")
            self.assertIn('("block_size"), RAD_C_EQ, (long long)(4)', page4)
            self.assertIn('("kv_dtype"), RAD_C_IN, 0, ("fp8_e4m3")', page4)
            self.assertIn('qk_norm_rope_gate_kv_store_h256_gqa6_page16_fp8', expanded)
            self.assertIn('r4d_rad_qk_norm_rope_gate_kv_store', expanded)

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


    def test_exact_frozen_composition_page16_and_gfx1100_rewrite(self):
        source = os.environ.get("RADIANCE_FROZEN_SOURCE")
        if not source:
            self.skipTest("set RADIANCE_FROZEN_SOURCE to the pre-RUM11 composed 2.2 source")
        frozen = Path(source)
        files = load_patch_files(PACKAGE)
        self.assertTrue(all((frozen / f.path).is_file() for f in files))
        with tempfile.TemporaryDirectory(prefix="rum11-") as tmp:
            root = Path(tmp)
            shutil.copytree(frozen / "libr4d", root / "libr4d")
            shutil.copytree(frozen / "abi", root / "abi")
            for name in ("arch/qwen4exp_fp8/qwen4exp_fp8.cpp", "tests/arch_test.cpp"):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(frozen / name, root / name)
            before = (root / "libr4d/r4d_rows.cpp").read_text()
            first = apply_all(files, root)
            self.assertTrue(all(result.ok for result in first), first)
            applied = {e.edit_id for result in first for e in result.results if e.status == "applied"}
            self.assertEqual(applied, {
                "q4-e4m3-page16-constraint", "q4-e4m3-page16-row",
                "q4-e4m3-page16-selftest-signature", "q4-e4m3-page16-selftest-row",
                "q4-e4m3-page16-selftest-geometry", "q4-e4m3-page16-selftest-invocation",
            })
            snapshot = {f.path: (root / f.path).read_bytes() for f in files}
            again = apply_all(files, root)
            self.assertTrue(all(result.ok for result in again), again)
            self.assertTrue(all(e.status == "already-applied"
                                for result in again for e in result.results), again)
            self.assertEqual(snapshot, {f.path: (root / f.path).read_bytes() for f in files})
            after = (root / "libr4d/r4d_rows.cpp").read_text()

            def region(text, begin, end):
                return text[text.index(begin):text.index(end, text.index(begin))]
            self.assertEqual(
                region(before, "static const RadConstraint cQkNormRopeGateKvQ4Fp8[]", "};"),
                region(after, "static const RadConstraint cQkNormRopeGateKvQ4Fp8[]", "};"))
            self.assertEqual(
                region(before, '{ "qk_norm_rope_gate_kv_store_fp8",',
                       '{ "qk_norm_rope_gate_kv_store_streamed_fp8",'),
                region(after, '{ "qk_norm_rope_gate_kv_store_fp8",',
                       '{ "qk_norm_rope_gate_kv_store_streamed_fp8",'))
            self.assertEqual(after.count('{ "qk_norm_rope_gate_kv_store_h256_gqa6_page16_fp8",'), 1)
            selftest = (root / "libr4d/r4d_selftest.cpp").read_text()
            self.assertIn("qk_kv_fold_case(pl, false, 0, true);", selftest)
            self.assertIn("nq = page16_e4m3 ? 6 : 12", selftest)
            self.assertIn("nkv = page16_e4m3 ? 1 : 2", selftest)
            cxx = shutil.which("c++")
            if cxx:
                subprocess.run([cxx, "-std=c++17", "-I", str(root / "abi"),
                                str(root / "libr4d/r4d_rows.cpp"), "-fsyntax-only"], check=True)
            rewritten = subprocess.run(
                [sys.executable, str(REPO / "engines/radiance/components/libr3/r3_rewrite.py"),
                 "--source", str(root / "libr4d"),
                 "--rules", str(REPO / "engines/radiance/components/libr3/rewrites.txt"),
                 "--native", str(REPO / "engines/radiance/components/libr3/native"), "--check"],
                capture_output=True, text=True)
            self.assertEqual(rewritten.returncode, 0, rewritten.stdout + rewritten.stderr)
            self.assertEqual(json.loads(rewritten.stdout)["failures"], {})


if __name__ == "__main__":
    unittest.main()
