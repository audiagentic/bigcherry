"""Exact candidate replay and registry checks for the retained packed caches and FP8 modes."""
from __future__ import annotations

import json
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
from test_radiance_rad0051 import PACKAGE as B16_KERNELS, PREREQUISITES
from test_radiance_rad0052 import PACKAGE as B16_GEOMETRIES
from test_vtp14_n1_rank_code_selection import radiance_checkout

V4_PACKAGE = "rad0053_v4_kv_layouts"
PREFILL_PACKAGE = "rad0054_attn_prefill_fp8"
LIBR3 = REPO / "engines/radiance/components/libr3"


class RetainedKVTests(unittest.TestCase):
    def test_composition_uses_the_existing_abi44_descriptor_owner(self):
        assert_promoted_recipe(
            self, *PREREQUISITES, B16_KERNELS, B16_GEOMETRIES, V4_PACKAGE, PREFILL_PACKAGE,
        )
        request = patchset.expand_composition("radiance", (PREFILL_PACKAGE,), directory=PATCHES)
        modules = patchset.resolve_exact("radiance", request.expanded, directory=PATCHES).modules
        self.assertEqual(
            [module.patch_id for module in modules[-7:]],
            [*PREREQUISITES, B16_KERNELS, B16_GEOMETRIES, V4_PACKAGE, PREFILL_PACKAGE],
        )
        for package in (V4_PACKAGE, PREFILL_PACKAGE):
            self.assertFalse(any(p.path.startswith("abi/") for p in load_patch_files(package)))

    def test_full_feature_prefix_replays_byte_identically(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("set RADIANCE_SRC to a checkout containing the frozen candidate")
        names = (*PREREQUISITES, B16_KERNELS, B16_GEOMETRIES, V4_PACKAGE, PREFILL_PACKAGE)
        with patched_pinned_source(checkout, "rad0036_prefill_governor", full_checkout=True) as root:
            touched = set()
            for name in names:
                files = load_patch_files(name)
                touched.update(item.path for item in files)
                results = apply_all(files, root)
                self.assertTrue(results and all(result.ok for result in results), (name, results))
            snapshot = {path: (root / path).read_bytes() for path in touched}
            for name in names:
                results = apply_all(load_patch_files(name), root)
                self.assertTrue(results and all(result.ok for result in results), (name, results))
                self.assertTrue(
                    all(edit.status in {"already-applied", "not-applicable"}
                        for result in results for edit in result.results),
                    (name, results),
                )
            self.assertEqual(snapshot, {path: (root / path).read_bytes() for path in touched})

    def test_exact_replay_and_mode_constraints(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("set RADIANCE_SRC to a checkout containing the frozen candidate")
        with patched_pinned_source(checkout, "rad0036_prefill_governor", full_checkout=True) as root:
            for name in (*PREREQUISITES, B16_KERNELS, B16_GEOMETRIES):
                results = apply_all(load_patch_files(name), root)
                self.assertTrue(results and all(result.ok for result in results), results)
            for name in (V4_PACKAGE, PREFILL_PACKAGE):
                files = load_patch_files(name)
                for expected in ("applied", "already-applied"):
                    results = apply_all(files, root)
                    self.assertTrue(results and all(result.ok for result in results), results)
                    self.assertTrue(
                        all(edit.status == expected for result in results for edit in result.results),
                        results,
                    )
                    current = {p.path: (root / p.path).read_bytes() for p in files}
                    if expected == "applied":
                        snapshot = current
                    else:
                        self.assertEqual(snapshot, current)

            # Legacy non-unit scales round at different points in the two writers.
            standalone = (root / "libr4d/r4d_model_bf16.hip").read_text()
            body = standalone.split("static int kv_store_v4(", 1)[1]
            body = body.split('extern "C" int r4d_rad_kv_store_fp8v4', 1)[0]
            self.assertIn("(float)(1.0 / ks), (float)(1.0 / vs)", body)
            fused = (root / "libr4d/r4d_qk_norm_rope_gate.hip").read_text()
            body = fused.split("if (r4d_v4_layout(layout)) {", 1)[1]
            body = body.split("  const bool fp8", 1)[0]
            self.assertIn("kvs.k_inv = 1.0f / (float)ks;", body)
            self.assertIn("kvs.v_inv = 1.0f / (float)vs;", body)
            self.check_libr3_rewrites(root)

            cxx = shutil.which("c++")
            if cxx is None:
                self.skipTest("a C++ compiler is required for the registry checks")
            command = [cxx, "-std=c++17", "-I", str(root / "abi"), str(root / "libr4d/r4d_rows.cpp")]
            subprocess.run([*command, "-fsyntax-only"], check=True)
            rows = subprocess.check_output([*command, "-E", "-P"], text=True)

            def constraints(name):
                begin = rows.index(f"{name}[]")
                return rows[begin:rows.index("};", begin)]

            for tag, dtype in (("fp8v4", "fp8_v4"), ("k6v4", "k6_v4")):
                for name, gqa, page, limit in (
                    (f"cB16Decode{tag.upper()}", 12, 4, 5),
                    (f"c_attn_decode_h256_gqa6_bs4_{tag}", 6, 4, 10),
                    (f"c_attn_decode_h256_gqa6_{tag}", 6, 16, 10),
                ):
                    leg = constraints(name)
                    self.assertIn(f'("gqa"), RAD_C_EQ, (long long)({gqa})', leg)
                    self.assertIn(f'("block_size"), RAD_C_EQ, (long long)({page})', leg)
                    self.assertIn(f'("q_len"), RAD_C_LE, (long long)({limit})', leg)
                    self.assertIn(f'("kv_dtype"), RAD_C_IN, 0, ("{dtype}")', leg)
                    self.assertNotIn("prefill_fp8", leg)

            for tag, dtype in (("fp8", "fp8_e4m3"), ("fp8v4", "fp8_v4"), ("k6v4", "k6_v4")):
                for mode, bits in (
                    ("qk8", "RAD_ATTN_FP8_QK"),
                    ("pv8", "RAD_ATTN_FP8_PV"),
                    ("qk8pv8", "RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV"),
                ):
                    leg = constraints(f"c_bc_{tag}_{mode}")
                    self.assertIn('("head_dim"), RAD_C_EQ, (long long)(256)', leg)
                    self.assertIn('("gqa"), RAD_C_EQ, (long long)(6)', leg)
                    self.assertIn('("block_size"), RAD_C_EQ, (long long)(16)', leg)
                    self.assertIn(f'("kv_dtype"), RAD_C_IN, 0, ("{dtype}")', leg)
                    self.assertIn(f'("prefill_fp8"), RAD_C_EQ, (long long)({bits})', leg)
                    self.assertIn(f"r4d_rad_attn_prefill_h256_{tag}_{mode}", rows)
            self.assertNotIn("prefill_fp8", constraints("cBcPrefillFp8"))
            self.assertNotIn("prefill_fp8", constraints("cBcDecodeFp8"))

    def check_libr3_rewrites(self, root):
        command = [sys.executable, str(LIBR3 / "r3_rewrite.py"),
                   "--source", str(root / "libr4d"), "--rules", str(LIBR3 / "rewrites.txt")]
        native = ["--native", str(LIBR3 / "native")]
        for options in ([], native):
            report = subprocess.run([*command, *options, "--check"], capture_output=True, text=True)
            self.assertEqual(report.returncode, 0, report.stderr + report.stdout)
            self.assertEqual(json.loads(report.stdout), {"failures": {}})

        generated = root.parent / "libr3-native"
        subprocess.run([*command, *native, "--out", str(generated)], check=True, capture_output=True)
        unit = (generated / "r4d_attn_paged_h256_gqa6.hip").read_text()
        launcher = unit.split("static int prefill_launch(", 1)[1].split('extern "C"', 1)[0]
        self.assertIn("constexpr bool CAN_SPLIT = !(XOPT & (O_QK8 | O_PV8));", launcher)
        self.assertIn("constexpr int R3X = (XOPT & (O_QK8 | O_PV8)) ? 0 : O_R3NATIVE;", launcher)
        for launch_opts in ("P_OPT | XOPT", "(P_OPT & ~R3KL) | XOPT | R3X"):
            for split in ("", " | O_PSPLIT"):
                self.assertEqual(launcher.count(f"KVP, {launch_opts}{split}, 0, Layout>"), 1)


if __name__ == "__main__":
    unittest.main()
