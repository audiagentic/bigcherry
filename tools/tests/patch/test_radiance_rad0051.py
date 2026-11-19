"""Mechanics checks for the matched ABI44 Flash-Next B16 row and store set."""
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
from _vtp_promotion_support import PATCHES, load_patch_files, patched_pinned_source, recipe
from test_vtp14_n1_rank_code_selection import radiance_checkout

PACKAGE = "rad0051_b16_kv_kernels"
PREREQUISITES = (
    "rad0048_abi44_kv_contract",
    "rad0049_qwen4exp_e4m3_kv",
    "rad0050_b16_kv_runtime_contract",
)

SHAPE_FIXTURE = r"""#include "rad_builder.h"
#include "r4d_plugin.h"

int main() {
    const struct { const char* name; uint32_t dtype; int row; } formats[] = {
        {"fp8_e4m3", RAD_F8E4M3, 512}, {"i8", RAD_I8, 516},
        {"bf16_fp8", RAD_U8, 768}, {"bf16_i8", RAD_U8, 784},
    };
    const struct { RadShapeFn shape; int operand; } bindings[] = {
        {r4d_shape_attn_paged, 1}, {r4d_shape_kv_store, 3},
        {r4d_shape_qk_norm_rope_gate_kv_store, 11},
    };
    for (auto f : formats) {
        const RadParam p[] = {
            RAD_INT("M", 2), RAD_INT("q_len", 2), RAD_INT("head_dim", 256),
            RAD_INT("gqa", 12), RAD_INT("n_head", 24), RAD_INT("n_head_kv", 2),
            RAD_INT("block_size", 4), RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", f.name),
        };
        for (auto b : bindings) {
            RadOpdDesc out{};
            if (b.shape(p, (int)(sizeof(p) / sizeof(p[0])), b.operand, &out) != RAD_OK ||
                out.dtype != f.dtype || out.rank != 4 || out.shape[1] != 2 ||
                out.shape[2] != 4 || out.shape[3] != f.row) return 1;
        }
    }
}
"""


class Qwen4expB16Tests(unittest.TestCase):
    def test_candidate_requires_both_composition_prerequisites(self):
        expanded = patchset.expand_composition("radiance", (PACKAGE,), directory=PATCHES)
        resolved = patchset.resolve_exact("radiance", expanded.expanded, directory=PATCHES)
        self.assertEqual([module.patch_id for module in resolved.modules[-4:]], [*PREREQUISITES, PACKAGE])
        self.assertEqual(resolved.modules[-1].state, "untested")
        for patch_set in recipe()["patch-set"].values():
            self.assertNotIn(PACKAGE, patch_set["patches"])

    def test_exact_candidate_apply_and_selected_rows(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("set RADIANCE_SRC to a checkout with the frozen ABI43 commit")
        with patched_pinned_source(checkout, "rad0036_prefill_governor", full_checkout=True) as root:
            for name in PREREQUISITES:
                applied = apply_all(load_patch_files(name), root)
                self.assertTrue(applied and all(result.ok for result in applied), applied)
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
            self.assertNotIn("O_KB16", (root / "libr4d/r4d_attn_decode_h256_gqa6.hip").read_text())

            cxx = shutil.which("c++")
            if cxx is None:
                self.skipTest("a host C++ compiler is required for the row constraints check")
            args = [cxx, "-std=c++17", "-I", str(root / "abi"), str(root / "libr4d/r4d_rows.cpp")]
            subprocess.run([*args, "-fsyntax-only"], check=True)
            expanded = subprocess.check_output([*args, "-E", "-P"], text=True)
            for tag, dtype in (("B16F8", "bf16_fp8"), ("B16I8", "bf16_i8")):
                for row in ("Prefill", "Decode", "Gate", "Store", "FusedStore"):
                    start = expanded.index(f"cB16{row}{tag}[]")
                    constraints = expanded[start:expanded.index("};", start)]
                    self.assertIn('("head_dim"), RAD_C_EQ, (long long)(256)', constraints)
                    self.assertIn('("block_size"), RAD_C_EQ, (long long)(4)', constraints)
                    self.assertIn(f'("kv_dtype"), RAD_C_IN, 0, ("{dtype}")', constraints)
                    if row in ("Prefill", "Decode", "Gate"):
                        self.assertIn('("gqa"), RAD_C_EQ, (long long)(12)', constraints)
                    if row in ("Decode", "Gate"):
                        self.assertIn('("q_len"), RAD_C_LE, (long long)(5)', constraints)
                for stage in ("prefill", "decode", "decode_gq", "decode_ghq"):
                    symbol = f"r4d_rad_attn_{stage}_h256_gqa12_{tag.lower()}"
                    self.assertGreaterEqual(len(re.findall(rf"\b{symbol}\b", expanded)), 2, symbol)

            shape_fixture = root / "shape_fixture.cpp"
            shape_fixture.write_text(SHAPE_FIXTURE, encoding="utf-8")
            executable = root / "shape_fixture"
            subprocess.run(
                [cxx, "-std=c++17", "-DNDEBUG", "-ffunction-sections", "-fdata-sections",
                 "-Wl,--gc-sections", "-I", str(root / "abi"), "-I", str(root / "libr4d"),
                 str(root / "libr4d/r4d_shapes.cpp"), str(shape_fixture), "-o", str(executable)],
                check=True,
            )
            subprocess.run([executable], check=True)


if __name__ == "__main__":
    unittest.main()
