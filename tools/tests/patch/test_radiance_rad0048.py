"""Exact-source replay and always-on checks for the local ABI45 header patch."""
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

PACKAGE = "rad0048_abi44_kv_contract"
PIPELINE = "rad0020_vtp14_tp3_pipelined_prefill"
BATCH = "rad0025_multiseq_pipelined_prefill"
OWNERSHIP = "rad0018_vtp14_n2_tp_ownership"

ABI_FIELDS = {
    "RadBuildCtx": (
        "tp_ownership", "pipelined_prefill", "pipelined_prefill_min_rows", "kv_layout",
        "attn_prefill_fp8",
    ),
    "RadKVGroupDecl": ("kv_layout",),
    "RadArchProbe": ("kv_layout_ok",),
    "RadBatch": ("cu_seqlens_h",),
}

FINGERPRINT_PROBE = r"""#include "rad_abi_layout.h"
#include <inttypes.h>
#include <stdio.h>

int main(void) {
    printf("%u %" PRIx64 "\n", RAD_ABI_VERSION, rad_abi_layout());
    return 0;
}
"""


def compile_release_probe(cc: str, root: Path, source: Path) -> Path:
    executable = root / source.stem
    subprocess.run(
        [cc, "-std=c11", "-DNDEBUG", "-O2", "-I", str(root / "abi"), str(source),
         "-o", str(executable)],
        check=True,
    )
    return executable


class Abi45ContractTests(unittest.TestCase):
    def test_promoted_package_dependency_and_recipe_membership(self):
        closure = patchset.expand_composition("radiance", (PACKAGE,), directory=PATCHES)
        resolved = patchset.resolve_exact("radiance", closure.expanded, directory=PATCHES)
        ordered = [module.patch_id for module in resolved.modules]
        self.assertLess(ordered.index(OWNERSHIP), ordered.index(PIPELINE))
        self.assertLess(ordered.index(PIPELINE), ordered.index(BATCH))
        self.assertLess(ordered.index(BATCH), ordered.index(PACKAGE))
        assert_promoted_recipe(self, PACKAGE)

    def test_frozen_abi44_prefix_extends_once_and_release_fixture_runs(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("set RADIANCE_SRC to a checkout with the frozen ABI44 commit")
        cc = shutil.which("cc")
        if cc is None:
            self.skipTest("a C compiler is required for the header fixture")

        with patched_pinned_source(checkout, BATCH, full_checkout=True) as root:
            fingerprint_probe = root / "fingerprint_probe.c"
            fingerprint_probe.write_text(FINGERPRINT_PROBE, encoding="utf-8")
            executable = compile_release_probe(cc, root, fingerprint_probe)
            version, previous_hash = subprocess.check_output([executable], text=True).split()
            self.assertEqual(version, "44")

            patches = load_patch_files(PACKAGE)
            first = apply_all(patches, root)
            self.assertTrue(first and all(result.ok for result in first), first)
            self.assertTrue(all(edit.status == "applied" for result in first for edit in result.results), first)
            snapshot = {patch.path: (root / patch.path).read_bytes() for patch in patches}

            second = apply_all(patches, root)
            self.assertTrue(second and all(result.ok for result in second), second)
            self.assertTrue(
                all(edit.status == "already-applied" for result in second for edit in result.results), second,
            )
            self.assertEqual(snapshot, {patch.path: (root / patch.path).read_bytes() for patch in patches})

            abi_header = (root / "abi/rad_abi.h").read_text()
            self.assertEqual(re.findall(r"^#define RAD_ABI_VERSION (\d+u)$", abi_header, re.M), ["45u"])
            fingerprint_rows = (root / "abi/rad_abi_layout_rows.h").read_text().splitlines()
            for struct, fields in ABI_FIELDS.items():
                rows = [row for row in fingerprint_rows if row.startswith(f"ABI_FIELD({struct}, ")]
                new_rows = [f"ABI_FIELD({struct}, {field})" for field in fields]
                self.assertEqual(rows[-len(new_rows):], new_rows, struct)
                for row in new_rows:
                    self.assertEqual(rows.count(row), 1, row)

            executable = compile_release_probe(cc, root, fingerprint_probe)
            version, current_hash = subprocess.check_output([executable], text=True).split()
            self.assertEqual(version, "45")
            self.assertEqual(current_hash, "6c431f97e8b670d4")
            self.assertNotEqual(current_hash, previous_hash)

            fixture = root / "tests/bc_kv_layout_contract_test.c"
            executable = compile_release_probe(cc, root, fixture)
            self.assertEqual(subprocess.check_output([executable], text=True).strip(), current_hash)

            # Run the upstream C/C++ ABI suite against actual composed headers/table.
            cxx = shutil.which("c++")
            self.assertIsNotNone(cxx, "requires C++ compiler for ABI parity")
            c_obj = root / "abi_c.o"
            subprocess.run([cc, "-std=c11", "-pedantic-errors", "-I", str(root / "abi"),
                            "-c", str(root / "tests/abi_c_test.c"), "-o", str(c_obj)], check=True)
            abi_suite = root / "abi-layout-test"
            subprocess.run([cxx, "-std=c++17", "-O2", "-I", str(root / "abi"),
                            str(root / "tests/abi_layout_test.cpp"), str(c_obj),
                            "-o", str(abi_suite)], check=True)
            checked = subprocess.run([str(abi_suite)], capture_output=True, text=True, timeout=10)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

            # A wrong expectation must fail in the same NDEBUG build.
            failed_fixture = root / "test_wrong_row_size.c"
            failed_fixture.write_text(fixture.read_text().replace("== 516", "== 512", 1), encoding="utf-8")
            executable = compile_release_probe(cc, root, failed_fixture)
            failure = subprocess.run([executable], capture_output=True, text=True)
            self.assertEqual(failure.returncode, 1)
            self.assertIn("check failed", failure.stderr)


if __name__ == "__main__":
    unittest.main()
