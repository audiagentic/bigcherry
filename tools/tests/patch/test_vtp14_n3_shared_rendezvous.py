"""VTP14 N3 (rad0017): one all-reduce rendezvous shared by libr3 and libr4d, as a managed source patch.

Card-free. Applies rad0016 then rad0017 to the pinned Radiance source when a checkout carrying that tag is
reachable (RADIANCE_SRC or the workspace's vendor/radiance), and checks that libr3's rewrite of the patched libr4d
source still matches. The shared slot itself is tested in C++ by tests/bc_shared_slot_test.cpp, which the patch adds.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vtp_promotion_support import assert_focal_guards_unique, assert_promoted_recipe, load_patch_files, patched_pinned_source
from test_vtp14_n1_rank_code_selection import radiance_checkout
from bigcherry.patcher import apply_all

PATCHES = REPO / "engines/radiance/patches"
N1 = PATCHES / "rad0016_vtp14_rank_code_selection"
N3 = PATCHES / "rad0017_vtp14_n3_shared_rendezvous"
LIBR3 = REPO / "engines/radiance/components/libr3"


def load(package: Path):
    return load_patch_files(package.name)


class SharedRendezvousTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (N3 / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
            recipe = tomllib.load(f)
        self.assertEqual(meta["order"], 17)
        self.assertEqual(meta["state"], "validated")
        self.assertEqual(meta["requires"], ["rad0016_vtp14_rank_code_selection"])
        # one patch set in the main build, inert until its options are given; the as-released source stays unpatched
        assert_promoted_recipe(self, N1.name, N3.name)
        self.assertNotIn("radiance-vtp14-rank-code-selection", recipe["source"])

    def test_guards_unique_and_created_files_from_src(self):
        assert_focal_guards_unique(self, N3.name)
        files = load(N3)
        guards = [e.guard for f in files for e in f.edits]
        self.assertEqual(len(guards), len(set(guards)))
        for f in files:
            if f.create:
                self.assertEqual(f.edits[0].text, (N3 / "src" / Path(f.path).name).read_text())

    def test_applies_over_n1_to_pinned_source_and_libr3_rewrite_matches(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no Radiance checkout with the declared pin; set RADIANCE_SRC")
        with patched_pinned_source(checkout, N3.name, full_checkout=True) as root:
            entry = (root / "libr4d/r4d_ar_entry.hip").read_text()
            self.assertIn('dlsym(RTLD_DEFAULT, "rad_bc_shared_slot")', entry)
            self.assertIn("Rendezvous& rv_for(long long group)", entry)
            self.assertNotRegex(entry, r"(?m)^Rendezvous g_rv;$")
            # libr3's literal anchors in this file must survive the patch.
            self.assertIn("#include <condition_variable>\n#include <cstdio>\n", entry)
            r = subprocess.run([sys.executable, str(LIBR3 / "r3_rewrite.py"), "--source", str(root / "libr4d"),
                                "--rules", str(LIBR3 / "rewrites.txt"), "--native", str(LIBR3 / "native"),
                                "--check"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertNotRegex(r.stdout, r'"failures": \{\s*"')


    def test_rank_buffer_allocation_failures_release_unpublished_storage(self):
        checkout = radiance_checkout()
        compiler = shutil.which("c++")
        if checkout is None or compiler is None:
            self.skipTest("requires a Radiance source checkout and a C++ compiler")
        with (REPO / "engines/radiance/recipes.toml").open("rb") as config:
            pin = tomllib.load(config)["pinned"]
        if subprocess.run(
            ["git", "-C", checkout, "cat-file", "-e", f"{pin}^{{commit}}"],
            capture_output=True,
        ).returncode:
            self.skipTest("Radiance checkout does not contain the current recipe pin")

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temp:
            root = Path(temp)
            archive = subprocess.check_output(
                ["git", "-C", checkout, "archive", pin, "abi", "libr4d"],
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(root, filter="data")

            patches = [file for file in load(N3) if file.path.startswith("libr4d/")]
            results = apply_all(patches, root)
            self.assertTrue(all(result.ok for result in results), results)
            hip = root / "hip"
            hip.mkdir()
            (hip / "hip_runtime.h").write_text(
                "#pragma once\n"
                "#define __host__\n"
                "#define __device__\n"
                "#define __forceinline__ inline\n"
                "using hipStream_t = void*;\n"
                "constexpr int hipSuccess = 0;\n"
                "inline int hipGetDevice(int* out) { *out = 0; return hipSuccess; }\n"
                "inline int hipDeviceGetPCIBusId(char*, int, int) { return 1; }\n"
            )
            # The IPC allocator lives in a HIP kernel translation unit. Compile its
            # actual function body with the fake runtime, alongside the full init shim.
            ipc = (root / "libr4d/r4d_ar_oneshot_2rank_exact.hip").read_text()
            start = ipc.index("long r4d_ar_ipc_alloc(")
            end = ipc.index("\n}", start) + 2
            (root / "ar_ipc_alloc.inc").write_text(ipc[start:end] + "\n")
            fixture = Path(__file__).parent / "fixtures/radiance_ar_allocation_test.cpp"
            executable = root / "ar_allocation_test"
            compiled = subprocess.run(
                [compiler, "-std=c++17", "-ffunction-sections", "-fdata-sections",
                 "-Wl,--gc-sections", "-pthread", "-I", str(root), "-I", str(root / "abi"),
                 "-I", str(root / "libr4d"), str(fixture), "-ldl", "-o", str(executable)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([executable], capture_output=True, text=True, timeout=5)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)


if __name__ == "__main__":
    unittest.main()
