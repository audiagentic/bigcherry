"""VTP14 N3 (rad0017): one all-reduce rendezvous shared by libr3 and libr4d, as a managed source patch.

Card-free. Applies rad0016 then rad0017 to the pinned Radiance source when a checkout carrying that tag is
reachable (RADIANCE_SRC or the workspace's vendor/radiance), and checks that libr3's rewrite of the patched libr4d
source still matches. The shared slot itself is tested in C++ by tests/bc_shared_slot_test.cpp, which the patch adds.
"""
from __future__ import annotations

import hashlib
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
from _vtp_promotion_support import (
    assert_focal_guards_unique,
    assert_promoted_recipe,
    load_patch_files,
    patched_pinned_source,
)
from bigcherry.patcher import apply_all
from test_vtp14_n1_rank_code_selection import radiance_checkout

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
            self.assertEqual(entry.count('#include "bc_ar_rendezvous_impl.h"'), 1)
            self.assertEqual(entry.count('#include "bc_ar_lifecycle_impl.h"'), 1)
            rv = (root / "libr4d/bc_ar_rendezvous_impl.h").read_text()
            life = (root / "libr4d/bc_ar_lifecycle_impl.h").read_text()
            self.assertIn('dlsym(RTLD_DEFAULT, "rad_bc_shared_slot")', rv)
            self.assertIn("Rendezvous& rv_for(long long group)", rv)
            self.assertIn("BC_VTP14_N3_POISON", rv)
            self.assertIn("BC_VTP14_N3_UNPUBLISHED", life)
            self.assertIn("BC_VTP14_N3_FAILED_RANK", life)
            self.assertNotRegex(entry, r"(?m)^Rendezvous g_rv;$")
            # Freeze the reviewed 2.3 composition (native traffic/weighted-four-rank
            # additions and local ownership/poison/teardown). See bump evidence.
            expanded = entry
            for name, text in (("bc_ar_rendezvous_impl.h", rv),
                               ("bc_ar_lifecycle_impl.h", life)):
                expanded = expanded.replace(
                    f'#include "{name}"\n\n',
                    text.split("#pragma once\n\n", 1)[1], 1)
            self.assertEqual(
                hashlib.sha256(expanded.encode()).hexdigest(),
                "2ee24606598d3fbb874a364bc78cd9bf083638939a0d8b6c75138a03726248b1",
            )
            # The native <condition_variable>/<cstdio> pair is in the owned
            # rendezvous header; its gfx11 rewrite must migrate with it.
            self.assertIn("#include <condition_variable>\n#include <cstdio>\n", rv)
            # This fixture only composes through rad0017. Check its three owned
            # native units; full-tree rewrite coverage remains the production
            # composition gate after all patches and renamed units are reconciled.
            native = root / "native-rules"
            native.mkdir()
            for name in ("r4d_ar_entry.hip", "bc_ar_rendezvous_impl.h", "bc_ar_lifecycle_impl.h"):
                self.assertTrue((root / "libr4d" / name).is_file())
                rule = LIBR3 / "native" / (name + ".rw")
                shutil.copyfile(rule, native / rule.name)
            rules = root / "focal-rewrites.txt"
            rules.write_text("")
            r = subprocess.run([sys.executable, str(LIBR3 / "r3_rewrite.py"), "--source", str(root / "libr4d"),
                                "--rules", str(rules), "--native", str(native),
                                "--check"], capture_output=True, text=True, check=False)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertNotRegex(r.stdout, r'"failures": \{\s*"')


    def test_rank_buffer_allocation_and_owner_retirement(self):
        checkout = radiance_checkout()
        compiler = shutil.which("c++")
        if checkout is None or compiler is None:
            self.skipTest("requires a Radiance source checkout and a C++ compiler")
        with (REPO / "engines/radiance/recipes.toml").open("rb") as config:
            pin = tomllib.load(config)["pinned"]
        if subprocess.run(
            ["git", "-C", checkout, "cat-file", "-e", f"{pin}^{{commit}}"],
            capture_output=True,
        check=False).returncode:
            self.skipTest("Radiance checkout does not contain the current recipe pin")

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temp:
            root = Path(temp)
            archive = subprocess.check_output(
                ["git", "-C", checkout, "archive", pin, "abi", "libr4d"],
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(root, filter="data")

            patches = [
                file for file in load(N3)
                if file.path.startswith(("libr4d/", "core/plugin/bc_shared_slot."))
            ]
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
                "extern thread_local int fake_hip_device;\n"
                "inline int hipGetDevice(int* out) { *out = fake_hip_device; return hipSuccess; }\n"
                "inline int hipSetDevice(int dev) { fake_hip_device = dev; return hipSuccess; }\n"
                "inline int hipDeviceGetPCIBusId(char*, int, int) { return 1; }\n"
            )
            # The IPC allocator lives in a HIP kernel translation unit. Compile its
            # actual function body with the fake runtime, alongside the full init shim.
            ipc = (root / "libr4d/r4d_ar_oneshot_2rank_exact.hip").read_text()
            start = ipc.index("long r4d_ar_ipc_alloc(")
            end = ipc.index("\n}", start) + 2
            (root / "ar_ipc_alloc.inc").write_text(ipc[start:end] + "\n")
            # The real bridge needs only Radiance's current engine owner from core.
            (root / "rad_internal.h").write_text("namespace rad { int engine_owner(); }\n")
            bridge = root / "core/plugin/bc_shared_slot.cpp"
            exports = [
                "rad_bc_shared_slot", "rad_bc_rendezvous_owner",
                "rad_bc_defer_rank_cleanup", "rad_bc_forget_rank_cleanup",
            ]
            fixture = Path(__file__).parent / "fixtures/radiance_ar_allocation_test.cpp"
            executable = root / "ar_allocation_test"
            compiled = subprocess.run(
                [compiler, "-std=c++17", "-ffunction-sections", "-fdata-sections",
                 "-Wl,--gc-sections", "-pthread", "-I", str(root), "-I", str(root / "abi"),
                 "-I", str(root / "libr4d"), "-I", str(bridge.parent),
                 *(f"-Wl,--export-dynamic-symbol={symbol}" for symbol in exports),
                 str(fixture), str(bridge), "-ldl", "-o", str(executable)],
                capture_output=True, text=True, timeout=30,
            check=False)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([executable], capture_output=True, text=True, timeout=5, check=False)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)


if __name__ == "__main__":
    unittest.main()
