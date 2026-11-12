"""VTP14 N3 (rad0017): one all-reduce rendezvous shared by libr3 and libr4d, as a managed source patch.

Card-free. Applies rad0016 then rad0017 to the pinned radiance v1.17.0 source when a checkout carrying that tag is
reachable (RADIANCE_SRC or the workspace's vendor/radiance), and checks that libr3's rewrite of the patched libr4d
source still matches. The shared slot itself is tested in C++ by tests/bc_shared_slot_test.cpp, which the patch adds.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcherry.patcher import apply_all
from test_vtp14_n1_rank_code_selection import PIN_COMMIT, radiance_checkout

PATCHES = REPO / "engines/radiance/patches"
N1 = PATCHES / "rad0016_vtp14_rank_code_selection"
N3 = PATCHES / "rad0017_vtp14_n3_shared_rendezvous"
LIBR3 = REPO / "engines/radiance/components/libr3"


def load(package: Path):
    spec = importlib.util.spec_from_file_location(package.name, package / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


class SharedRendezvousTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (N3 / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
            recipe = tomllib.load(f)
        self.assertEqual(meta["order"], 17)
        self.assertEqual(meta["state"], "untested")
        self.assertEqual(meta["requires"], ["rad0016_vtp14_rank_code_selection"])
        # one patch set in the main build, inert until its options are given; the as-released source stays unpatched
        self.assertEqual(recipe["patch-set"]["vtp14-mixed-tp"]["patches"],
                         ["rad0016_vtp14_rank_code_selection", "rad0017_vtp14_n3_shared_rendezvous", "rad0018_vtp14_n2_tp_ownership",
                          "rad0019_vtp14_n2_tp_ownership_fp8", "rad0020_vtp14_tp3_pipelined_prefill",
                          "rad0021_vtp14_weighted_twoshot_3rank", "rad0022_tune_pin", "rad0023_ar_timeline",
                          "rad0024_ts3_decode", "rad0025_multiseq_pipelined_prefill"])
        self.assertEqual(recipe["source"]["radiance"]["patch-sets"][-1], "vtp14-mixed-tp")
        self.assertNotIn("vtp14-mixed-tp", recipe["source"]["radiance-native"]["patch-sets"])
        self.assertNotIn("radiance-vtp14-rank-code-selection", recipe["source"])

    def test_guards_unique_and_created_files_from_src(self):
        files = load(N3)
        guards = [e.guard for f in files for e in f.edits]
        self.assertEqual(len(guards), len(set(guards)))
        for f in files:
            if f.create:
                self.assertEqual(f.edits[0].text, (N3 / "src" / Path(f.path).name).read_text())

    def test_applies_over_n1_to_pinned_source_and_libr3_rewrite_matches(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no radiance checkout with the pinned v1.17.0 commit; set RADIANCE_SRC")
        n1, n3 = load(N1), load(N3)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "radiance"
            root.mkdir()
            subprocess.run(f"git -C {checkout} archive {PIN_COMMIT} | tar -x -C {root}", shell=True, check=True)
            first = apply_all(n1 + n3, root)
            self.assertTrue(all(f.ok for f in first), first)
            again = apply_all(n1 + n3, root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))
            entry = (root / "libr4d/r4d_ar_entry.hip").read_text()
            self.assertIn('dlsym(RTLD_DEFAULT, "rad_bc_shared_slot")', entry)
            self.assertIn("#define g_rv (rv_slot())", entry)
            self.assertNotRegex(entry, r"(?m)^Rendezvous g_rv;$")
            # libr3's literal anchors in this file must survive the patch.
            self.assertIn("#include <condition_variable>\n#include <cstdio>\n", entry)
            r = subprocess.run([sys.executable, str(LIBR3 / "r3_rewrite.py"), "--source", str(root / "libr4d"),
                                "--rules", str(LIBR3 / "rewrites.txt"), "--native", str(LIBR3 / "native"),
                                "--check"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertNotRegex(r.stdout, r'"failures": \{\s*"')


if __name__ == "__main__":
    unittest.main()
