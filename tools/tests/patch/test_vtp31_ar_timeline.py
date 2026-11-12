"""VTP31 (rad0023): the exact one-shot all-reduce's diagnostic timeline.

Card-free. The pinned-source test applies rad0023 to radiance v1.17.0 when a checkout with that tag is reachable
(RADIANCE_SRC, or the workspace's vendor/radiance), and is skipped otherwise.
"""
from __future__ import annotations

import importlib.util
import os
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

PKG = REPO / "engines/radiance/patches/rad0023_ar_timeline"


def load():
    spec = importlib.util.spec_from_file_location(PKG.name, PKG / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


class ArTimelineTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
            recipe = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"], meta["kind"]), (23, "untested", "diagnostic"))
        self.assertIn("rad0023_ar_timeline", recipe["patch-set"]["vtp14-mixed-tp"]["patches"])

    def test_inert_without_the_variable(self):
        edits = {e.id: e.text for f in load() for e in f.edits}
        self.assertIn('std::getenv("R4D_AR_TIMELINE")', edits["artl-host"])
        self.assertIn("return (d && *d) ? d : nullptr;", edits["artl-host"])
        self.assertIn("if (!artl_dir()) return nullptr;", edits["artl-host"])
        self.assertIn("if (tlring && b < R4D_ARTL_BLOCKS)", edits["artl-kstart"])
        self.assertIn("if (tl) {", edits["artl-kend"])
        # the stamps index the ring by the device sequence number, not a host counter (replay-safe)
        self.assertIn("(s % R4D_ARTL_RING)", edits["artl-kstart"])

    def test_applies_to_pinned_radiance_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no pinned radiance checkout; set RADIANCE_SRC")
        files = load()
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            root = Path(d)
            for f in files:
                blob = subprocess.run(["git", "-C", str(checkout), "show", f"{PIN_COMMIT}:{f.path}"],
                                      capture_output=True, check=True).stdout
                (root / f.path).parent.mkdir(parents=True, exist_ok=True)
                (root / f.path).write_bytes(blob)
            first = apply_all(files, root)
            self.assertTrue(all(r.ok for r in first), first)
            again = apply_all(files, root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))


if __name__ == "__main__":
    unittest.main()
