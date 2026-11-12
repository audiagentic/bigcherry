"""VTP14 (rad0021): link-weighted shards for the exact three-rank two-shot all-reduce.

Card-free. The pinned-source test applies rad0021 to radiance v1.17.0 when a checkout with that tag is reachable
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

PKG = REPO / "engines/radiance/patches/rad0021_vtp14_weighted_twoshot_3rank"


def load():
    spec = importlib.util.spec_from_file_location(PKG.name, PKG / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


class WeightedTwoshot3Tests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
            recipe = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"]), (21, "untested"))
        self.assertEqual(meta["requires"], ["rad0017_vtp14_n3_shared_rendezvous"])
        self.assertIn("rad0021_vtp14_weighted_twoshot_3rank", recipe["patch-set"]["vtp14-mixed-tp"]["patches"])

    def test_settles_at_three_ranks_and_launches_with_the_table(self):
        edits = {e.id: e.text for f in load() for e in f.edits}
        self.assertIn("if (world == 8 || world == 3)", edits["ts3w-settle"])
        self.assertIn("16, world, in->shard_w)", edits["ts3w-size"])
        launch = edits["ts3w-launch"]
        self.assertIn("ar8_weights_for(c.in, c.numel)", launch)
        self.assertIn("r4d_ar_shard_table(words16, c.in->world, weights, off)", launch)
        self.assertIn("r4d_ar_twoshot_3rank_exact_w(weights,", launch)
        self.assertIn("shard_weights,", edits["ts3w-wrapper"])

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
            entry = (root / "libr4d/r4d_ar_entry.hip").read_text()
            self.assertNotIn(", 8, in->shard_w);", entry)


if __name__ == "__main__":
    unittest.main()
