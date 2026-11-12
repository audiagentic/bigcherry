"""VTP31 (rad0024): the three-rank host one-shot hands decode-sized messages to the weighted two-shot.

Card-free. The pinned-source test applies the vtp14-mixed-tp chain up to rad0024 to radiance v1.17.0 when a checkout
with that tag is reachable (RADIANCE_SRC, or the workspace's vendor/radiance), and is skipped otherwise.
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

PATCHES = REPO / "engines/radiance/patches"
PKG = PATCHES / "rad0024_ts3_decode"


def load(pkg: Path):
    spec = importlib.util.spec_from_file_location(pkg.name, pkg / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def chain():
    with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
        names = tomllib.load(f)["patch-set"]["vtp14-mixed-tp"]["patches"]
    return names[: names.index(PKG.name) + 1]


class Ts3DecodeTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"], meta["kind"]), (24, "untested", "enhancement"))
        self.assertIn("rad0021_vtp14_weighted_twoshot_3rank", meta["requires"])
        names = chain()
        self.assertLess(names.index("rad0021_vtp14_weighted_twoshot_3rank"), names.index(PKG.name))

    def test_handover_is_gated(self):
        edits = {e.id: e.text for f in load(PKG) for e in f.edits}
        hand = edits["ts3d-handoff"]
        # only a host mailbox with settled weights, only from ts_min, and 0 turns it off
        self.assertIn("if (in->host && in->shard_weighted) {", hand)
        self.assertIn("if (ts_min > 0 && c.numel >= ts_min) {", hand)
        # a shape or slot the two-shot refuses falls through to the one-shot
        self.assertIn("if (rc2 != RAD_E_SHAPE && rc2 != RAD_E_SCRATCH) return rc2;", hand)
        # the row's own entry keeps the weights' floor
        self.assertIn("return ts3_exact(a, s, false);", edits["ts3d-split"])
        self.assertIn("ar8_weights_for(c.in, c.numel)", edits["ts3d-weights"])

    def test_applies_to_pinned_radiance_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no pinned radiance checkout; set RADIANCE_SRC")
        files = [f for name in chain() for f in load(PATCHES / name)]
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            root = Path(d)
            for path in {f.path for f in files if not f.create}:
                blob = subprocess.run(["git", "-C", str(checkout), "show", f"{PIN_COMMIT}:{path}"],
                                      capture_output=True, check=True).stdout
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_bytes(blob)
            for f in files:
                result = apply_all([f], root)
                self.assertTrue(all(r.ok for r in result), result)
            text = (root / "libr4d/r4d_ar_entry.hip").read_text()
            # declared before its first use, defined once
            self.assertLess(text.index("static int ts3_exact(const RadArgs* a, RadStream s, bool any_size);"),
                            text.index("ts3_exact(a, s, true)"))
            self.assertEqual(text.count("static int ts3_exact(const RadArgs* a, RadStream s, bool any_size) {"), 1)
            again = apply_all(load(PKG), root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))


if __name__ == "__main__":
    unittest.main()
