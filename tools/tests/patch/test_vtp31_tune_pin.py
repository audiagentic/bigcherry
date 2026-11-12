"""VTP31 (rad0022): `--tune-pin SPEC`, radiance's kernel tune pin as an engine option.

Card-free. The pinned-source test applies rad0022 to radiance v1.17.0 when a checkout with that tag is reachable
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

PKG = REPO / "engines/radiance/patches/rad0022_tune_pin"


def load():
    spec = importlib.util.spec_from_file_location(PKG.name, PKG / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


class TunePinTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
            recipe = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"], meta["requires"]), (22, "untested", []))
        self.assertIn("rad0022_tune_pin", recipe["patch-set"]["vtp14-mixed-tp"]["patches"])

    def test_sets_the_environment_pin_and_refuses_a_disagreeing_one(self):
        edits = {e.id: e.text for f in load() for e in f.edits}
        apply = edits["tp-apply"]
        self.assertIn('std::getenv("RADIANCE_TUNE")', apply)
        self.assertIn('::setenv("RADIANCE_TUNE", cfg_.tune_pin.c_str(), 1)', apply)
        self.assertIn("disagree", apply)
        self.assertIn("c->tune_pin = v;", edits["tp-parse"])
        self.assertNotIn("nblocks", apply)  # nothing about a kernel is built in

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
            self.assertIn("--tune-pin", (root / "core/config.cpp").read_text())


if __name__ == "__main__":
    unittest.main()
