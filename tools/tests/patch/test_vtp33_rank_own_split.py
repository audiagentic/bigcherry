"""VTP33 (rad0029): in an uneven dense world, every rank gets its own VRAM split.

Card-free. The pinned-source test applies the vtp14-mixed-tp chain up to rad0029 to radiance v1.17.0 when a checkout
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
PKG = PATCHES / "rad0029_rank_own_split"


def load(pkg: Path):
    spec = importlib.util.spec_from_file_location(pkg.name, pkg / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def chain():
    with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
        names = tomllib.load(f)["patch-set"]["vtp14-mixed-tp"]["patches"]
    return names[: names.index(PKG.name) + 1]


class RankOwnSplitTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"], meta["kind"]), (29, "untested", "enhancement"))
        self.assertIn("rad0028_rank_consistent_block_drafts", meta["requires"])
        names = chain()
        self.assertLess(names.index("rad0028_rank_consistent_block_drafts"), names.index(PKG.name))

    def test_own_split_rules(self):
        text = {e.id: e.text for f in load(PKG) for e in f.edits}["own-split"]
        self.assertIn("facts.size() == ranks_.size() &&\n                       cfg_.gpu_memory_mib <= 0;", text)
        self.assertNotIn("kv_planned_tokens_ == 0", text)
        self.assertIn("own = own && f.experts == 0 && !f.unified;", text)
        self.assertIn("std::max<int64_t>(facts[i].free - free_min, 0) / kMiB", text)
        self.assertIn("if (own && T_own > T_common && T_common > 0) {", text)
        self.assertIn("need[i] = own_kv[i]->pool_mib_for_tokens(T_own, own_pool[i]);", text)
        self.assertIn("if (own) kv_planned_tokens_ = T_own;", text)

    def test_guards_are_not_prefixes(self):
        guards = [e.guard for f in load(PKG) for e in f.edits]
        for a in guards:
            for b in guards:
                self.assertFalse(a != b and b.startswith(a), (a, b))

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
            for f in load(PKG):
                text = (root / f.path).read_text()
                for e in f.edits:
                    self.assertIn(e.text, text, e.id)
            bringup = (root / "core/engine_bringup.cpp").read_text()
            self.assertEqual(bringup.count("BC_VTP33_OWN_SPLIT"), 1)
            again = apply_all(load(PKG), root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))


if __name__ == "__main__":
    unittest.main()
