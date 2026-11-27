"""VTP33 (rad0029): in an uneven dense world, every rank gets its own VRAM split.

Card-free. The pinned-source test applies the production serving-core chain up to rad0029 to the pinned Radiance source when a checkout
with that tag is reachable (RADIANCE_SRC, or the workspace's vendor/radiance), and is skipped otherwise.
"""
from __future__ import annotations

import sys
import shutil
import subprocess
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcherry.patcher import apply_all
from _vtp_promotion_support import assert_focal_guards_unique, assert_promoted_recipe, serving_chain_to, load_patch_files, patched_pinned_source
from test_vtp14_n1_rank_code_selection import radiance_checkout

PATCHES = REPO / "engines/radiance/patches"
PKG = PATCHES / "rad0029_rank_own_split"


def load(pkg: Path):
    return load_patch_files(pkg.name)


def chain():
    return serving_chain_to(PKG.name)


class RankOwnSplitTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        assert_promoted_recipe(self, PKG.name)
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"], meta["kind"]), (29, "validated", "enhancement"))
        self.assertIn("rad0028_rank_consistent_block_drafts", meta["requires"])
        names = chain()
        self.assertLess(names.index("rad0028_rank_consistent_block_drafts"), names.index(PKG.name))

    def test_own_split_rules(self):
        text = {e.id: e.text for f in load(PKG) for e in f.edits}["own-split"]
        self.assertIn("facts.size() == ranks_.size() &&\n                       !broker_ && cfg_.vram_claim <= 0 && cfg_.gpu_memory_mib <= 0;", text)
        self.assertNotIn("kv_planned_tokens_ == 0", text)
        self.assertIn("own = own && f.experts == 0 && !f.unified;", text)
        self.assertIn("vram_budget_rank_spare_mib(cfg_, facts, i, false)", text)
        self.assertIn("if (own && T_own > T_common && T_common > 0) {", text)
        self.assertIn("need[i] = own_kv[i]->pool_mib_for_tokens(T_own, own_pool[i]);", text)
        self.assertIn("if (own) kv_planned_tokens_ = T_own;", text)

    def test_guards_are_not_prefixes(self):
        assert_focal_guards_unique(self, PKG.name)
        guards = [e.guard for f in load(PKG) for e in f.edits]
        for a in guards:
            for b in guards:
                self.assertFalse(a != b and b.startswith(a), (a, b))

    def test_applies_to_pinned_radiance_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no pinned radiance checkout; set RADIANCE_SRC")
        with patched_pinned_source(checkout, PKG.name, full_checkout=True) as root:
            for f in load(PKG):
                text = (root / f.path).read_text()
                for e in f.edits:
                    self.assertIn(e.text, text, e.id)
            bringup = (root / "core/engine_bringup.cpp").read_text()
            self.assertEqual(bringup.count("BC_VTP33_OWN_SPLIT"), 1)
            again = apply_all(load(PKG), root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))
            cxx = shutil.which("c++")
            self.assertIsNotNone(cxx, "requires C++ compiler for rank spare contract")
            fixture = Path(__file__).parent / "fixtures/radiance_rank_spare_test.cpp"
            exe = root / "rank-spare-test"
            build = subprocess.run([cxx, "-std=c++17", "-O0", "-ffunction-sections",
                                    "-fdata-sections", "-Wl,--gc-sections",
                                    "-I", str(root / "abi"), "-I", str(root / "core"),
                                    str(fixture), str(root / "core/mem/vram_budget.cpp"), "-o", str(exe)],
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(build.returncode, 0, build.stderr[-6000:])
            subprocess.run([str(exe)], check=True, timeout=5)



if __name__ == "__main__":
    unittest.main()
