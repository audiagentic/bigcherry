"""VTP42/rad0036: card-free patch checks for the prefill governor on the pinned Radiance source."""
from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vtp_promotion_support import assert_promoted_recipe, serving_chain_to, load_patch_files, patched_pinned_source
from test_vtp14_n1_rank_code_selection import radiance_checkout

PATCHES = ROOT / "engines/radiance/patches"
PKG = PATCHES / "rad0036_prefill_governor"
RETIRED = ("rad0034_prefill_concurrency", "rad0035_prefill_order")


def meta(name: str) -> dict:
    # A retired patch lives, unchanged, under archive/ (never composed); an active one at the patches root.
    package = PATCHES / "archive" / name if name in RETIRED else PATCHES / name
    with (package / "patch.toml").open("rb") as f:
        return tomllib.load(f)


def patch_files(name: str):
    return load_patch_files(name)


def chain():
    return serving_chain_to(PKG.name)


class Vtp42PatchTests(unittest.TestCase):
    def test_package_replaces_rad0034_and_rad0035(self):
        assert_promoted_recipe(self, PKG.name)
        m = meta(PKG.name)
        self.assertEqual((m["id"], m["order"], m["state"], m["plan-ids"]), (PKG.name, 36, "validated", ["VTP42"]))
        self.assertEqual(chain().count(PKG.name), 1)
        for old in RETIRED:
            self.assertEqual(meta(old)["state"], "superseded")
            self.assertIn(PKG.name, meta(old)["retirement"])
            self.assertNotIn(old, chain())
        files = patch_files(PKG.name)
        ids = [(p.path, e.id) for p in files for e in p.edits]
        self.assertEqual(len(ids), len(set(ids)))
        field = next(e.text for p in files for e in p.edits if p.path == "core/rad_core.h")
        self.assertIn("std::string prefill_governor;", field)

    def test_chain_applies_and_is_idempotent(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("pristine Radiance checkout not present; provide RADIANCE_SRC")
        with patched_pinned_source(checkout, PKG.name) as root:
            scheduler = (root / "core/sched/scheduler.cpp").read_text()
            for hook in ("gov_.gate(", "gov_reserved_ = false;", "gov_.full(prefill_load(false)"):
                self.assertIn(hook, scheduler)
            self.assertIn('"--prefill-governor"', (root / "core/config.cpp").read_text())
            self.assertIn("struct PrefillGovernor", (root / "core/sched/prefill_governor.h").read_text())


if __name__ == "__main__":
    unittest.main()
