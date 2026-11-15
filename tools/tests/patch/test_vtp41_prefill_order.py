"""VTP41/rad0035: card-free patch checks on the pristine pinned Radiance source plus rad0034."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcherry.patcher import apply_all
from test_vtp14_n1_rank_code_selection import PIN_COMMIT, radiance_checkout

PKG = ROOT / "engines/radiance/patches/rad0035_prefill_order"
BASE = ROOT / "engines/radiance/patches/rad0034_prefill_concurrency"


def patch_files(pkg: Path):
    spec = importlib.util.spec_from_file_location(f"{pkg.name}_patch", pkg / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


class Vtp41PatchTests(unittest.TestCase):
    def test_package_recipe_and_default(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (ROOT / "engines/radiance/recipes.toml").open("rb") as f:
            recipes = tomllib.load(f)
        self.assertEqual(meta["id"], PKG.name)
        self.assertEqual(meta["order"], 35)
        self.assertEqual(meta["state"], "untested")
        self.assertEqual(meta["plan-ids"], ["VTP41"])
        self.assertEqual(meta["requires"], [BASE.name])
        chain = recipes["patch-set"]["vtp14-mixed-tp"]["patches"]
        self.assertEqual(chain.count(PKG.name), 1)
        self.assertEqual(chain[chain.index(PKG.name) - 1], BASE.name)
        field = next(e.text for p in patch_files(PKG) for e in p.edits if p.path == "core/rad_core.h")
        for default in ("int64_t long_prefill_tokens = 0;", "int     prefill_order = 0;",
                        "int64_t prefill_age_ms = 0;"):
            self.assertIn(default, field)
        ids = [(p.path, e.id) for p in patch_files(PKG) for e in p.edits]
        self.assertEqual(len(ids), len(set(ids)))

    def test_applies_after_rad0034_and_is_idempotent(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("pristine Radiance v1.17.0 checkout not present; provide RADIANCE_SRC")
        base, mine = patch_files(BASE), patch_files(PKG)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            root = Path(d)
            for path in {p.path for p in base + mine}:
                dest = root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(subprocess.run(
                    ["git", "-C", str(checkout), "show", f"{PIN_COMMIT}:{path}"],
                    check=True, capture_output=True).stdout)
            self.assertTrue(all(f.ok for f in apply_all(base, root)))
            first = apply_all(mine, root)
            for f in first:
                self.assertTrue(f.ok, f)
                for edit in f.results:
                    self.assertEqual(edit.status, "applied", edit)
            second = apply_all(mine, root)
            self.assertTrue(all(e.status == "already-applied" for f in second for e in f.results), second)
            scheduler = (root / "core/sched/scheduler.cpp").read_text()
            self.assertIn("long_tok()", scheduler)
            self.assertNotIn("> geo_.shared_tok)", scheduler)
            self.assertIn("order_queues();", scheduler)
            config = (root / "core/config.cpp").read_text()
            for flag in ("--long-prefill-tokens", "--prefill-order", "--prefill-age-ms"):
                self.assertIn(flag, config)


if __name__ == "__main__":
    unittest.main()
