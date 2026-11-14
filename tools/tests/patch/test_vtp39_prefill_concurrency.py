"""VTP39/rad0034: card-free patch checks on the pristine pinned Radiance source."""
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

PKG = ROOT / "engines/radiance/patches/rad0034_prefill_concurrency"


def patch_files():
    spec = importlib.util.spec_from_file_location("vtp39_prefill_patch", PKG / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


class Vtp39PatchTests(unittest.TestCase):
    def test_package_recipe_and_default(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (ROOT / "engines/radiance/recipes.toml").open("rb") as f:
            recipes = tomllib.load(f)
        self.assertEqual(meta["id"], PKG.name)
        self.assertEqual(meta["order"], 34)
        self.assertEqual(meta["state"], "untested")
        self.assertEqual(meta["kind"], "enhancement")
        self.assertEqual(meta["plan-ids"], ["VTP39"])
        chain = recipes["patch-set"]["vtp14-mixed-tp"]["patches"]
        self.assertEqual(chain[-1], PKG.name)
        self.assertEqual(chain.count(PKG.name), 1)
        self.assertIn("int64_t max_prefill_seqs = 0;", next(
            e.text for p in patch_files() for e in p.edits if e.id == "config-field"))
        self.assertEqual(len({p.path for p in patch_files()}), len(patch_files()))

    def test_source_pin_marker_and_idempotent_apply(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("pristine Radiance v1.17.0 checkout not present; provide RADIANCE_SRC")
        patches = patch_files()
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            root = Path(d)
            for p in patches:
                dest = root / p.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                blob = subprocess.run(
                    ["git", "-C", str(checkout), "show", f"{PIN_COMMIT}:{p.path}"],
                    check=True, capture_output=True,
                ).stdout
                dest.write_bytes(blob)
            first = apply_all(patches, root)
            for f in first:
                self.assertTrue(f.ok, f)
                for edit in f.results:
                    self.assertEqual(edit.status, "applied", edit)
            for p in patches:
                content = (root / p.path).read_text()
                for e in p.edits:
                    self.assertIn(e.text, content)
                    self.assertIn(e.id, {e.id for e in p.edits})
            second = apply_all(patches, root)
            self.assertTrue(all(f.ok for f in second), second)
            self.assertTrue(all(e.status == "already-applied" for f in second for e in f.results), second)
            config = (root / "core/config.cpp").read_text()
            self.assertIn("--max-prefill-seqs", config)
            self.assertIn("0 (unlimited)", config)
            scheduler = (root / "core/sched/scheduler.cpp").read_text()
            self.assertIn("return kAdmitWait;", scheduler)
            self.assertIn("prefilling_running()", scheduler)


if __name__ == "__main__":
    unittest.main()
