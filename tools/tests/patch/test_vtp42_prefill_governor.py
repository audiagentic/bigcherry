"""VTP42/rad0036: card-free patch checks for the prefill governor on the pinned Radiance source."""
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

PATCHES = ROOT / "engines/radiance/patches"
PKG = PATCHES / "rad0036_prefill_governor"
RETIRED = ("rad0034_prefill_concurrency", "rad0035_prefill_order")


def meta(name: str) -> dict:
    with (PATCHES / name / "patch.toml").open("rb") as f:
        return tomllib.load(f)


def patch_files(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_patch", PATCHES / name / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def chain() -> list[str]:
    with (ROOT / "engines/radiance/recipes.toml").open("rb") as f:
        return tomllib.load(f)["patch-set"]["vtp14-mixed-tp"]["patches"]


class Vtp42PatchTests(unittest.TestCase):
    def test_package_replaces_rad0034_and_rad0035(self):
        m = meta(PKG.name)
        self.assertEqual((m["id"], m["order"], m["state"], m["plan-ids"]), (PKG.name, 36, "untested", ["VTP42"]))
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
        names = chain()[: chain().index(PKG.name) + 1]
        before = [p for n in names[:-1] for p in patch_files(n)]
        mine = patch_files(PKG.name)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            root = Path(d)
            for path in {p.path for p in before + mine if not p.create}:
                dest = root / path
                if dest.exists():
                    continue
                shown = subprocess.run(["git", "-C", str(checkout), "show", f"{PIN_COMMIT}:{path}"],
                                       capture_output=True)
                if shown.returncode == 0:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(shown.stdout)
            for name in names[:-1]:
                self.assertTrue(all(f.ok for f in apply_all(patch_files(name), root)), name)
            first = apply_all(mine, root)
            for f in first:
                self.assertTrue(f.ok, f)
                for edit in f.results:
                    self.assertEqual(edit.status, "applied", edit)
            second = apply_all(mine, root)
            self.assertTrue(all(e.status == "already-applied" for f in second for e in f.results), second)
            scheduler = (root / "core/sched/scheduler.cpp").read_text()
            for hook in ("gov_.gate(", "gov_reserved_ = false;", "gov_.full(prefill_load(false)"):
                self.assertIn(hook, scheduler)
            self.assertIn('"--prefill-governor"', (root / "core/config.cpp").read_text())
            self.assertIn("struct PrefillGovernor", (root / "core/sched/prefill_governor.h").read_text())


if __name__ == "__main__":
    unittest.main()
