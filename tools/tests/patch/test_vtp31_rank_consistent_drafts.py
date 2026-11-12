"""VTP31 (rad0028): every rank verifies rank 0's DFlash2 draft.

Card-free. The pinned-source test applies the vtp14-mixed-tp chain up to rad0028 to radiance v1.17.0 when a checkout
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
PKG = PATCHES / "rad0028_rank_consistent_block_drafts"


def load(pkg: Path):
    spec = importlib.util.spec_from_file_location(pkg.name, pkg / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def chain():
    with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
        names = tomllib.load(f)["patch-set"]["vtp14-mixed-tp"]["patches"]
    return names[: names.index(PKG.name) + 1]


class RankConsistentDraftTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"], meta["kind"]), (28, "untested", "enhancement"))
        self.assertIn("rad0027_mixed_one_pipelined_prefill", meta["requires"])
        names = chain()
        self.assertLess(names.index("rad0027_mixed_one_pipelined_prefill"), names.index(PKG.name))

    def test_gathered_tokens_feed_the_proposal(self):
        edits = {e.id: e.text for f in load(PKG) for e in f.edits}
        self.assertIn("{w.tokens}, {w.tokg}", edits["rcd-decl"])
        self.assertIn('RAD_INT("numel", w.tok_rows * cfg.steps())', edits["rcd-decl"])
        self.assertIn("RAD_ISSUE(c, op_tgather, RAD_B(w.tokens), RAD_B(w.tokg))", edits["rcd-issue"])
        self.assertIn("while (rcd && (tok_rows * dc.steps()) % 4) ++tok_rows;", edits["rcd-buf"])
        self.assertIn("drf.proposal       = dw.tokg ? dw.tokg : dw.tokens;", edits["rcd-prop"])

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
            block = (root / "arch/common/rad_block_dflash2.h").read_text()
            self.assertEqual(block.count("BC_VTP31_RCD_ISSUE"), 1)
            again = apply_all(load(PKG), root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))


if __name__ == "__main__":
    unittest.main()
