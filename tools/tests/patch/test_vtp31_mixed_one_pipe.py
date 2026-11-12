"""VTP31 (rad0027): the pipelined prefill over a mixed step of decode rows and one prefill sequence.

Card-free. The pinned-source test applies the vtp14-mixed-tp chain up to rad0026 to radiance v1.17.0 when a checkout
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
PKG = PATCHES / "rad0027_mixed_one_pipelined_prefill"


def load(pkg: Path):
    spec = importlib.util.spec_from_file_location(pkg.name, pkg / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def chain():
    with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
        names = tomllib.load(f)["patch-set"]["vtp14-mixed-tp"]["patches"]
    return names[: names.index(PKG.name) + 1]


class MixedOnePipeTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"], meta["kind"]), (27, "untested", "enhancement"))
        self.assertIn("rad0026_mixed_pipelined_prefill", meta["requires"])
        names = chain()
        self.assertLess(names.index("rad0026_mixed_pipelined_prefill"), names.index(PKG.name))

    def test_one_sequence_cut_after_the_decode_rows(self):
        edits = {e.id: e.text for f in load(PKG) for e in f.edits}
        ready = edits["mx1-ready"]
        self.assertIn("R = dT + pipe_rows(T - dT);", ready)
        self.assertIn("if (T - R < 64) return nullptr;", ready)
        self.assertIn("if (batch->n_seq > 1 && !mix1) {", ready)
        # the bounds come from pinned memory made once per decode height, for replay
        self.assertIn("tb[3 * t]     = (int32_t)dT;", edits["mx1-tbl"])
        # the prefill piece is sequence D, its first piece from row DT
        self.assertIn("praw2(table + oD * tpitch, RAD_I32, 1, tpitch)", edits["mx1-attn-launch"])
        self.assertIn("p == 0 ? n - oDT : n", edits["mx1-attn-launch"])
        self.assertIn("praw(batch->positions + r0 + o0, RAD_I32, 1)", edits["mx1-gdn-prep"])
        self.assertIn("st_idx + mD * st_w", edits["mx1-gdn-state"])

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
            gdn = (root / "arch/common/rad_block_gdn_fp8.h").read_text()
            self.assertEqual(gdn.count("BC_VTP31_MX1_GDN_PREP"), 1)
            again = apply_all(load(PKG), root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))


if __name__ == "__main__":
    unittest.main()
