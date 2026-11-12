"""VTP31 (rad0026): the pipelined prefill over a mixed step, its decode sequences in the first piece.

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
PKG = PATCHES / "rad0026_mixed_pipelined_prefill"


def load(pkg: Path):
    spec = importlib.util.spec_from_file_location(pkg.name, pkg / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def chain():
    with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
        names = tomllib.load(f)["patch-set"]["vtp14-mixed-tp"]["patches"]
    return names[: names.index(PKG.name) + 1]


class MixedPipeTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"], meta["kind"]), (26, "untested", "enhancement"))
        self.assertIn("rad0025_multiseq_pipelined_prefill", meta["requires"])
        names = chain()
        self.assertLess(names.index("rad0025_multiseq_pipelined_prefill"), names.index(PKG.name))

    def test_decode_prefix_rides_in_the_first_piece(self):
        edits = {e.id: e.text for f in load(PKG) for e in f.edits}
        # the cut is between two prefill sequences, after the decode ones
        self.assertIn("for (int64_t s = dS + 1; s < batch->n_seq; ++s)", edits["mxp-ready-loop"])
        self.assertIn("batch->n_seq - dS < 2", edits["mxp-ready-dec"])
        for k in ("mxp-attn-takes", "mxp-gdn-takes"):
            self.assertIn("pieces->seq_cut > dS", edits[k])
        # the first piece issues the serial decode launches over [0, D)
        self.assertIn("praw(batch->cu_seqlens, RAD_I32, mD + 1)", edits["mxp-attn-seqs"])
        self.assertIn("msplit ? (int64_t)g.max_tok", edits["mxp-attn-seqs"])
        gdn = edits["mxp-gdn-dec"]
        self.assertIn("RAD_ISSUE_N(c, op_cr, qd,", gdn)
        self.assertIn("RAD_ISSUE_N(c, op_recur, qd,", gdn)
        self.assertIn("praw(batch->num_accepted, RAD_I32, mD)", gdn)
        # the gated norm skips the decode rows the recurrent update already gated
        self.assertIn("const int64_t g0 = p == 0 ? r0 + gDT : r0;", edits["mxp-gdn-gnorm"])
        self.assertIn("RAD_PHASE_MIXED", edits["mxp-key"])

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
            self.assertEqual(gdn.count("BC_VTP31_MXP_GDN_SEQS"), 1)
            again = apply_all(load(PKG), root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))


if __name__ == "__main__":
    unittest.main()
