"""VTP37 (rad0030): the pipelined prefill on a rotated-int8 trunk, under --pipelined-prefill on.

Card-free. The pinned-source test applies the production serving-core chain up to rad0030 to the pinned Radiance source when a checkout
with that tag is reachable (RADIANCE_SRC, or the workspace's vendor/radiance), and is skipped otherwise.
"""
from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcherry.patcher import apply_all
from _vtp_promotion_support import assert_promoted_recipe, serving_chain_to, load_patch_files, patched_pinned_source
from test_vtp14_n1_rank_code_selection import radiance_checkout

PATCHES = REPO / "engines/radiance/patches"
PKG = PATCHES / "rad0030_int8_pipelined_prefill"


def load(pkg: Path):
    return load_patch_files(pkg.name)


def chain():
    return serving_chain_to(PKG.name)


class Int8PipelinedPrefillTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        assert_promoted_recipe(self, PKG.name)
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"], meta["kind"]), (30, "validated", "enhancement"))
        self.assertEqual(meta["plan-ids"], ["VTP37"])
        self.assertEqual(meta["requires"], ["rad0029_rank_own_split"])
        names = chain()
        # after every patch whose text its anchors match: the mode (rad0020) and the pipelined steps (rad0025-rad0027)
        for earlier in ("rad0020_vtp14_tp3_pipelined_prefill", "rad0027_mixed_one_pipelined_prefill",
                        "rad0029_rank_own_split"):
            self.assertLess(names.index(earlier), names.index(PKG.name))
        self.assertIn("**Plan item:** VTP37", (PKG / "SUMMARY.md").read_text())

    def test_plain_norm_writes_the_pair_step_writes(self):
        text = {e.id: e.text for f in load(PKG) for e in f.edits}["plain-norm-fed-pair"]
        self.assertIn("brow_slice(hi8 ? a.q8 : a.q, r0, rows, n),", text)
        self.assertIn("brow_slice(hi8 ? a.s8 : a.s, r0, rows, n / (hi8 ? a.g8 : (int64_t)RAD_SCALE_GROUP)),", text)

    def test_int8_trunk_is_opt_in(self):
        edits = {e.id: e.text for f in load(PKG) for e in f.edits}
        # auto (the default) keeps a rotated-int8 trunk serial; only `on` pipelines it, and never unrotated int8 codes
        self.assertIn("if (m.a_h.q8_fed && (ctx->pipelined_prefill != BC_PIPE_ON || !m.a_h.q8_rot)) {",
                      edits["pipe-i8-mode"])
        self.assertIn("under --pipelined-prefill on", edits["pipe-i8-note"])
        # the second lane's buffers are the pair the norms write, not the E4M3 pair by name
        self.assertIn("rad_buf_concurrent(b, m.a_h.cq())", edits["pipe-i8-concurrent"])
        self.assertIn("rad_buf_concurrent(b, m.a_h.cs())", edits["pipe-i8-concurrent"])
        self.assertNotIn("m.a_h.q)", edits["pipe-i8-concurrent"])

    def test_guards_are_unique_and_not_prefixes(self):
        guards = [e.guard for f in load(PKG) for e in f.edits]
        self.assertEqual(len(guards), len(set(guards)))
        for a in guards:
            for b in guards:
                self.assertFalse(a != b and b.startswith(a), (a, b))
        others = [e.guard for name in chain()[:-1] for f in load(PATCHES / name) for e in f.edits]
        for a in guards:
            for b in others:
                self.assertFalse(a == b or a.startswith(b) or b.startswith(a), (a, b))

    def test_applies_to_pinned_radiance_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no pinned radiance checkout; set RADIANCE_SRC")
        with patched_pinned_source(checkout, PKG.name) as root:
            for f in load(PKG):
                text = (root / f.path).read_text()
                for e in f.edits:
                    self.assertIn(e.text, text, e.id)
            norm = (root / "arch/common/rad_linear.h").read_text()
            plugin = (root / "arch/qwen35_fp8/qwen35_fp8.cpp").read_text()
            self.assertEqual(norm.count("BC_VTP37_PLAIN_FED"), 1)
            # no issue of the plain norm names the E4M3 pair outright any more
            self.assertNotIn("brow_slice(a.q, r0, rows, n),", norm)
            for guard in ("BC_VTP37_PIPE_I8", "BC_VTP37_PIPE_NOTE", "BC_VTP37_PIPE_BUFS"):
                self.assertEqual(plugin.count(guard), 1, guard)
            self.assertNotIn("rad_buf_concurrent(b, m.a_h.q)", plugin)
            again = apply_all(load(PKG), root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))


if __name__ == "__main__":
    unittest.main()
