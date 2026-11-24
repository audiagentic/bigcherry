"""rad0064: the uneven-ownership vocabulary declare accepts a vocabulary-weighted head."""
from __future__ import annotations

import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcherry.patch import patchset  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from _vtp_promotion_support import CANDIDATE_COMMIT, PATCHES, load_patch_files, seed_pinned_paths  # noqa: E402
from test_vtp14_n1_rank_code_selection import radiance_checkout  # noqa: E402

PACKAGE = "rad0064_owned_weighted_head"
TARGET = "arch/qwen35_fp8/qwen35_fp8.cpp"


def source_after(chain, checkout, root):
    for name in chain:
        files = load_patch_files(name)
        seed_pinned_paths(checkout, root, files, CANDIDATE_COMMIT)
        results = apply_all(files, root)
        assert all(r.ok for r in results), (name, results)


class OwnedWeightedHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkout = radiance_checkout()
        if cls.checkout is None:
            raise unittest.SkipTest("set RADIANCE_SRC to a checkout with the pinned Radiance commit")
        closure = patchset.expand_composition("radiance", (PACKAGE,), directory=PATCHES).expanded
        cls.chain = [p for p in closure]
        cls.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        cls.root = Path(cls.tmp.name) / "radiance"
        cls.root.mkdir()
        source_after(cls.chain[:-1], cls.checkout, cls.root)
        cls.before = (cls.root / TARGET).read_text()
        files = load_patch_files(PACKAGE)
        seed_pinned_paths(cls.checkout, cls.root, files, CANDIDATE_COMMIT)
        cls.first = apply_all(files, cls.root)
        cls.after = (cls.root / TARGET).read_text()
        cls.files = files

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_metadata(self):
        meta = tomllib.loads((PATCHES / PACKAGE / "patch.toml").read_text())
        self.assertEqual(meta["requires"], ["rad0019_vtp14_n2_tp_ownership_fp8"])
        self.assertEqual(meta["state"], "untested")
        self.assertEqual(self.chain[-1], PACKAGE)
        self.assertIn("rad0019_vtp14_n2_tp_ownership_fp8", self.chain)

    def test_applies_once_idempotent_and_fails_closed_without_its_requirement(self):
        self.assertTrue(self.first and all(r.ok for r in self.first), self.first)
        again = apply_all(self.files, self.root)
        self.assertTrue(all(r.ok for r in again), again)
        self.assertEqual((self.root / TARGET).read_text(), self.after)
        bare = Path(self.tmp.name) / "bare"
        bare.mkdir()
        seed_pinned_paths(self.checkout, bare, self.files, CANDIDATE_COMMIT)
        self.assertFalse(all(r.ok for r in apply_all(self.files, bare)))

    def test_weighted_head_declares_the_vocabulary_spans_and_the_plain_path_is_unchanged(self):
        old = "reads a plain output.weight, not a "
        self.assertIn(old, self.before)
        self.assertNotIn(old, self.after)
        block = self.after[self.after.index("BC_VTP57_OWNED_VHEAD"):]
        block = block[:block.index("m.op_embed")]
        # the token table keeps its span for both heads
        self.assertIn("rad_weight_shard_span(b, m.w_tok, g.vocab_off, g.vocab_off + g.n_vocab)", block)
        # a weighted head gets the id-to-row map's span, a plain head the lm_head span, as before
        self.assertIn("if (m.n_vcls)\n            RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_vrow,", block)
        self.assertIn("else\n            RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_lm_head,", block)
        # the plain path's two spans are the same calls as before the patch
        for call in ("rad_weight_shard_span(b, m.w_tok, g.vocab_off, g.vocab_off + g.n_vocab)",
                     "rad_weight_shard_span(b, m.w_lm_head, g.vocab_off, g.vocab_off + g.n_vocab)"):
            self.assertIn(call, self.before)
        # nothing outside the ownership block changed
        head = self.before[:self.before.index("reads a plain output.weight")]
        self.assertEqual(self.after[:len(head) - 400], head[:-400])
        tail = self.before[self.before.index("m.op_embed = rw(b, RAD_OP(b, \"embed_lookup\""):]
        self.assertTrue(self.after.endswith(tail))

    def test_only_the_architecture_file_is_edited(self):
        self.assertEqual({f.path for f in self.files}, {TARGET})


if __name__ == "__main__":
    unittest.main()
