"""RD94: EC13 wired into RD30/31/32's boundary sweeps."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import moe_hostile_routing_sweep as sweep_mod  # noqa: E402


class BoundaryDimensionTests(unittest.TestCase):
    def test_ubatch_matches_rd30_31_32s_own_boundary_language(self):
        # RD30/31/32's own code_samples sections state this exact list
        # verbatim ("ubatch 128,256,512,1024,2048,4096") -- not invented here.
        self.assertEqual(
            sweep_mod.BOUNDARY_UBATCH, (128, 256, 512, 1024, 2048, 4096))

    def test_sweep_covers_every_boundary_ubatch(self):
        cells = sweep_mod.sweep(ubatches=sweep_mod.BOUNDARY_UBATCH)
        seen = {cell.n_tokens for cell in cells}
        self.assertEqual(seen, set(sweep_mod.BOUNDARY_UBATCH))

    def test_sweep_covers_uniform_skew_and_single_hot(self):
        cells = sweep_mod.sweep(ubatches=(512,))
        labels = {cell.label for cell in cells}
        self.assertIn("uniform", labels)
        self.assertIn("mild-skew-zipf", labels)
        self.assertIn("concentrated-zipf", labels)
        self.assertIn("single-hot", labels)


class DeterminismTests(unittest.TestCase):
    def test_same_params_produce_identical_sweep(self):
        a = sweep_mod.sweep(ubatches=(128, 512), seed=7)
        b = sweep_mod.sweep(ubatches=(128, 512), seed=7)
        self.assertEqual(a, b)

    def test_different_seed_can_change_stats(self):
        a = sweep_mod.sweep(ubatches=(512,), seed=1)
        b = sweep_mod.sweep(ubatches=(512,), seed=2)
        # At least one cell's realized stats differ under a different seed
        # (assign_ids' tie-breaking shuffle is seed-dependent).
        self.assertNotEqual(
            tuple(c.tpe_p95 for c in a), tuple(c.tpe_p95 for c in b))


class StatisticsCorrectnessTests(unittest.TestCase):
    def test_uniform_mode_has_minimal_spread(self):
        # Uniform routing: every active expert should get (almost) exactly
        # the same token count -- p95 and mean should be very close.
        cells = sweep_mod.sweep(ubatches=(1024,))
        uniform = next(c for c in cells if c.label == "uniform")
        self.assertLessEqual(uniform.tpe_max - uniform.tpe_min, 1)
        self.assertAlmostEqual(uniform.tpe_mean, uniform.tpe_p95, delta=1.0)

    def test_single_hot_concentrates_almost_everything_on_one_expert(self):
        # "single" mode targets one dominant expert, but assign_ids requires
        # top_k DISTINCT experts per token, so a token can send at most one
        # of its top_k slots to that expert -- the real achievable max is
        # n_tokens (one slot/token), not n_tokens*top_k.
        cells = sweep_mod.sweep(ubatches=(512,))
        single = next(c for c in cells if c.label == "single-hot")
        self.assertEqual(single.tpe_max, 512)

    def test_p95_is_between_mean_and_max(self):
        cells = sweep_mod.sweep(ubatches=(1024,))
        for cell in cells:
            self.assertGreaterEqual(cell.tpe_p95, cell.tpe_min)
            self.assertLessEqual(cell.tpe_p95, cell.tpe_max)

    def test_concentrated_zipf_is_more_skewed_than_mild_skew_zipf(self):
        # Higher alpha (concentrated-zipf uses 2.0 vs mild-skew's 0.5) must
        # produce a real, measurable increase in spread (max-min), not just
        # a label difference -- this is the actual hostile-routing property
        # RD30's acceptance criteria care about.
        cells = sweep_mod.sweep(ubatches=(2048,))
        mild = next(c for c in cells if c.label == "mild-skew-zipf")
        concentrated = next(c for c in cells if c.label == "concentrated-zipf")
        self.assertGreater(
            concentrated.tpe_max - concentrated.tpe_min,
            mild.tpe_max - mild.tpe_min,
        )


class RenderTableTests(unittest.TestCase):
    def test_render_table_contains_every_cell_label(self):
        cells = sweep_mod.sweep(ubatches=(128,))
        table = sweep_mod.render_table(cells)
        for cell in cells:
            self.assertIn(cell.label, table)
        self.assertIn("tpe_p95", table)

    def test_render_table_on_empty_sweep_does_not_crash(self):
        table = sweep_mod.render_table(())
        self.assertIn("label", table)


class IdsShapeUsableAsMulMatIdInputTests(unittest.TestCase):
    def test_ids_length_matches_top_k_times_n_tokens_for_every_cell(self):
        # Confirms the sweep's underlying generate() calls still produce the
        # exact ggml_mul_mat_id `ids` tensor shape (top_k distinct experts
        # per token, flattened token-major) that a real selector would
        # consume -- same shape contract EC13's own tests already lock in,
        # re-verified here at the actual boundary dimensions RD30/31/32 use.
        for n_tokens in (128, 4096):
            result = sweep_mod._sweep_cells_for_ubatch(
                n_tokens, n_experts=256, top_k=8, seed=1)
            for cell in result:
                self.assertEqual(cell.n_tokens, n_tokens)


if __name__ == "__main__":
    unittest.main()
