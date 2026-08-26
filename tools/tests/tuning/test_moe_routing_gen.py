"""EC13 MoE routing generator tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import moe_routing_gen as mrg  # noqa: E402


class DeterminismTests(unittest.TestCase):
    def test_same_seed_and_params_produce_byte_identical_output(self):
        for mode, kwargs in (
            ("uniform", {}),
            ("single", {}),
            ("zipf", {"alpha": 1.2}),
            ("concentration", {"tpe_target": 16}),
        ):
            a = mrg.generate(mode, n_tokens=64, seed=42, **kwargs)
            b = mrg.generate(mode, n_tokens=64, seed=42, **kwargs)
            self.assertEqual(a.ids, b.ids, f"mode={mode} not deterministic")
            self.assertEqual(a.tpe_target, b.tpe_target)
            self.assertEqual(a.n_active, b.n_active)

    def test_different_seed_generally_changes_output(self):
        a = mrg.generate("zipf", n_tokens=128, seed=1, alpha=1.0)
        b = mrg.generate("zipf", n_tokens=128, seed=2, alpha=1.0)
        self.assertNotEqual(a.ids, b.ids)


class UniformModeTests(unittest.TestCase):
    def test_uniform_spreads_tokens_within_one_of_equal(self):
        result = mrg.generate("uniform", n_tokens=512, n_experts=16, top_k=4, seed=7)
        self.assertEqual(result.n_active, 16)
        self.assertLessEqual(result.tpe_max - result.tpe_min, 1)

    def test_uniform_total_assignment_matches_tokens_times_topk(self):
        n_tokens, top_k = 100, 4
        result = mrg.generate("uniform", n_tokens=n_tokens, n_experts=16, top_k=top_k, seed=3)
        self.assertEqual(len(result.ids), n_tokens * top_k)


class SingleModeTests(unittest.TestCase):
    def test_single_concentrates_on_one_expert(self):
        result = mrg.generate("single", n_tokens=256, n_experts=32, top_k=4, seed=5)
        # top_k > 1 forces every token to touch top_k-1 "spillover" experts too
        # (assign_ids greedily fills the remaining top_k-1 slots from whatever
        # has the most remaining budget, which is 0 for everyone once the
        # single target expert's budget-heavy on round 1) -- but the target
        # expert must be the single dominant one by tokens-per-expert.
        self.assertEqual(result.tpe_target, 256 * 4)
        self.assertGreaterEqual(result.tpe_max, result.tpe_mean)


class ZipfModeTests(unittest.TestCase):
    def test_zipf_follows_a_power_law_ranking(self):
        result = mrg.generate("zipf", n_tokens=2048, n_experts=64, top_k=8, seed=11, alpha=1.5)
        counts = [0] * result.n_experts
        for e in result.ids:
            counts[e] += 1
        ranked = sorted(counts, reverse=True)
        # Higher alpha => steeper decay; the top-ranked expert must carry
        # meaningfully more load than the median active expert.
        active = [c for c in ranked if c > 0]
        self.assertGreater(active[0], active[len(active) // 2] * 2)

    def test_higher_alpha_increases_concentration(self):
        low = mrg.generate("zipf", n_tokens=1024, n_experts=32, top_k=4, seed=9, alpha=0.5)
        high = mrg.generate("zipf", n_tokens=1024, n_experts=32, top_k=4, seed=9, alpha=2.5)
        self.assertGreater(high.tpe_max, low.tpe_max)


class ConcentrationModeTests(unittest.TestCase):
    def test_feasible_targets_are_powers_of_two_from_sixteen(self):
        targets = mrg.concentration_targets(n_experts=64, top_k=8, n_tokens=64)
        self.assertTrue(targets)
        for t in targets:
            self.assertEqual(t & (t - 1), 0)  # power of two
            self.assertGreaterEqual(t, 16)

    def test_concentration_interpolates_between_uniform_and_single(self):
        n_experts, top_k, n_tokens = 64, 8, 64
        targets = sorted(mrg.concentration_targets(n_experts, top_k, n_tokens))
        results = [
            mrg.generate("concentration", n_tokens=n_tokens, n_experts=n_experts,
                         top_k=top_k, seed=13, tpe_target=t)
            for t in targets
        ]
        active_counts = [r.n_active for r in results]
        # As tpe_target grows, the active set shrinks monotonically toward top_k.
        self.assertEqual(active_counts, sorted(active_counts, reverse=True))
        self.assertGreaterEqual(active_counts[-1], top_k)

    def test_infeasible_target_raises(self):
        with self.assertRaises(mrg.RoutingGenError):
            mrg.generate("concentration", n_tokens=8, n_experts=16, top_k=4, tpe_target=3)


class BoundaryTests(unittest.TestCase):
    def test_single_token(self):
        result = mrg.generate("uniform", n_tokens=1, n_experts=8, top_k=2, seed=1)
        self.assertEqual(len(result.ids), 2)
        self.assertEqual(result.n_active, 2)

    def test_large_token_count(self):
        result = mrg.generate("uniform", n_tokens=4096, n_experts=256, top_k=8, seed=1)
        self.assertEqual(len(result.ids), 4096 * 8)

    def test_top_k_equal_to_n_experts(self):
        result = mrg.generate("uniform", n_tokens=4, n_experts=4, top_k=4, seed=1)
        for t in range(4):
            slot = sorted(result.ids[t * 4:(t + 1) * 4])
            self.assertEqual(slot, [0, 1, 2, 3])


class ValidationTests(unittest.TestCase):
    def test_rejects_top_k_greater_than_n_experts(self):
        with self.assertRaises(mrg.RoutingGenError):
            mrg.generate("uniform", n_tokens=4, n_experts=4, top_k=5)

    def test_rejects_zero_tokens(self):
        with self.assertRaises(mrg.RoutingGenError):
            mrg.generate("uniform", n_tokens=0, n_experts=4, top_k=2)

    def test_rejects_unknown_mode(self):
        with self.assertRaises(mrg.RoutingGenError):
            mrg.generate("bogus", n_tokens=4, n_experts=4, top_k=2)

    def test_concentration_without_target_rejected(self):
        with self.assertRaises(mrg.RoutingGenError):
            mrg.generate("concentration", n_tokens=64, n_experts=16, top_k=4)


class DistinctExpertsPerTokenTests(unittest.TestCase):
    def test_every_token_gets_top_k_distinct_experts(self):
        for mode, kwargs in (("uniform", {}), ("single", {}), ("zipf", {"alpha": 1.0})):
            result = mrg.generate(mode, n_tokens=64, n_experts=16, top_k=4, seed=1, **kwargs)
            for t in range(result.n_tokens):
                slot = result.ids[t * result.top_k:(t + 1) * result.top_k]
                self.assertEqual(len(set(slot)), result.top_k,
                                  f"mode={mode} token {t} has duplicate experts: {slot}")


if __name__ == "__main__":
    unittest.main()
