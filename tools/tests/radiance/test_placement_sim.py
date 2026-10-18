"""RHP07-01: the placement simulator's invariants on deterministic synthetic traces (host only, no GPU)."""
import io
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[3] / "tools/lab/radiance/heat-placement"
sys.path.insert(0, str(HERE))

import placement_sim as ps  # noqa: E402
import placement_trace as pt  # noqa: E402


def _sim(kind, **policy):
    meta, routes = pt.synth(kind, steps=4000)
    return ps.simulate(meta, routes, policy=ps.Policy(**policy)), meta, routes


class PlacementSim(unittest.TestCase):
    def test_trace_roundtrip_and_validation(self):
        meta, routes = pt.synth("stationary", steps=20)
        meta2, routes2 = pt.parse(io.StringIO(pt.dumps(meta, routes)))
        self.assertEqual((meta, routes), (meta2, routes2))
        bad = pt.dumps(meta, routes).replace('"phase": "decode"', '"phase": "warp"', 1)
        with self.assertRaises(pt.TraceError):
            pt.parse(io.StringIO(bad))
        with self.assertRaises(pt.TraceError):
            pt.parse(io.StringIO('{"v":2,"type":"meta"}\n'))

    def test_stationary_makes_no_unnecessary_moves(self):
        r, _, _ = _sim("stationary")
        d, st = r["dynamic"], r["static"]
        self.assertLessEqual(d.moves_after_warmup, 4)                  # a 15% training window is not the true hot set
        self.assertLessEqual(d.mean_us(), st.mean_us() * 1.005)        # and what it moves must not cost more than it saves

    def test_sustained_shift_adapts_and_pays_for_it(self):
        r, _, _ = _sim("shift")
        d, s = r["dynamic"], r["static"]
        self.assertLess(d.mark_mean_us("shift"), s.mark_mean_us("shift"))
        self.assertLess(d.mean_us(True), s.mean_us())               # still ahead with the migration stall charged
        self.assertGreater(d.moves, 0)
        self.assertEqual(d.capacity_violations, 0)

    def test_flipflop_is_not_chased(self):
        r, meta, _ = _sim("flipflop")
        d = r["dynamic"]
        self.assertLessEqual(d.moves, 4)
        self.assertEqual(d.capacity_violations, 0)
        self.assertLessEqual(d.mean_us(True), r["static"].mean_us() * 1.01)

    def test_unprofitable_shift_is_declined(self):
        """With a move costing more than any saving can repay, the policy must stay put."""
        r, _, _ = _sim("shift", gain_factor=1e9)
        self.assertEqual(r["dynamic"].moves, 0)
        self.assertAlmostEqual(r["dynamic"].mean_us(), r["static"].mean_us(), places=6)

    def test_concurrent_sessions_respect_capacity(self):
        r, _, _ = _sim("concurrent")
        for res in r.values():
            self.assertEqual(res.capacity_violations, 0)

    def test_every_route_is_unique_and_validated(self):
        meta, routes = pt.synth("concurrent", steps=30)
        again = pt.dumps(meta, routes + [routes[0]])
        with self.assertRaises(pt.TraceError):
            pt.parse(io.StringIO(again))                       # a duplicate (session, step, layer) is refused

    def test_budget_is_strict(self):
        r, _, _ = _sim("shift", budget_fraction_per_epoch=0.0)
        self.assertEqual(r["dynamic"].moves, 0)                # no floor: a zero budget moves nothing

    def test_budget_bounds_moves_per_epoch(self):
        r, _, _ = _sim("shift")
        d = r["dynamic"]
        per_epoch = {}
        for epoch, *_ in d.history:
            per_epoch[epoch] = per_epoch.get(epoch, 0) + 1
        model, policy = ps.Model(), ps.Policy()
        cap = int(8 * 64 * model.helper_fraction) // 2 * 2 * model.expert_bytes
        most = int(policy.budget_fraction_per_epoch * cap) // model.expert_bytes
        self.assertTrue(per_epoch and max(per_epoch.values()) <= max(1, most + 1))

    def test_capacity_is_never_overcommitted_by_free_slots(self):
        """Two promotions into one free slot must not both be accepted (shadow capacity)."""
        meta, routes = pt.synth("shift", steps=3000)
        model = ps.Model(helper_fraction=0.30)
        policy = ps.Policy(budget_fraction_per_epoch=0.5, margin=0.0, min_residency_epochs=0, gain_factor=0.0)
        res = ps.run(routes, meta, model, policy, True, "d")
        self.assertEqual(res.capacity_violations, 0)

    def test_static_reproduces_hot_first(self):
        meta, routes = pt.synth("stationary", steps=600)
        model = ps.Model()
        place = ps.hot_first(routes, meta, model, 200)
        total = meta["layers"] * meta["experts"]
        self.assertEqual(len(place.owner), int(total * model.helper_fraction))
        hist = {}
        for r in routes[:200]:
            for x, n in zip(r["experts"], r["expert_rows"]):
                k = ps.key_of(r["layer"], x, meta["experts"])
                hist[k] = hist.get(k, 0) + n
        ranked = sorted(hist, key=lambda k: (-hist[k], k))
        self.assertEqual([place.owner[k] for k in ranked[:6]], [0, 1, 0, 1, 0, 1])      # alternating, hottest first
        self.assertEqual(place.violations(), 0)

    def test_deterministic(self):
        a = {k: v.as_dict() for k, v in _sim("shift")[0].items()}
        b = {k: v.as_dict() for k, v in _sim("shift")[0].items()}
        self.assertEqual(a, b)

    def test_no_lookahead(self):
        """The same starting placement and the same observed prefix give the same moves, whatever follows the prefix."""
        meta, routes = pt.synth("shift", steps=3000)
        train = int(len(routes) * 0.15)
        model, policy = ps.Model(), ps.Policy()
        full = ps.run(routes, meta, model, policy, True, "d", train_count=train)
        cut = ps.run(routes[: int(len(routes) * 0.7)], meta, model, policy, True, "d", train_count=train)
        done = cut.epochs
        self.assertGreater(done, 0)
        self.assertEqual([m for m in full.history if m[0] <= done], cut.history)

if __name__ == "__main__":
    unittest.main()
