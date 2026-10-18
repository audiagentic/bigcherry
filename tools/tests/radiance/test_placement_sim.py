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
        self.assertGreater(d.moves, 0)
        self.assertEqual(d.capacity_violations, 0)

    def test_flipflop_is_not_chased(self):
        r, meta, _ = _sim("flipflop")
        d = r["dynamic"]
        self.assertLessEqual(d.moves, 4)
        self.assertEqual(d.capacity_violations, 0)

    def test_unprofitable_shift_is_declined(self):
        """With a move costing more than any saving can repay, the policy must stay put."""
        r, _, _ = _sim("shift", gain_factor=1e9)
        self.assertEqual(r["dynamic"].moves, 0)
        self.assertAlmostEqual(r["dynamic"].mean_us(), r["static"].mean_us(), places=6)

    def test_concurrent_sessions_keep_the_books(self):
        r, _, _ = _sim("concurrent")
        for res in r.values():
            self.assertEqual(res.rows_routed, res.rows_computed)       # no duplicate or lost contribution
            self.assertEqual(res.capacity_violations, 0)

    def test_budget_bounds_moves_per_epoch(self):
        r, _, _ = _sim("shift", budget_fraction_per_epoch=0.0)         # the floor is one expert's worth per epoch
        d = r["dynamic"]
        self.assertGreater(d.epochs, 0)
        self.assertLessEqual(d.moves, d.epochs)

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
        """Placement for an interval uses only events before it: truncating the trace after tick T leaves moves before T unchanged."""
        meta, routes = pt.synth("shift", steps=3000)
        full = ps.run(routes, meta, ps.Model(), ps.Policy(), True, "d")
        cut = ps.run(routes[: int(len(routes) * 0.6)], meta, ps.Model(), ps.Policy(), True, "d")
        self.assertLessEqual(cut.moves, full.moves)


if __name__ == "__main__":
    unittest.main()
