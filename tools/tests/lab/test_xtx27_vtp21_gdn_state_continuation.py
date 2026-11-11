"""VTP21 original checkpoint GDN recurrent+conv chunk continuation contracts."""
from __future__ import annotations

import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

LAB = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(LAB))

import xtx27_vtp14_gdn_exact as exact
import xtx27_vtp21_gdn_state_continuation as continuation


class GdnStateContinuationTests(unittest.TestCase):
    def test_causal_convolution_ring_matches_full_original_reference(self):
        x = [
            [1.0, -0.5],
            [2.0, 0.75],
            [-0.5, 1.5],
            [0.25, -1.0],
        ]
        weights = [0.25, -0.125, 0.5, 1.0, -0.25, 0.75, 0.125, 0.5]
        full = [
            [exact._bf16_conv(weights, c, t, synthetic=lambda tt, cc: x[tt][cc])
             for c in range(2)]
            for t in range(4)
        ]
        for parts in continuation.PARTITIONS:
            streamed, receipt_hashes = continuation.cached_conv(x, weights, parts)
            self.assertEqual(streamed, full)
            self.assertEqual(len(receipt_hashes), len(parts))
            self.assertTrue(all(len(h) == 64 for h in receipt_hashes))
        first, _ = continuation.cached_conv(x[:2], weights, (2,))
        reset, _ = continuation.cached_conv(x[2:], weights, (2,))
        self.assertNotEqual([*first, *reset], full)

    def test_recurrent_state_layout_round_trip_and_fail_closed(self):
        state = [[0.0] * 128 for _ in range(128)]
        state[0][0] = 1.234567890123
        state[9][17] = -0.25
        resumed, digest = continuation.f64_checkpoint(state)
        self.assertEqual(resumed, state)
        self.assertEqual(len(digest), 64)
        self.assertIsNot(state, resumed)
        self.assertIsNot(state[0], resumed[0])
        resumed[0][0] = 0.0
        self.assertEqual(state[0][0], 1.234567890123)
        with self.assertRaises(ValueError):
            continuation.f64_checkpoint(state, tamper=True)
        with self.assertRaises(ValueError):
            continuation.f64_checkpoint(state[:127])
        state[0][1] = float("nan")
        with self.assertRaises(ValueError):
            continuation.f64_checkpoint(state)

    def test_exact_chunk_splits_and_physical_ownership(self):
        self.assertEqual(continuation.PARTITIONS, (
            (4,), (1, 3), (2, 2), (3, 1), (1, 1, 1, 1),
        ))
        self.assertEqual(continuation.ROLE_SPECS, (
            ("xtx0", 4, 12), ("xtx1", 4, 12), ("r9700", 8, 24),
        ))
        self.assertEqual(sum(nv for _, _, nv in continuation.ROLE_SPECS), 48)
        self.assertEqual(sum(nk for _, nk, _ in continuation.ROLE_SPECS), 16)
        self.assertEqual(continuation.F64_STATE_LEN, 128 * 128 * 8)
        self.assertEqual([sum(p) for p in continuation.PARTITIONS], [4] * 5)

    def test_rejects_missing_real_checkpoint_receipts_and_gpu_free_plan(self):
        with (redirect_stdout(io.StringIO()) as out,
              patch.object(continuation, "verify",
                           side_effect=AssertionError("plan cannot read weights"))):
            self.assertEqual(continuation.main([]), 0)
        record = json.loads(out.getvalue())
        self.assertFalse(record["gpu_required"])
        self.assertFalse(record["model_serving"])
        self.assertEqual(len(record["partitions"]), 5)
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            continuation.main(["--phase", "verify"])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            continuation.main(["--phase", "verify", "--run"])
        with tempfile.TemporaryDirectory() as tmp:
            malformed = pathlib.Path(tmp) / "fake.json"
            malformed.write_text("{}")
            with self.assertRaises((KeyError, ValueError)):
                continuation.validate(malformed, malformed, malformed, malformed)

    def test_causal_convolution_guard_rejects_bad_segments(self):
        rows = [[1.0, 2.0] for _ in range(4)]
        weights = [1.0] * 8
        with self.assertRaises(ValueError):
            continuation.cached_conv(rows, weights, (1, 1))
        with self.assertRaises(ValueError):
            continuation.cached_conv(rows, weights, (0, 4))
        with self.assertRaises(ValueError):
            continuation.cached_conv(rows, weights[:4], (4,))
        with self.assertRaises(ValueError):
            continuation.cached_conv([[1.0], [1.0, 2.0]], weights, (2,))


if __name__ == "__main__":
    unittest.main()
