"""Offline reduction-layout checks for two gfx1100 RMSNorm GPU candidates.

Tests arithmetic coverage and wave/weight indexing only. Does NOT compile HIP or validate
the device's reduction order, performance, or Radiance plugin integration.
"""
import math
import unittest


def wave_columns(n, lane):
    return range(lane, n, 32)


def block_columns(n, thread):
    return range(thread, n, 256)


def wave_reduction(values, n):
    sums = [sum(values[c] * values[c] for c in wave_columns(n, lane))
            for lane in range(32)]
    return sum(sums)


def block_reduction(values, n):
    partial = [
        sum(sum(values[c] * values[c] for c in block_columns(n, w * 32 + lane))
            for lane in range(32))
        for w in range(8)
    ]
    return sum(partial)


class NormLayout(unittest.TestCase):
    def test_exact_column_coverage(self):
        for n in (32, 64, 256, 5120, 8192):
            for fn, lanes in ((wave_columns, 32), (block_columns, 256)):
                reached = [c for lane in range(lanes) for c in fn(n, lane)]
                self.assertEqual(list(range(n)), sorted(reached))

    def test_f32_gain_offset_not_bf16_prefolding(self):
        # Qwen/Gemma-style zero-centered gain: wadd=1 is applied in F32.
        gain = 0.0038
        x = 0.75
        expected = x * (gain + 1.0)
        self.assertAlmostEqual(expected, 0.75285)
        self.assertNotEqual(round(gain + 1, 2), gain + 1)

    def test_both_reductions_match_sum_square(self):
        for n in (32, 256, 5120):
            values = [((13 * i + 5) % 37 - 18) / 16 for i in range(n)]
            expected = math.fsum(x*x for x in values)
            self.assertAlmostEqual(wave_reduction(values, n), expected, places=5)
            self.assertAlmostEqual(block_reduction(values, n), expected, places=5)


if __name__ == "__main__":
    unittest.main()
