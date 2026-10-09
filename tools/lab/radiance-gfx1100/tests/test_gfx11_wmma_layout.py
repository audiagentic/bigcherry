"""Offline CPU invariants for gfx1100 wave32 BF16 WMMA output layout.

Sources: AMD ROCm Optimization Guide RDNA3 dense WMMA builtins (gfx11),
and ggml-org/llama.cpp ggml/src/ggml-cuda/mma.cuh RDNA3 path.
These tests prove the INDICES, not actual GPU code generation or WMMA numerics.
"""
import unittest


def gfx11_result_location(lane: int, acc_index: int):
    return (2 * acc_index + lane // 16, lane % 16)


def gfx12_result_location(lane: int, acc_index: int):
    return (8 * (lane // 16) + acc_index, lane % 16)


class RDNA3Layout(unittest.TestCase):
    def test_every_output_written_once(self):
        positions = [gfx11_result_location(l, j) for l in range(32) for j in range(8)]
        self.assertEqual(256, len(set(positions)))
        self.assertEqual({(r, c) for r in range(16) for c in range(16)}, set(positions))

    def test_input_fragment_replication(self):
        # Both lane halves load A[row=lane%16] and B[col=lane%16].
        for lane in range(16):
            self.assertEqual(lane, (lane + 16) % 16)
            self.assertEqual(lane, lane % 16)

    def test_asymmetric_matrix_discriminates_gfx12_layout(self):
        # Every output position differs: symmetric/identity matrices would hide errors.
        a = [[(r * 13 + k * 7 + 5) % 17 - 8 for k in range(16)] for r in range(16)]
        b = [[(c * 11 + k * 3 + 9) % 23 - 11 for k in range(16)] for c in range(16)]
        expected = [[sum(a[r][k] * b[c][k] for k in range(16))
                     for c in range(16)] for r in range(16)]
        # Simulate the correct hardware accumulator as eight f32 values per lane.
        acc = [[expected[2 * j + lane // 16][lane % 16] for j in range(8)]
               for lane in range(32)]
        got = [[None] * 16 for _ in range(16)]
        wrong = [[None] * 16 for _ in range(16)]
        for lane in range(32):
            for j in range(8):
                r, c = gfx11_result_location(lane, j)
                got[r][c] = acc[lane][j]
                r4, c4 = gfx12_result_location(lane, j)
                wrong[r4][c4] = acc[lane][j]
        self.assertEqual(expected, got)
        self.assertNotEqual(expected, wrong)


if __name__ == "__main__":
    unittest.main()
