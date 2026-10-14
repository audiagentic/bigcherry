"""No-GPU numerical checks for RWQ01's non-production codec proxy."""
import importlib.util
from pathlib import Path
import random
import unittest

MOD = Path(__file__).resolve().parents[2] / 'lab' / 'radiance' / 'wire-quality' / 'codec_mock.py'
spec = importlib.util.spec_from_file_location('codec_mock', MOD)
assert spec and spec.loader
import sys
codec = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = codec
spec.loader.exec_module(codec)


class WireMockTest(unittest.TestCase):
    def test_pack_roundtrip_and_size(self):
        rng = random.Random(201)
        for bits in (4, 6, 8):
            for count in (64, 128, 512):
                codes = [rng.randrange(1 << bits) for _ in range(count)]
                b = codec.pack_codes(codes, bits)
                self.assertEqual(codec.unpack_codes(b, bits, count), codes)
                self.assertEqual(len(b), count * bits // 8)
                self.assertEqual(len(b) + count // 64 * 2, count * bits // 8 + count // 32)

    def test_round_even_bf16(self):
        self.assertEqual(codec.bf16_bits(1.0), 0x3F80)
        self.assertEqual(codec.bf16_bits(1.00390625), 0x3F80)  # halfway to 1+1/128
        self.assertEqual(codec.bf16_bits(1.01171875), 0x3F82)  # halfway to 1+2/128

    def test_wht_self_inverse_f32_tolerance(self):
        rng = random.Random(103)
        x = [rng.uniform(-5, 5) for _ in range(64)]
        y = codec.transform64(codec.transform64(x))
        self.assertLess(max(abs(a-b) for a, b in zip(x, y)), 3e-6)

    def test_zero_and_rank_symmetry(self):
        z = [0.0] * 128
        for bits in (4, 6, 8):
            p = codec.encode(z, bits)
            self.assertEqual(codec.reduce_packets(p, p), z)
            a, b = codec.make_inputs(14, 'sparse', groups=2)
            pa, pb = codec.encode(a, bits), codec.encode(b, bits)
            self.assertEqual(codec.reduce_packets(pa, pb), codec.reduce_packets(pb, pa))

    def test_input_shape_and_packet_validation(self):
        with self.assertRaises(ValueError):
            codec.encode([1.0] * 63, 6)
        with self.assertRaises(ValueError):
            codec.encode([1.0] * 64, 5)
        with self.assertRaises(ValueError):
            codec.unpack_codes(b'\x00', 6, 64)
        with self.assertRaises(ValueError):
            codec.reduce_packets(codec.encode([0.0] * 64, 6), codec.encode([0.0] * 128, 6))

    def test_nonzero_accuracy_and_wht8_gaussian(self):
        a, b = codec.make_inputs(7107, 'gaussian', groups=8)
        ref = codec.exact_sum(a, b)
        e6 = codec.metrics(ref, codec.reduce_packets(codec.encode(a, 6), codec.encode(b, 6)))
        e8 = codec.metrics(ref, codec.reduce_packets(codec.encode(a, 8), codec.encode(b, 8)))
        self.assertGreater(e6['rel_l2'], 0)
        self.assertLess(e8['rel_l2'], e6['rel_l2'])
        self.assertEqual(codec.encode(a, 8).wire_bytes, 8 * (64 + 2))
        self.assertEqual(codec.encode(a, 6).wire_bytes, 8 * (48 + 2))

    def test_mock_metadata_disclaims_real_quality(self):
        d = codec.run_mock()
        self.assertFalse(d['gpu_tested'])
        self.assertFalse(d['model_kl_tested'])
        self.assertEqual(set(d['results']), {'gaussian', 'heavy_tail', 'sparse', 'scale_mixed'})


if __name__ == '__main__':
    unittest.main()
