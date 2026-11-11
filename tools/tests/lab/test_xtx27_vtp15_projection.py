"""VTP15 original BF16 GDN projection packing and GPU-path card-free tests."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

LAB = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(LAB))
import xtx27_vtp15_gdn_projection as projection
import xtx27_vtp15_projection_gpu as gpu
import xtx27_vtp15_projection_ref as ref


def bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


class ProjectionContractTests(unittest.TestCase):
    def test_original_head_shard_coverage_is_exact(self):
        parts = projection.ranges()
        self.assertEqual(parts["xtx0"], {"a": (0, 12), "b": (0, 12), "z": (0, 1536)})
        self.assertEqual(parts["xtx1"], {"a": (12, 12), "b": (12, 12), "z": (1536, 1536)})
        self.assertEqual(parts["r9700"], {"a": (24, 24), "b": (24, 24), "z": (3072, 3072)})
        self.assertEqual([sum(n for _, n in projection.gdn.layout()[rank])
                          for rank in projection.gdn.ROLES], [2560, 2560, 5120])
        for op, n in (("a", 48), ("b", 48), ("z", 6144)):
            intervals = sorted((begin, begin + width)
                               for by_rank in parts.values()
                               for key, (begin, width) in by_rank.items() if key == op)
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][-1], n)
            for previous, current in itertools.pairwise(intervals):
                self.assertEqual(previous[1], current[0])

    def test_original_weighted_rmsnorm_uses_one_plus_weight(self):
        # Zero checkpoint weights imply multiplier = 1, not zero.
        zeros = struct.pack("<5120H", *([0] * 5120))
        h, norm = projection.norm_input(zeros, 2)
        self.assertEqual(len(h), 2 * 5120 * 2)
        self.assertEqual(len(norm), len(h))
        original = struct.unpack("<5120H", h[:10240])
        normalized = struct.unpack("<5120H", norm[:10240])
        self.assertNotEqual(original, normalized)
        self.assertTrue(any(v != 0 for v in normalized))
        vals = [projection.initial_hidden(0, i) for i in range(5120)]
        expect_scale = 1 / math.sqrt(sum(x * x for x in vals) / 5120 + 1e-6)
        self.assertEqual(normalized[0], bf16_bits(vals[0] * expect_scale))
        with self.assertRaises(ValueError):
            projection.norm_input(b"\x00", 1)
        with self.assertRaises(ValueError):
            projection.norm_input(zeros, 9)

    def test_double_accum_cpu_small_fixture(self):
        ref.SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ref.SCRATCH) as temp:
            root = pathlib.Path(temp)
            inp = [1, 0.5, -2, 0.25, 2, 1, -1, 0.5]
            weights = [
                1, 1, 1, 1,
                0.5, 0, 0, 0,
                0, -1, 0.5, 1,
            ]
            (root / "x.bf16").write_bytes(struct.pack("<8H", *map(bf16_bits, inp)))
            (root / "w.bf16").write_bytes(struct.pack("<12H", *map(bf16_bits, weights)))
            exe = root / "oracle"
            subprocess.run(
                ["g++", "-O2", "-std=c++17", str(ref.SOURCE), "-o", str(exe)],
                check=True, capture_output=True,
            )
            cmd = [
                str(exe), "--m", "2", "--n", "3", "--k", "4",
                "--input", str(root / "x.bf16"), "--weights", str(root / "w.bf16"),
                "--out-bf16", str(root / "actual.bf16"),
                "--out-f32", str(root / "actual.f32"),
            ]
            done = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertIn("CPU PASS", done.stdout)
            vals = struct.unpack("<6f", (root / "actual.f32").read_bytes())
            for row in range(2):
                for col in range(3):
                    expected = sum(inp[row * 4 + i] * weights[col * 4 + i]
                                   for i in range(4))
                    self.assertAlmostEqual(vals[row * 3 + col], expected, places=6)
            rounded = struct.unpack("<6H", (root / "actual.bf16").read_bytes())
            self.assertEqual(rounded, tuple(bf16_bits(x) for x in vals))
            fail = subprocess.run(cmd, check=False, capture_output=True)
            self.assertNotEqual(fail.returncode, 0, "refuse to overwrite oracle output")

    def test_no_gpu_by_default_and_gpu_phase_fail_closed(self):
        with (patch.object(gpu.subprocess, "run", side_effect=AssertionError("would launch")),
              redirect_stdout(io.StringIO()) as out):
            self.assertEqual(gpu.main([]), 0)
        plan = json.loads(out.getvalue())
        self.assertTrue(plan["not_full_model"])
        self.assertTrue(plan["not_native_optimized_gemm"])
        self.assertIsNone(plan["source_receipts"])
        for cli in (["--phase", "gpu"],
                    ["--phase", "gpu", "--run"],
                    ["--phase", "compile"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                gpu.main(cli)

    def test_gpu_source_enforces_real_bf16_payload_and_numeric_oracle(self):
        source = gpu.SOURCE.read_text()
        for item in (
            "constexpr std::array<int,3> NKEYS{4,4,8}",
            "constexpr std::array<int,3> NVALUES{12,12,24}",
            "original_bf16_gemv<<<",
            "const uint16_t* x,const uint16_t* weights",
            "accum+=bf16_float(w[k])*bf16_float(input[k])",
            "o.expected32[i]",
            "o.expected16[i]",
            "original BF16 Qwen GDN projection FP32 oracle FAILED",
            "physical_devices()",
        ):
            self.assertIn(item, source)
        self.assertNotIn("hipMemcpyPeer", source)
        self.assertNotIn("hipDeviceEnablePeerAccess", source)

    def test_external_reference_tampering_fails_before_gpu_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            d = pathlib.Path(temp)
            weights = d / "weights.json"
            outputs = d / "expected.json"
            weights.write_text(json.dumps({"schema": "not_original"}))
            outputs.write_text(json.dumps({"schema": "not_verified"}))
            with self.assertRaises(ValueError):
                gpu.validate(weights, outputs)
            self.assertEqual(ref.sha(weights), hashlib.sha256(weights.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
