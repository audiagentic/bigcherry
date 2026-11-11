"""Card-free VTP16 original BF16 GDN output sharding and residual tests."""
from __future__ import annotations

import io
import json
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

import xtx27_vtp16_gdn_output_oracle as oracle
import xtx27_vtp16_gdn_output_pack as output_pack


def bf16(value: float) -> int:
    return struct.unpack("<H", output_pack.round_to_bf16(value))[0]


class OriginalGdnOutputTests(unittest.TestCase):
    def test_true_unequal_input_column_partition(self):
        self.assertEqual(output_pack.split_columns(), {
            "xtx0": (0, 1536), "xtx1": (1536, 1536), "r9700": (3072, 3072),
        })
        self.assertEqual(sum(output_pack.SPLITS), 6144)
        self.assertEqual(output_pack.SPLITS[2], 2 * output_pack.SPLITS[0])
        self.assertEqual(output_pack.HEADS, (12, 12, 24))
        self.assertEqual(output_pack.K * output_pack.HIDDEN * 2, 62914560)

    def test_bf16_rounding_is_nearest_even(self):
        self.assertEqual(bf16(1.0), 0x3F80)
        self.assertEqual(bf16(-1.0), 0xBF80)
        self.assertEqual(bf16(1.00390625), 0x3F80)
        self.assertEqual(bf16(1.01171875), 0x3F82)

    def test_cpu_full_vs_column_sharded_reference_and_corruption(self):
        oracle.SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=oracle.SCRATCH) as tmp:
            root = pathlib.Path(tmp)
            packet = root / "pack"
            packet.mkdir()
            result = root / "result"
            result.mkdir()
            # Two input rows, three output neurons; 1/2/3-column shards.
            rows, n, k = 2, 3, 6
            widths = (1, 2, 3)
            weights = [
                1., 0.5, -1., 0.25, 2., -0.5,
                0.5, -1., 0.25, 0.5, -1., 2.,
                -0.5, -0.25, 1., 2., 0.25, -0.5,
            ]
            x = [
                1., 0.5, -2., 0.25, 1., 0.5,
                -0.5, 2., 0.5, -1., 0.25, 1.,
            ]
            residual = [0.25, 1., -0.5, 1., -1., 0.75]
            as_bytes = lambda values: struct.pack("<" + "H" * len(values), *map(bf16, values))
            (packet / "out-original.bf16").write_bytes(as_bytes(weights))
            (packet / "gdn-original.bf16").write_bytes(as_bytes(x))
            (packet / "residual.bf16").write_bytes(as_bytes(residual))
            off = 0
            for role, width in zip(output_pack.ROLES, widths, strict=True):
                shard_w = [weights[i * k + off + d] for i in range(n) for d in range(width)]
                shard_x = [x[i * k + off + d] for i in range(rows) for d in range(width)]
                (packet / f"out-{role}.bf16").write_bytes(as_bytes(shard_w))
                (packet / f"gdn-{role}.bf16").write_bytes(as_bytes(shard_x))
                off += width
            binary = root / "cpu-oracle"
            subprocess.run(
                ["g++", "-O2", "-std=c++17", str(oracle.SOURCE), "-o", str(binary)],
                check=True, capture_output=True,
            )
            command = [
                str(binary), "--m", "2", "--n", "3", "--k", "6",
                "--k0", "1", "--k1", "2", "--k2", "3",
                "--package-dir", str(packet), "--residual", str(packet / "residual.bf16"),
                "--output-dir", str(result),
            ]
            passed = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("status=PASS", passed.stdout)
            self.assertIn("bf16_proj_exact=6/6", passed.stdout)
            full = struct.unpack("<6f", (result / "unsharded-original.f32").read_bytes())
            summed = struct.unpack("<6f", (result / "ordered-three-rank.f32").read_bytes())
            for m in range(rows):
                for i in range(n):
                    expected = sum(weights[i * k + j] * x[m * k + j] for j in range(k))
                    self.assertAlmostEqual(full[m * n + i], expected, places=5)
                    self.assertAlmostEqual(summed[m * n + i], expected, places=5)
            self.assertEqual(
                (result / "ordered-three-rank-residual.bf16").read_bytes(),
                (result / "unsharded-residual.bf16").read_bytes(),
            )
            (packet / "out-xtx1.bf16").write_bytes(
                b"\xFF\x7F" + (packet / "out-xtx1.bf16").read_bytes()[2:]
            )
            negative = root / "rejected"
            negative.mkdir()
            corrupted = [*command]
            corrupted[-1] = str(negative)
            failure = subprocess.run(corrupted, check=False, capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("column shards diverge", failure.stderr)

    def test_fail_closed_without_original_model_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = pathlib.Path(temp) / "record.json"
            fake.write_text(json.dumps({"schema": output_pack.SCHEMA, "status": "INCOMPLETE"}))
            with self.assertRaises(ValueError):
                oracle.validate(fake)
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                output_pack.main([
                    "--weights-receipt", str(fake), "--core-receipt", str(fake),
                    "--phase", "pack",
                ])
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                oracle.main(["--packet-receipt", str(fake), "--phase", "run"])

    def test_projected_hidden_is_bf16_and_stage_is_not_engine(self):
        with (patch.object(oracle, "validate", return_value={"rows": 1, "layer": 0}),
              redirect_stdout(io.StringIO()) as capture,
              patch.object(oracle.subprocess, "run",
                           side_effect=AssertionError("no external execution on plan"))):
            self.assertEqual(oracle.main(["--packet-receipt", "not-accessed"]), 0)
        state = json.loads(capture.getvalue())
        self.assertFalse(state["gpu_required"])
        self.assertFalse(state["model_serving"])
        cpp = oracle.SOURCE.read_text()
        self.assertIn("const std::array<int,3> widths{A,B,C}", cpp)
        self.assertIn("reduced+=partial[1][pos]", cpp)
        self.assertIn("reduced+=partial[2][pos]", cpp)
        self.assertIn("final_tp3[pos]=rn(f32(physical_bf16)+res)", cpp)
        self.assertNotIn("hipMemcpyPeer", cpp)


if __name__ == "__main__":
    unittest.main()
