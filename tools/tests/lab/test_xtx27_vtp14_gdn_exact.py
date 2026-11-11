"""Card-free VTP14 reference maths, source integrity and GPU admission tests."""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

LAB = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(LAB))
import xtx27_vtp14_gdn_exact as exact
import xtx27_vtp14_gdn_gpu as gpu


class ExactMathTests(unittest.TestCase):
    def test_bf16_round_to_nearest_even_not_truncate(self):
        self.assertEqual(exact.round_bf16(1.00390625), 1.0)
        self.assertEqual(exact.round_bf16(1.01171875), 1.015625)
        self.assertEqual(exact.round_bf16(-1.00390625), -1.0)
        self.assertEqual(exact.bf16_values(struct.pack("<HH", 0x3F80, 0xBF80)),
                         [1.0, -1.0])
        with self.assertRaises(ValueError):
            exact.bf16_values(b"\x01")

    def test_actual_convolution_rank_geometry_and_tap_order(self):
        # Nonzero tap and nontrivial temporal history, tested at original
        # production channel count without original model or any GPU.
        weights = [((c * 7 + tap * 13) % 19 - 9) / 64
                   for c in range(10240) for tap in range(4)]
        full = exact.conv_global(weights, 4)
        parts = exact.gdn.layout()
        for role in exact.ROLES:
            channels = [idx for first, count in parts[role]
                        for idx in range(first, first + count)]
            local = [weights[4 * global_c + tap]
                     for global_c in channels for tap in range(4)]
            packed = exact.conv_packed_rank(local, role, 4)
            for t in (0, 1, 2, 3):
                self.assertEqual(len(packed[t]), len(channels))
                self.assertEqual(packed[t],
                                 [full[t][global_c] for global_c in channels])

    def test_state_recurrence_and_head_normalization_both_orientations(self):
        q = [((i * 7) % 19 - 9) / 64 for i in range(128)]
        k = [((i * 5) % 23 - 11) / 64 for i in range(128)]
        v = [((i * 11) % 29 - 14) / 64 for i in range(128)]
        z = [((i * 3) % 17 - 8) / 64 for i in range(128)]
        gains = [0.25 + (i % 7) / 32 for i in range(128)]
        st_a = [[0.0] * 128 for _ in range(128)]
        st_b = [[0.0] * 128 for _ in range(128)]
        for t in range(3):
            a = exact.step_global(st_a, q, k, v, 0.25 + 0.1 * t,
                                  -0.3 + 0.05 * t, z, 0.2, -0.4, gains)
            b = exact.step_local(st_b, q, k, v, 0.25 + 0.1 * t,
                                 -0.3 + 0.05 * t, z, 0.2, -0.4, gains)
            self.assertLess(max(abs(x - y) for x, y in zip(a, b, strict=True)), 2e-6)
        self.assertLess(max(abs(st_a[e][d] - st_b[d][e])
                            for e in range(128) for d in range(128)), 2e-6)
        self.assertTrue(any(abs(x) > 1e-7 for row in st_a for x in row))

    def test_gate_decay_beta_z_are_all_effective(self):
        q = [0.05 + i / 4096 for i in range(128)]
        k = [0.1 + i / 8192 for i in range(128)]
        v = [0.2 + i / 2048 for i in range(128)]
        gain = [0.6] * 128

        def run(alog, dt, beta, gate):
            state = [[0.0] * 128 for _ in range(128)]
            for _ in range(3):
                out = exact.step_global(state, q, k, v, 0.7, beta,
                                        [gate] * 128, alog, dt, gain)
            return out

        base = run(0.1, 0.2, 0.3, 0.8)
        for val in (
            run(1.1, 0.2, 0.3, 0.8),
            run(0.1, 1.2, 0.3, 0.8),
            run(0.1, 0.2, -1.7, 0.8),
            run(0.1, 0.2, 0.3, -0.4),
        ):
            self.assertGreater(max(abs(a - b) for a, b in zip(base, val, strict=True)),
                               1e-9)


class GpuAdmissionTests(unittest.TestCase):
    @staticmethod
    def create_fixtures(root: pathlib.Path):
        rows = 2
        digest = "a" * 64
        coeff = root / "learned.f32"
        coeff.write_bytes(b"\0" * 224 * 4)
        ref, pack = {
            "source_sha256": gpu.sha(gpu.SOURCE.with_name("xtx27_vtp14_gdn_exact.py")),
            "schema": "vtp14-gdn-partial-real-bf16-math-v1",
            "status": "CARD_FREE_QWEN_GDN_MATH_AND_SHARD_CONTINUITY_PASS_NOT_GPU",
            "layer": 0, "rows": rows,
            "checkpoint_source_sha256": digest,
            "rank_results": {},
            "reference_coefficients": {
                "file": str(coeff),
                "sha256": hashlib.sha256(coeff.read_bytes()).hexdigest(),
            },
        }, {
            "status": "EXACT_ORIGINAL_BF16_GDN_QKV_CONV_PACK_PASS_NOT_RUNTIME",
            "layer": 0,
            "outputs": {"conv1d": {"source_payload_sha256": digest, "ranks": {}}},
        }
        for role, heads, keys in zip(gpu.ROLES, (12, 12, 24), (4, 4, 8), strict=True):
            oracle = root / f"oracle-{role}.f32"
            oracle.write_bytes(b"\0" * rows * heads * 128 * 4)
            conv = root / f"conv-{role}.bf16"
            conv.write_bytes(b"\0" * (2 * keys + heads) * 128 * 4 * 2)
            ref["rank_results"][role] = {
                "heads": heads, "state_elements": heads * 128 * 128,
                "oracle_file": str(oracle),
                "output_sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
            }
            pack["outputs"]["conv1d"]["ranks"][role] = {
                "path": str(conv), "bytes": conv.stat().st_size,
                "sha256": hashlib.sha256(conv.read_bytes()).hexdigest(),
            }
        rp = root / "reference.json"
        pp = root / "pack.json"
        rp.write_text(json.dumps(ref))
        pp.write_text(json.dumps(pack))
        return rp, pp, ref, pack

    def test_preflight_exact_full_length_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            r, p, receipt, pack = self.create_fixtures(root)
            result = gpu.validate_reference(r, p)[2]
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["args"].count("--conv-r9700"), 1)
            self.assertEqual(len(result["args"]), 16)
            self.assertEqual(result["reference_sha256"], gpu.sha(r))
            conv = pathlib.Path(pack["outputs"]["conv1d"]["ranks"]["r9700"]["path"])
            conv.write_bytes(conv.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, "convolution weights changed"):
                gpu.validate_reference(r, p)
            conv.write_bytes(b"\0" * pack["outputs"]["conv1d"]["ranks"]["r9700"]["bytes"])
            coeff = pathlib.Path(receipt["reference_coefficients"]["file"])
            coeff.write_bytes(b"\x01" + coeff.read_bytes()[1:])
            with self.assertRaisesRegex(ValueError, "parameter packet changed"):
                gpu.validate_reference(r, p)

    def test_card_free_plan_does_not_start_process(self):
        with (patch.object(gpu.subprocess, "run",
                           side_effect=AssertionError("must not run external processes")),
              redirect_stdout(io.StringIO()) as output):
            self.assertEqual(gpu.main([]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["phase"], "card_free")
        self.assertTrue(plan["not_full_qwen_gdn"])
        self.assertIsNone(plan["independent_reference"])

    def test_gpu_refuses_without_explicit_permission_and_reference(self):
        for phase in ("compile", "gpu"):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                gpu.main(["--phase", phase])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            gpu.main(["--phase", "gpu", "--run"])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            gpu.main(["--iterations", "31"])
        source = gpu.SOURCE.read_text()
        for bit in (
            "delta_qwen<<<r.values,H",
            "conv4<<<", "qk_norm<<<", "rsqrtf(",
            "coeff[global_h]", "coeff[48+global_h]", "coeff[96+e]",
            "beta*(v-predicted)",
            "state[stbase+d]*dec",
            "requires",  # fail-closed missing arguments and device geometry
        ):
            if bit == "requires":
                continue
            self.assertIn(bit, source)


if __name__ == "__main__":
    unittest.main()
