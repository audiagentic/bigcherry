"""XDP04: card-free quant/packing, oracle, and GPU lease-control tests."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lab.radiance import xtx27_w4a4 as w4


def args(*items: str) -> list[str]:
    return list(items)


class W4A4ProbeTests(unittest.TestCase):
    def test_signed_nibbles_roundtrip_and_boundaries(self):
        raw = [-8, -7, -1, 0, 1, 7]
        encoded = w4.pack_nibbles(raw, signed=True)
        self.assertEqual(encoded, bytes((0x98, 0x0F, 0x71)))
        self.assertEqual(w4.unpack_nibbles(encoded, signed=True), raw)
        self.assertEqual(w4.unpack_nibbles(encoded, signed=False), [8, 9, 15, 0, 1, 7])
        for bad in ([8, 0], [-9, 0], [0, 1, 2], [True, 0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                w4.pack_nibbles(bad, signed=True)

    def test_unsigned_nibbles_roundtrip(self):
        raw = [0, 15, 8, 7, 5, 4]
        self.assertEqual(
            w4.unpack_nibbles(w4.pack_nibbles(raw, signed=False), signed=False), raw
        )
        with self.assertRaises(ValueError):
            w4.pack_nibbles([-1, 0], signed=False)
        with self.assertRaises(ValueError):
            w4.pack_nibbles([0, 16], signed=False)

    def test_activation_quantization_zero_groups_and_saturation(self):
        row = [0.0] * w4.GROUP + [1.0] * (w4.GROUP - 1) + [-1.0]
        codes, scales, sums = w4.quant_activation(row, bits=4)
        self.assertEqual(len(scales), 2)
        self.assertEqual(scales[0], 1e-12)
        self.assertEqual(sums[0], 0)
        self.assertEqual(codes[-1], -7)
        self.assertEqual(codes[-2], 7)
        self.assertEqual(sums[1], (w4.GROUP - 2) * 7)
        self.assertEqual(
            w4.unpack_nibbles(w4.pack_nibbles(codes, signed=True), signed=True), codes
        )
        c8, _, _ = w4.quant_activation(row, bits=8)
        self.assertEqual(c8[-1], -127)
        self.assertEqual(c8[-2], 127)

    def test_weight_quantization_offsets_and_signs(self):
        row = [float(i % 17 - 8) for i in range(128)]
        codes, scales, zeros = w4.quant_weight(row)
        self.assertGreater(zeros[0], 0)
        self.assertLess(zeros[0], 15)
        self.assertEqual(len(codes), 128)
        self.assertTrue(all(0 <= v <= 15 for v in codes))
        self.assertEqual(
            w4.unpack_nibbles(w4.pack_nibbles(codes, signed=False), signed=False), codes
        )
        _, _, zp = w4.quant_weight([1.0] * 128)
        _, _, zn = w4.quant_weight([-1.0] * 128)
        self.assertEqual(zp[0], 0)
        self.assertEqual(zn[0], 15)
        self.assertTrue(math.isfinite(scales[0]))

    def test_group128_zero_point_correction_matches_float_reference(self):
        # Signed q4 +/-1 and unsigned q4 codes for W=-7,+8 are exact.
        # Nonzero sum(A) makes the unsigned weight zero correction essential.
        a = [[-1.0 if i % 3 == 0 else 1.0 for i in range(256)]]
        w = [[8.0 if i % 2 else -7.0 for i in range(256)]]
        want = w4.float_matmul(a, w)[0][0]
        self.assertNotEqual(sum(a[0]), 0.0)
        self.assertNotEqual(want, 0.0)
        for bits in (4, 8):
            with self.subTest(bits=bits):
                got = w4.quant_matmul(a, w, activation_bits=bits)[0][0]
                self.assertAlmostEqual(got, want, places=5)

    def test_zero_activation_gives_exact_zero_after_correction(self):
        a = [[0.0] * 256]
        w = [[float((i % 19) - 8) for i in range(256)]]
        self.assertEqual(w4.quant_matmul(a, w, activation_bits=4), [[0.0]])

    def test_reject_non_finite_and_partial_groups(self):
        with self.assertRaises(ValueError):
            w4.quant_activation([1.0] * 127)
        with self.assertRaises(ValueError):
            w4.quant_activation([float("nan")] * 128)
        with self.assertRaises(ValueError):
            w4.quant_weight([float("inf")] * 128)
        with self.assertRaises(ValueError):
            w4.quant_activation([1.0] * 128, bits=3)
        with self.assertRaises(ValueError):
            w4.quant_matmul([[1.0] * 128], [[0.0] * 256], activation_bits=4)

    def test_cpu_normal_and_outlier_probes_report_numeric_error(self):
        for dist in ("normal", "outlier"):
            with self.subTest(dist=dist):
                got = w4.cpu_probe(m=6, n=13, k=256, seed=19, distribution=dist)
                self.assertEqual(got["status"], "SYNTHETIC_NUMERICS_ONLY")
                self.assertGreater(got["w4a4_vs_float"]["relative_l2"], 0)
                self.assertGreater(got["w4a8_vs_float"]["relative_l2"], 0)
                self.assertEqual(
                    got, w4.cpu_probe(m=6, n=13, k=256, seed=19, distribution=dist)
                )

    def test_cpu_excessive_work_rejected(self):
        for shape in ((0, 16, 128), (257, 16, 128), (16, 16, 129), (256, 512, 8192)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                w4.cpu_probe(
                    m=shape[0], n=shape[1], k=shape[2], seed=0, distribution="normal"
                )

    def test_plan_uses_selected_rocm_compiler_without_build(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.dict(w4.os.environ, {"RAD_HIPCC": "/selected/rocm-7.2/bin/hipcc"}),
            mock.patch.object(
                w4.subprocess, "run", side_effect=AssertionError("subprocess invoked")
            ),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(w4.main(["--phase", "plan", "--output-dir", temp]), 0)
        record = json.loads(stdout.getvalue())
        self.assertEqual(
            record["commands"]["compiler"][0], "/selected/rocm-7.2/bin/hipcc"
        )

    def test_plan_mode_never_builds_or_acquires_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "scratch"
            with (
                mock.patch.object(
                    w4.subprocess,
                    "run",
                    side_effect=AssertionError("subprocess invoked"),
                ),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(
                    w4.main(
                        args(
                            "--phase",
                            "plan",
                            "--output-dir",
                            str(out),
                        )
                    ),
                    0,
                )
            self.assertFalse(out.exists())
            record = json.loads(stdout.getvalue())
            self.assertEqual(record["schema"], "xtx27.w4a4.execution.v1")
            self.assertEqual(
                record["commands"]["gpu"][0:3], ["agent-lease", "acquire", "gpu"]
            )
            self.assertIn("--offload-arch=gfx1100", record["commands"]["compiler"])
            self.assertIn("xtx27_w4a4_probe.hip", record["commands"]["compiler"][-3])

    def test_cpu_mode_never_uses_gpu_or_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "scratch"
            with (
                mock.patch.object(
                    w4.subprocess,
                    "run",
                    side_effect=AssertionError("subprocess invoked"),
                ),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(
                    w4.main(
                        args(
                            "--phase",
                            "cpu",
                            "--m",
                            "2",
                            "--n",
                            "8",
                            "--k",
                            "128",
                            "--output-dir",
                            str(out),
                            "--distribution",
                            "outlier",
                        )
                    ),
                    0,
                )
            self.assertFalse(out.exists())
            self.assertEqual(
                json.loads(stdout.getvalue())["numeric"]["distribution"], "outlier"
            )

    def test_gpu_and_build_need_opt_in_and_scratch_outside_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            for phase in ("compile", "gpu"):
                with (
                    self.subTest(phase=phase),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    w4.main(args("--phase", phase, "--output-dir", temp))
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                w4.main(
                    args("--phase", "plan", "--output-dir", str(w4.ROOT / "scratch"))
                )
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                w4.main(
                    args("--phase", "gpu", "--run", "--m", "1", "--output-dir", temp)
                )

    def test_six_case_suite_uses_one_lease_command(self):
        with tempfile.TemporaryDirectory() as temp:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(
                    w4.main(
                        args(
                            "--phase",
                            "plan",
                            "--suite",
                            "--output-dir",
                            temp,
                        )
                    ),
                    0,
                )
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["commands"]["gpu"].count("agent-lease"), 1)
            self.assertEqual(plan["commands"]["gpu"][-1], "--suite")
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                w4.main(args("--phase", "cpu", "--suite", "--output-dir", temp))

    def test_tiled_plan_source_hash_and_separate_binary(self):
        with tempfile.TemporaryDirectory() as temp:
            with (
                mock.patch.object(
                    w4.subprocess,
                    "run",
                    side_effect=AssertionError("subprocess invoked"),
                ),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(
                    w4.main(
                        [
                            "--phase",
                            "plan",
                            "--kernel",
                            "tiled",
                            "--suite",
                            "--output-dir",
                            temp,
                        ]
                    ),
                    0,
                )
            record = json.loads(stdout.getvalue())
            self.assertEqual(record["source_sha256"], w4.sha256_file(w4.TILED_SOURCE))
            self.assertEqual(record["kernel"], "tiled")
            self.assertEqual(record["commands"]["gpu"][-1], "--suite")
            self.assertEqual(record["commands"]["gpu"].count("agent-lease"), 1)
            self.assertIn("tiled", record["commands"]["binary"])
            self.assertIn(str(w4.TILED_SOURCE), record["commands"]["compiler"])
            self.assertFalse((Path(temp) / "w4a4-tiled-manifest.json").exists())

    def test_tiled_27b_explicit_geometry_in_one_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            for tm in (2, 4, 8):
                with (
                    self.subTest(tile_m=tm),
                    contextlib.redirect_stdout(io.StringIO()) as stdout,
                ):
                    self.assertEqual(
                        w4.main(
                            [
                                "--phase",
                                "plan",
                                "--kernel",
                                "tiled",
                                "--shape-27b",
                                "--tile-m",
                                str(tm),
                                "--output-dir",
                                temp,
                            ]
                        ),
                        0,
                    )
                rec = json.loads(stdout.getvalue())
                self.assertEqual(rec["commands"]["gpu"][-2:], ["--shape-27b", str(tm)])
                self.assertEqual(rec["tile_m"], tm)

    def test_tiled_reject_shape_27b_on_oracle_and_mixed_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            for flags in (
                ["--kernel", "oracle", "--shape-27b"],
                ["--kernel", "tiled", "--shape-27b", "--suite"],
                ["--kernel", "tiled", "--phase", "cpu", "--shape-27b"],
                ["--kernel", "oracle", "--tile-m", "4"],
            ):
                with (
                    self.subTest(flags=flags),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    w4.main(["--output-dir", temp, *flags])

    def test_tiled_source_layout_and_l2_reuse_are_explicit(self):
        code = w4.TILED_SOURCE.read_text()
        self.assertIn("template <int WM, int WN>", code)
        self.assertIn("WM * WN == 8", code)
        self.assertIn("wm * 256 + step * 32", code)
        self.assertIn("WM * 256 + wn * 256 + step * 32", code)
        self.assertIn("__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32", code)
        self.assertIn("XDP05_TILED_SUITE_PASS cases=27", code)
        self.assertIn("XDP05_TILED_PASS", code)
        self.assertNotIn("gemm_mxfp4", code)

    def test_isolated_hip_contains_real_native_opcode_and_group_fix(self):
        code = w4.SOURCE.read_text()
        self.assertIn("__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32", code)
        self.assertIn("false, wp, true, ap", code)
        self.assertIn("w_zero[n * G + g] * a_sum[m * G + g]", code)
        self.assertIn("XDP04_GPU_PASS", code)
        self.assertNotIn("__builtin_amdgcn_wmma_i32_16x16x16_iu8", code)


if __name__ == "__main__":
    unittest.main()
