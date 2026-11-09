"""XDP04: card-free quant/packing, oracle, and GPU lease-control tests."""

from __future__ import annotations

import contextlib
import hashlib
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
            mock.patch.dict(
                w4.os.environ, {"RAD_HIPCC": "/selected/rocm-7.2/bin/hipcc"}
            ),
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

    def test_wide8_output_and_skewed_stage_ownership_card_free(self):
        """Eight-wave M32xN64 tile covers 128x128 exactly once."""
        seen = set()
        for wave in range(8):
            wm_pair, wn0 = wave >> 1, (wave & 1) * 4
            for lane in range(32):
                for mi in range(2):
                    m = 32 * wm_pair + 16 * mi + (lane & 15)
                    for c in range(4):
                        for e in range(8):
                            n = 16 * (wn0 + c) + 2 * e + (lane >> 4)
                            self.assertTrue(0 <= m < 128 and 0 <= n < 128)
                            self.assertNotIn((m, n), seen)
                            seen.add((m, n))
        self.assertEqual(len(seen), 16384)

        for slice_id in range(4):
            staged = set()
            for wave in range(8):
                wp, wn0 = wave >> 1, (wave & 1) * 4
                if wp != slice_id:
                    continue
                for lane in range(32):
                    for mi in range(2):
                        lm = 16 * mi + (lane & 15)
                        for c in range(4):
                            for e in range(8):
                                ln = 16 * (wn0 + c) + 2 * e + (lane >> 4)
                                at = 129 * lm + ln
                                self.assertNotIn(at, staged)
                                staged.add(at)
            self.assertEqual(len(staged), 4096)
            consumed = {
                129 * (t // 128) + t % 128
                for tid in range(256)
                for t in range(tid, 4096, 256)
            }
            self.assertEqual(staged, consumed)

    def test_wide8_source_native_opcode_and_no_model_route(self):
        code = w4.WIDE8_SOURCE.read_text()
        self.assertIn("template <bool COALESCED>", code)
        self.assertIn("__launch_bounds__(256)", code)
        self.assertIn("float sums[2][4][8]", code)
        self.assertIn("__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32", code)
        self.assertIn("XDP06_WIDE8_SUITE_PASS cases=24", code)
        self.assertIn("XDP06_WIDE8_PAIRED_PASS modes=2", code)
        self.assertNotIn("gemm_mxfp4", code)

    def test_wide8_plan_is_distinct_and_single_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            for store in ("direct", "coalesced", "paired"):
                with (
                    self.subTest(store=store),
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
                                "wide8",
                                "--shape-27b",
                                "--wide-store",
                                store,
                                "--output-dir",
                                temp,
                            ]
                        ),
                        0,
                    )
                rec = json.loads(stdout.getvalue())
                self.assertEqual(rec["source_sha256"], w4.sha256_file(w4.WIDE8_SOURCE))
                self.assertEqual(rec["wave32_per_block"], 8)
                self.assertEqual(rec["wide_store"], store)
                self.assertEqual(rec["commands"]["gpu"][-2:], ["--shape-27b", store])
                self.assertEqual(
                    rec["commands"]["gpu"][0:3], ["agent-lease", "acquire", "gpu"]
                )
                self.assertTrue(rec["commands"]["binary"].endswith("-wide8"))
            self.assertNotEqual(
                w4.binaries(Path(temp), "wide8"),
                w4.binaries(Path(temp), "wide"),
            )
            self.assertFalse(Path(temp, "w4a4-wide8-manifest.json").exists())

    def test_wide16_output_ownership_card_free(self):
        """Each of 16 wave32s owns 32 distinct f32 results per lane."""
        written = set()
        for wave in range(16):
            wm, wn0 = wave >> 1, (wave & 1) * 4
            for lane in range(32):
                m = 16 * wm + (lane & 15)
                for c in range(4):
                    for e in range(8):
                        n = 16 * (wn0 + c) + 2 * e + (lane >> 4)
                        self.assertTrue(0 <= m < 128 and 0 <= n < 128)
                        self.assertNotIn((m, n), written)
                        written.add((m, n))
        self.assertEqual(len(written), 128 * 128)

    def test_wide16_coalesced_stage_and_global_store_ownership(self):
        # Model exactly one 32x128 skewed-LDS tile per slice; 4 waves
        # publish, all 16 waves read 8 contiguous columns per workitem.
        seen_global = set()
        for slice_id in range(4):
            written_lds = set()
            for wave in range(16):
                wm, wn0 = wave >> 1, (wave & 1) * 4
                if wm >> 1 != slice_id:
                    continue
                for lane in range(32):
                    local_m = 16 * (wm - 2 * slice_id) + (lane & 15)
                    for c in range(4):
                        for e in range(8):
                            local_n = 16 * (wn0 + c) + 2 * e + (lane >> 4)
                            at = local_m * 129 + local_n
                            self.assertNotIn(at, written_lds)
                            written_lds.add(at)
            self.assertEqual(len(written_lds), 4096)
            for tid in range(512):
                for offset in range(0, 4096, 512):
                    t = tid + offset
                    local_m, local_n = divmod(t, 128)
                    self.assertIn(local_m * 129 + local_n, written_lds)
                    global_cell = (32 * slice_id + local_m, local_n)
                    self.assertNotIn(global_cell, seen_global)
                    seen_global.add(global_cell)
        self.assertEqual(len(seen_global), 128 * 128)

    def test_wide16_coalesced_plan_arg_and_reject_wrong_variants(self):
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
                            "wide",
                            "--shape-27b",
                            "--wide-store",
                            "coalesced",
                            "--output-dir",
                            temp,
                        ]
                    ),
                    0,
                )
            record = json.loads(stdout.getvalue())
            self.assertEqual(record["wide_store"], "coalesced")
            self.assertEqual(
                record["commands"]["gpu"][-2:], ["--shape-27b", "coalesced"]
            )
            for flags in (
                ["--kernel", "tiled", "--wide-store", "coalesced"],
                ["--kernel", "oracle", "--wide-store", "coalesced"],
                ["--kernel", "wide", "--suite", "--wide-store", "coalesced"],
            ):
                with (
                    self.subTest(flags=flags),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    w4.main(["--output-dir", temp, *flags])

    def test_wide16_plan_is_separate_fail_closed_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            for flags in (["--suite"], ["--shape-27b"], []):
                with (
                    self.subTest(flags=flags),
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
                                "wide",
                                "--output-dir",
                                temp,
                                *flags,
                            ]
                        ),
                        0,
                    )
                rec = json.loads(stdout.getvalue())
                self.assertEqual(rec["source_sha256"], w4.sha256_file(w4.WIDE_SOURCE))
                self.assertTrue(
                    any(
                        x.endswith("xtx27_w4a4_wide16.hip")
                        for x in rec["commands"]["compiler"]
                    )
                )
                self.assertTrue(rec["commands"]["binary"].endswith("-wide16"))
                self.assertEqual(rec["kernel"], "wide")
                self.assertEqual(rec["output_tile"], [128, 128])
                self.assertEqual(rec["wave32_per_block"], 16)
                self.assertEqual(
                    rec["commands"]["gpu"][0:3], ["agent-lease", "acquire", "gpu"]
                )
                self.assertTrue(
                    any(x.startswith("XDP06 wide16") for x in rec["commands"]["gpu"])
                )
                self.assertEqual(
                    rec["commands"]["gpu"][-1] if flags else "",
                    ("direct" if flags == ["--shape-27b"] else flags[-1])
                    if flags
                    else "",
                )
            self.assertFalse(Path(temp, "w4a4-wide16-manifest.json").exists())

    def test_wide16_paired_full_shape_uses_one_lease(self):
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
                            "wide",
                            "--shape-27b",
                            "--wide-store",
                            "paired",
                            "--output-dir",
                            temp,
                        ]
                    ),
                    0,
                )
            rec = json.loads(stdout.getvalue())
            self.assertEqual(rec["commands"]["gpu"][-2:], ["--shape-27b", "paired"])
            self.assertEqual(rec["commands"]["gpu"].count("agent-lease"), 1)
            self.assertEqual(rec["wide_store"], "paired")
            self.assertFalse(Path(temp, "w4a4-wide16-manifest.json").exists())
            for flags in (
                ["--kernel", "wide", "--wide-store", "paired"],
                ["--kernel", "wide", "--suite", "--wide-store", "paired"],
                ["--kernel", "tiled", "--shape-27b", "--wide-store", "paired"],
            ):
                with (
                    self.subTest(flags=flags),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    w4.main(["--output-dir", temp, *flags])

    def test_wide16_reject_unsupported_flags_and_avoids_other_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            for flags in (
                ["--phase", "gpu", "--kernel", "wide", "--tile-m", "8"],
                ["--phase", "cpu", "--kernel", "wide", "--shape-27b"],
                ["--phase", "gpu", "--kernel", "wide", "--shape-27b", "--suite"],
                ["--phase", "gpu", "--kernel", "oracle", "--shape-27b"],
            ):
                with (
                    self.subTest(flags=flags),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    w4.main(["--output-dir", temp, *flags])
        self.assertNotEqual(
            w4.binaries(Path("/tmp"), "wide"),
            w4.binaries(Path("/tmp"), "tiled"),
        )
        self.assertNotEqual(
            w4.binaries(Path("/tmp"), "wide"),
            w4.binaries(Path("/tmp"), "oracle"),
        )

    def test_wide16_group_major_meta_repacking_exact_card_free(self):
        # Unlike the regular row-major [row,G] metadata, the new physical
        # sidecar groups 128 adjacent rows for each g. Its only extra
        # storage is right-padding on incomplete M/N tiles.
        for rows, groups in ((1, 1), (17, 3), (129, 5), (257, 40)):
            with self.subTest(rows=rows, groups=groups):
                src = [row * 1000 + g for row in range(rows) for g in range(groups)]
                packed = [0] * (((rows + 127) // 128) * groups * 128)
                locations = set()
                for row in range(rows):
                    for g in range(groups):
                        dst = ((row // 128) * groups + g) * 128 + row % 128
                        self.assertNotIn(dst, locations)
                        locations.add(dst)
                        packed[dst] = src[row * groups + g]
                for row in range(rows):
                    for g in range(groups):
                        dst = ((row // 128) * groups + g) * 128 + row % 128
                        self.assertEqual(src[row * groups + g], packed[dst])
                self.assertEqual(len(locations), rows * groups)
        code = w4.WIDE_SOURCE.read_text()
        for phrase in (
            "#if defined(XDP06_META_TILE_MAJOR)",
            "meta_group_offset(ma, g, G)",
            "meta_group_offset(nw, g, G)",
            "pack_group_headers(h.Sa, h.M, h.G)",
            "pack_group_headers(h.Sw, h.N, h.G)",
            "pack_group_headers(h.Asum, h.M, h.G)",
            "pack_group_headers(h.Wzero, h.N, h.G)",
        ):
            self.assertIn(phrase, code)
        self.assertIn(
            "#define XDP06_META_TILE_MAJOR 1",
            w4.WIDE16METATILE_SOURCE.read_text(),
        )

    def test_wide16_header_staging_cohorts_and_source(self):
        # Four nonoverlapping 128-thread cohorts load exactly 128 A scales,
        # 128 A sums, 128 W scales and 128 W zeros per K128 group.
        cohorts = {(t >> 7, t & 127) for t in range(512)}
        self.assertEqual(len(cohorts), 4 * 128)
        for group in range(4):
            self.assertEqual({row for g, row in cohorts if g == group}, set(range(128)))
        source = w4.WIDE_SOURCE.read_text()
        for code in (
            "#if defined(XDP06_STAGE_META)",
            "tile_sa[local]",
            "tile_asum[local]",
            "tile_sw[local]",
            "tile_wz[local]",
            "const int adjusted = acc[e] - tile_wz[ln] * tile_asum[lm]",
            "__builtin_amdgcn_sched_barrier(0)",
        ):
            self.assertIn(code, source)
        self.assertEqual(
            w4.WIDE16META_SOURCE.read_text().count("#define XDP06_STAGE_META 1"),
            1,
        )

    def test_wide16_wrappers_track_included_source_in_binary_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "base.hip"
            source.write_text("old-base")
            for kernel in ("wide16sched", "wide16meta", "wide16metatile"):
                wrapper = Path(temp) / (kernel + ".hip")
                wrapper.write_text("wrapper " + kernel)
                field = (
                    "WIDE16SCHED_SOURCE"
                    if kernel == "wide16sched"
                    else "WIDE16META_SOURCE"
                    if kernel == "wide16meta"
                    else "WIDE16METATILE_SOURCE"
                )
                with (
                    self.subTest(kernel=kernel),
                    mock.patch.object(w4, "WIDE_SOURCE", source),
                    mock.patch.object(w4, field, wrapper),
                ):
                    first = w4.source_digest(kernel)
                    expected = hashlib.sha256(
                        wrapper.read_bytes() + bytes([0]) + source.read_bytes()
                    ).hexdigest()
                    self.assertEqual(first, expected)
                    source.write_text("new-base")
                    self.assertNotEqual(first, w4.source_digest(kernel))
                    source.write_text("old-base")
                    wrapper.write_text("changed-wrapper")
                    self.assertNotEqual(first, w4.source_digest(kernel))
                    wrapper.write_text("wrapper " + kernel)

    def test_wide16_wrapped_variants_one_lease_and_isolated_builds(self):
        with tempfile.TemporaryDirectory() as temp:
            variants = ("wide16sched", "wide16meta", "wide16metatile")
            for kernel in variants:
                for store in ("direct", "paired"):
                    with (
                        self.subTest(kernel=kernel, store=store),
                        mock.patch.object(
                            w4.subprocess,
                            "run",
                            side_effect=AssertionError("subprocess invoked"),
                        ),
                        contextlib.redirect_stdout(io.StringIO()) as out,
                    ):
                        self.assertEqual(
                            w4.main(
                                [
                                    "--phase",
                                    "plan",
                                    "--kernel",
                                    kernel,
                                    "--shape-27b",
                                    "--wide-store",
                                    store,
                                    "--output-dir",
                                    temp,
                                ]
                            ),
                            0,
                        )
                    report = json.loads(out.getvalue())
                    self.assertEqual(report["source_sha256"], w4.source_digest(kernel))
                    self.assertEqual(
                        report["commands"]["gpu"][0:3],
                        ["agent-lease", "acquire", "gpu"],
                    )
                    self.assertEqual(
                        report["commands"]["gpu"][-2:], ["--shape-27b", store]
                    )
                    self.assertIn(kernel, report["commands"]["binary"])
                    self.assertEqual(report["wave32_per_block"], 16)
                    self.assertFalse(Path(report["commands"]["manifest"]).exists())
            self.assertNotEqual(
                w4.binaries(Path(temp), "wide16meta"),
                w4.binaries(Path(temp), "wide16sched"),
            )

    def test_wide16_kernel_explicit_native_wmma_and_two_barriers(self):
        code = w4.WIDE_SOURCE.read_text()
        self.assertIn("template <bool COALESCED>", code)
        self.assertIn("__launch_bounds__(512)", code)
        self.assertIn("constexpr int OUT_STRIDE = 129", code)
        self.assertIn("if constexpr (COALESCED)", code)
        self.assertIn("float sums[4][8]", code)
        self.assertIn(
            "__shared__ unsigned stage[COALESCED ? OUT_STAGE_WORDS : STAGE_WORDS]", code
        )
        self.assertIn("__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32", code)
        self.assertIn("XDP06_WIDE_PASS", code)
        self.assertIn("XDP06_WIDE_SUITE_PASS cases=24", code)
        self.assertIn("XDP06_WIDE_PAIRED_PASS modes=2", code)
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
