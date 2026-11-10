"""VTP05 test of legal dense 27B quant kernel shapes; card-free only."""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance" / "xtx27_quant_kernel.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("xtx27_quant_kernel", SOURCE)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class QuantKernelGeometryTests(unittest.TestCase):
    def test_all_candidates_are_shape_legal_and_complete(self):
        expect = {
            "capacity": [5120, 5120, 3584, 3584],
            "intermediate": [6144, 6144, 2560, 2560],
            "bandwidth": [6656, 6656, 2048, 2048],
        }
        for name, widths in expect.items():
            with self.subTest(candidate=name):
                g = mod.geometry(name, "xtx0", "gate_up")
                self.assertEqual(g["intermediate_shards"], widths)
                self.assertEqual(sum(widths), mod.INTERMEDIATE)
                self.assertEqual(g["N"], 2 * widths[0])
                self.assertEqual(g["K"], mod.HIDDEN)
                self.assertEqual(mod.geometry(name, "xtx1", "down")["K"], widths[1])
                self.assertEqual(
                    mod.geometry(name, "r9700", "gate_up")["N"],
                    2 * (widths[2] + widths[3]),
                )
                self.assertEqual(
                    mod.geometry(name, "r9700", "down")["K"],
                    widths[2] + widths[3],
                )
                for role in ("xtx0", "xtx1", "r9700"):
                    for projection in ("gate_up", "down"):
                        geo = mod.geometry(name, role, projection)
                        self.assertEqual(geo["N"] % 256, 0)
                        self.assertEqual(geo["K"] % 256, 0)

    def test_full_and_logical_r9700_both_legal(self):
        self.assertEqual(mod.geometry("capacity", "r9700", "gate_up", "full")["N"], 34816)
        self.assertEqual(mod.geometry("capacity", "r9700", "down", "logical2")["K"], 3584)
        self.assertEqual(mod.geometry("capacity", "r9700", "down", "logical3")["K"], 3584)
        with self.assertRaises(ValueError):
            mod.geometry("capacity", "xtx1", "down", "logical2")

    def test_device_discovery_fails_closed_on_bad_counts(self):
        example = (
            "GPU[0] : GFX Version: gfx1100\nGPU[1] : GFX Version: gfx1100\n"
            "GPU[2] : GFX Version: gfx1201\nGPU[3] : GFX Version: gfx1030\n"
        )
        self.assertEqual(mod.parse_devices(example), {"xtx0": 0, "xtx1": 1, "r9700": 2})
        with self.assertRaises(ValueError):
            mod.parse_devices(example.replace("gfx1201", "gfx1100"))
        with self.assertRaises(ValueError):
            mod.parse_devices("GPU[0] : GFX Version: gfx1100")

    def test_quant_probes_select_only_matching_real_kernel_rows(self):
        self.assertEqual(mod.selected_row("iq6r", 1), "iqr k6 m64")
        self.assertEqual(mod.selected_row("iq6r", 128), "iqr k6 tiled")
        self.assertEqual(mod.selected_row("w8a8", 4), "gemm_i8a8_nt_m16")
        self.assertEqual(mod.selected_row("w8a8", 512), "gemm_i8a8_tiled")
        with self.assertRaises(ValueError):
            mod.selected_row("q6_k", 4)
        text = "  gemm_fp8a8_tiled 123.5 us\n  gemm_i8a8_tiled 289.2 us   17 TFLOP/s\n"
        self.assertEqual(mod.extract_timing(text, quant="w8a8", m=128), 289.2)
        with self.assertRaises(ValueError):
            mod.extract_timing(text, quant="w8a8", m=4)
        self.assertEqual(mod.extract_timing("  iqr k6 m64 51.40 us  23 GB/s",
                                            quant="iq6r", m=4), 51.4)
        with self.assertRaises(ValueError):
            mod.extract_timing("  iqr k6 m64 row or hooks absent", quant="iq6r", m=4)
        with self.assertRaises(ValueError):
            mod.extract_timing("  iqr k6 m64 51.4 us\nFAIL kernel",
                               quant="iq6r", m=4)

    def test_command_is_bounded_and_knows_quant_encoding(self):
        geo = mod.geometry("capacity", "r9700", "gate_up")
        command = mod.command(pathlib.Path("test"), pathlib.Path("plugin"),
                              quant="iq6r", geo=geo, m=4, iters=3, cold_mib=256)
        self.assertEqual(command[2:8], ["--perfiqr", "14336", "5120", "4", "6", "3"])
        self.assertEqual(command[-2:], ["--cold", "256"])
        w8 = mod.command(pathlib.Path("test"), pathlib.Path("plugin"),
                         quant="w8a8", geo=geo, m=4, iters=3, cold_mib=0)
        self.assertEqual(w8[2:7], ["--perfw8", "14336", "5120", "4", "3"])
        with self.assertRaises(ValueError):
            mod.command(pathlib.Path("test"), pathlib.Path("plugin"),
                        quant="w8a8", geo=geo, m=3, iters=3, cold_mib=0)
        with self.assertRaises(ValueError):
            mod.command(pathlib.Path("test"), pathlib.Path("plugin"),
                        quant="iq6r", geo=geo, m=4, iters=100, cold_mib=0)

    def test_plan_never_executes_subprocess_or_checks_build(self):
        buffer = io.StringIO()
        with (
            patch.object(mod.subprocess, "run", side_effect=AssertionError("external process invoked")),
            redirect_stdout(buffer),
        ):
            self.assertEqual(mod.main(["--phase", "plan"]), 0)
        result = json.loads(buffer.getvalue())
        self.assertEqual(result["status"], "CARD_FREE_PLAN_NO_GPU")
        self.assertTrue(result["timing_only"])

    def test_run_requires_opt_in_and_explicit_existing_build(self):
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as stopped:
            mod.main(["--phase", "run"])
        self.assertEqual(stopped.exception.code, 2)
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as stopped:
            mod.main(["--phase", "run", "--run", "--build-dir", "/nonexistent"])
        self.assertEqual(stopped.exception.code, 2)

    def test_gpu_internal_entry_requires_real_lease_environment(self):
        import argparse

        args = argparse.Namespace(_under_lease=True)
        with patch.dict(mod.os.environ, {}, clear=True), self.assertRaises(ValueError):
            mod.execute_one(args, mod.geometry("capacity", "xtx0", "gate_up"))


if __name__ == "__main__":
    unittest.main()
