"""Card-free negative/positive trials of the actual VTP05 receipt contract.

Mock the GPU launch and ROCm discovery. NEVER import HIP or run a model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

DIRECTORY = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location("vtp05_execution", DIRECTORY / "xtx27_quant_kernel.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

ROCM_SMI_INVENTORY = """\
GPU[0]\t: Card Series: AMD Radeon RX 7900 XTX
GPU[0]\t: GFX Version:\tgfx1100
GPU[1]\t: Card Series: AMD Radeon RX 7900 XTX
GPU[1]\t: GFX Version:\tgfx1100
GPU[2]\t: Card Series: AMD Radeon Graphics
GPU[2]\t: GFX Version:\tgfx1201
GPU[3]\t: Card Series: AMD Radeon RX 6900 XT
GPU[3]\t: GFX Version:\tgfx1030
"""


def fixture_build(base: pathlib.Path) -> pathlib.Path:
    build = base / "build"
    exe = build / "bin" / "r4d_selftest"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"--perfiqr --perfw8")
    for plugin_name in ("libr3.so", "libr4d.so"):
        p = build / "radiance_home" / "kernels" / plugin_name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(
            b"gemm_iqr_nt_m64 gemm_iqr_tiled "
            b"gemm_i8a8_nt_m16 gemm_i8a8_tiled"
        )
    return build


def args_for(build: pathlib.Path, scratch: pathlib.Path, *,
             gpu: str = "xtx1", quant: str = "iq6r") -> argparse.Namespace:
    return argparse.Namespace(
        _under_lease=True, build_dir=build, gpu=gpu,
        quant=quant, candidate="capacity",
        m=4, iters=3, cold_mib=256 if quant == "iq6r" else 0,
        timeout_s=30, scratch=scratch, part="physical", projection="gate_up",
    )


class ExecutionSafetyTests(unittest.TestCase):
    def test_rocr_only_visibility_and_inherited_masks_fail_closed(self):
        for key in runner.MASK_VARIABLES:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "inherited GPU masks"):
                runner.isolated_device_env({key: ""}, 2)
        result = runner.isolated_device_env({"PATH": "/bin"}, 2)
        self.assertEqual(result["ROCR_VISIBLE_DEVICES"], "2")
        self.assertNotIn("HIP_VISIBLE_DEVICES", result)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", result)
        self.assertNotIn("GPU_DEVICE_ORDINAL", result)
        with self.assertRaisesRegex(ValueError, "negative physical GPU"):
            runner.isolated_device_env({}, -1)

    def test_actual_multiline_rocm_smi_inventory(self):
        self.assertEqual(
            runner.parse_devices(ROCM_SMI_INVENTORY),
            {"xtx0": 0, "xtx1": 1, "r9700": 2},
        )

    def test_output_shape_header_checked_not_just_matching_kernel_name(self):
        good_iqr = "N=10240 K=5120 M=4\n  iqr k6 m64 85.5 us 10 GB/s\n"
        self.assertEqual(runner.extract_measured_shape(good_iqr, quant="iq6r"),
                         (10240, 5120, 4))
        good_w8 = ("8-bit trunk GEMMs  N=10240 K=5120 M=4  3 iters\n"
                   "  gemm_i8a8_nt_m16 30.1 us\n")
        self.assertEqual(runner.extract_measured_shape(good_w8, quant="w8a8"),
                         (10240, 5120, 4))
        for text in (
            "  iqr k6 m64 85.5 us\n",
            "N=10240 K=5120 M=4\nN=10240 K=5120 M=4\n",
            "N=10240 K=5120 M=4\nN=10240 K=5120 M=8\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                runner.extract_measured_shape(text, quant="iq6r")

    def _simulated_trial(self, *, gpu: str, quant: str,
                         output_override: str | None = None,
                         expected_success: bool = True) -> tuple[dict, list]:
        with tempfile.TemporaryDirectory() as temp:
            base = pathlib.Path(temp)
            build = fixture_build(base)
            args = args_for(build, base / "scratch", gpu=gpu, quant=quant)
            geo = runner.geometry("capacity", gpu, "gate_up")
            if quant == "iq6r":
                stdout = (f'N={geo["N"]} K={geo["K"]} M=4\n'
                          "  iqr k6 m64 101.5 us 13.2 GB/s\n")
            else:
                stdout = (f'8-bit trunk GEMMs  N={geo["N"]} K={geo["K"]} M=4  3 iters\n'
                          "  gemm_i8a8_nt_m16 56.4 us 1.3 TFLOP/s\n")
            if output_override is not None:
                stdout = output_override
            actual_cmds = []

            def fake_run(cmd, **kwargs):
                actual_cmds.append((cmd, kwargs))
                if cmd[:2] == ["rocm-smi", "--showproductname"]:
                    return Mock(returncode=0, stdout=ROCM_SMI_INVENTORY, stderr="")
                if cmd[:3] == ["git", "-C", str(runner.ROOT)]:
                    return Mock(returncode=0, stdout="test-head\n", stderr="")
                self.assertEqual(cmd[0], str(build / "bin" / "r4d_selftest"))
                visible = kwargs["env"]
                self.assertEqual(visible["ROCR_VISIBLE_DEVICES"],
                                 {"xtx0": "0", "xtx1": "1", "r9700": "2"}[gpu])
                self.assertNotIn("HIP_VISIBLE_DEVICES", visible)
                self.assertNotIn("CUDA_VISIBLE_DEVICES", visible)
                self.assertNotIn("GPU_DEVICE_ORDINAL", visible)
                return Mock(returncode=0, stdout=stdout, stderr="")

            with (
                patch.dict(runner.os.environ, {"VTP05_LEASED": "1"}, clear=True),
                patch.object(runner.subprocess, "run", side_effect=fake_run),
            ):
                if expected_success:
                    path = runner.execute_one(args, geo)
                else:
                    with self.assertRaisesRegex(RuntimeError, "failed/unqualified case"):
                        runner.execute_one(args, geo)
                    paths = list((base / "scratch" / "runs").glob("*/receipt.json"))
                    self.assertEqual(len(paths), 1)
                    path = paths[0]
                receipt = json.loads(path.read_text())
            return receipt, actual_cmds

    def test_two_physical_cards_and_two_formats_mocked(self):
        for gpu, quant, expected in (
            ("xtx0", "iq6r", 101.5),
            ("xtx1", "iq6r", 101.5),
            ("r9700", "w8a8", 56.4),
        ):
            with self.subTest(gpu=gpu, quant=quant):
                receipt, executed = self._simulated_trial(gpu=gpu, quant=quant)
                self.assertEqual(receipt["status"], "TIMING_OBTAINED_NUMERICAL_UNCHECKED")
                self.assertEqual(receipt["device_us"], expected)
                self.assertEqual(receipt["observed_nkm"],
                                 [receipt["geometry"]["N"], receipt["geometry"]["K"], 4])
                self.assertEqual(receipt["numerical_oracle"],
                                 "NOT_TESTED_SELFTEST_PERF_MODE_TIMING_ONLY")
                self.assertEqual(len(executed), 3)

    def test_wrong_shape_must_not_be_accepted(self):
        bad = "N=9000 K=5120 M=4\n  iqr k6 m64 1.0 us\n"
        receipt, _ = self._simulated_trial(gpu="xtx0", quant="iq6r",
                                           output_override=bad, expected_success=False)
        self.assertEqual(receipt["status"], "FAILED_NO_SELECTED_KERNEL_TIMING")
        self.assertIn("GEMM shape differs", receipt["error"])
        self.assertNotIn("device_us", receipt)

    def test_good_kernel_row_without_shape_must_not_be_accepted(self):
        receipt, _ = self._simulated_trial(
            gpu="xtx1", quant="w8a8",
            output_override="  gemm_i8a8_nt_m16 0.1 us\n",
            expected_success=False,
        )
        self.assertEqual(receipt["status"], "FAILED_NO_SELECTED_KERNEL_TIMING")
        self.assertNotIn("device_us", receipt)


if __name__ == "__main__":
    unittest.main()
