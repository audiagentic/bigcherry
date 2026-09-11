"""PVPS02 step 2: direct tests for run_paired_llama_benchmark() and
_paired_llama_bench_command() -- the shared execution shape extracted
from run_rd04_benchmark_evidence()/run_rd08_validation_lanes(). RD04's
and RD08's own existing test files already exercise this primitive
indirectly through those wrappers; this file targets the primitive's
own behavior directly, per the extraction's design (docs/planning/
active/patching-validation-package-standard/PVPS02.md).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class _Result:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class PairedLlamaBenchCommandTests(unittest.TestCase):
    def test_patch_args_land_before_ngl(self) -> None:
        command = vc._paired_llama_bench_command(
            Path("bin"), Path("m.gguf"), "decode", patch_args=("-fa", "on"),
        )
        self.assertLess(command.index("-fa"), command.index("-ngl"))

    def test_runtime_args_land_after_ngl(self) -> None:
        command = vc._paired_llama_bench_command(
            Path("bin"), Path("m.gguf"), "decode", runtime_args=("-sm", "tensor"),
        )
        self.assertGreater(command.index("-sm"), command.index("-ngl"))

    def test_reproduces_rd04_exact_historical_argv_shape(self) -> None:
        command = vc._paired_llama_bench_command(
            Path("control_bin"), Path("m.gguf"), "decode",
            patch_args=("-fa", "on", "-ctk", "bf16", "-ctv", "bf16"),
        )
        self.assertEqual(
            command,
            ["control_bin", "-m", "m.gguf", "-p", "0", "-n", "128",
             "-fa", "on", "-ctk", "bf16", "-ctv", "bf16", "-ngl", "99"],
        )

    def test_reproduces_rd08_exact_historical_argv_shape(self) -> None:
        command = vc._paired_llama_bench_command(
            Path("control_bin"), Path("m.gguf"), "decode",
        )
        self.assertEqual(
            command, ["control_bin", "-m", "m.gguf", "-p", "0", "-n", "128", "-ngl", "99"],
        )
        self.assertEqual(
            command,
            vc.rd08_validation_lane_commands(
                control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
                model=Path("m.gguf"), workload="decode",
            )[0],
        )

    def test_unmapped_workload_raises(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc._paired_llama_bench_command(Path("bin"), Path("m.gguf"), "bogus")


class RunPairedLlamaBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_subprocess_run = vc.subprocess.run

    def tearDown(self) -> None:
        vc.subprocess.run = self._real_subprocess_run

    def test_default_workloads_run_both_decode_and_prefill(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            metric = "tg128" if "-n" in command and command[command.index("-n") + 1] == "128" else "pp512"
            return _Result(0, f"ggml_cuda_init: found 1 ROCm devices\n{metric} | 100.0 t/s\n")

        vc.subprocess.run = fake_run
        outcome = vc.run_paired_llama_benchmark(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), pairs=1, log_context="test",
        )
        self.assertEqual(set(outcome.runs), {"decode", "prefill"})
        self.assertEqual(set(outcome.commands), {"decode", "prefill"})
        self.assertTrue(outcome.raw_logs)

    def test_single_requested_workload_only_runs_that_one(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            return _Result(0, "ggml_cuda_init: found 1 ROCm devices\ntg128 | 100.0 t/s\n")

        vc.subprocess.run = fake_run
        outcome = vc.run_paired_llama_benchmark(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), workloads=("decode",),
            pairs=1, log_context="test",
        )
        self.assertEqual(set(outcome.runs), {"decode"})
        self.assertEqual(set(outcome.commands), {"decode"})

    def test_log_context_reaches_the_gpu_execution_guard_error_message(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            return _Result(0, "no rocm devices here\ntg128 | 100.0 t/s\n")

        vc.subprocess.run = fake_run
        with self.assertRaisesRegex(Exception, "distinctive-context"):
            vc.run_paired_llama_benchmark(
                control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
                model=Path("m.gguf"), hip_path=Path("H:/hip"), workloads=("decode",),
                pairs=1, log_context="distinctive-context",
            )


if __name__ == "__main__":
    unittest.main()
