"""VA21 fix: fail-closed real-GPU-execution enforcement.

GPT review (session ses_8a915986b0e64312): a llama.cpp binary whose
ROCm/HIP device init fails silently falls back to CPU execution while
still printing a normal-looking benchmark table under a "backend: ROCm"
label -- exit code 0 and a plausible metric line are NOT proof the
process actually touched the GPU. Real-hardware finding (2026-09-01,
Brutus, gfx1030/device 3): the HIP runtime failed to detect a real,
rocminfo-visible GPU, and nothing caught it before RD08's trigger check
quietly "ran" and reported a negative -- confounding a real dispatch-
routing question (VA21) with an unrelated environment failure.

Fix: _require_real_gpu_execution() demands a positive "ggml_cuda_init:
found N ROCm devices" signature (not merely the absence of an error) and
explicitly rejects known ROCm-init-failure signatures. Wired into
_run_one_trace_probe() (the shared generic activation-probe primitive
RD08's trigger check and RD12/RD13/etc's generic probes all go through),
rd08_validation_lane_commands()'s runner, and run_rd04_benchmark_evidence()'s
runner -- all three also gained an explicit -ngl 99 flag.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class RequireRealGpuExecutionTests(unittest.TestCase):
    def test_valid_gpu_output_passes(self) -> None:
        vc._require_real_gpu_execution(
            "ggml_cuda_init: found 2 ROCm devices (Total VRAM: 49120 MiB):\ntg128 | 100.0 t/s\n",
            "", context="test",
        )  # must not raise

    def test_rocm_init_failure_rejected(self) -> None:
        with self.assertRaises(vc.PatchCampaignError) as ctx:
            vc._require_real_gpu_execution(
                "tg128 | 16.33 t/s\n", "ggml_cuda_init: failed to initialize ROCm: no ROCm-capable "
                "device is detected\n", context="test",
            )
        self.assertIn("ROCm/HIP device initialization failed", str(ctx.exception))

    def test_cpu_only_silent_fallback_rejected(self) -> None:
        # Exit code 0, a plausible-looking metric line, but NO real
        # positive GPU-init evidence anywhere in the output -- exactly
        # the real 2026-09-01 confound.
        with self.assertRaises(vc.PatchCampaignError) as ctx:
            vc._require_real_gpu_execution("tg128 | 16.33 t/s\n", "", context="test")
        self.assertIn("no real GPU execution evidence", str(ctx.exception))

    def test_zero_devices_found_is_rejected(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc._require_real_gpu_execution(
                "ggml_cuda_init: found 0 ROCm devices\n", "", context="test",
            )

    def test_signature_may_appear_in_stderr(self) -> None:
        vc._require_real_gpu_execution(
            "", "ggml_cuda_init: found 1 ROCm devices\n", context="test",
        )  # must not raise


class TraceProbeGpuGuardIntegrationTests(unittest.TestCase):
    """Real integration through _run_one_trace_probe() -- not mocking that
    function itself, unlike the other trace-probe test file, so this
    actually exercises the -ngl flag and the guard together."""

    def setUp(self) -> None:
        self._real_subprocess_run = vc.subprocess.run
        self._tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self._tmp.name)
        self.binary = self.workdir / "fake-binary"
        self.binary.write_text("", encoding="utf-8")
        self.model = self.workdir / "model.gguf"
        self.model.write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        vc.subprocess.run = self._real_subprocess_run
        self._tmp.cleanup()

    def _fake_run(self, stdout: str, stderr: str = "", returncode: int = 0):
        class _Result:
            pass

        def fake_run(command, **kwargs):
            self.last_command = command
            result = _Result()
            result.returncode = returncode
            result.stdout = stdout
            result.stderr = stderr
            return result

        vc.subprocess.run = fake_run

    def test_command_includes_ngl_99(self) -> None:
        self._fake_run("ggml_cuda_init: found 1 ROCm devices\n")
        vc._run_one_trace_probe(
            name="test", binary=self.binary, model=self.model, hip_path=Path("H:/hip"),
            workdir=self.workdir, bench_prompt=0, bench_gen=128, disable_fusion=False,
        )
        self.assertIn("-ngl", self.last_command)
        self.assertEqual(self.last_command[self.last_command.index("-ngl") + 1], "99")

    def test_command_includes_verbose(self) -> None:
        # VA21 real-hardware finding: llama-bench.cpp gates ggml's log
        # level on its OWN --verbose flag (ERROR without it, filtering
        # BOTH GGML_LOG_INFO and GGML_LOG_WARN) -- this probe's entire
        # purpose is observing log-based activation markers, so it must
        # always request verbose output.
        self._fake_run("ggml_cuda_init: found 1 ROCm devices\n")
        vc._run_one_trace_probe(
            name="test", binary=self.binary, model=self.model, hip_path=Path("H:/hip"),
            workdir=self.workdir, bench_prompt=0, bench_gen=128, disable_fusion=False,
        )
        self.assertIn("--verbose", self.last_command)

    def test_rocm_init_failure_raises_even_with_returncode_zero(self) -> None:
        self._fake_run(
            "tg128 | 16.33 t/s\n",
            stderr="ggml_cuda_init: failed to initialize ROCm: no ROCm-capable device is detected\n",
        )
        with self.assertRaises(vc.PatchCampaignError):
            vc._run_one_trace_probe(
                name="test", binary=self.binary, model=self.model, hip_path=Path("H:/hip"),
                workdir=self.workdir, bench_prompt=0, bench_gen=128, disable_fusion=False,
            )

    def test_valid_gpu_run_passes(self) -> None:
        self._fake_run("ggml_cuda_init: found 2 ROCm devices\nBIGCHERRY_PATCH_HIT patch=x\n")
        combined = vc._run_one_trace_probe(
            name="test", binary=self.binary, model=self.model, hip_path=Path("H:/hip"),
            workdir=self.workdir, bench_prompt=0, bench_gen=128, disable_fusion=False,
        )
        self.assertIn("BIGCHERRY_PATCH_HIT", combined)

    def test_nonzero_returncode_still_raises_its_own_error_not_the_gpu_guard(self) -> None:
        # A genuine process crash (nonzero exit) must surface as the
        # existing "activation probe ... failed with exit code" error,
        # not be masked or reclassified by the GPU-execution guard.
        self._fake_run("", stderr="segfault", returncode=1)
        with self.assertRaises(vc.PatchCampaignError) as ctx:
            vc._run_one_trace_probe(
                name="test", binary=self.binary, model=self.model, hip_path=Path("H:/hip"),
                workdir=self.workdir, bench_prompt=0, bench_gen=128, disable_fusion=False,
            )
        self.assertIn("failed with exit code", str(ctx.exception))


class Rd08LaneCommandsGpuFlagTests(unittest.TestCase):
    def test_rd08_lane_commands_include_ngl_99(self) -> None:
        control_cmd, subject_cmd = vc.rd08_validation_lane_commands(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), workload="decode",
        )
        for command in (control_cmd, subject_cmd):
            self.assertIn("-ngl", command)
            self.assertEqual(command[command.index("-ngl") + 1], "99")


class Rd04CommandGpuFlagTests(unittest.TestCase):
    def test_rd04_command_includes_ngl_99(self) -> None:
        import inspect

        source = inspect.getsource(vc.run_rd04_benchmark_evidence)
        self.assertIn('"-ngl", "99"', source)


if __name__ == "__main__":
    unittest.main()
