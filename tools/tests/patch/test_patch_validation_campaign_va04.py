"""VA04 hardware-free preflight slice: run_rd04_benchmark_evidence() --
an RD04-scoped validation-domain paired benchmark producer, analogous to
RD08's real lanes but without contract promotion/generalisation (GPT
session ses_5bbee8ce5c9a4265, req_da015a1366044ad1). Hardware-free via
an injected fake subprocess.run(), consistent with VA14's established
pattern.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import execution as ex  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402


class _Result:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class RunRd04BenchmarkEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_subprocess_run = vc.subprocess.run

    def tearDown(self) -> None:
        vc.subprocess.run = self._real_subprocess_run

    def _run_dir(self) -> Path:
        return Path(tempfile.mkdtemp())

    def test_alternates_control_subject_and_uses_exact_rd04_flags(self) -> None:
        seen_commands: list[list[str]] = []

        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            seen_commands.append(command)
            binary = command[0]
            metric = "tg128" if "-n" in command and command[command.index("-n") + 1] == "128" else "pp512"
            value = {"control_bin": 100.0, "subject_bin": 103.39}[binary]
            return _Result(0, f"{metric} | {value} t/s\n")

        vc.subprocess.run = fake_run
        result = vc.run_rd04_benchmark_evidence(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=self._run_dir(),
            campaign_id="campaign123", amdgpu_targets="gfx1100",
            control_build_identity={"effective_build_id": "c1"},
            subject_build_identity={"effective_build_id": "s1"},
            pairs=2,
        )
        self.assertTrue(result["passed"])
        # decode: pair0 control,subject ; pair1 subject,control ; then prefill same pattern.
        self.assertEqual(
            [c[0] for c in seen_commands],
            ["control_bin", "subject_bin", "subject_bin", "control_bin",
             "control_bin", "subject_bin", "subject_bin", "control_bin"],
        )
        for command in seen_commands:
            self.assertIn("-fa", command)
            self.assertEqual(command[command.index("-fa") + 1], "on")
            self.assertIn("-ctk", command)
            self.assertEqual(command[command.index("-ctk") + 1], "bf16")
            self.assertIn("-ctv", command)
            self.assertEqual(command[command.index("-ctv") + 1], "bf16")
        decode_commands = seen_commands[:4]
        prefill_commands = seen_commands[4:]
        for command in decode_commands:
            self.assertEqual(command[command.index("-p") + 1], "0")
            self.assertEqual(command[command.index("-n") + 1], "128")
        for command in prefill_commands:
            self.assertEqual(command[command.index("-p") + 1], "512")
            self.assertEqual(command[command.index("-n") + 1], "0")

    def test_nonzero_arm_fails_closed(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            if command[0] == "subject_bin":
                return _Result(1, "", "boom")
            return _Result(0, "tg128 | 100.0 t/s\n")

        vc.subprocess.run = fake_run
        with self.assertRaises(ex.LaneExecutionError):
            vc.run_rd04_benchmark_evidence(
                control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
                model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=self._run_dir(),
                campaign_id="campaign123", amdgpu_targets="gfx1100",
                control_build_identity={}, subject_build_identity={}, pairs=1,
            )

    def test_artifact_hash_is_real(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            metric = "tg128" if "-n" in command and command[command.index("-n") + 1] == "128" else "pp512"
            return _Result(0, f"{metric} | 100.0 t/s\n")

        vc.subprocess.run = fake_run
        run_dir = self._run_dir()
        result = vc.run_rd04_benchmark_evidence(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=run_dir,
            campaign_id="campaign123", amdgpu_targets="gfx1100",
            control_build_identity={}, subject_build_identity={}, pairs=1,
        )
        import hashlib

        performance_path = run_dir / "performance.json"
        self.assertTrue(performance_path.is_file())
        real_hash = hashlib.sha256(performance_path.read_bytes()).hexdigest()
        self.assertEqual(result["artifact"]["sha256"], real_hash)
        self.assertEqual(result["artifact"]["path"], "performance.json")
        doc = json.loads(performance_path.read_text(encoding="utf-8"))
        self.assertTrue(doc["passed"])
        self.assertEqual(doc["campaign_id"], "campaign123")
        self.assertIn("decode", doc["metrics"])
        self.assertIn("prefill", doc["metrics"])

    def test_generic_s6_bench_json_is_never_consumed(self) -> None:
        # This producer must be entirely self-contained -- it never reads
        # any pre-existing campaign/bench.json (S6's stock/native/replay
        # measurement, which cannot prove RD04's patch effect at all).
        import inspect
        source = inspect.getsource(vc.run_rd04_benchmark_evidence)
        self.assertNotIn("bench.json", source)


class BenchmarkArtifactBindingTests(unittest.TestCase):
    """Proves binding run_rd04_benchmark_evidence()'s real artifact into
    ctx.performance_evidence makes both the real "performance" and
    "controls" checks (validator="benchmark") reach PASS."""

    def test_performance_and_controls_checks_both_pass_from_the_bound_artifact(self) -> None:
        from bigcherry.patch import validation as pv

        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            metric = "tg128" if "-n" in command and command[command.index("-n") + 1] == "128" else "pp512"
            return _Result(0, f"{metric} | 100.0 t/s\n")

        real_subprocess_run = vc.subprocess.run
        vc.subprocess.run = fake_run
        try:
            run_dir = Path(tempfile.mkdtemp())
            result = vc.run_rd04_benchmark_evidence(
                control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
                model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=run_dir,
                campaign_id="campaign123", amdgpu_targets="gfx1100",
                control_build_identity={}, subject_build_identity={}, pairs=1,
            )
        finally:
            vc.subprocess.run = real_subprocess_run

        performance_evidence = {"artifact": result["artifact"]}
        ctx = pv.ValidationContext(
            descriptor=None, base_revision="a" * 40, control_source=None, subject_source=None,
            run_dir=run_dir, performance_evidence=performance_evidence,
        )
        performance_spec = pv.CheckSpec("performance", "performance", "benchmark", True, {})
        controls_spec = pv.CheckSpec("controls", "controls", "benchmark", True, {})
        performance_result = pv.evaluate_check(performance_spec, ctx)
        controls_result = pv.evaluate_check(controls_spec, ctx)
        self.assertEqual(performance_result.status, pv.PASS, performance_result.summary)
        self.assertEqual(controls_result.status, pv.PASS, controls_result.summary)

    def test_missing_benchmark_evidence_is_blocked_not_fabricated(self) -> None:
        from bigcherry.patch import validation as pv

        ctx = pv.ValidationContext(
            descriptor=None, base_revision="a" * 40, control_source=None, subject_source=None,
            run_dir=Path(tempfile.mkdtemp()), performance_evidence={},
        )
        spec = pv.CheckSpec("performance", "performance", "benchmark", True, {})
        result = pv.evaluate_check(spec, ctx)
        self.assertNotEqual(result.status, pv.PASS)


class Rd04CliWiringTests(unittest.TestCase):
    """run() is a large real-hardware integration entry point (source
    materialization, 7 real cmake builds) -- consistent with VA14/VA15's
    established scope boundary, these prove the real committed wiring via
    source inspection rather than mocking the entire pipeline."""

    def setUp(self) -> None:
        import inspect
        self.source = inspect.getsource(vc.run)

    def test_run_rd04_benchmark_flag_exists_and_defaults_false(self) -> None:
        import argparse
        parser = argparse.ArgumentParser()
        # main()'s own parser construction is easiest to check indirectly:
        # confirm the CLI wiring string is present in main()'s source.
        import inspect
        main_source = inspect.getsource(vc.main)
        self.assertIn('"--run-rd04-benchmark"', main_source)
        self.assertIn('action="store_true", default=False', main_source.replace("'", '"'))

    def test_mutually_exclusive_with_rd08_modes(self) -> None:
        self.assertIn("--run-rd04-benchmark is mutually exclusive with the", self.source)
        self.assertIn("if args.run_rd08_lanes or args.run_rd08_contract:", self.source)

    def test_rd04_only_gating(self) -> None:
        self.assertIn('descriptor.experiment_contract != "RD04-BF16-FLASH-ATTN-TILE"', self.source)

    def test_binds_only_performance_evidence_never_contract_promotions(self) -> None:
        # RD04 mode must never write into contract_promotions -- eligibility
        # must stay false regardless of a PASS.
        rd04_block_start = self.source.index("if args.run_rd04_benchmark:")
        rd04_block = self.source[rd04_block_start:self.source.index("validation_check_results: dict")]
        self.assertIn("performance_evidence = {\"artifact\": rd04_result[\"artifact\"]}", rd04_block)
        self.assertNotIn("contract_promotions[", rd04_block)


if __name__ == "__main__":
    unittest.main()
