"""VA06 slice: RD73 observability primitives -- graph-cache resource
telemetry parsing/reduction and the real subject-hit/control-miss
activation probe. GPT scoping (session ses_8a915986b0e64312,
req_fbd1784639a54858): land the two missing RD73-specific evidence
producers only; no full --run-rd73-contract orchestration yet.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as ec  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402


class ParseRd73ResourceTelemetryTests(unittest.TestCase):
    def test_parses_all_readings_in_emission_order(self) -> None:
        text = (
            "some other log line\n"
            "BIGCHERRY_RD73_RESOURCE graph_cache_entries=386\n"
            "more noise\n"
            "BIGCHERRY_RD73_RESOURCE graph_cache_entries=651\n"
        )
        self.assertEqual(vc.parse_rd73_resource_telemetry(text), (386, 651))

    def test_no_readings_returns_empty_tuple(self) -> None:
        self.assertEqual(vc.parse_rd73_resource_telemetry("nothing here\n"), ())

    def test_ignores_unrelated_bigcherry_lines(self) -> None:
        text = "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
        self.assertEqual(vc.parse_rd73_resource_telemetry(text), ())


class PeakRd73ResourceResultTests(unittest.TestCase):
    def test_real_contract_800_limit_passes_at_651(self) -> None:
        registry = ec.load_contracts("config/experiment-contracts.toml")
        contract = registry.contracts["RD73-STABLE-GRAPH-CACHE-KEY"]
        result = vc.peak_rd73_resource_result((300, 651, 400))
        self.assertEqual(result.metric, "graph_cache_entries")
        self.assertEqual(result.unit, "count")
        self.assertEqual(result.subject_value, 651.0)
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_entries": result})
        self.assertTrue(gate["passed"], gate)

    def test_real_contract_800_limit_fails_at_801(self) -> None:
        registry = ec.load_contracts("config/experiment-contracts.toml")
        contract = registry.contracts["RD73-STABLE-GRAPH-CACHE-KEY"]
        result = vc.peak_rd73_resource_result((801,))
        gate = ec.evaluate_resource_gate(contract, {"graph_cache_entries": result})
        self.assertFalse(gate["passed"])
        self.assertIn("graph_cache_entries", gate["failed_metrics"])

    def test_peak_is_the_maximum_not_the_last(self) -> None:
        result = vc.peak_rd73_resource_result((651, 200, 400))
        self.assertEqual(result.subject_value, 651.0)

    def test_empty_subject_readings_raises(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc.peak_rd73_resource_result(())

    def test_malformed_subject_reading_raises(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc.peak_rd73_resource_result((386, -1))

    def test_control_readings_populate_control_value_as_peak(self) -> None:
        result = vc.peak_rd73_resource_result((651, 651), control_readings=(300, 386))
        self.assertEqual(result.control_value, 386.0)

    def test_no_control_readings_leaves_control_value_none(self) -> None:
        result = vc.peak_rd73_resource_result((651,))
        self.assertIsNone(result.control_value)


class RunRd73ActivationEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_subprocess_run = vc.subprocess.run
        self._tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self._tmp.name)
        self.control_bin = self.workdir / "control_bin"
        self.control_bin.write_text("", encoding="utf-8")
        self.subject_bin = self.workdir / "subject_bin"
        self.subject_bin.write_text("", encoding="utf-8")
        self.model = self.workdir / "model.gguf"
        self.model.write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        vc.subprocess.run = self._real_subprocess_run
        self._tmp.cleanup()

    def _fake_run(self, stdout_for_binary):
        def fake_run(command, **kwargs):
            class _Result:
                pass

            result = _Result()
            result.returncode = 0
            result.stdout = stdout_for_binary(command[0])
            result.stderr = "ggml_cuda_init: found 1 ROCm devices\n"
            return result

        vc.subprocess.run = fake_run

    def test_subject_hit_control_miss(self) -> None:
        marker = "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key"
        self._fake_run(lambda binary: f"{marker}\n" if "subject" in binary else "nothing\n")
        result = vc.run_rd73_activation_evidence(
            marker_regex=marker, control_binary=self.control_bin,
            subject_binary=self.subject_bin, model=self.model,
            hip_path=Path("H:/hip"), workdir=self.workdir, run_dir=self.workdir,
        )
        self.assertTrue(result["subject_hit"])
        self.assertFalse(result["control_hit"])

    def test_neither_hit(self) -> None:
        marker = "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key"
        self._fake_run(lambda binary: "nothing\n")
        result = vc.run_rd73_activation_evidence(
            marker_regex=marker, control_binary=self.control_bin,
            subject_binary=self.subject_bin, model=self.model,
            hip_path=Path("H:/hip"), workdir=self.workdir, run_dir=self.workdir,
        )
        self.assertFalse(result["subject_hit"])
        self.assertFalse(result["control_hit"])

    def test_command_includes_verbose_and_ngl(self) -> None:
        # Confirms this real producer reuses the fixed _run_one_trace_probe
        # (VA21's --verbose/-ngl fixes), not a bespoke re-implementation.
        seen_commands = []

        def fake_run(command, **kwargs):
            seen_commands.append(command)

            class _Result:
                returncode = 0
                stdout = "ggml_cuda_init: found 1 ROCm devices\n"
                stderr = ""

            return _Result()

        vc.subprocess.run = fake_run
        vc.run_rd73_activation_evidence(
            marker_regex="x", control_binary=self.control_bin,
            subject_binary=self.subject_bin, model=self.model,
            hip_path=Path("H:/hip"), workdir=self.workdir, run_dir=self.workdir,
        )
        for command in seen_commands:
            self.assertIn("--verbose", command)
            self.assertIn("-ngl", command)


if __name__ == "__main__":
    unittest.main()
