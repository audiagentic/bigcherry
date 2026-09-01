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

    def test_malformed_resource_line_raises(self) -> None:
        text = "BIGCHERRY_RD73_RESOURCE graph_cache_entries=notanumber\n"
        with self.assertRaises(vc.PatchCampaignError):
            vc.parse_rd73_resource_telemetry(text)

    def test_mixed_valid_and_malformed_raises(self) -> None:
        text = (
            "BIGCHERRY_RD73_RESOURCE graph_cache_entries=386\n"
            "BIGCHERRY_RD73_RESOURCE graph_cache_entries=\n"
            "BIGCHERRY_RD73_RESOURCE graph_cache_entries=651\n"
        )
        with self.assertRaises(vc.PatchCampaignError):
            vc.parse_rd73_resource_telemetry(text)


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


class EvaluateRd73ActivationEvidenceTests(unittest.TestCase):
    """User redirect (2026-09-01): activation evidence is now read from
    the MTP server lane's own control/subject log files (always
    BIGCHERRY_PATCH_TRACE=1) rather than launched via a separate
    llama-bench probe -- llama-bench itself proved unworkable for RD73's
    real 27B/dual-GPU config on real hardware."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        (self.run_dir / "logs").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_log(self, name: str, text: str) -> Path:
        path = self.run_dir / "logs" / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_subject_hit_control_miss(self) -> None:
        marker = "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key"
        subject_log = self._write_log("subject.log", f"{marker}\n")
        control_log = self._write_log("control.log", "nothing\n")
        result = vc.evaluate_rd73_activation_evidence(
            marker_regex=marker, control_log_path=control_log, subject_log_path=subject_log,
            run_dir=self.run_dir,
        )
        self.assertTrue(result["subject_hit"])
        self.assertFalse(result["control_hit"])

    def test_neither_hit(self) -> None:
        marker = "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key"
        subject_log = self._write_log("subject.log", "nothing\n")
        control_log = self._write_log("control.log", "nothing\n")
        result = vc.evaluate_rd73_activation_evidence(
            marker_regex=marker, control_log_path=control_log, subject_log_path=subject_log,
            run_dir=self.run_dir,
        )
        self.assertFalse(result["subject_hit"])
        self.assertFalse(result["control_hit"])

    def test_artifact_bound_with_relative_log_paths(self) -> None:
        marker = "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key"
        subject_log = self._write_log("subject.log", f"{marker}\n")
        control_log = self._write_log("control.log", "nothing\n")
        result = vc.evaluate_rd73_activation_evidence(
            marker_regex=marker, control_log_path=control_log, subject_log_path=subject_log,
            run_dir=self.run_dir,
        )
        self.assertEqual(result["subject_log_path"], "logs/subject.log")
        self.assertEqual(result["control_log_path"], "logs/control.log")
        artifact_path = self.run_dir / result["artifact"]["path"]
        self.assertTrue(artifact_path.is_file())


if __name__ == "__main__":
    unittest.main()
