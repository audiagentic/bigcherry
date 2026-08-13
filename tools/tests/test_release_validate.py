from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigcherry import release_validate  # noqa: E402


class SafeNameTests(unittest.TestCase):
    def test_safe_name_cannot_escape_staging_root(self):
        self.assertEqual(release_validate.safe_name("../../etc/passwd"), "etc-passwd")
        self.assertEqual(release_validate.safe_name("b10362"), "b10362")
        self.assertEqual(release_validate.safe_name("..."), "upstream")


class ReleaseGateTests(unittest.TestCase):
    def _evidence(self) -> dict:
        return {
            "claim": "validated",
            "architectures": ["gfx1201"],
            "required_architectures": ["gfx1201"],
            "architecture_coverage": {
                "required": ["gfx1201"],
                "observed": ["gfx1201"],
                "validated": ["gfx1201"],
                "by_architecture": {
                    "gfx1201": {
                        "observed": True, "validated": True,
                        "candidate_coverage": True,
                    },
                },
            },
            "candidate_coverage": {
                "variant_set": "workload-max",
                "observed_types": ["q8_0"],
                "by_type": {
                    "q8_0": {"observed": True, "candidate_count": 2,
                              "alternative_count": 1},
                },
            },
        }

    def test_compatibility_record_does_not_need_hardware_evidence(self):
        release_validate.validate_release_claim({"outcome": "compatible"})

    def test_validated_claim_requires_and_accepts_consistent_evidence(self):
        release_validate.validate_release_claim(self._evidence())

    def test_validated_claim_rejects_missing_architecture_evidence(self):
        record = self._evidence()
        del record["architecture_coverage"]
        with self.assertRaisesRegex(ValueError, "architecture_coverage"):
            release_validate.validate_release_claim(record)

    def test_validated_claim_rejects_mismatched_candidate_coverage(self):
        record = self._evidence()
        record["candidate_coverage"]["observed_types"].append("q6_k")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            release_validate.validate_release_claim(record)

    def test_validated_claim_rejects_unvalidated_architecture(self):
        record = self._evidence()
        record["architecture_coverage"]["validated"] = []
        record["architecture_coverage"]["by_architecture"]["gfx1201"]["validated"] = False
        with self.assertRaisesRegex(ValueError, "gfx1201"):
            release_validate.validate_release_claim(record)

    def test_optimized_claim_rejects_missing_required_architecture(self):
        record = self._evidence()
        record["architectures"] = ["gfx1100", "gfx1201"]
        record["required_architectures"] = ["gfx1100", "gfx1201"]
        record["architecture_coverage"]["required"] = ["gfx1100", "gfx1201"]
        with self.assertRaisesRegex(ValueError, "missing"):
            release_validate.validate_release_claim(record)

    def test_inventory_legacy_flat_architecture_map_remains_diagnostic(self):
        record = self._evidence()
        record["candidate_coverage"]["variant_set"] = "inventory"
        record["architecture_coverage"] = {
            "gfx1201": {"status": "observed", "candidate_coverage": True},
        }
        del record["required_architectures"]
        release_validate.validate_release_claim(record)

    def test_optimized_claim_rejects_legacy_flat_architecture_map(self):
        record = self._evidence()
        record["architecture_coverage"] = {
            "gfx1201": {"status": "validated", "candidate_coverage": True},
        }
        with self.assertRaisesRegex(ValueError, "explicit"):
            release_validate.validate_release_claim(record)

    def test_optimized_claim_rejects_zero_alternatives(self):
        record = self._evidence()
        coverage = record["candidate_coverage"]
        coverage["by_type"]["q8_0"]["alternative_count"] = 0
        coverage["by_type"]["q8_0"]["zero_alternative_reason"] = (
            "no supported alternative was generated for this type")
        with self.assertRaisesRegex(ValueError, "zero alternatives"):
            release_validate.validate_release_claim(record)

    def test_inventory_native_only_profile_remains_diagnostic(self):
        record = self._evidence()
        coverage = record["candidate_coverage"]
        coverage["variant_set"] = "inventory"
        coverage["by_type"]["q8_0"]["alternative_count"] = 0
        coverage["by_type"]["q8_0"]["native_count"] = 1
        coverage["by_type"]["q8_0"]["zero_alternative_reason"] = (
            "inventory profile contains native wrappers only")
        release_validate.validate_release_claim(record)

    def test_validated_claim_rejects_missing_variant_set(self):
        record = self._evidence()
        del record["candidate_coverage"]["variant_set"]
        with self.assertRaisesRegex(ValueError, "variant_set"):
            release_validate.validate_release_claim(record)


class ProbeTests(unittest.TestCase):
    def test_run_already_exists_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            (staging / "dup").mkdir()
            with self.assertRaises(FileExistsError):
                release_validate.probe("dup", staging, "master", "bigcherry")

    def test_pull_failure_short_circuits_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            with mock.patch.object(release_validate, "_run_logged", side_effect=[False]) as run:
                code, path = release_validate.probe("r1", staging, "master", "bigcherry")
            self.assertEqual(code, 1)
            self.assertEqual(run.call_count, 1)
            record = path.read_text(encoding="utf-8")
            self.assertIn('"outcome": "pull-failed"', record)
            self.assertIn('"failure_class": "pull-failed"', record)
            self.assertIn('"stage": "pull"', record)

    def test_build_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            with mock.patch.object(release_validate, "_run_logged", side_effect=[True, False]):
                code, path = release_validate.probe("r2", staging, "master", "bigcherry")
            self.assertEqual(code, 1)
            record = path.read_text(encoding="utf-8")
            self.assertIn('"outcome": "patch-drift-or-build-failed"', record)
            self.assertIn('"failure_class": "patch-drift"', record)
            self.assertIn('"command":', record)

    def test_clean_probe_reports_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            with mock.patch.object(release_validate, "_run_logged", side_effect=[True, True]) as run:
                code, path = release_validate.probe("r3", staging, "master", "bigcherry")
            self.assertEqual(code, 0)
            pull_command = run.call_args_list[0].args[0]
            build_command = run.call_args_list[1].args[0]
            self.assertLess(pull_command.index("--llama-root"), pull_command.index("pull"))
            self.assertLess(build_command.index("--llama-root"), build_command.index("build"))
            record = path.read_text(encoding="utf-8")
            self.assertIn('"outcome": "compatible"', record)

    def test_inventory_is_forwarded_to_isolated_build(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            inventory = staging / "inventory.json"
            inventory.write_text("{}", encoding="utf-8")
            with mock.patch.object(release_validate, "_run_logged", side_effect=[True, True]) as run:
                code, _ = release_validate.probe("r4", staging, "master", "workstation", inventory)
            self.assertEqual(code, 0)
            build_command = run.call_args_list[1].args[0]
            self.assertEqual(build_command[-2:], ["--inventory", str(inventory)])


if __name__ == "__main__":
    unittest.main()
