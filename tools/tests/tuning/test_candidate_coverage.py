import unittest

from bigcherry.tuning.catalog import (
    Inventory, candidate_coverage, supported_candidate_coverage,
)


class CandidateCoverageTests(unittest.TestCase):
    def test_inventory_profile_reports_zero_alternatives_explicitly(self):
        candidates = [{
            "stable_name": "native:mmq:v1", "family": "mmq",
            "source_class": "native_wrapper", "config": {"type": "q8_0"},
        }]
        report = candidate_coverage(
            candidates, Inventory(mmq_types={"q8_0"}), "inventory")
        row = report["by_type"]["q8_0"]
        self.assertEqual(row["native_count"], 1)
        self.assertEqual(row["alternative_count"], 0)
        self.assertIn("zero_alternative_reason", row)

    def test_workload_profile_distinguishes_supported_alternatives(self):
        candidates = [
            {"stable_name": "native:mmq:v1", "family": "mmq",
             "source_class": "native_wrapper", "config": {"type": "q8_0"}},
            {"stable_name": "mmvq:q8_0:v1", "family": "mmvq",
             "source_class": "new_generated_variant", "config": {"type": "q8_0"}},
        ]
        report = candidate_coverage(
            candidates, Inventory(mmq_types={"q8_0"}), "workload-max")
        row = report["by_type"]["q8_0"]
        self.assertEqual(row["alternative_count"], 1)
        self.assertEqual(row["alternative_families"], ["mmvq"])
        self.assertNotIn("zero_alternative_reason", row)

    def test_supported_coverage_distinguishes_unobserved_and_unsupported(self):
        candidates = [
            {"stable_name": "mmq:native:v1", "family": "mmq",
             "source_class": "native_wrapper", "architectures": ["gfx1100"],
             "config": {}},
            {"stable_name": "mmvq:q8_0:v1", "family": "mmvq",
             "source_class": "new_generated_variant", "architectures": ["gfx1100"],
             "config": {"type": "q8_0"}},
            {"stable_name": "mmvf:f32:v1", "family": "mmvf",
             "source_class": "existing_runtime", "architectures": ["gfx1100"],
             "config": {"type": "f32"}},
        ]
        report = supported_candidate_coverage(
            candidates, Inventory(mmq_types={"q4_0", "q8_0"}))

        self.assertEqual(report["supported_types"], ["f32", "q8_0"])
        self.assertFalse(report["by_type"]["q4_0"]["supported"])
        self.assertTrue(report["by_type"]["q8_0"]["observed"])
        self.assertFalse(report["by_type"]["f32"]["observed"])
        self.assertIn("zero_alternative_reason", report["by_type"]["q4_0"])
        self.assertEqual(
            report["by_type"]["f32"]["by_architecture"]["gfx1100"]["alternative_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
