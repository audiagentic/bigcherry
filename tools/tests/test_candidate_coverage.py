import unittest

from bigcherry.autotune_catalog import Inventory, candidate_coverage


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


if __name__ == "__main__":
    unittest.main()
