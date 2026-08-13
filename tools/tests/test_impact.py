import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import impact  # noqa: E402


class ImpactTests(unittest.TestCase):
    def _data(self):
        observations = [{"signature": "a", "calls": 10, "native": "blas:native:v1"}]
        results = [{
            "signature": "a", "native": "blas:native:v1", "winner": "mmq:candidate:v1",
            "candidates": [
                {"name": "blas:native:v1", "status": "ok", "median_us": 10.0},
                {"name": "mmq:candidate:v1", "status": "ok", "median_us": 8.0},
            ],
        }]
        return observations, results

    def test_call_weighted_saving_and_coverage_are_explicit(self):
        observations, results = self._data()
        report = impact.predicted_saving(observations, results)
        self.assertEqual(report["coverage"].calls_covered, 10)
        self.assertAlmostEqual(report["saving_pct"], 20.0)
        self.assertEqual(report["rows"][0]["saved_us"], 20.0)

    def test_missing_candidate_does_not_claim_coverage(self):
        observations, results = self._data()
        results[0]["winner"] = "missing"
        report = impact.predicted_saving(observations, results)
        self.assertEqual(report["coverage"].calls_covered, 0)

    def test_decode_effect_requires_fraction_in_range(self):
        self.assertAlmostEqual(impact.expected_decode_effect(10.0, 0.3), 3.0)
        with self.assertRaisesRegex(impact.ImpactError, "between"):
            impact.expected_decode_effect(10.0, 1.1)


if __name__ == "__main__":
    unittest.main()
