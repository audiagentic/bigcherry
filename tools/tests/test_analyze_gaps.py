import unittest

from bigcherry.analyze_gaps import GapAnalysisError, analyze_gaps


def _gap(signature: str, *, pattern: str = "f32-small", family: str = "blas",
         calls: int = 0, est_bytes: int = 0, evidence: str = "evidence") -> dict:
    return {
        "kind": "transform-gap", "source_signature": signature,
        "hardware_provenance": {"digest": "b" * 32, "architecture": "gfx1201"},
        "build_provenance": {"source_revision": "c" * 40, "manifest_hash": "d" * 64,
                              "build_descriptor_hash": "e" * 64},
        "evidence_references": [evidence], "reason": "no transform served the signature",
        "pattern": pattern, "native_family": family,
        "transformations_tried": [{"id": 1, "name": "transpose", "reason": "rejected"}],
        "calls": calls, "est_bytes": est_bytes,
    }


class TestAnalyzeGaps(unittest.TestCase):
    def test_groups_deterministically_and_sums_impact(self):
        records = [_gap("a" * 32, pattern="wide", family="MMVF", calls=2,
                         est_bytes=100, evidence="run/b"),
                   _gap("b" * 32, pattern="wide", family="mmvf", calls=3,
                         est_bytes=200, evidence="run/a"),
                   _gap("c" * 32, pattern="small", family="blas", calls=7,
                         est_bytes=50, evidence="run/c")]
        report = analyze_gaps(records)
        self.assertEqual(report["record_count"], 3)
        self.assertEqual([(g["pattern"], g["native_family"]) for g in report["groups"]],
                         [("small", "blas"), ("wide", "mmvf")])
        self.assertEqual(report["groups"][1]["count"], 2)
        self.assertEqual(report["groups"][1]["calls"], 5)
        self.assertEqual(report["groups"][1]["estimated_bytes"], 300)
        self.assertEqual(report["groups"][1]["source_signatures"], ["a" * 32, "b" * 32])

    def test_attempts_are_ignored_but_still_validated(self):
        attempt = _gap("a" * 32, evidence="attempt")
        attempt.update({"kind": "transform-attempt", "transformation": {
            "id": 1, "name": "t", "source": "predefined"}, "outcome": "rejected"})
        for key in ("pattern", "native_family", "transformations_tried"):
            attempt.pop(key)
        report = analyze_gaps([attempt])
        self.assertEqual(report["groups"], [])
        self.assertIsNone(report["provenance"])

    def test_mixed_provenance_is_rejected(self):
        other = _gap("b" * 32, evidence="other")
        other["hardware_provenance"]["architecture"] = "gfx1100"
        with self.assertRaisesRegex(GapAnalysisError, "mixed"):
            analyze_gaps([_gap("a" * 32), other])

    def test_duplicate_evidence_is_rejected_case_insensitively(self):
        with self.assertRaisesRegex(GapAnalysisError, "duplicated"):
            analyze_gaps([_gap("a" * 32, evidence="Run/Evidence"),
                          _gap("b" * 32, evidence="run/evidence")])

    def test_invalid_records_fail_closed(self):
        bad = _gap("a" * 32)
        bad["calls"] = -1
        with self.assertRaisesRegex(GapAnalysisError, "calls"):
            analyze_gaps([bad])


if __name__ == "__main__":
    unittest.main()
