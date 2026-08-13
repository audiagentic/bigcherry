from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigcherry.multi_gpu_validate import (  # noqa: E402
    MultiGPUEvidenceError,
    validate_multi_gpu_claim,
    validate_multi_gpu_evidence,
)


class MultiGPUEvidenceTests(unittest.TestCase):
    def _evidence(self) -> dict:
        return {
            "topology": {
                "device_count": 2,
                "ordinals": [0, 1],
                "devices": [
                    {"ordinal": 0, "identity": "0000:03:00.0"},
                    {"ordinal": 1, "identity": "0000:43:00.0"},
                ],
            },
            "graph": {"mode": "enabled", "capture_observed": True},
            "per_device": [
                {"ordinal": 0, "dispatches": 12, "signatures": 3, "winners": ["a"]},
                {"ordinal": 1, "dispatches": 12, "signatures": 3, "winners": ["b"]},
            ],
        }

    def test_complete_evidence_is_accepted(self):
        validate_multi_gpu_evidence(self._evidence())

    def test_disabled_graph_mode_is_explicitly_accepted(self):
        evidence = self._evidence()
        evidence["graph"] = {"mode": "disabled", "capture_observed": False}
        validate_multi_gpu_evidence(evidence)

    def test_missing_device_evidence_is_rejected(self):
        evidence = self._evidence()
        evidence["per_device"].pop()
        with self.assertRaisesRegex(MultiGPUEvidenceError, "per_device"):
            validate_multi_gpu_evidence(evidence)

    def test_mismatched_ordinals_are_rejected(self):
        evidence = self._evidence()
        evidence["per_device"][1]["ordinal"] = 2
        with self.assertRaisesRegex(MultiGPUEvidenceError, "cover topology"):
            validate_multi_gpu_evidence(evidence)

    def test_graph_claim_without_capture_is_rejected(self):
        evidence = self._evidence()
        evidence["graph"]["capture_observed"] = False
        with self.assertRaisesRegex(MultiGPUEvidenceError, "observed capture"):
            validate_multi_gpu_evidence(evidence)

    def test_release_validation_only_checks_present_multi_gpu_claim(self):
        validate_multi_gpu_claim({"claim": "validated"})
        with self.assertRaises(MultiGPUEvidenceError):
            validate_multi_gpu_claim({"multi_gpu": {}})


if __name__ == "__main__":
    unittest.main()
