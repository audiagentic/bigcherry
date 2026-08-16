"""Tuner artifact serialization source-contract tests.

The measurements JSONL must stay machine-readable for every row, including
the stub rows written when tuning is disabled (poisoned configuration,
invalid policy). A single empty field serialised as `"field":` corrupts the
whole downstream parse, so the contract has two halves:

- every early-return stub populates the fields the flush prints that have no
  valid JSON default (canonical_json in particular), and
- the flush itself fails closed: an empty canonical emits null instead of
  nothing.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"


class TunerArtifactJsonContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = TUNER.read_text(encoding="utf-8")

    def test_poisoned_stub_sets_canonical_json(self):
        # The poisoned stub skips the cold-path setup; it must still fill
        # canonical_json from the in-scope signature or every row written
        # after a fatal failure is invalid JSON.
        self.assertIn(
            "poisoned.canonical_json = ggml_hip_signature_json(sig, true);",
            self.tuner)

    def test_invalid_config_stub_sets_canonical_json(self):
        self.assertIn(
            "invalid.canonical_json = ggml_hip_signature_json(sig, true);",
            self.tuner)

    def test_flush_emits_null_for_empty_canonical(self):
        # Fail closed at the single sink: no future stub path may be able to
        # emit `"canonical":,` -- the only flush field without a valid JSON
        # default.
        self.assertIn(
            'r.canonical_json.empty() ? "null" : r.canonical_json.c_str(),',
            self.tuner)


if __name__ == "__main__":
    unittest.main()
