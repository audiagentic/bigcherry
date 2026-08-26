"""HI19 namespace separation contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.identity_separation import (  # noqa: E402
    IdentitySeparationError, validate_measurement_identity,
)


class IdentitySeparationTests(unittest.TestCase):
    def test_legacy_result_without_component_identities_remains_valid(self):
        row = {"kind": "result", "dispatch": "a" * 32,
               "winner": "mmq:native:v1",
               "candidates": [{"name": "mmq:native:v1"}]}
        self.assertIs(validate_measurement_identity(row), row)

    def test_legacy_signature_is_not_substituted_as_hardware(self):
        row = {"dispatch": "a" * 32, "signature": "b" * 32}
        self.assertIs(validate_measurement_identity(row), row)
        self.assertNotIn("hardware", row)

    def test_candidate_cannot_carry_signature_identity(self):
        with self.assertRaisesRegex(IdentitySeparationError, "conflates"):
            validate_measurement_identity({
                "dispatch": "a" * 32, "winner": "c",
                "candidates": [{"name": "c", "signature": "b" * 32}],
            })

    def test_observation_cannot_carry_hardware_identity(self):
        with self.assertRaisesRegex(IdentitySeparationError, "durable identity"):
            validate_measurement_identity({
                "dispatch": "a" * 32,
                "observations": [{"hardware": "b" * 32}],
            })

    def test_header_is_build_context_not_operation_identity(self):
        with self.assertRaisesRegex(IdentitySeparationError, "per-operation"):
            validate_measurement_identity(
                {"dispatch": "a" * 32},
                header={"signature": "b" * 32},
            )

    def test_manifest_hash_is_validated_but_legacy_descriptor_is_not(self):
        validate_measurement_identity(
            {"dispatch": "a" * 32},
            header={"manifest_hash": "b" * 32,
                    "build_descriptor_hash": "legacy-label"},
        )
        with self.assertRaises(IdentitySeparationError):
            validate_measurement_identity(
                {"dispatch": "a" * 32}, header={"manifest_hash": "bad"})


if __name__ == "__main__":
    unittest.main()
