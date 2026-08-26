"""HI26 offline production-readiness gates for experimental fused candidates."""

import unittest

from bigcherry.tuning import schema as schema


def _candidate(**overrides):
    candidate = {
        "stable_name": "fused:mmq:q8_0:v1",
        "family": "mmq",
        "source_class": "new_generated_variant",
        "implementation_version": 1,
        "architectures": ["gfx1100"],
        "graph_safe": True,
        "deterministic": True,
        "config": {"type": "q8_0"},
        "experimental_fused": True,
        "graph_safety_evidence": {
            "status": "validated", "capture_observed": True,
        },
        "correctness_evidence": {
            "status": "validated", "reference": "native",
            "comparison": "bitwise",
            "reference_path": "evidence/hi26/correctness.json",
            "error_metrics": {"max_abs": 0.0},
            "tolerances": {"max_abs": 0.0},
            "signature_namespace": {
                "signature_schema_version": 1,
                "hardware_schema_version": 1,
            },
            "build_namespace": {
                "source_revision": "a" * 40,
                "manifest_hash": "b" * 32,
            },
        },
        "workspace_evidence": {
            "status": "validated", "peak_bytes": 4096,
        },
        "provenance_evidence": {
            "status": "validated", "source_revision": "a" * 40,
            "evidence_references": ["run:hi26:001"],
        },
    }
    candidate.update(overrides)
    return candidate


def _manifest(variant_set, candidate):
    native = {
        "stable_name": "native:mmq:v1",
        "family": "mmq",
        "source_class": "native_wrapper",
        "implementation_version": 1,
        "architectures": ["gfx1100"],
        "graph_safe": True,
        "deterministic": True,
        "config": {},
    }
    return {
        "artifact_version": 1,
        "variant_set": variant_set,
        "source_revision": "a" * 40,
        "manifest_hash": "b" * 32,
        "signature_schema_version": 1,
        "hardware_schema_version": 1,
        "architectures": ["gfx1100"],
        "candidates": [native, candidate],
    }


class HI26OfflineReadinessTests(unittest.TestCase):
    def test_experimental_fused_candidate_requires_all_evidence_in_production(self):
        for field in (
            "graph_safety_evidence", "correctness_evidence",
            "workspace_evidence", "provenance_evidence",
        ):
            candidate = _candidate()
            del candidate[field]
            with self.subTest(field=field), self.assertRaisesRegex(
                    schema.SchemaError, "requires explicit"):
                schema.validate_manifest(_manifest("replay-slim", candidate))

    def test_experimental_fused_candidate_is_rejected_without_graph_capture(self):
        candidate = _candidate(graph_safety_evidence={
            "status": "validated", "capture_observed": False,
        })
        with self.assertRaisesRegex(schema.SchemaError, "capture_observed"):
            schema.validate_manifest(_manifest("replay-full", candidate))

    def test_experimental_fused_candidate_requires_validated_correctness_and_workspace(self):
        candidate = _candidate(correctness_evidence={
            "status": "validated", "reference": "native",
        })
        with self.assertRaisesRegex(schema.SchemaError, "reference and comparison"):
            schema.validate_manifest(_manifest("replay-slim", candidate))

        candidate = _candidate(workspace_evidence={
            "status": "validated", "peak_bytes": -1,
        })
        with self.assertRaisesRegex(schema.SchemaError, "peak_bytes"):
            schema.validate_manifest(_manifest("replay-slim", candidate))

    def test_experimental_fused_candidate_requires_traceable_provenance(self):
        candidate = _candidate(provenance_evidence={
            "status": "validated", "source_revision": "a" * 40,
            "evidence_references": ["run:hi26:001", "run:hi26:001"],
        })
        with self.assertRaisesRegex(schema.SchemaError, "unique"):
            schema.validate_manifest(_manifest("replay-full", candidate))

    def test_experimental_marker_is_allowed_in_nonproduction_profiles(self):
        candidate = _candidate()
        schema.validate_manifest(_manifest("workload-max", candidate))

    def test_unmarked_catalog_candidates_keep_existing_production_contract(self):
        candidate = _candidate(experimental_fused=False)
        schema.validate_manifest(_manifest("replay-slim", candidate))


if __name__ == "__main__":
    unittest.main()
