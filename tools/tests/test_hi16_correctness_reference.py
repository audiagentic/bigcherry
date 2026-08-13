"""HI16 offline correctness-reference evidence contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import autotune_schema as schema  # noqa: E402


def _native():
    return {
        "stable_name": "mmq:native:v1", "family": "mmq",
        "source_class": "native_wrapper", "implementation_version": 1,
        "architectures": ["gfx1100"], "graph_safe": True,
        "deterministic": True, "config": {},
    }


def _candidate():
    return {
        "stable_name": "mmq:q8_0:j64:fb0:v1", "family": "mmq",
        "source_class": "existing_runtime", "implementation_version": 1,
        "architectures": ["gfx1100"], "graph_safe": True,
        "deterministic": True, "config": {},
        "correctness_evidence": {
            "reference_path": "evidence/hi16/mmq-q8.json",
            "error_metrics": {"max_abs": 0.0, "max_rel": 0.0},
            "tolerances": {"max_abs": 1e-5, "max_rel": 1e-5},
            "signature_namespace": {
                "signature_schema_version": 1,
                "hardware_schema_version": 1,
            },
            "build_namespace": {
                "source_revision": "a" * 40,
                "manifest_hash": "b" * 32,
                "build_descriptor_hash": "c" * 64,
            },
        },
    }


def _manifest(variant_set="replay-slim", candidates=None):
    return {
        "artifact_version": 1, "variant_set": variant_set,
        "source_revision": "a" * 40, "manifest_hash": "b" * 32,
        "build_descriptor_hash": "c" * 64,
        "signature_schema_version": 1, "hardware_schema_version": 1,
        "architectures": ["gfx1100"],
        "candidates": candidates or [_native(), _candidate()],
    }


def test_non_native_candidate_with_namespace_bound_reference_is_valid():
    schema.validate_manifest(_manifest())


@pytest.mark.parametrize("field", ["reference_path", "error_metrics", "tolerances"])
def test_non_native_candidate_requires_correctness_proof_fields(field):
    manifest = _manifest()
    del manifest["candidates"][1]["correctness_evidence"][field]
    with pytest.raises(schema.SchemaError, match="correctness"):
        schema.validate_manifest(manifest)


@pytest.mark.parametrize("namespace", ["signature_namespace", "build_namespace"])
def test_non_native_candidate_requires_matching_namespace(namespace):
    manifest = _manifest()
    evidence = manifest["candidates"][1]["correctness_evidence"]
    if namespace == "signature_namespace":
        evidence[namespace]["signature_schema_version"] = 2
    else:
        evidence[namespace]["source_revision"] = "d" * 40
    with pytest.raises(schema.SchemaError, match="namespace"):
        schema.validate_manifest(manifest)


def test_inventory_native_only_profile_remains_valid_without_correctness_evidence():
    schema.validate_manifest(_manifest("inventory", [_native()]))


def test_non_native_candidate_without_evidence_is_rejected_in_production():
    manifest = _manifest()
    del manifest["candidates"][1]["correctness_evidence"]
    with pytest.raises(schema.SchemaError, match="non-native"):
        schema.validate_manifest(manifest)
