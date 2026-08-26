"""HI25 offline readiness contracts for future custom MMQ/MMVQ kernels."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import schema as schema # noqa: E402


def _candidate() -> dict:
    return {
        "stable_name": "mmq:q8_0:gfx1100:k5120:m6144:n512:v1",
        "family": "mmq",
        "source_class": "new_generated_variant",
        "implementation_version": 1,
        "architectures": ["gfx1100"],
        "architecture_mask": schema.architecture_mask(["gfx1100"]),
        "graph_safe": False,
        "deterministic": True,
        "config": {
            "custom_kernel": True,
            "candidate_identity": "mmq:q8_0:gfx1100:k5120:m6144:n512:v1",
            "correctness_reference": {
                "candidate": "mmq:native:v1",
                "path": "evidence/hi25/correctness.json",
            },
            "resource_report": {
                "path": "evidence/hi25/resources.json",
                "sha256": "a" * 64,
                "architectures": ["gfx1100"],
            },
            "benchmark_evidence": {
                "path": "evidence/hi25/benchmark.json",
                "sha256": "b" * 64,
                "status": "passed",
                "metric": "median_us",
                "baseline": "mmq:native:v1",
            },
        },
    }


def test_complete_custom_candidate_is_enableable():
    schema.validate_candidate(_candidate(), "candidate[0]")


@pytest.mark.parametrize(
    "field",
    ["correctness_reference", "resource_report", "benchmark_evidence"],
)
def test_custom_candidate_requires_each_evidence_record(field):
    candidate = _candidate()
    del candidate["config"][field]
    with pytest.raises(schema.SchemaError, match=field):
        schema.validate_candidate(candidate, "candidate[0]")


def test_custom_candidate_requires_matching_architecture_mask():
    candidate = _candidate()
    candidate["architecture_mask"] = 0
    with pytest.raises(schema.SchemaError, match="architecture_mask"):
        schema.validate_candidate(candidate, "candidate[0]")


def test_custom_candidate_is_limited_to_mmq_or_mmvq():
    candidate = _candidate()
    candidate["family"] = "blas"
    with pytest.raises(schema.SchemaError, match="MMQ or MMVQ"):
        schema.validate_candidate(candidate, "candidate[0]")


def test_custom_candidate_requires_stable_identity_binding():
    candidate = _candidate()
    candidate["config"]["candidate_identity"] = "mmq:other:v1"
    with pytest.raises(schema.SchemaError, match="candidate_identity"):
        schema.validate_candidate(candidate, "candidate[0]")


def test_custom_candidate_requires_passed_benchmark_and_reference_baseline():
    candidate = _candidate()
    candidate["config"]["benchmark_evidence"]["status"] = "failed"
    with pytest.raises(schema.SchemaError, match="status"):
        schema.validate_candidate(candidate, "candidate[0]")

    candidate = _candidate()
    candidate["config"]["correctness_reference"]["candidate"] = candidate["stable_name"]
    with pytest.raises(schema.SchemaError, match="distinct"):
        schema.validate_candidate(candidate, "candidate[0]")


def test_unmarked_existing_candidates_keep_legacy_schema():
    candidate = _candidate()
    candidate["config"] = {}
    candidate["source_class"] = "existing_runtime"
    schema.validate_candidate(candidate, "candidate[0]")
