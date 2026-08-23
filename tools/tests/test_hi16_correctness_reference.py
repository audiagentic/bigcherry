"""HI16 offline correctness-reference evidence contracts.

HI67 (RV49) superseded the candidate-level check this file originally
covered: `_validate_correctness_reference()` validated a `correctness_evidence`
blob embedded on the CANDIDATE, but a Candidate is reusable across many
signatures/architectures, so "candidate C has evidence" was never a
meaningful production statement. The real gate now lives at replay-cache
export time (replay_cache.build()'s dispatch_db gate, tested in
test_replay_cache_promotion_gate.py and test_promotion_correctness_gate.py),
at the (dispatch, signature, hardware, candidate) binding granularity a
replay decision actually needs. Schema validation no longer requires or
inspects a `correctness_evidence` blob on the candidate at all -- these
tests now document that absence is a deliberate design decision, not a gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import autotune_schema as schema  # noqa: E402


def _native():
    return {
        "stable_name": "mmq:native:v1", "family": "mmq",
        "source_class": "native_wrapper", "implementation_version": 1,
        "architectures": ["gfx1100"], "graph_safe": True,
        "deterministic": True, "config": {},
    }


def _candidate(with_evidence=True):
    candidate = {
        "stable_name": "mmq:q8_0:j64:fb0:v1", "family": "mmq",
        "source_class": "existing_runtime", "implementation_version": 1,
        "architectures": ["gfx1100"], "graph_safe": True,
        "deterministic": True, "config": {},
    }
    if with_evidence:
        candidate["correctness_evidence"] = {
            "reference_path": "evidence/hi16/mmq-q8.json",
            "error_metrics": {"max_abs": 0.0, "max_rel": 0.0},
            "tolerances": {"max_abs": 1e-5, "max_rel": 1e-5},
        }
    return candidate


def _manifest(variant_set="replay-slim", candidates=None):
    return {
        "artifact_version": 1, "variant_set": variant_set,
        "source_revision": "a" * 40, "manifest_hash": "b" * 32,
        "build_descriptor_hash": "c" * 64,
        "signature_schema_version": 1, "hardware_schema_version": 1,
        "architectures": ["gfx1100"],
        "candidates": candidates or [_native(), _candidate()],
    }


def test_non_native_candidate_with_correctness_evidence_is_still_valid():
    # A candidate MAY still carry a `correctness_evidence` blob (harmless
    # extra metadata) -- schema validation neither requires nor inspects it.
    schema.validate_manifest(_manifest())


def test_non_native_candidate_without_evidence_is_also_valid_in_production():
    # HI67: this is the behavior change from HI16's original contract --
    # a production-variant-set manifest no longer needs any
    # `correctness_evidence` blob on its candidates. Real correctness proof
    # is enforced later, at replay-cache export time, against a real
    # dispatch_db and the exact binding about to ship.
    manifest = _manifest(candidates=[_native(), _candidate(with_evidence=False)])
    schema.validate_manifest(manifest)


def test_inventory_native_only_profile_remains_valid_without_correctness_evidence():
    schema.validate_manifest(_manifest("inventory", [_native()]))
