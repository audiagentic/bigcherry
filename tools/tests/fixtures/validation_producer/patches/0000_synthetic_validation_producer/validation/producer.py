"""PA36-F step 7 synthetic producer fixture (GPT design req_8ec9b90c05f84a30,
section 7): proves a brand-new producer can be declared/selected/executed/
validated/evidence-bound through execute_validation_producer() with ZERO
shared implementation-code change.

Imports only bigcherry.patch.validation_producer and bigcherry.patch.
validation, per the design's explicit import restriction -- this module
must never import bigcherry.patch.validation_campaign (see
test_producer_modules_cannot_import_validation_campaign,
tools/tests/patch/test_validation_producer_structure.py).
"""

from __future__ import annotations

from bigcherry.patch import validation as pv
from bigcherry.patch import validation_producer as vp

# Patch-local knowledge only: which check_id is scoped to which of this
# fixture's two synthetic contracts. A real producer's own validation.toml
# declares the same scoping (see validation.toml alongside this file) --
# this dict just lets the producer emit ProducerCheckResults that match it.
_CHECK_CONTRACTS = {
    "syn-c1-correctness": "SYN-C1",
    "syn-c2-correctness": "SYN-C2",
}


def _fake_build_identity(role: str, token: str) -> dict[str, object]:
    # This synthetic producer never builds anything real -- it proves the
    # dispatcher's typed plumbing, not a real compile -- so this is a
    # well-FORMED (matches patch_validation_evidence.BUILD_IDENTITY_KEYS),
    # not real, CompletedBuildEvidence.campaign_identity() shape. A real
    # producer that materializes/builds its own isolated control/subject
    # worktrees (RD12's shape) returns the genuine one instead.
    return {
        "effective_build_id": f"{role}-{token}-build-id",
        "compile_verification_id": f"{role}-compile-verification",
        "compile_commands_digest": f"{role}-compile-commands-digest",
        "hip_compile_commands_digest": f"{role}-hip-compile-commands-digest",
        "runtime_bundle_hash": f"{role}-runtime-bundle-hash",
        "runtime_artifacts": {f"{role}.bin": "a" * 64},
    }


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    token = ctx.inputs["token"]
    artifact = ctx.runtime.write_artifact(
        name="synthetic-correctness.json",
        payload={"token": token, "checks": sorted(_CHECK_CONTRACTS)},
    )

    check_results = tuple(
        vp.ProducerCheckResult(
            check_id=check_id,
            contract_ids=(contract_id,),
            validation_result=pv.ValidationResult(
                check_id=check_id, capability="correctness",
                status=pv.PASS, summary=f"synthetic pass for {contract_id}",
                artifacts=(artifact,),
            ),
            disposition={"passed": True, "contract_id": contract_id},
        )
        for check_id, contract_id in sorted(_CHECK_CONTRACTS.items())
    )

    return vp.ProducerResult(
        correctness=None,
        # A real producer that materializes/builds its OWN isolated
        # control/subject worktrees (RD12's shape) replaces these; this
        # synthetic producer never builds anything real -- it proves the
        # dispatcher's TYPED plumbing, not a real compile.
        validation_build_identities={
            "control": _fake_build_identity("control", token),
            "subject": _fake_build_identity("subject", token),
        },
        activation_evidence=None,
        performance_evidence=None,
        trace_evidence=None,
        check_results=check_results,
        lane_effects=(),
        emitted_artifacts=frozenset({"synthetic-correctness.json"}),
    )
