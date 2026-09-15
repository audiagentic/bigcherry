"""Hardware-free framework-configuration apply/build evidence checks.

These read the ``configuration_evidence`` bound by
``--framework-configuration`` (single-composition apply proof, plus
production/diagnostic build completion) -- not a control/subject source
diff, which is what the generic builtin ``apply``/``build`` validators
require and this patch does not have (it is a local framework-configuration
package, not an RD/contract-bound patch). No patch-specific "configuration"
capability check is added here: GPT review (dev-gpt-agent session
ses_c2892cdae7f14feb, req_84fb9ad1e65c4ec0) confirmed apply+build alone is
the correct bar for these packages, and explicitly warned against adding a
lightweight custom check just to look stronger than the real behavioral
evidence apply+build actually proves.
"""

from __future__ import annotations

from typing import Any

from bigcherry.patch.validation import (
    BLOCKED,
    FAIL,
    PASS,
    ArtifactRef,
    ValidationResult,
    _artifact_is_bound,
)


def check_apply(ctx: Any) -> ValidationResult:
    evidence = getattr(ctx, "configuration_evidence", None)
    if not isinstance(evidence, dict):
        return ValidationResult("apply", "apply", BLOCKED, "single-composition apply evidence is required")
    apply = evidence.get("apply")
    if not isinstance(apply, dict):
        return ValidationResult("apply", "apply", BLOCKED, "single-composition apply evidence is missing")
    if apply.get("single_composition") is not True:
        return ValidationResult("apply", "apply", FAIL, "apply evidence is not for one framework composition")
    if apply.get("verified") is not True or apply.get("idempotent") is not True:
        return ValidationResult("apply", "apply", FAIL, "framework source apply/idempotence proof failed")
    artifact = apply.get("artifact")
    if not isinstance(artifact, dict) or not _artifact_is_bound(artifact, getattr(ctx, "run_dir", None)):
        return ValidationResult("apply", "apply", BLOCKED, "bound single-composition apply artifact is required")
    return ValidationResult("apply", "apply", PASS, "single framework composition apply and idempotence are verified",
                            artifacts=(ArtifactRef("apply", str(artifact["path"]), str(artifact["sha256"])),))


def check_build(ctx: Any) -> ValidationResult:
    evidence = getattr(ctx, "configuration_evidence", None)
    if not isinstance(evidence, dict):
        return ValidationResult("build", "build", BLOCKED, "framework production/diagnostic build evidence is required")
    builds = evidence.get("builds")
    if not isinstance(builds, dict):
        return ValidationResult("build", "build", BLOCKED, "production and diagnostic build evidence is missing")
    artifacts: list[ArtifactRef] = []
    for role in ("production", "diagnostic"):
        item = builds.get(role)
        if not isinstance(item, dict) or item.get("completed") is not True:
            return ValidationResult("build", "build", FAIL, f"{role} framework build is not completed")
        artifact = item.get("artifact")
        if not isinstance(artifact, dict) or not _artifact_is_bound(artifact, getattr(ctx, "run_dir", None)):
            return ValidationResult("build", "build", BLOCKED, f"bound {role} framework build artifact is required")
        artifacts.append(ArtifactRef(role, str(artifact["path"]), str(artifact["sha256"])))
    return ValidationResult("build", "build", PASS, "completed production and diagnostic framework builds are verified",
                            artifacts=tuple(artifacts))
