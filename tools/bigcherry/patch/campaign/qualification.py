"""Qualification-evidence manifests for the patch-validation campaign:
execution/artifact binding, manifest build and validation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import shutil
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment.attestation import (
    compare_execution_identity,
    ExecutionAttestation,
    ExecutionIdentity,
)
from bigcherry.patch.campaign.build import (
    _canonical_json,
    _canonical_json_mapping,
    _sha256_file,
    PatchCampaignError,
)


# ---------------------------------------------------------------------------
# Externally measured contract-evidence provenance
#
# RD39-43 can consume measurements produced outside this module, but those
# measurements are promotion-eligible only when their complete provenance is
# bound here.  This is deliberately campaign-local rather than part of
# experiment.contract: contracts define WHAT must be proven; this structure
# proves WHICH concrete executions supplied the facts. (GPT design,
# req_ea5a7ab7d6634e09.)
# ---------------------------------------------------------------------------

_QUALIFICATION_EVIDENCE_SCHEMA_VERSION = 1


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


_REQUIRED_BUILD_IDENTITY_KEYS = frozenset(
    {
        "effective_build_id",
        "compile_verification_id",
        "compile_commands_digest",
        "hip_compile_commands_digest",
        "runtime_bundle_hash",
        "runtime_artifacts",
    }
)


_REQUIRED_SOURCE_IDENTITY_KEYS = frozenset(
    {
        "resolved_revision",
        "source_tree",
        "composition",
    }
)


_RAW_EVIDENCE_KINDS = frozenset(
    {
        "benchmark_raw",
        "correctness_raw",
        "logits_raw",
        "trigger_raw",
    }
)


@dataclass(frozen=True)
class QualificationEvidenceExecution:
    """One concrete process execution contributing qualification evidence."""

    name: str
    role: str
    command: tuple[str, ...]
    # Explicit selector state, including keys deliberately unset.
    selector_env: tuple[tuple[str, str | None], ...]
    expected_execution: ExecutionIdentity
    observed_execution: ExecutionAttestation

    def document(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "command": list(self.command),
            "selector_env": {k: v for k, v in self.selector_env},
            "expected_execution": {
                "backend": self.expected_execution.backend,
                "architectures": list(self.expected_execution.architectures),
                "locators": (
                    list(self.expected_execution.locators)
                    if self.expected_execution.locators is not None
                    else {}
                ),
            },
            "observed_execution": self.observed_execution.document(),
        }


def make_qualification_execution(
    *,
    name: str,
    role: str,
    command: Iterable[str],
    selector_env: Mapping[str, str | None],
    expected_execution: ExecutionIdentity,
    observed_execution: ExecutionAttestation,
) -> QualificationEvidenceExecution:
    if not name:
        raise PatchCampaignError("qualification execution name must be non-empty")
    if not role:
        raise PatchCampaignError(
            f"qualification execution {name!r}: role must be non-empty"
        )

    argv = tuple(str(arg) for arg in command)
    if not argv:
        raise PatchCampaignError(f"qualification execution {name!r}: command is empty")

    # HIP/ROCR selector state is identity-relevant even when one is explicitly
    # absent.  Recording only HIP_VISIBLE_DEVICES would lose the double-filter
    # condition that has caused real invalid runs in this project.
    missing_selector_keys = {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"} - set(
        selector_env
    )
    if missing_selector_keys:
        raise PatchCampaignError(
            f"qualification execution {name!r}: selector_env missing "
            f"{sorted(missing_selector_keys)}"
        )

    if expected_execution.locators is None:
        raise PatchCampaignError(
            f"qualification execution {name!r}: expected device locators are "
            "required for promotion-grade provenance"
        )

    reasons = compare_execution_identity(expected_execution, observed_execution)
    if reasons:
        raise PatchCampaignError(
            f"qualification execution {name!r}: execution attestation mismatch: "
            f"{', '.join(reasons)}"
        )

    return QualificationEvidenceExecution(
        name=name,
        role=role,
        command=argv,
        selector_env=tuple(sorted(selector_env.items())),
        expected_execution=expected_execution,
        observed_execution=observed_execution,
    )


@dataclass(frozen=True)
class QualificationEvidenceArtifact:
    kind: str
    execution: str
    path: str
    sha256: str
    size_bytes: int

    def document(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "execution": self.execution,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def bind_qualification_raw_artifact(
    *,
    run_dir: Path,
    kind: str,
    execution: str,
    source: Path,
) -> QualificationEvidenceArtifact:
    """Copy immutable raw evidence into run_dir and bind its SHA-256."""

    if kind not in _RAW_EVIDENCE_KINDS:
        raise PatchCampaignError(
            f"unknown qualification raw-artifact kind {kind!r}; "
            f"expected one of {sorted(_RAW_EVIDENCE_KINDS)}"
        )
    if not execution:
        raise PatchCampaignError("raw evidence execution name must be non-empty")
    if not source.is_file():
        raise PatchCampaignError(f"raw evidence file does not exist: {source}")

    digest = _sha256_file(source)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.name)
    target = run_dir / "artifacts" / "raw" / f"{kind}-{digest[:16]}-{safe_name}"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if _sha256_file(target) != digest:
            raise PatchCampaignError(
                f"existing bound evidence artifact has wrong digest: {target}"
            )
    else:
        shutil.copyfile(source, target)

    return QualificationEvidenceArtifact(
        kind=kind,
        execution=execution,
        path=target.relative_to(run_dir).as_posix(),
        sha256=digest,
        size_bytes=target.stat().st_size,
    )


@dataclass(frozen=True)
class QualificationEvidenceManifest:
    schema_version: int
    campaign_id: str

    contract_id: str
    contract_hash: str
    patch_ids: tuple[str, ...]
    base_revision: str

    target_metric: str
    positive_pct_deltas: tuple[float, ...]
    control_pct_deltas: tuple[float, ...]

    correctness_results: tuple[object, ...]
    trigger_evidence: tuple[object, ...]

    # Canonical JSON strings make these flexible identity documents deeply
    # immutable despite the outer dataclass being frozen.
    subject_source_identity_json: str
    control_source_identity_json: str
    subject_build_identity_json: str
    control_build_identity_json: str

    model_ref: str
    model_sha256: str
    model_size_bytes: int

    executions: tuple[QualificationEvidenceExecution, ...]
    artifacts: tuple[QualificationEvidenceArtifact, ...]

    manifest_sha256: str

    def document(self, *, include_manifest_sha256: bool = True) -> dict[str, object]:
        correctness_doc = [
            {
                "check": r.check,
                "passed": r.passed,
                "detail": r.detail,
            }
            for r in self.correctness_results
        ]
        trigger_doc = [
            {
                "role": e.role,
                "lane_id": e.lane_id,
                "candidate_launches": e.candidate_launches,
                "expected_route_selected": e.expected_route_selected,
            }
            for e in self.trigger_evidence
        ]

        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "patch_ids": list(self.patch_ids),
            "base_revision": self.base_revision,
            "target_metric": self.target_metric,
            "positive_pct_deltas": list(self.positive_pct_deltas),
            "control_pct_deltas": list(self.control_pct_deltas),
            "correctness_results": correctness_doc,
            "trigger_evidence": trigger_doc,
            "source_identity": {
                "subject": json.loads(self.subject_source_identity_json),
                "control": json.loads(self.control_source_identity_json),
            },
            "build_identity": {
                "subject": json.loads(self.subject_build_identity_json),
                "control": json.loads(self.control_build_identity_json),
            },
            "model": {
                "ref": self.model_ref,
                "sha256": self.model_sha256,
                "size_bytes": self.model_size_bytes,
            },
            "executions": [e.document() for e in self.executions],
            "artifacts": [a.document() for a in self.artifacts],
        }

        if include_manifest_sha256:
            result["manifest_sha256"] = self.manifest_sha256

        return result


def _qualification_manifest_digest(
    manifest: QualificationEvidenceManifest,
) -> str:
    payload = manifest.document(include_manifest_sha256=False)
    return hashlib.sha256(
        _canonical_json(payload, label="qualification evidence manifest").encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_qualification_manifest_inputs(
    *,
    campaign_id: str,
    patch_ids_t: tuple[str, ...],
    base_revision: str,
    target_metric: str,
    positive_t: tuple[float, ...],
    control_t: tuple[float, ...],
    correctness_t: tuple,
    trigger_t: tuple,
    model_ref: str,
    model_path: Path,
    executions_t: tuple[QualificationEvidenceExecution, ...],
    artifacts_t: tuple[QualificationEvidenceArtifact, ...],
) -> None:
    """Fail closed on any missing, malformed or unbound manifest input."""
    if not campaign_id:
        raise PatchCampaignError("qualification manifest campaign_id is required")
    if not patch_ids_t or len(set(patch_ids_t)) != len(patch_ids_t):
        raise PatchCampaignError(
            "qualification manifest patch_ids must be non-empty and unique"
        )
    if not re.fullmatch(r"[0-9a-fA-F]{40}", base_revision):
        raise PatchCampaignError(
            "qualification manifest base_revision must be a full 40-hex SHA"
        )
    if not target_metric:
        raise PatchCampaignError("qualification manifest target_metric is required")

    if not positive_t or not control_t:
        raise PatchCampaignError(
            "qualification manifest requires both positive and control measurements"
        )
    if len(positive_t) != len(control_t):
        raise PatchCampaignError(
            "qualification manifest positive/control measurement counts differ"
        )
    if not all(math.isfinite(v) for v in (*positive_t, *control_t)):
        raise PatchCampaignError(
            "qualification manifest percentage deltas must all be finite"
        )

    if not correctness_t:
        raise PatchCampaignError(
            "qualification manifest requires explicit correctness results"
        )
    if not trigger_t:
        raise PatchCampaignError(
            "qualification manifest requires explicit trigger evidence"
        )

    if not model_ref:
        raise PatchCampaignError("qualification manifest model_ref is required")
    if not model_path.is_file():
        raise PatchCampaignError(
            f"qualification manifest model does not exist: {model_path}"
        )

    execution_names = [e.name for e in executions_t]
    if not execution_names or len(set(execution_names)) != len(execution_names):
        raise PatchCampaignError(
            "qualification manifest executions must be non-empty and uniquely named"
        )

    # These four evidence classes must all have raw files bound.  A logit file
    # can satisfy the correctness obligation while correctness_raw can carry
    # the parsed checker output; require at least one of those two.
    artifact_kinds = {a.kind for a in artifacts_t}
    if "benchmark_raw" not in artifact_kinds:
        raise PatchCampaignError("qualification manifest has no benchmark_raw artifact")
    if not ({"correctness_raw", "logits_raw"} & artifact_kinds):
        raise PatchCampaignError(
            "qualification manifest has no correctness_raw/logits_raw artifact"
        )
    if "trigger_raw" not in artifact_kinds:
        raise PatchCampaignError("qualification manifest has no trigger_raw artifact")

    known_executions = set(execution_names)
    for artifact in artifacts_t:
        if artifact.execution not in known_executions:
            raise PatchCampaignError(
                f"artifact {artifact.path!r} references unknown execution "
                f"{artifact.execution!r}"
            )


def build_qualification_evidence_manifest(
    *,
    campaign_id: str,
    contract: object,
    patch_ids: Iterable[str],
    base_revision: str,
    target_metric: str,
    positive_pct_deltas: Iterable[float],
    control_pct_deltas: Iterable[float],
    correctness_results: Mapping[str, object],
    trigger_evidence: Iterable[object],
    subject_source_identity: Mapping[str, object],
    control_source_identity: Mapping[str, object],
    subject_build_identity: Mapping[str, object],
    control_build_identity: Mapping[str, object],
    model_ref: str,
    model_path: Path,
    executions: Iterable[QualificationEvidenceExecution],
    artifacts: Iterable[QualificationEvidenceArtifact],
) -> QualificationEvidenceManifest:
    """Construct one immutable, content-bound promotion evidence manifest."""

    patch_ids_t = tuple(patch_ids)
    positive_t = tuple(float(v) for v in positive_pct_deltas)
    control_t = tuple(float(v) for v in control_pct_deltas)
    correctness_t = tuple(correctness_results[k] for k in sorted(correctness_results))
    trigger_t = tuple(trigger_evidence)
    executions_t = tuple(executions)
    artifacts_t = tuple(artifacts)

    _validate_qualification_manifest_inputs(
        campaign_id=campaign_id,
        patch_ids_t=patch_ids_t,
        base_revision=base_revision,
        target_metric=target_metric,
        positive_t=positive_t,
        control_t=control_t,
        correctness_t=correctness_t,
        trigger_t=trigger_t,
        model_ref=model_ref,
        model_path=model_path,
        executions_t=executions_t,
        artifacts_t=artifacts_t,
    )


    def _checked_identity(
        value: Mapping[str, object],
        *,
        label: str,
        required_keys: frozenset[str],
    ) -> str:
        missing = required_keys - set(value)
        if missing:
            raise PatchCampaignError(
                f"{label} missing identity fields: {sorted(missing)}"
            )
        return _canonical_json_mapping(value, label=label)

    subject_source_json = _checked_identity(
        subject_source_identity,
        label="subject_source_identity",
        required_keys=_REQUIRED_SOURCE_IDENTITY_KEYS,
    )
    control_source_json = _checked_identity(
        control_source_identity,
        label="control_source_identity",
        required_keys=_REQUIRED_SOURCE_IDENTITY_KEYS,
    )
    subject_build_json = _checked_identity(
        subject_build_identity,
        label="subject_build_identity",
        required_keys=_REQUIRED_BUILD_IDENTITY_KEYS,
    )
    control_build_json = _checked_identity(
        control_build_identity,
        label="control_build_identity",
        required_keys=_REQUIRED_BUILD_IDENTITY_KEYS,
    )

    manifest = QualificationEvidenceManifest(
        schema_version=_QUALIFICATION_EVIDENCE_SCHEMA_VERSION,
        campaign_id=campaign_id,
        contract_id=contract.id,
        contract_hash=contract.contract_hash,
        patch_ids=patch_ids_t,
        base_revision=base_revision.lower(),
        target_metric=target_metric,
        positive_pct_deltas=positive_t,
        control_pct_deltas=control_t,
        correctness_results=correctness_t,
        trigger_evidence=trigger_t,
        subject_source_identity_json=subject_source_json,
        control_source_identity_json=control_source_json,
        subject_build_identity_json=subject_build_json,
        control_build_identity_json=control_build_json,
        model_ref=model_ref,
        model_sha256=_sha256_file(model_path),
        model_size_bytes=model_path.stat().st_size,
        executions=executions_t,
        artifacts=artifacts_t,
        manifest_sha256="",
    )

    return dataclasses.replace(
        manifest,
        manifest_sha256=_qualification_manifest_digest(manifest),
    )


def validate_qualification_evidence_manifest(
    manifest: QualificationEvidenceManifest,
    *,
    contract: object,
    run_dir: Path,
) -> None:
    """Fail closed before manually gathered measurements reach contract gates."""

    if manifest.schema_version != _QUALIFICATION_EVIDENCE_SCHEMA_VERSION:
        raise PatchCampaignError(
            f"unsupported qualification evidence schema {manifest.schema_version!r}"
        )
    if manifest.contract_id != contract.id:
        raise PatchCampaignError(
            f"qualification manifest contract mismatch: "
            f"{manifest.contract_id!r} != {contract.id!r}"
        )
    if manifest.contract_hash != contract.contract_hash:
        raise PatchCampaignError(
            "qualification manifest was produced for a different contract hash"
        )

    expected_digest = _qualification_manifest_digest(manifest)
    if (
        not _SHA256_RE.fullmatch(manifest.manifest_sha256)
        or manifest.manifest_sha256 != expected_digest
    ):
        raise PatchCampaignError(
            "qualification manifest content hash is missing or invalid"
        )

    if not _SHA256_RE.fullmatch(manifest.model_sha256):
        raise PatchCampaignError("qualification manifest model SHA-256 is invalid")

    for execution in manifest.executions:
        reasons = compare_execution_identity(
            execution.expected_execution,
            execution.observed_execution,
        )
        if reasons:
            raise PatchCampaignError(
                f"qualification execution {execution.name!r} no longer "
                f"validates: {', '.join(reasons)}"
            )

    for artifact in manifest.artifacts:
        path = run_dir / artifact.path
        if not path.is_file():
            raise PatchCampaignError(f"bound qualification artifact is missing: {path}")
        if path.stat().st_size != artifact.size_bytes:
            raise PatchCampaignError(
                f"bound qualification artifact size changed: {path}"
            )
        actual = _sha256_file(path)
        if actual != artifact.sha256:
            raise PatchCampaignError(
                f"bound qualification artifact digest changed: {path}"
            )


def _lane_effect_from_pct_deltas(
    *,
    role: str,
    metric: str,
    pct_deltas: list[float],
) -> "experiment_contract.LaneEffect":
    """Binds real, already-measured per-round percentage deltas (the shape
    manually-run paired llama-bench campaigns report, e.g. patches
    1215/1216's README round-by-round evidence) into a LaneEffect, without
    re-running anything. ``pct_deltas`` is one signed percentage change per
    paired round (subject vs control, same round); converted to the
    ``pair_ratios`` sufficient statistic (1 + delta/100) that
    bootstrap_fixed_composite_mean()/aggregate_contract_effects() require
    for an interval, and to the geometric mean effect for the point
    estimate -- the same estimator block_bootstrap_effect() uses elsewhere
    in this module, just computed here directly since there is no raw
    per-pair timing to re-derive it from."""
    from bigcherry.experiment import contract as experiment_contract

    if not pct_deltas:
        raise PatchCampaignError(
            "_lane_effect_from_pct_deltas: pct_deltas must be non-empty"
        )
    ratios = tuple(1.0 + (delta / 100.0) for delta in pct_deltas)
    if any(ratio <= 0 for ratio in ratios):
        raise PatchCampaignError(
            f"_lane_effect_from_pct_deltas: non-positive ratio derived from deltas {pct_deltas!r}"
        )
    log_ratios = [math.log(ratio) for ratio in ratios]
    point_pct = 100.0 * (math.exp(statistics.mean(log_ratios)) - 1.0)
    lane = experiment_contract.LaneEffect(
        role=role,
        metric=metric,
        geometric_effect_pct=point_pct,
        paired_rounds=len(ratios),
        pair_ratios=ratios,
    )
    ci = experiment_contract.bootstrap_fixed_composite_mean([lane])
    if ci is not None:
        lane = dataclasses.replace(lane, ci95_low_pct=ci[0], ci95_high_pct=ci[1])
    return lane
