"""HI83: tracked evidence contract for patch STATE="validated".

GPT review of the completed HI82 body of work (gpt-auto-agent,
req_6b1466ee8369406c) found the most consequential remaining gap: BigCherry
can now produce strong, real evidence that a patch was genuinely built,
activated, and tested (HI82) -- but the repository's authoritative patch
STATE="validated" is not mechanically tied to that evidence. A developer can
still hand-edit a patch module's STATE with no machine check that matching
evidence exists, or that it still matches the CURRENT patch implementation.

This module is the tracked evidence contract itself: a JSON record per
patch, stored with packaged patches under their ``evidence/`` directory and
with the legacy baseline under ``patches/_validation/``. The authority must
be resolvable from the repository alone, not from ``artifacts/``, which is
gitignored, or the external ledger, which offline pytest/CI/a fresh checkout
cannot resolve.

Design: GPT (gpt-auto-agent, req_487497b28d444d50), applied per plan item
HI83. Deliberately does NOT wire hard enforcement into any real build/apply
path in this pass -- see HI83's plan item notes for why (patch_catalog.py's
own module docstring states it is read-only/additive and does not change
selection; hard-wiring this module into campaign_lane.py or
__main__.py's _apply_selection() would need to resolve that tension plus
two real open policy questions GPT flagged -- a correctness-evidence
producer per patch class, and a per-patch validation-architectures
obligation -- neither of which this module can decide on its own).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from ..core import paths
from . import patchset
from .activation import ActivationEvidence

SCHEMA_VERSION = 4
READABLE_SCHEMA_VERSIONS = (1, 2, 3, 4)
CONTRACT_VERSION = "hi83-v1"
FRAMEWORK_CONFIGURATION_SCHEMA_VERSION = 5
FRAMEWORK_CONFIGURATION_KIND = "framework-configuration-v1"
CORRECTNESS_SCHEMA_VERSION = 1

# One-time migration contract -- see generate_legacy_baseline().
LEGACY_SCHEMA_VERSION = 1
LEGACY_CONTRACT = "hi83-legacy-grandfather-v1"

BUILD_ROLES = ("tune", "replay", "stock")
# VA07: two genuinely distinct provenance domains (GPT round-4 correction,
# req_82dc1c3dcc744fb2 -- tune/replay/stock are real, distinct CAMPAIGN
# build variants, not mislabeled validation roles). CAMPAIGN_BUILD_ROLES is
# an alias of the historical BUILD_ROLES kept for v1/v2 records and
# call sites; VALIDATION_BUILD_ROLES is new in schema v3.
CAMPAIGN_BUILD_ROLES = BUILD_ROLES
VALIDATION_BUILD_ROLES = ("control", "subject")

# Exact builds.CompletedBuildEvidence.campaign_identity() shape -- also
# enforced in e2e_smoke_campaign.py's _require_completed_build_identity_shape()
# and e2e_smoke_report.py's BENCH_BUILD_IDENTITY_REQUIRED_KEYS. Kept as a
# separate literal here (not imported) so this module stays usable without
# pulling in the campaign runtime.
BUILD_IDENTITY_KEYS = (
    "effective_build_id", "compile_verification_id", "compile_commands_digest",
    "hip_compile_commands_digest", "runtime_bundle_hash", "runtime_artifacts",
)

_STATE_RE = re.compile(
    r"""(?m)^STATE[ \t]*=[ \t]*(?P<q>["'])(?:validated|rejected|untested)(?P=q)"""
    r"""(?P<tail>[ \t]*(?:#.*)?)$"""
)
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


class ValidationEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceCheck:
    status: str
    problems: tuple[str, ...] = ()
    campaign_digests: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status in {
            "not-required", "validated-evidence", "legacy-grandfathered",
            "ported-benched-evidence", "deferred-hardware-evidence",
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationEvidenceError(f"{field} must be a non-empty string")
    return value


def _require_hex(value: object, field: str, lengths: tuple[int, ...]) -> str:
    text = _require_string(value, field)
    if len(text) not in lengths or _HEX_RE.fullmatch(text) is None:
        raise ValidationEvidenceError(f"{field} must be hex of length {lengths}, got {text!r}")
    return text.lower()


def _architectures(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = re.split(r"[;,\s]+", value)
    else:
        parts = []
        for item in value:
            parts.extend(re.split(r"[;,\s]+", str(item)))
    return tuple(sorted({item for item in parts if item}))


def patch_validation_subject_digest(path: Path) -> str:
    """Digest stable across only the intended STATE transition.

    The validation campaign normally executes while STATE="untested".
    Changing exactly that literal assignment to STATE="validated" must
    therefore not invalidate the proof that authorized the transition.
    Every other byte of the patch module remains identity-relevant --
    this is NOT a replacement for patch_source_isolation.
    patch_implementation_digest(), which still records the exact bytes a
    given hardware campaign actually used.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    matches = list(_STATE_RE.finditer(text))
    if len(matches) == 1:
        match = matches[0]
        normalized = (
            text[:match.start()] + 'STATE = "<validation-state>"' + match.group("tail")
            + text[match.end():]
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    # Packaged patches keep lifecycle state in patch.toml, not patch.py.
    manifest = path.parent / "patch.toml"
    if len(matches) == 0 and manifest.is_file():
        manifest_text = manifest.read_text(encoding="utf-8")
        state = re.compile(r'(?m)^state\s*=\s*["\'][^"\']+["\']\s*$')
        if len(state.findall(manifest_text)) != 1:
            raise ValidationEvidenceError(f"{manifest}: expected exactly one state field")
        normalized_manifest = state.sub('state = "<validation-state>"', manifest_text)
        payload = {"implementation": text, "metadata": normalized_manifest}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    # Legacy flat patches may keep lifecycle state solely in catalog.toml.
    # Their implementation bytes are already state-independent, so the raw
    # implementation digest is the stable validation subject identity.
    if len(matches) == 0:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    raise ValidationEvidenceError(
        f"{path}: expected exactly one literal STATE assignment or a packaged patch.toml state"
    )


def _validate_build_identity(value: object, *, field: str) -> dict[str, object]:
    """Full CompletedBuildEvidence.campaign_identity() shape validation,
    parameterized by the caller's field path (VA07: reused identically for
    all five build roles across both provenance domains -- v1/v2's
    build_identities.{tune,replay,stock} and v3's
    campaign_build_identities.{tune,replay,stock} /
    validation_build_identities.{control,subject}. No weaker validation-
    build identity is defined; every role uses the same BUILD_IDENTITY_KEYS."""
    if not isinstance(value, Mapping):
        raise ValidationEvidenceError(f"{field} must be an object")
    missing = [key for key in BUILD_IDENTITY_KEYS if key not in value]
    if missing:
        raise ValidationEvidenceError(f"{field} missing field(s) {missing!r}")

    result = dict(value)
    for key in BUILD_IDENTITY_KEYS:
        subfield = f"{field}.{key}"
        if key == "runtime_artifacts":
            artifacts = result[key]
            if not isinstance(artifacts, Mapping) or not artifacts:
                raise ValidationEvidenceError(f"{subfield} must be a non-empty object")
            for name, digest in artifacts.items():
                _require_string(name, f"{subfield} artifact name")
                _require_hex(digest, f"{subfield}.{name}", (64,))
        else:
            _require_string(result[key], subfield)
    return result


def evidence_path(patch_id: str, *, root: Path | None = None) -> Path:
    if root is not None:
        return Path(root) / f"{patch_id}.json"
    # Package-local evidence follows the patch's own identity. Legacy flat
    # callers use the central quarantine directory until their package is
    # migrated; the production tree is package-only after PA18.
    try:
        from . import registry as patch_registry
        descriptor = patch_registry.load_registry(paths.PATCHES).get(patch_id)
    except (OSError, ValueError, KeyError):
        descriptor = None
    if descriptor is not None and descriptor.package_root is not None:
        return (
            paths.PATCHES / descriptor.package_root / "evidence" / "validation.json"
        )
    return paths.PATCHES / "_validation" / f"{patch_id}.json"


def _artifact_refs(campaign_workdir: Path) -> list[dict[str, str]]:
    root = Path(campaign_workdir).resolve()
    names = (
        "status.json", "activation.json", "correctness.json", "bench.json", "report.md",
        "tune.jsonl.measurements.jsonl", "promoted.jsonl", "coverage.json", "dispatch.cache",
        "performance.json",
        "artifacts/validation-lanes.json", "artifacts/rd08-correctness.json",
        "artifacts/rd08-trigger.json", "artifacts/contract-qualification.json",
        "logs/activation-rd08-trigger-subject.log", "logs/activation-rd08-trigger-control.log",
        # VA23: RD73's contract artifacts. This list is the record's own
        # AUTHORITATIVE artifact_hashes map -- verify_evidence() only accepts
        # a passing performance/controls check whose artifact appears here,
        # so a contract whose artifacts are absent reports "no recorded
        # benchmark execution" no matter how real the run was. The
        # enumeration is deliberate (only known artifact names count, so an
        # arbitrary file dropped in the workdir cannot become evidence), so
        # each new contract's artifacts must be added explicitly, exactly as
        # RD08's are above.
        "artifacts/rd73-performance.json", "artifacts/rd73-correctness.json",
        "artifacts/rd73-contract-qualification.json", "artifacts/rd73-activation.json",
        "artifacts/rd73-mtp-lane.json", "artifacts/rd73-decode-control.json",
        "artifacts/rd73-resource.json",
        "logs/rd73-mtp-subject-server.log", "logs/rd73-mtp-control-server.log",
    )
    return [{"path": name, "sha256": _sha256_file(root / name)} for name in names if (root / name).is_file()]


def load_correctness_summary(
    path: Path, *, patch_id: str, subject_digest: str, base_revision: str,
    patched_source_tree: str, campaign_identity_digest: str, gpu_architectures: Iterable[str],
) -> dict[str, object]:
    """Load one patch-level machine-readable correctness summary.

    This intentionally does NOT infer correctness from bench.json or
    promotion status. A producer (e.g. HI67's correctness_evidence.py where
    applicable, or a patch-specific direct-op validator) must explicitly
    create it -- there is no universal patch-level correctness oracle in
    this repository today.
    """
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationEvidenceError(f"cannot read correctness evidence {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValidationEvidenceError(f"{path}: correctness root must be an object")
    if doc.get("schema_version") != CORRECTNESS_SCHEMA_VERSION:
        raise ValidationEvidenceError(f"{path}: unsupported correctness schema")

    expected = {
        "patch_id": patch_id, "patch_validation_subject_digest": subject_digest,
        "base_revision": base_revision, "patched_source_tree": patched_source_tree,
        "campaign_identity_digest": campaign_identity_digest,
    }
    for key, value in expected.items():
        if doc.get(key) != value:
            raise ValidationEvidenceError(f"{path}: correctness {key}={doc.get(key)!r}, expected {value!r}")

    if doc.get("disposition") not in {"passed", "failed", "not-applicable"}:
        raise ValidationEvidenceError(f"{path}: invalid correctness disposition")
    _require_string(doc.get("mechanism"), "correctness.mechanism")
    _require_string(doc.get("detail"), "correctness.detail")

    covered = set(_architectures(doc.get("gpu_architectures") or ()))
    required = set(_architectures(gpu_architectures))
    if not required.issubset(covered):
        raise ValidationEvidenceError(
            f"{path}: correctness evidence does not cover {sorted(required - covered)!r}"
        )
    return doc


def _record_digest(record: Mapping[str, object]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validation_digest(patch_path: Path) -> str:
    manifest = patch_path.parent / "validation.toml"
    if manifest.is_file():
        return _sha256_file(manifest)
    return hashlib.sha256(b"no-validation-manifest").hexdigest()


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False).encode()).hexdigest()


def make_framework_configuration_record(
    *, descriptor, patch_path: Path, base_ref: str, base_revision: str,
    source_name: str, source_composition, source_tree: str, source_slice_id: str,
    compiled_targets, builds, generated_inputs, check_results, artifact_hashes,
    campaign_workdir: Path,
) -> dict[str, object]:
    """Construct the schema-5, configuration-only evidence record.

    This is intentionally separate from :func:`make_record`; the latter is
    the historical schema-4 campaign writer and must remain unchanged.
    """
    patch_path = Path(patch_path)
    patch_digest = _sha256_file(patch_path)
    subject_digest = patch_validation_subject_digest(patch_path)
    composition = [{"id": str(pair[0]), "digest": str(pair[1])} for pair in source_composition]
    targets = [str(target) for target in compiled_targets]
    if not targets:
        raise ValidationEvidenceError("compiled_targets must not be empty")
    normalized_builds = {role: _validate_build_identity(value, field=f"builds.{role}")
                         for role, value in builds.items()}
    if set(normalized_builds) != {"production", "diagnostic"}:
        raise ValidationEvidenceError("builds must contain production and diagnostic")
    normalized_inputs = json.loads(json.dumps(generated_inputs, sort_keys=True))
    for role in ("production", "diagnostic"):
        entry = normalized_inputs.get(role)
        if not isinstance(entry, dict) or entry.get("proof") != "compiled-copy-v1":
            raise ValidationEvidenceError(f"generated_inputs.{role} requires compiled-copy-v1")
    normalized_checks = json.loads(json.dumps(check_results, sort_keys=True))
    normalized_artifacts = {str(path): _require_hex(value, f"artifact_hashes.{path}", (64,))
                            for path, value in artifact_hashes.items()}
    eligible = bool(normalized_checks) and all(
        isinstance(value, dict) and value.get("status") == "pass" and
        all(isinstance(artifact, dict) and artifact.get("path") in normalized_artifacts and
            artifact.get("sha256") == normalized_artifacts[artifact["path"]]
            for artifact in value.get("artifacts", ()))
        for value in normalized_checks.values()
    )
    descriptor_id = getattr(descriptor, "patch_id", getattr(descriptor, "id", ""))
    implementation_digest = getattr(descriptor, "implementation_digest", patch_digest)
    record: dict[str, object] = {
        "record_schema_version": FRAMEWORK_CONFIGURATION_SCHEMA_VERSION,
        "kind": FRAMEWORK_CONFIGURATION_KIND,
        "patch_id": _require_string(descriptor_id, "descriptor.patch_id"),
        "patch_implementation_digest": _require_hex(implementation_digest, "patch_implementation_digest", (64,)),
        "patch_validation_subject_digest": subject_digest,
        "validation_digest": _validation_digest(patch_path),
        "base_ref": _require_string(base_ref, "base_ref"), "base_revision": _require_string(base_revision, "base_revision"),
        "source_name": _require_string(source_name, "source_name"), "source_composition": composition,
        "source_tree": _require_string(source_tree, "source_tree"), "source_slice_id": _require_string(source_slice_id, "source_slice_id"),
        "compiled_targets": targets, "builds": normalized_builds, "generated_inputs": normalized_inputs,
        "check_results": normalized_checks, "artifact_hashes": normalized_artifacts,
        "claim_scope": "configuration-only", "runtime_performance_qualified": False,
        "hardware_execution_qualified": False, "eligible_for_validated_state": eligible,
        "campaign_identity": _canonical_digest({"builds": normalized_builds, "source_tree": source_tree, "targets": targets}),
        "campaign_workdir": str(Path(campaign_workdir)),
    }
    record["record_digest"] = _record_digest(record)
    return record


def verify_framework_configuration_record(
    record: Mapping[str, object], *, descriptor, patch_path: Path, pinned_ref: str,
    required_compiled_targets=(), resolved_base_revision: str | None = None,
    source_composition=None,
) -> tuple[bool, tuple[str, ...]]:
    """Strict offline verifier for schema-5 framework configuration proof."""
    problems: list[str] = []
    try:
        from . import validation_policy
        if not validation_policy.is_framework_configuration_patch(descriptor):
            problems.append("descriptor is not a framework configuration patch")
    except Exception as exc:
        problems.append(f"cannot classify descriptor: {exc}")
    if not isinstance(record, Mapping):
        return False, ("record must be an object",)
    if record.get("record_schema_version") != FRAMEWORK_CONFIGURATION_SCHEMA_VERSION or record.get("kind") != FRAMEWORK_CONFIGURATION_KIND:
        problems.append("wrong schema or kind")
    if record.get("claim_scope") != "configuration-only" or record.get("runtime_performance_qualified") is not False or record.get("hardware_execution_qualified") is not False:
        problems.append("forbidden runtime or hardware claim")
    if record.get("record_digest") != _record_digest(record): problems.append("record_digest mismatch")
    if record.get("base_ref") != pinned_ref: problems.append("stale base_ref")
    if resolved_base_revision is not None and record.get("base_revision") != resolved_base_revision: problems.append("stale base_revision")
    if record.get("patch_id") != getattr(descriptor, "patch_id", None): problems.append("patch identity mismatch")
    try:
        if record.get("patch_implementation_digest") != getattr(descriptor, "implementation_digest", None): problems.append("implementation digest mismatch")
        if record.get("patch_validation_subject_digest") != patch_validation_subject_digest(Path(patch_path)): problems.append("subject digest mismatch")
    except Exception as exc: problems.append(f"cannot recompute patch digest: {exc}")
    targets = record.get("compiled_targets")
    if not isinstance(targets, list) or not targets or not set(required_compiled_targets).issubset(targets): problems.append("compiled target coverage incomplete")
    builds = record.get("builds")
    if not isinstance(builds, Mapping) or set(builds) != {"production", "diagnostic"}:
        problems.append("build role identity set is not production/diagnostic")
    else:
        for role, value in builds.items():
            try: _validate_build_identity(value, field=f"builds.{role}")
            except ValidationEvidenceError as exc: problems.append(str(exc))
    inputs = record.get("generated_inputs")
    if not isinstance(inputs, Mapping): problems.append("generated_inputs missing")
    else:
        for role in ("production", "diagnostic"):
            if not isinstance(inputs.get(role), Mapping) or inputs[role].get("proof") != "compiled-copy-v1": problems.append(f"generated_inputs.{role} invalid")
    checks, artifacts = record.get("check_results"), record.get("artifact_hashes")
    if not isinstance(checks, Mapping) or not checks: problems.append("check_results missing")
    elif not all(isinstance(v, Mapping) and v.get("status") == "pass" and v.get("artifacts") for v in checks.values()): problems.append("required check missing or not pass")
    if not isinstance(artifacts, Mapping) or not artifacts: problems.append("artifact_hashes missing")
    else:
        for value in checks.values() if isinstance(checks, Mapping) else ():
            for artifact in value.get("artifacts", ()) if isinstance(value, Mapping) else ():
                if not isinstance(artifact, Mapping) or artifacts.get(artifact.get("path")) != artifact.get("sha256"): problems.append("check artifact missing or tampered")
    if source_composition is not None and record.get("source_composition") != [{"id": str(p[0]), "digest": str(p[1])} for p in source_composition]: problems.append("source composition mismatch")
    return not problems, tuple(dict.fromkeys(problems))


def make_record(
    *, patch_id: str, patch_path: Path, patch_implementation_digest: str, base_ref: str,
    base_revision: str, framework_baseline_digest: str, patched_source_tree: str,
    gpu_architectures: str | Iterable[str], activation_evidence: ActivationEvidence | None,
    activation_disposition: str | None, correctness: Mapping[str, object] | None,
    campaign_identity_digest: str, build_identities: Mapping[str, Mapping[str, object]],
    validation_build_identities: Mapping[str, Mapping[str, object]],
    campaign_workdir: Path,
    representation: str = "simple", validation_implementation_digest: str | None = None,
    contract_id: str | None = None, contract_hash: str | None = None,
    contracts: Iterable[Mapping[str, str]] = (),
    contract_verdicts: Mapping[str, Mapping[str, object]] | None = None,
    baseline_composition: Mapping[str, object] | None = None,
    control_composition: Mapping[str, object] | None = None,
    subject_composition: Mapping[str, object] | None = None,
    control_tree: str | None = None, subject_tree: str | None = None,
    stock_tree: str | None = None, blockers: Iterable[str] = (),
    check_results: Mapping[str, object] | None = None,
    validation_eligible: bool | None = None,
    lane_effects: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """``build_identities`` is the campaign-build domain
    ({tune,replay,stock}); ``validation_build_identities`` is the
    validation-build domain ({control,subject}) -- two genuinely distinct
    provenance domains (VA07, GPT round-4 correction req_82dc1c3dcc744fb2),
    both mandatory, always producing a schema-v4 record with both domains
    as real top-level fields. GPT round 5 (req_48c36d9e3d324ec5): making
    this optional preserved an accidental v2-downgrade path -- every NEW
    validation now writes v4; v1/v2/v3 backward-COMPATIBILITY (reading old
    records) lives in load_records()/_record_qualifies(), not here. It is
    intentional and expected that validation.subject may carry the same
    physical build identity as campaign.tune; the schema does not require
    or assume equality, since a future validation executor may use a
    genuinely distinct subject build.

    VA18: contract identity is now PLURAL. Pass either ``contract_id``/
    ``contract_hash`` (0/1-contract convenience, wrapped internally into a
    single-entry ``contracts`` list) or ``contracts`` (canonical, a list of
    ``{"id": ..., "hash": ...}`` -- sorted by id at write time), never
    both. ``contract_verdicts`` is a dict keyed by contract id; every key
    must reference a contract actually present in ``contracts`` (an
    unbound verdict id is a structural error, fail-closed) -- a bound
    contract with NO verdict entry is not itself an error here (the
    caller may not have produced one yet), but it does mean that contract
    reads as an implicit incomplete/BLOCKED verdict at qualification time
    (_record_qualifies), never an implicit PASS."""
    subject_digest = patch_validation_subject_digest(patch_path)
    archs = _architectures(gpu_architectures)
    if not archs:
        raise ValidationEvidenceError("gpu_architectures must not be empty")

    if set(build_identities) != set(CAMPAIGN_BUILD_ROLES):
        raise ValidationEvidenceError(
            f"build_identities role set must be exactly {set(CAMPAIGN_BUILD_ROLES)!r}, "
            f"got {set(build_identities)!r}"
        )
    builds: dict[str, object] = {
        role: _validate_build_identity(build_identities[role], field=f"build_identities.{role}")
        for role in CAMPAIGN_BUILD_ROLES
    }

    if set(validation_build_identities) != set(VALIDATION_BUILD_ROLES):
        raise ValidationEvidenceError(
            f"validation_build_identities role set must be exactly {set(VALIDATION_BUILD_ROLES)!r}, "
            f"got {set(validation_build_identities)!r}"
        )
    validation_builds: dict[str, object] = {
        role: _validate_build_identity(
            validation_build_identities[role], field=f"validation_build_identities.{role}"
        )
        for role in VALIDATION_BUILD_ROLES
    }

    if activation_evidence is None:
        activation = {"status": "unknown", "disposition": "unknown", "mechanism": "", "detail": ""}
    else:
        activation = {
            "status": activation_evidence.status,
            "disposition": activation_disposition or "unknown",
            "mechanism": activation_evidence.mechanism,
            "detail": activation_evidence.detail,
        }

    correctness_doc = (
        dict(correctness) if correctness is not None else {
            "schema_version": CORRECTNESS_SCHEMA_VERSION, "patch_id": patch_id,
            "patch_validation_subject_digest": subject_digest, "base_revision": base_revision,
            "patched_source_tree": patched_source_tree,
            "campaign_identity_digest": campaign_identity_digest,
            "gpu_architectures": list(archs), "disposition": "unknown", "mechanism": "", "detail": "",
        }
    )

    if contract_id is not None and contracts:
        raise ValidationEvidenceError(
            "make_record: pass either contract_id/contract_hash or contracts=, not both"
        )
    if contracts:
        resolved_contracts = tuple(
            sorted(
                (
                    {
                        "id": _require_string(entry.get("id"), "contracts[].id"),
                        "hash": _require_string(entry.get("hash"), "contracts[].hash"),
                    }
                    for entry in contracts
                ),
                key=lambda entry: entry["id"],
            )
        )
    elif contract_id is not None:
        resolved_contracts = ({"id": contract_id, "hash": contract_hash or ""},)
    elif correctness_doc.get("contract_id"):
        resolved_contracts = (
            {"id": correctness_doc["contract_id"], "hash": correctness_doc.get("contract_hash") or ""},
        )
    else:
        resolved_contracts = ()

    contract_ids = {entry["id"] for entry in resolved_contracts}
    resolved_verdicts = dict(contract_verdicts or {})
    unbound_verdicts = sorted(set(resolved_verdicts) - contract_ids)
    if unbound_verdicts:
        raise ValidationEvidenceError(
            f"contract_verdicts references unbound contract id(s): {unbound_verdicts!r}"
        )
    for verdict_id, verdict in resolved_verdicts.items():
        if not isinstance(verdict, Mapping) or not isinstance(verdict.get("passed"), bool):
            raise ValidationEvidenceError(
                f"contract_verdicts[{verdict_id!r}] must be an object with a boolean 'passed' field"
            )

    # A gate-proved "not applicable on this GPU" result is useful evidence,
    # but is not enough to make a globally STATE="validated" patch.
    eligible = (
        activation.get("status") == "executed"
        and activation.get("disposition") == "activation-verified"
        and correctness_doc.get("disposition") == "passed"
    )
    if validation_eligible is not None:
        eligible = validation_eligible

    result = {
        # Always v4: every new validation writes the current schema.
        # v1/v2/v3 records already on disk remain readable unchanged --
        # backward compatibility is a read-side (load_records/
        # _record_qualifies) concern, never a writer concern (GPT round 5,
        # req_48c36d9e3d324ec5; VA18 extends this to the v3->v4 contract
        # pluralization the same way).
        "record_schema_version": 4,
        "validation_contract_version": CONTRACT_VERSION,
        "representation": representation,
        "validation_implementation_digest": validation_implementation_digest or _validation_digest(patch_path),
        # VA18: plural, canonical-sorted -- replaces v1-v3's singular
        # contract_id/contract_hash fields (which stay readable, never
        # written, on old records only).
        "contracts": list(resolved_contracts),
        "contract_verdicts": resolved_verdicts,
        "baseline_composition": dict(baseline_composition or {}),
        "control_composition": dict(control_composition or correctness_doc.get("control_composition", {})),
        "subject_composition": dict(subject_composition or correctness_doc.get("subject_composition", {})),
        "control_tree": control_tree or correctness_doc.get("control_tree"),
        "subject_tree": subject_tree or correctness_doc.get("subject_tree", patched_source_tree),
        "stock_tree": stock_tree,
        "check_results": dict(check_results or correctness_doc.get("check_results") or {
            "activation": dict(activation), "correctness": dict(correctness_doc),
        }),
        "hardware": {"architectures": list(archs)},
        "artifact_hashes": {str(item.get("path")): item.get("sha256") for item in _artifact_refs(campaign_workdir) if isinstance(item, Mapping)},
        "blockers": list(blockers),
        "final_eligibility": eligible,
        "patch_id": patch_id,
        # Exact bytes the hardware campaign used.
        "patch_implementation_digest": _require_hex(
            patch_implementation_digest, "patch_implementation_digest", (64,)
        ),
        # STATE-normalized freshness identity.
        "patch_validation_subject_digest": subject_digest,
        "base_ref": _require_string(base_ref, "base_ref"),
        "base_revision": _require_hex(base_revision, "base_revision", (40, 64)),
        # Provenance only, not a per-patch freshness gate: otherwise
        # promoting one patch into the validated baseline would trigger a
        # validation avalanche across every unrelated already-validated patch.
        "framework_baseline_digest": _require_hex(
            framework_baseline_digest, "framework_baseline_digest", (64,)
        ),
        "patched_source_tree": _require_hex(patched_source_tree, "patched_source_tree", (40, 64)),
        "gpu_architectures": list(archs),
        "activation": activation,
        "correctness": correctness_doc,
        "campaign_identity_digest": _require_hex(
            campaign_identity_digest, "campaign_identity_digest", (64,)
        ),
        "campaign_build_identities": builds,
        "validation_build_identities": validation_builds,
        "campaign_artifacts": _artifact_refs(campaign_workdir),
        # RV99: the MEASUREMENTS, not just the verdict derived from them.
        # Before this the record kept identity, provenance, check verdicts and
        # artifact hashes, but the per-lane effects and their pair_ratios lived
        # only in the campaign's own artifacts under artifacts/, which is
        # gitignored. So from committed evidence alone an interval could not be
        # re-derived, re-aggregated across sessions or lanes, re-analysed under
        # a new estimator, or audited against the data that produced it -- the
        # record asserted a number whose inputs were unavailable, and on any
        # machine that had not run the campaign they were simply gone.
        #
        # block_bootstrap_effect() already names pair_ratios "the SUFFICIENT
        # STATISTIC for recomputing an AGGREGATE interval ... without
        # re-running the benchmark"; this is where that intent becomes real.
        # The vector is one float per paired round (10 for RD73), so retaining
        # it is a decision about what to keep, not new measurement.
        #
        # Required rather than defaulted: a campaign that measured lanes and
        # recorded none of them should be a call-site error, not a silently
        # thinner record. An empty tuple is legitimate for a campaign with no
        # measured lanes at all.
        "lane_effects": [dict(effect) for effect in lane_effects],
        "validation_disposition": "validated" if eligible else "incomplete",
        "eligible_for_validated_state": eligible,
    }
    result["record_digest"] = _record_digest(result)
    return result


def write_record(record: Mapping[str, object], *, root: Path | None = None) -> Path:
    patch_id = _require_string(record.get("patch_id"), "patch_id")
    campaign_digest = _require_hex(record.get("campaign_identity_digest"), "campaign_identity_digest", (64,))
    path = evidence_path(patch_id, root=root)

    if path.is_file():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationEvidenceError(f"cannot read {path}: {exc}") from exc
        if (
            not isinstance(document, dict) or document.get("schema_version") not in READABLE_SCHEMA_VERSIONS
            or document.get("patch_id") != patch_id or not isinstance(document.get("records"), list)
        ):
            raise ValidationEvidenceError(f"{path}: invalid evidence file")
        # Reading a v1 or v2 container is supported; any successful write
        # upgrades the container header to the current SCHEMA_VERSION (3)
        # without rewriting any legacy record already present.
        document["schema_version"] = SCHEMA_VERSION
    else:
        document = {"schema_version": SCHEMA_VERSION, "patch_id": patch_id, "records": []}

    records = document["records"]
    assert isinstance(records, list)
    new_record = dict(record)

    # RV96: campaign_identity_digest is a BUILD identity, not a run identity.
    # e2e_smoke_campaign.Campaign derives it (_make_campaign_identity ->
    # _stable_json_sha256) from the content identities of the built
    # executables plus patch identity -- so two independent measurements of
    # the same binaries necessarily share it.
    #
    # This function used to reject the second such record ("campaign digest X
    # already has different evidence"), which made the digest behave as though
    # a build could only ever be measured once. Three consequences, all real:
    #
    #   * the frozen re-run policy (EXPERIMENT_CONTRACT.md, "Re-running")
    #     requires extending a run to a pre-declared N_max and estimating over
    #     all valid pairs -- i.e. producing further measurements of the SAME
    #     build. Its output was unstorable, so the policy was doctrine only.
    #   * independent replication on unchanged code was impossible.
    #   * whichever run was written FIRST owned the digest permanently, so the
    #     record silently favoured first measurements. RD73 hit exactly this:
    #     a passing run (+1.717%) was stored and a later confirming run that
    #     failed the gate (+1.249%) could not be -- the direction that
    #     flatters a patch.
    #
    # Evidence stays append-only: an existing record is never mutated or
    # replaced here. Re-writing an identical record is still idempotent, and
    # tampering with a stored record is still caught at READ time by the
    # record_digest checks in _record_qualifies()/_record_qualifies_for_benched().
    for old in records:
        if not isinstance(old, dict):
            raise ValidationEvidenceError(f"{path}: non-object record")
        if old.get("campaign_identity_digest") != campaign_digest:
            continue
        if old == new_record:
            return path
        # Same build, different measurement -- but the fields the campaign
        # identity provably determines must still agree. If they do not, the
        # digest is not identifying what it claims to, which is corruption
        # rather than replication.
        for field in ("patch_id", "patch_implementation_digest", "patched_source_tree"):
            if field in old and field in new_record and old[field] != new_record[field]:
                raise ValidationEvidenceError(
                    f"{path}: campaign digest {campaign_digest} has records disagreeing "
                    f"on {field} ({old[field]!r} vs {new_record[field]!r}) -- the campaign "
                    f"identity is derived from built-binary and patch identity, so records "
                    f"sharing it cannot have come from different sources"
                )

    records.append(new_record)
    # Sort by campaign identity first (grouping a build's measurements
    # together), then by record_digest so ordering stays deterministic across
    # the multiple records a single build may now legitimately have.
    records.sort(key=lambda row: (
        str(row.get("campaign_identity_digest", "")), str(row.get("record_digest", "")),
    ))
    _atomic_json(path, document)
    return path


def load_records(patch_id: str, *, root: Path | None = None) -> tuple[dict[str, object], ...]:
    path = evidence_path(patch_id, root=root)
    if not path.is_file():
        return ()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationEvidenceError(f"cannot read {path}: {exc}") from exc
    if (
        not isinstance(document, dict) or document.get("schema_version") not in READABLE_SCHEMA_VERSIONS
        or document.get("patch_id") != patch_id or not isinstance(document.get("records"), list)
        or not all(isinstance(row, dict) for row in document["records"])
    ):
        raise ValidationEvidenceError(f"{path}: invalid evidence file")
    return tuple(document["records"])


def _record_qualifies(
    record: Mapping[str, object], *, module: patchset.PatchModule, pinned_ref: str, subject_digest: str,
    resolved_base_revision: str | None = None, validation_digest: str | None = None,
    contracts: tuple[Mapping[str, str], ...] = (),
) -> tuple[bool, tuple[str, ...]]:
    problems: list[str] = []
    expected = {
        "validation_contract_version": CONTRACT_VERSION,
        "patch_id": module.patch_id, "patch_validation_subject_digest": subject_digest,
        "base_ref": pinned_ref, "validation_disposition": "validated",
        "eligible_for_validated_state": True,
    }
    record_version = record.get("record_schema_version")
    if record_version not in READABLE_SCHEMA_VERSIONS:
        problems.append(f"record_schema_version={record_version!r} is unsupported")

    # VA18: a currently multi-contract patch can NEVER be qualified by a
    # pre-v4 record -- those schemas have no way to express more than one
    # contract's identity/verdict at all, so "matches" would be meaningless
    # (which of RD05/RD06/RD07 would a lone contract_id even mean?).
    if record_version in (1, 2, 3) and len(contracts) > 1:
        problems.append(
            f"record_schema_version={record_version!r} cannot qualify a current "
            f"{len(contracts)}-contract patch (schema v4 required for multi-contract identity)"
        )

    single_contract_id = contracts[0]["id"] if len(contracts) == 1 else None
    single_contract_hash = contracts[0].get("hash") if len(contracts) == 1 else None

    if record_version in (2, 3):
        if validation_digest is not None and record.get("validation_implementation_digest") != validation_digest:
            problems.append("validation implementation digest is stale")
        if record.get("contract_id") != single_contract_id:
            problems.append("contract identity is stale")
        if single_contract_hash is not None and record.get("contract_hash") != single_contract_hash:
            problems.append("contract hash is stale")
        if record.get("record_digest") != _record_digest(record):
            problems.append("record_digest does not match the evidence payload")
        required_v2 = ("representation", "validation_implementation_digest",
                       "baseline_composition", "control_composition", "subject_composition",
                       "control_tree", "subject_tree", "stock_tree", "check_results",
                       "hardware", "artifact_hashes", "blockers", "final_eligibility")
        for key in required_v2:
            value = record.get(key)
            if value is None or value == {}:
                problems.append(f"provenance field {key!r} is missing")
        try:
            _require_hex(record.get("validation_implementation_digest"),
                         "validation_implementation_digest", (64,))
        except ValidationEvidenceError as exc:
            problems.append(str(exc))
        if record.get("contract_id") is not None and not record.get("contract_hash"):
            problems.append("contract_hash is required when contract_id is present")
    elif record_version == 4:
        if validation_digest is not None and record.get("validation_implementation_digest") != validation_digest:
            problems.append("validation implementation digest is stale")
        if record.get("record_digest") != _record_digest(record):
            problems.append("record_digest does not match the evidence payload")
        required_v2 = ("representation", "validation_implementation_digest",
                       "baseline_composition", "control_composition", "subject_composition",
                       "control_tree", "subject_tree", "stock_tree", "check_results",
                       "hardware", "artifact_hashes", "blockers", "final_eligibility")
        for key in required_v2:
            value = record.get(key)
            if value is None or value == {}:
                problems.append(f"provenance field {key!r} is missing")
        try:
            _require_hex(record.get("validation_implementation_digest"),
                         "validation_implementation_digest", (64,))
        except ValidationEvidenceError as exc:
            problems.append(str(exc))

        # VA18: exact current contract ID/hash SET (order-independent),
        # complete verdict set, and every verdict must have passed --
        # missing/extra/stale-hash on even ONE bound contract is
        # nonqualification for the WHOLE patch (never partial credit).
        expected_contracts = {entry["id"]: entry.get("hash") for entry in contracts}
        record_contracts_raw = record.get("contracts")
        record_contracts: dict[str, object] = {}
        if isinstance(record_contracts_raw, list) and all(
            isinstance(entry, Mapping) for entry in record_contracts_raw
        ):
            record_contracts = {
                str(entry.get("id")): entry.get("hash") for entry in record_contracts_raw
            }
        else:
            problems.append("contracts must be a list of {id,hash} objects")
        if record_contracts != expected_contracts:
            problems.append(
                f"contract identity set is stale: record has {sorted(record_contracts)!r}, "
                f"current bound set is {sorted(expected_contracts)!r} (or a hash differs)"
            )
        verdicts = record.get("contract_verdicts")
        if not isinstance(verdicts, Mapping):
            problems.append("contract_verdicts must be an object")
        else:
            missing_verdicts = sorted(set(expected_contracts) - set(verdicts))
            if missing_verdicts:
                problems.append(f"missing contract_verdicts for: {missing_verdicts!r}")
            failing_verdicts = sorted(
                contract_id for contract_id in expected_contracts
                if contract_id in verdicts
                and not (
                    isinstance(verdicts[contract_id], Mapping)
                    and verdicts[contract_id].get("passed") is True
                )
            )
            if failing_verdicts:
                problems.append(f"contract_verdicts did not pass for: {failing_verdicts!r}")
    if record_version in (3, 4):
        # VA07: v3 records use campaign_build_identities/
        # validation_build_identities exclusively -- the legacy top-level
        # build_identities field must be absent to prevent a mixed/
        # ambiguous record (which domain would a stray legacy field even
        # belong to?).
        if "build_identities" in record:
            problems.append("v3 record must not carry the legacy top-level build_identities field")
        campaign_builds = record.get("campaign_build_identities")
        if not isinstance(campaign_builds, Mapping):
            problems.append("campaign_build_identities must be an object")
        elif set(campaign_builds) != set(CAMPAIGN_BUILD_ROLES):
            problems.append(
                f"campaign_build_identities role set must be exactly "
                f"{set(CAMPAIGN_BUILD_ROLES)!r}, got {set(campaign_builds)!r}"
            )
        else:
            for role in CAMPAIGN_BUILD_ROLES:
                try:
                    _validate_build_identity(
                        campaign_builds.get(role), field=f"campaign_build_identities.{role}"
                    )
                except ValidationEvidenceError as exc:
                    problems.append(str(exc))
        validation_builds = record.get("validation_build_identities")
        if not isinstance(validation_builds, Mapping):
            problems.append("validation_build_identities must be an object")
        elif set(validation_builds) != set(VALIDATION_BUILD_ROLES):
            problems.append(
                f"validation_build_identities role set must be exactly "
                f"{set(VALIDATION_BUILD_ROLES)!r}, got {set(validation_builds)!r}"
            )
        else:
            for role in VALIDATION_BUILD_ROLES:
                try:
                    _validate_build_identity(
                        validation_builds.get(role), field=f"validation_build_identities.{role}"
                    )
                except ValidationEvidenceError as exc:
                    problems.append(str(exc))
    for key, wanted in expected.items():
        if record.get(key) != wanted:
            problems.append(f"{key}={record.get(key)!r}, expected {wanted!r}")

    # GPT review (req_b87ea92609fa45fe): base_ref matching alone only proves
    # the record was made against a ref with the same NAME -- if that ref is
    # ever a moving target (unlike this project's current tag-pin practice),
    # a record could still "match" a base_ref string that now resolves to a
    # different commit. Only checked when a caller actually supplies the
    # real resolved commit (patch_catalog.py does not yet do this -- see its
    # module docstring); existing callers are unaffected.
    if resolved_base_revision is not None and record.get("base_revision") != resolved_base_revision:
        problems.append(
            f"base_revision={record.get('base_revision')!r} does not match the currently "
            f"resolved pin {resolved_base_revision!r}"
        )

    activation = record.get("activation")
    if (
        not isinstance(activation, Mapping) or activation.get("status") != "executed"
        or activation.get("disposition") != "activation-verified"
    ):
        problems.append("activation is not executed+activation-verified")

    correctness = record.get("correctness")
    if not isinstance(correctness, Mapping) or correctness.get("disposition") != "passed":
        problems.append("correctness did not pass")

    try:
        _require_hex(record.get("patch_implementation_digest"), "patch_implementation_digest", (64,))
        _require_hex(record.get("base_revision"), "base_revision", (40, 64))
        _require_hex(record.get("framework_baseline_digest"), "framework_baseline_digest", (64,))
        _require_hex(record.get("patched_source_tree"), "patched_source_tree", (40, 64))
        _require_hex(record.get("campaign_identity_digest"), "campaign_identity_digest", (64,))
        if record.get("record_schema_version") not in (3, 4):
            # v3/v4's own build_identities check (campaign_/validation_
            # split) already ran above -- this is the v1/v2 legacy shape only.
            builds = record.get("build_identities")
            if not isinstance(builds, Mapping):
                raise ValidationEvidenceError("build_identities must be an object")
            for role in BUILD_ROLES:
                _validate_build_identity(builds.get(role), field=f"build_identities.{role}")
    except ValidationEvidenceError as exc:
        problems.append(str(exc))

    if not _architectures(record.get("gpu_architectures") or ()):
        problems.append("gpu_architectures is empty")

    return (not problems, tuple(problems))


def _record_qualifies_for_benched(
    record: Mapping[str, object], *, module: patchset.PatchModule, pinned_ref: str, subject_digest: str,
    resolved_base_revision: str | None = None, validation_digest: str | None = None,
    contracts: tuple[Mapping[str, str], ...] = (),
) -> tuple[bool, tuple[str, ...]]:
    """VA08: the 'ported-benched' tracked-status obligation -- the same
    identity/freshness/schema-v3-or-v4 provenance discipline as
    _record_qualifies(), but requires only that a real control/subject
    benchmark actually ran -- validation_build_identities populated with
    real build identities, real hardware architectures recorded, AND a
    real check_results entry proving a performance/controls check
    actually passed with an artifact that is cross-checked against the
    record's OWN authoritative artifact_hashes map (build/hardware
    provenance alone only proves binaries were built, not that they were
    benchmarked -- GPT round 2 req_ecaa87b450294084; a bare non-empty
    artifacts list is not proof either -- GPT round 4
    req_73faeb08760c42fd) -- with no recorded correctness FAILURE.
    Deliberately NOT full
    eligible_for_validated_state (STATE='validated' is a strictly higher
    bar) and NOT activation executed+verified (activation is an
    orthogonal claim from "a real paired benchmark ran").

    VA18: contract identity is checked for STALENESS only (the current
    bound contract SET must match, same as _record_qualifies()) -- unlike
    validated-state qualification, a v4 record's contract_verdicts may be
    entirely BLOCKED/incomplete and still qualify ported-benched. Contract
    PASS is required only for validated promotion, never for ported-
    benched (a real benchmark having run is a real, independent claim from
    whether any bound contract's own promotion gate has passed)."""
    problems: list[str] = []
    record_version = record.get("record_schema_version")
    if record_version not in (3, 4):
        return False, ("ported-benched requires a schema-v3 or v4 record",)
    if record_version == 3 and len(contracts) > 1:
        return False, (
            f"record_schema_version=3 cannot qualify a current {len(contracts)}-contract patch",
        )
    expected = {
        "patch_id": module.patch_id, "patch_validation_subject_digest": subject_digest,
        "base_ref": pinned_ref,
    }
    for key, wanted in expected.items():
        if record.get(key) != wanted:
            problems.append(f"{key}={record.get(key)!r}, expected {wanted!r}")
    if validation_digest is not None and record.get("validation_implementation_digest") != validation_digest:
        problems.append("validation implementation digest is stale")
    if record_version == 4:
        expected_contracts = {entry["id"]: entry.get("hash") for entry in contracts}
        record_contracts_raw = record.get("contracts")
        record_contracts: dict[str, object] = {}
        if isinstance(record_contracts_raw, list) and all(
            isinstance(entry, Mapping) for entry in record_contracts_raw
        ):
            record_contracts = {
                str(entry.get("id")): entry.get("hash") for entry in record_contracts_raw
            }
        else:
            problems.append("contracts must be a list of {id,hash} objects")
        if record_contracts != expected_contracts:
            problems.append("contract identity set is stale")
    else:
        single_contract_id = contracts[0]["id"] if len(contracts) == 1 else None
        single_contract_hash = contracts[0].get("hash") if len(contracts) == 1 else None
        if record.get("contract_id") != single_contract_id:
            problems.append("contract identity is stale")
        if single_contract_hash is not None and record.get("contract_hash") != single_contract_hash:
            problems.append("contract hash is stale")
    if record.get("record_digest") != _record_digest(record):
        problems.append("record_digest does not match the evidence payload")
    if resolved_base_revision is not None and record.get("base_revision") != resolved_base_revision:
        problems.append(
            f"base_revision={record.get('base_revision')!r} does not match the currently "
            f"resolved pin {resolved_base_revision!r}"
        )
    validation_builds = record.get("validation_build_identities")
    if not isinstance(validation_builds, Mapping):
        problems.append("validation_build_identities must be an object")
    elif set(validation_builds) != set(VALIDATION_BUILD_ROLES):
        problems.append(
            f"validation_build_identities role set must be exactly "
            f"{set(VALIDATION_BUILD_ROLES)!r}, got {set(validation_builds)!r}"
        )
    else:
        for role in VALIDATION_BUILD_ROLES:
            try:
                _validate_build_identity(
                    validation_builds.get(role), field=f"validation_build_identities.{role}"
                )
            except ValidationEvidenceError as exc:
                problems.append(str(exc))
    hardware = record.get("hardware")
    if not isinstance(hardware, Mapping) or not hardware.get("architectures"):
        problems.append("hardware provenance (architectures) is missing")
    correctness = record.get("correctness")
    if isinstance(correctness, Mapping) and correctness.get("disposition") == "failed":
        problems.append("correctness recorded a failure")
    # GPT round 2 (req_ecaa87b450294084): build/hardware provenance alone
    # only proves binaries were BUILT, not that a benchmark actually RAN --
    # a build-only record (no benchmark) would otherwise silently qualify
    # as "ported-benched-evidence". Require at least one real
    # performance/controls check_results entry (validation_campaign.run()
    # serializes its actual evaluated ValidationResults there) with a
    # successful status AND a bound artifact -- an unbound/missing
    # artifact is not real evidence of execution either.
    # GPT round 4 (req_73faeb08760c42fd): a non-empty artifacts list alone
    # is not proof either -- anything could be typed into check_results by
    # hand (or corrupted). record["artifact_hashes"] is the record's own
    # authoritative path->sha256 map (make_record() builds it from the real
    # files _artifact_refs() found on disk at write time); cross-check each
    # candidate artifact against it rather than trusting the check_results
    # entry's own claimed hash.
    artifact_hashes = record.get("artifact_hashes")
    check_results = record.get("check_results")
    benchmark_executed = False
    if isinstance(check_results, Mapping) and isinstance(artifact_hashes, Mapping):
        for entry in check_results.values():
            if not (
                isinstance(entry, Mapping) and entry.get("capability") in ("performance", "controls")
                and entry.get("status") == "pass"
            ):
                continue
            for artifact in entry.get("artifacts") or ():
                if not isinstance(artifact, Mapping):
                    continue
                path = artifact.get("path")
                sha256 = artifact.get("sha256")
                if (
                    isinstance(path, str) and path and isinstance(sha256, str)
                    and len(sha256) == 64 and all(c in "0123456789abcdef" for c in sha256.lower())
                    and artifact_hashes.get(path) == sha256
                ):
                    benchmark_executed = True
                    break
            if benchmark_executed:
                break
    if not benchmark_executed:
        problems.append(
            "no recorded benchmark execution -- check_results has no passing "
            "performance/controls check with an artifact that matches the "
            "record's own authoritative artifact_hashes"
        )
    try:
        _require_hex(record.get("patch_implementation_digest"), "patch_implementation_digest", (64,))
        _require_hex(record.get("base_revision"), "base_revision", (40, 64))
        _require_hex(record.get("campaign_identity_digest"), "campaign_identity_digest", (64,))
    except ValidationEvidenceError as exc:
        problems.append(str(exc))
    return (not problems, tuple(problems))


def _record_qualifies_for_deferred_hardware(
    record: Mapping[str, object], *, module: patchset.PatchModule, pinned_ref: str, subject_digest: str,
    resolved_base_revision: str | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """VA08: the 'deferred-hardware' tracked-status obligation -- requires
    only a fresh, structured BLOCKED evidence record (real patch identity
    + resolved base revision, both current). Deliberately does NOT require
    completed hardware/build/benchmark evidence -- the entire point of
    this status is that hardware was unavailable, so demanding hardware
    evidence to qualify it would be self-defeating."""
    problems: list[str] = []
    if record.get("patch_id") != module.patch_id:
        problems.append(f"patch_id={record.get('patch_id')!r}, expected {module.patch_id!r}")
    if record.get("patch_validation_subject_digest") != subject_digest:
        problems.append("patch implementation digest is stale")
    if record.get("base_ref") != pinned_ref:
        problems.append(f"base_ref={record.get('base_ref')!r}, expected {pinned_ref!r}")
    if resolved_base_revision is not None and record.get("base_revision") != resolved_base_revision:
        problems.append(
            f"base_revision={record.get('base_revision')!r} does not match the currently "
            f"resolved pin {resolved_base_revision!r}"
        )
    if not record.get("blockers"):
        problems.append("no structured blockers recorded")
    try:
        _require_hex(record.get("patch_implementation_digest"), "patch_implementation_digest", (64,))
        _require_hex(record.get("base_revision"), "base_revision", (40, 64))
    except ValidationEvidenceError as exc:
        problems.append(str(exc))
    return (not problems, tuple(problems))


def _legacy_hashes(root: Path | None) -> dict[str, str]:
    path = Path(root or paths.PATCHES / "_validation") / "legacy-baseline.json"
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationEvidenceError(f"cannot read {path}: {exc}") from exc
    if (
        not isinstance(document, dict) or document.get("schema_version") != LEGACY_SCHEMA_VERSION
        or document.get("contract") != LEGACY_CONTRACT or not isinstance(document.get("patches"), dict)
    ):
        raise ValidationEvidenceError(f"{path}: invalid legacy baseline")
    return {
        str(patch_id): _require_hex(digest, f"legacy.{patch_id}", (64,))
        for patch_id, digest in document["patches"].items()
    }


def _resolve_contract_identities(
    module: patchset.PatchModule,
) -> tuple[str | None, tuple[dict[str, str], ...]]:
    """VA18: shared plural descriptor/contract lookup, used by every
    status-obligation verifier. Returns ``(validation_digest, contracts)``
    where ``contracts`` is a canonical id-sorted tuple of ``{"id":...,
    "hash":...}`` for every experiment contract bound to this patch (0, 1,
    or many).

    Only genuine "this module has no registry at all" (legacy/synthetic
    callers, OSError) or "this exact patch id isn't registered"
    (registry.get()'s own PatchRegistryError, or KeyError) are swallowed
    -- both pre-existing, narrow tolerances for legacy/synthetic test
    callers. A real contract-resolution failure (validation.py's
    ConfigurationError, from a malformed or unresolvable contract
    reference) is NEVER caught here and always propagates: degrading a
    real error into silent no-contract identity would let a broken
    multi-contract patch's evidence quietly stop being checked at all,
    rather than fail loudly (GPT review, session ses_d1759a9471d443d5).
    Note this is narrower than a blanket ValueError catch would be --
    ConfigurationError IS a ValueError subclass, so it must be excluded
    explicitly rather than by broadening the catch."""
    validation_digest = None
    contracts: tuple[dict[str, str], ...] = ()
    from . import registry as patch_registry

    try:
        registry = patch_registry.load_registry(module.catalog_root or paths.PATCHES)
        descriptor = registry.get(module.patch_id)
    except (OSError, KeyError, patch_registry.PatchRegistryError):
        return validation_digest, contracts

    validation_digest = descriptor.validation_digest or _validation_digest(module.path)
    if descriptor.experiment_contracts:
        from . import validation as patch_validation
        bindings = tuple(
            patch_validation.bind_contract(contract)
            for contract in patch_validation.load_contracts_for_descriptor(descriptor)
        )
        contracts = tuple(
            sorted(
                ({"id": b.contract_id, "hash": b.contract_hash} for b in bindings),
                key=lambda entry: entry["id"],
            )
        )
    return validation_digest, contracts


def verify_validated_patch(
    module: patchset.PatchModule, *, pinned_ref: str, required_architectures: Iterable[str] = (),
    root: Path | None = None, allow_legacy_grandfather: bool = True,
    resolved_base_revision: str | None = None,
) -> EvidenceCheck:
    """resolved_base_revision (optional): the actual commit `pinned_ref`
    currently resolves to. When supplied, a record's base_revision must
    match it exactly, catching the case where pinned_ref names a moving
    ref that has since advanced past what the record was validated
    against. Omitted by default since no caller resolves this yet (see
    patch_catalog.py's module docstring on why hard enforcement, which is
    where this would matter, is still deferred)."""
    if module.state != "validated":
        return EvidenceCheck("not-required")

    subject_digest = patch_validation_subject_digest(module.path)
    validation_digest, contracts = _resolve_contract_identities(module)
    qualifying: list[dict[str, object]] = []
    stale: list[str] = []

    for record in load_records(module.patch_id, root=root):
        ok, why = _record_qualifies(
            record, module=module, pinned_ref=pinned_ref, subject_digest=subject_digest,
            resolved_base_revision=resolved_base_revision,
            validation_digest=validation_digest, contracts=contracts,
        )
        if ok:
            qualifying.append(record)
        else:
            digest_prefix = str(record.get("campaign_identity_digest", "?"))[:12]
            stale.append(f"{digest_prefix}: " + "; ".join(why))

    required = set(_architectures(required_architectures))
    covered: set[str] = set()
    for record in qualifying:
        covered.update(_architectures(record.get("gpu_architectures") or ()))
    missing_architectures = sorted(required - covered)

    if qualifying and not missing_architectures:
        return EvidenceCheck(
            "validated-evidence",
            # RV96: a build may now legitimately have several qualifying
            # measurement records, so de-duplicate -- this reports which
            # BUILDS are qualified, not how many records exist.
            campaign_digests=tuple(
                sorted({str(record["campaign_identity_digest"]) for record in qualifying})
            ),
        )

    if allow_legacy_grandfather:
        legacy = _legacy_hashes(root)
        if legacy.get(module.patch_id) == module.content_hash:
            return EvidenceCheck("legacy-grandfathered")

    problems: list[str] = []
    if not qualifying:
        problems.append("no current qualifying HI83 validation record")
    if missing_architectures:
        problems.append("missing architecture(s): " + ", ".join(missing_architectures))
    if stale:
        problems.append("stale/ineligible: " + " | ".join(stale[:3]))
    return EvidenceCheck("missing-or-stale", tuple(problems))


def verify_ported_benched_patch(
    module: patchset.PatchModule, *, pinned_ref: str, root: Path | None = None,
    resolved_base_revision: str | None = None,
) -> EvidenceCheck:
    """VA08: current-pin qualification for the 'ported-benched'
    tracked-status (docs/reference/testing/PATCH_VALIDATION.md's table) --
    a real control/subject benchmark actually ran, with build and hardware
    identities recorded, and no recorded correctness failure. A failing
    required correctness result forbids qualification at this level; a
    generic missing/unattempted correctness claim does not (STATE=
    'validated' is the status that demands full correctness passing)."""
    subject_digest = patch_validation_subject_digest(module.path)
    validation_digest, contracts = _resolve_contract_identities(module)
    qualifying: list[dict[str, object]] = []
    stale: list[str] = []
    for record in load_records(module.patch_id, root=root):
        ok, why = _record_qualifies_for_benched(
            record, module=module, pinned_ref=pinned_ref, subject_digest=subject_digest,
            resolved_base_revision=resolved_base_revision, validation_digest=validation_digest,
            contracts=contracts,
        )
        if ok:
            qualifying.append(record)
        else:
            digest_prefix = str(record.get("campaign_identity_digest", "?"))[:12]
            stale.append(f"{digest_prefix}: " + "; ".join(why))
    if qualifying:
        return EvidenceCheck(
            "ported-benched-evidence",
            campaign_digests=tuple(sorted({str(r["campaign_identity_digest"]) for r in qualifying})),
        )
    problems = ["no current qualifying ported-benched evidence"]
    if stale:
        problems.append("stale/ineligible: " + " | ".join(stale[:3]))
    return EvidenceCheck("missing-or-stale", tuple(problems))


def verify_deferred_hardware_patch(
    module: patchset.PatchModule, *, pinned_ref: str, root: Path | None = None,
    resolved_base_revision: str | None = None,
) -> EvidenceCheck:
    """VA08: current-pin qualification for the 'deferred-hardware'
    tracked-status -- requires only a fresh, structured BLOCKED/deferred
    evidence record whose patch identity and resolved base revision are
    current. Never requires completed hardware/build/benchmark evidence
    (that would be self-defeating for a status whose whole meaning is
    "hardware was unavailable")."""
    subject_digest = patch_validation_subject_digest(module.path)
    qualifying: list[dict[str, object]] = []
    stale: list[str] = []
    for record in load_records(module.patch_id, root=root):
        ok, why = _record_qualifies_for_deferred_hardware(
            record, module=module, pinned_ref=pinned_ref, subject_digest=subject_digest,
            resolved_base_revision=resolved_base_revision,
        )
        if ok:
            qualifying.append(record)
        else:
            digest_prefix = str(record.get("campaign_identity_digest", "?"))[:12]
            stale.append(f"{digest_prefix}: " + "; ".join(why))
    if qualifying:
        return EvidenceCheck(
            "deferred-hardware-evidence",
            campaign_digests=tuple(sorted({str(r["campaign_identity_digest"]) for r in qualifying})),
        )
    problems = ["no current qualifying deferred-hardware (BLOCKED) evidence"]
    if stale:
        problems.append("stale/ineligible: " + " | ".join(stale[:3]))
    return EvidenceCheck("missing-or-stale", tuple(problems))


def generate_legacy_baseline(*, root: Path | None = None) -> Path:
    """One-time migration snapshot: every currently-STATE='validated' patch's
    exact content hash, as project history already accepted it pre-HI83.
    This does NOT claim those patches have HI82 hardware evidence -- it only
    prevents HI83 from retroactively invalidating the whole existing
    validated baseline the moment it lands. Editing any grandfathered patch
    by even one byte immediately requires real HI83 evidence."""
    modules = patchset.catalog()
    payload = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "contract": LEGACY_CONTRACT,
        "patches": {
            module.patch_id: module.content_hash
            for module in modules if module.state == "validated"
        },
    }
    output = Path(root or paths.PATCHES / "_validation") / "legacy-baseline.json"
    _atomic_json(output, payload)
    return output
