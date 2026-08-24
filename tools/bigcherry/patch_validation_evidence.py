"""HI83: tracked evidence contract for patch STATE="validated".

GPT review of the completed HI82 body of work (gpt-auto-agent,
req_6b1466ee8369406c) found the most consequential remaining gap: BigCherry
can now produce strong, real evidence that a patch was genuinely built,
activated, and tested (HI82) -- but the repository's authoritative patch
STATE="validated" is not mechanically tied to that evidence. A developer can
still hand-edit a patch module's STATE with no machine check that matching
evidence exists, or that it still matches the CURRENT patch implementation.

This module is the tracked evidence contract itself: a JSON record per
patch, one file per patch under docs/reference/patch-validation-evidence/
(tracked in git -- the authority must be resolvable from the repository
alone, not from artifacts/, which is gitignored, or the external ledger,
which offline pytest/CI/a fresh checkout cannot resolve).

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

from . import paths, patchset
from .patch_activation import ActivationEvidence

SCHEMA_VERSION = 2
READABLE_SCHEMA_VERSIONS = (1, 2)
CONTRACT_VERSION = "hi83-v1"
CORRECTNESS_SCHEMA_VERSION = 1

# One-time migration contract -- see generate_legacy_baseline().
LEGACY_SCHEMA_VERSION = 1
LEGACY_CONTRACT = "hi83-legacy-grandfather-v1"

BUILD_ROLES = ("tune", "replay", "stock")

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
        return self.status in {"not-required", "validated-evidence", "legacy-grandfathered"}


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
    raise ValidationEvidenceError(
        f"{path}: expected exactly one literal STATE assignment or a packaged patch.toml state"
    )


def _validate_build_identity(role: str, value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationEvidenceError(f"build_identities.{role} must be an object")
    missing = [key for key in BUILD_IDENTITY_KEYS if key not in value]
    if missing:
        raise ValidationEvidenceError(f"build_identities.{role} missing field(s) {missing!r}")

    result = dict(value)
    for key in BUILD_IDENTITY_KEYS:
        field = f"build_identities.{role}.{key}"
        if key == "runtime_artifacts":
            artifacts = result[key]
            if not isinstance(artifacts, Mapping) or not artifacts:
                raise ValidationEvidenceError(f"{field} must be a non-empty object")
            for name, digest in artifacts.items():
                _require_string(name, f"{field} artifact name")
                _require_hex(digest, f"{field}.{name}", (64,))
        else:
            _require_string(result[key], field)
    return result


def evidence_path(patch_id: str, *, root: Path | None = None) -> Path:
    return Path(root or paths.DOCS / "reference" / "patch-validation-evidence") / f"{patch_id}.json"


def _artifact_refs(campaign_workdir: Path) -> list[dict[str, str]]:
    root = Path(campaign_workdir).resolve()
    names = (
        "status.json", "activation.json", "correctness.json", "bench.json", "report.md",
        "tune.jsonl.measurements.jsonl", "promoted.jsonl", "coverage.json", "dispatch.cache",
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


def make_record(
    *, patch_id: str, patch_path: Path, patch_implementation_digest: str, base_ref: str,
    base_revision: str, framework_baseline_digest: str, patched_source_tree: str,
    gpu_architectures: str | Iterable[str], activation_evidence: ActivationEvidence | None,
    activation_disposition: str | None, correctness: Mapping[str, object] | None,
    campaign_identity_digest: str, build_identities: Mapping[str, Mapping[str, object]],
    campaign_workdir: Path,
    representation: str = "simple", validation_implementation_digest: str | None = None,
    contract_id: str | None = None, baseline_composition: Mapping[str, object] | None = None,
    control_composition: Mapping[str, object] | None = None,
    subject_composition: Mapping[str, object] | None = None,
    control_tree: str | None = None, subject_tree: str | None = None,
    stock_tree: str | None = None, blockers: Iterable[str] = (),
    check_results: Mapping[str, object] | None = None,
    validation_eligible: bool | None = None,
) -> dict[str, object]:
    subject_digest = patch_validation_subject_digest(patch_path)
    archs = _architectures(gpu_architectures)
    if not archs:
        raise ValidationEvidenceError("gpu_architectures must not be empty")

    builds: dict[str, object] = {}
    for role in BUILD_ROLES:
        if role not in build_identities:
            raise ValidationEvidenceError(f"missing build identity role {role!r}")
        builds[role] = _validate_build_identity(role, build_identities[role])

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
        "record_schema_version": SCHEMA_VERSION,
        "validation_contract_version": CONTRACT_VERSION,
        "representation": representation,
        "validation_implementation_digest": validation_implementation_digest or _validation_digest(patch_path),
        "contract_id": contract_id or correctness_doc.get("contract_id"),
        "contract_hash": correctness_doc.get("contract_hash"),
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
        "build_identities": builds,
        "campaign_artifacts": _artifact_refs(campaign_workdir),
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
        # Reading a v1 container is supported; any successful write upgrades
        # the container header to v2 without rewriting legacy records.
        document["schema_version"] = SCHEMA_VERSION
    else:
        document = {"schema_version": SCHEMA_VERSION, "patch_id": patch_id, "records": []}

    records = document["records"]
    assert isinstance(records, list)
    new_record = dict(record)

    for old in records:
        if not isinstance(old, dict):
            raise ValidationEvidenceError(f"{path}: non-object record")
        if old.get("campaign_identity_digest") != campaign_digest:
            continue
        if old == new_record:
            return path
        raise ValidationEvidenceError(
            f"{path}: campaign digest {campaign_digest} already has different evidence"
        )

    records.append(new_record)
    records.sort(key=lambda row: str(row.get("campaign_identity_digest", "")))
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
    contract_id: str | None = None, contract_hash: str | None = None,
) -> tuple[bool, tuple[str, ...]]:
    problems: list[str] = []
    expected = {
        "validation_contract_version": CONTRACT_VERSION,
        "patch_id": module.patch_id, "patch_validation_subject_digest": subject_digest,
        "base_ref": pinned_ref, "validation_disposition": "validated",
        "eligible_for_validated_state": True,
    }
    if record.get("record_schema_version") not in READABLE_SCHEMA_VERSIONS:
        problems.append(f"record_schema_version={record.get('record_schema_version')!r} is unsupported")
    if record.get("record_schema_version") == 2:
        if validation_digest is not None and record.get("validation_implementation_digest") != validation_digest:
            problems.append("validation implementation digest is stale")
        if record.get("contract_id") != contract_id:
            problems.append("contract identity is stale")
        if contract_hash is not None and record.get("contract_hash") != contract_hash:
            problems.append("contract hash is stale")
        if record.get("record_digest") != _record_digest(record):
            problems.append("record_digest does not match the v2 evidence payload")
        required_v2 = ("representation", "validation_implementation_digest",
                       "baseline_composition", "control_composition", "subject_composition",
                       "control_tree", "subject_tree", "stock_tree", "check_results",
                       "hardware", "artifact_hashes", "blockers", "final_eligibility")
        for key in required_v2:
            value = record.get(key)
            if value is None or value == {}:
                problems.append(f"v2 provenance field {key!r} is missing")
        try:
            _require_hex(record.get("validation_implementation_digest"),
                         "validation_implementation_digest", (64,))
        except ValidationEvidenceError as exc:
            problems.append(str(exc))
        if record.get("contract_id") is not None and not record.get("contract_hash"):
            problems.append("contract_hash is required when contract_id is present")
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
        builds = record.get("build_identities")
        if not isinstance(builds, Mapping):
            raise ValidationEvidenceError("build_identities must be an object")
        for role in BUILD_ROLES:
            _validate_build_identity(role, builds.get(role))
    except ValidationEvidenceError as exc:
        problems.append(str(exc))

    if not _architectures(record.get("gpu_architectures") or ()):
        problems.append("gpu_architectures is empty")

    return (not problems, tuple(problems))


def _legacy_hashes(root: Path | None) -> dict[str, str]:
    path = Path(root or paths.DOCS / "reference" / "patch-validation-evidence") / "legacy-baseline.json"
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
    validation_digest = None
    contract_id = None
    contract_hash = None
    try:
        from . import patch_registry
        registry = patch_registry.load_registry(module.catalog_root or paths.PATCHES)
        descriptor = registry.get(module.patch_id)
        validation_digest = descriptor.validation_digest or _validation_digest(module.path)
        contract_id = descriptor.experiment_contract
        if descriptor.experiment_contract is not None:
            from . import patch_validation
            binding = patch_validation.load_contract_for_descriptor(descriptor)
            if binding is not None:
                contract_hash = binding.contract_hash
    except (OSError, ValueError, KeyError):
        # Legacy/synthetic callers may not have a registry; structural checks
        # still apply, while production catalog verification remains strict.
        pass
    qualifying: list[dict[str, object]] = []
    stale: list[str] = []

    for record in load_records(module.patch_id, root=root):
        ok, why = _record_qualifies(
            record, module=module, pinned_ref=pinned_ref, subject_digest=subject_digest,
            resolved_base_revision=resolved_base_revision,
            validation_digest=validation_digest, contract_id=contract_id,
            contract_hash=contract_hash,
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
            campaign_digests=tuple(
                sorted(str(record["campaign_identity_digest"]) for record in qualifying)
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
    output = Path(root or paths.DOCS / "reference" / "patch-validation-evidence") / "legacy-baseline.json"
    _atomic_json(output, payload)
    return output
