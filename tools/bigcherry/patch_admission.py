"""HI102 post-composition admission policy.

Composition answers *which* patches were selected.  This module answers
whether that already-resolved selection may proceed in a named operation.  It
deliberately consumes ``patch.catalog`` and ``patch.evidence``; it does not
reimplement parsing, dependency expansion, or evidence verification.

Policy decisions:

* The current authoritative architecture default is ``gfx1100``: it is the
  only architecture for which this repository has an actual validated-state
  campaign history.  Catalog/package metadata can narrow or expand that set.
  Human confirmation is still required before declaring RDNA3+RDNA4 a global
  obligation; making that expansion here without records would block every
  future promotion.
* A validated patch is admissible with current evidence (qualifying record or
  retained legacy-grandfather record at the live pin/composition).
* Performance/qualification evidence is re-established on request, not at
  every pin bump: when the evidence is stale only because the pin or the
  surrounding composition moved, but an eligible record exists for this
  exact patch implementation digest, the patch is admitted as
  ``carried-forward`` with a warning.  A changed patch implementation, or no
  eligible record at all, stays a hard failure.  Hard mechanical validity
  (applies without conflict, not absorbed upstream) is enforced separately by
  patch-rebase-check and the apply transaction.
* Direct apply is hard-fail by default.  ``allow_stale_validation_evidence`` is
  an explicit development escape hatch and returns a warning; it never affects
  the production campaign/build gate.
* The production gate is activation-aware: until one non-grandfathered
  ``eligible_for_validated_state=true`` record exists, the gate reports
  ``not-ready`` and allows existing work to continue.  Once that bootstrap
  condition is met, failures are hard errors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .core import paths
from .patch import catalog as patch_catalog
from .patch import evidence as patch_evidence
from .patch import patchset
from .source import identity as source_identity

DEFAULT_VALIDATION_ARCHITECTURES = ("gfx1100",)
AdmissionMode = Literal["production", "apply"]


@dataclass(frozen=True)
class AdmissionResult:
    admissible: bool
    gate_active: bool
    status: str
    failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def live_revision(root: Path) -> str:
    """Read the live checkout revision through the shared identity authority."""
    return source_identity.git_revision(Path(root))


def verify_live_revision(root: Path, expected_revision: str) -> None:
    """Fail closed if a resolved pin is not the checkout being inspected.

    ``git_revision`` is intentionally reused from ``source.identity`` rather
    than adding another subprocess/git identity implementation here.  Full
    tree/source-slice attestation remains available to callers that have a
    ``SourceAttestation``; this helper covers the pinned-revision obligation.
    """
    actual = live_revision(root)
    if actual != expected_revision:
        raise ValueError(
            "live source revision does not match resolved validation pin: "
            f"{actual!r} != {expected_revision!r}"
        )


def _has_non_grandfathered_eligible(*, evidence_root: Path | None = None) -> bool:
    root = Path(evidence_root) if evidence_root is not None else paths.PATCHES
    candidates = root.glob("*.json") if evidence_root is not None else root.glob("**/evidence/*.json")
    for path in candidates:
        if path.name == "legacy-baseline.json":
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for record in document.get("records", []) if isinstance(document, dict) else ():
            if isinstance(record, dict) and record.get("eligible_for_validated_state") is True:
                return True
    return False


def _eligible_record_for_digest(
    patch_id: str, implementation_digest: str | None, *, evidence_root: Path | None,
) -> str | None:
    """base_ref of the newest eligible record for exactly this implementation."""
    if not implementation_digest:
        return None
    try:
        records = patch_evidence.load_records(patch_id, root=evidence_root)
    except patch_evidence.ValidationEvidenceError:
        return None
    for record in reversed(records):
        if (
            record.get("eligible_for_validated_state") is True
            and record.get("patch_implementation_digest") == implementation_digest
        ):
            return str(record.get("base_ref") or "unknown pin")
    return None


def admit(
    patch_ids: Iterable[str],
    *,
    mode: AdmissionMode = "production",
    catalog_path: Path | None = None,
    patches_dir: Path | None = None,
    pinned_ref: str | None = None,
    resolved_base_revision: str | None = None,
    evidence_root: Path | None = None,
    allow_stale_validation_evidence: bool = False,
    allow_legacy_grandfather: bool = True,
) -> AdmissionResult:
    """Admit an already-resolved patch composition.

    No patch is removed or substituted.  ``not-ready`` is deliberately
    additive bootstrap behavior because the current repository has no real
    non-grandfathered eligible record yet.
    """
    ids = tuple(patch_ids)
    statuses = patch_catalog.validation_evidence_statuses(
        ids, catalog_path=catalog_path, patches_dir=patches_dir,
        pinned_ref=pinned_ref, evidence_root=evidence_root,
        allow_legacy_grandfather=allow_legacy_grandfather,
        resolved_base_revision=resolved_base_revision,
        default_validation_architectures=DEFAULT_VALIDATION_ARCHITECTURES,
    )
    digests = {m.patch_id: m.content_hash for m in patchset.catalog(patches_dir)}
    failures: list[str] = []
    carried: list[str] = []
    for patch_id, check in statuses.items():
        if check.ok:
            continue
        qualified_at = _eligible_record_for_digest(
            patch_id, digests.get(patch_id), evidence_root=evidence_root,
        )
        if qualified_at is not None:
            carried.append(
                f"{patch_id}: carried-forward (qualified at {qualified_at} for this "
                "implementation; stale for the current pin/composition -- revalidate on request)"
            )
        else:
            failures.append(f"{patch_id}: {('; '.join(check.problems) or check.status)}")
    failures = tuple(failures)
    bootstrap_ready = _has_non_grandfathered_eligible(evidence_root=evidence_root)
    if mode == "production" and not bootstrap_ready:
        return AdmissionResult(True, False, "not-ready", warnings=failures)
    if failures and mode == "apply" and allow_stale_validation_evidence:
        return AdmissionResult(True, False, "escape-hatch", warnings=failures + tuple(carried))
    return AdmissionResult(
        not failures, True, "admitted" if not failures else "rejected",
        failures=failures, warnings=tuple(carried),
    )


def require_admission(*args, **kwargs) -> AdmissionResult:
    result = admit(*args, **kwargs)
    if not result.admissible:
        detail = "; ".join(result.failures) or result.status
        raise ValueError(f"patch admission failed: {detail}")
    return result
