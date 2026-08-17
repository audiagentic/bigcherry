"""Explicit campaign-backed release promotion pointer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .campaign_lane import CampaignLaneResult


class PromotionError(ValueError):
    pass


@dataclass(frozen=True)
class PromotionPointer:
    schema_version: int
    release_tag: str
    revision: str
    campaign_plan_id: str
    campaign_run_id: str
    report_hash: str
    source_slice_id: str
    build_id: str
    binary_hash: str
    #: Architectures the validated campaign report actually covered (RE13
    #: requirement 2) -- e.g. ("gfx1100", "gfx1201"). A promotion pointer
    #: without this is a claim of production-readiness with no record of
    #: which hardware it was ever proven on.
    required_architectures: tuple[str, ...]
    #: content_hash of the published runtime-bundle artifact the replay
    #: build actually loads (RE13 requirement 2) -- what gets deployed is
    #: the replay build's runtime closure, not the tune build's binary_hash
    #: above (which identifies the campaign's *measuring* build, not what
    #: ships). Distinct from binary_hash for the same reason RE07 keeps
    #: binary and runtime-bundle as separate ArtifactRefs: the launcher
    #: alone does not prove the .so closure it loads is unmodified.
    replay_artifact_hash: str

    def document(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "release_tag": self.release_tag,
            "revision": self.revision,
            "validated_campaign": {
                "campaign_plan_id": self.campaign_plan_id,
                "campaign_run_id": self.campaign_run_id,
                "report_hash": self.report_hash,
            },
            "promoted_source": {
                "source_slice_id": self.source_slice_id,
                "build_id": self.build_id,
                "binary_hash": self.binary_hash,
            },
            "required_architectures": list(self.required_architectures),
            "replay_artifact_hash": self.replay_artifact_hash,
        }


def make_pointer(*, release_tag: str, revision: str, campaign_plan_id: str,
                 campaign_run_id: str, report: bytes, source_slice_id: str,
                 build_id: str, binary_hash: str,
                 required_architectures: tuple[str, ...],
                 replay_artifact_hash: str, valid: bool) -> PromotionPointer:
    if not valid:
        raise PromotionError("campaign report is not valid for promotion")
    if not all(isinstance(value, str) and value for value in (
        release_tag, revision, campaign_plan_id, campaign_run_id,
        source_slice_id, build_id, binary_hash, replay_artifact_hash)):
        raise PromotionError("promotion identities must be non-empty strings")
    if (not required_architectures
            or not all(isinstance(arch, str) and arch for arch in required_architectures)):
        raise PromotionError(
            "promotion requires at least one non-empty required architecture")
    return PromotionPointer(
        schema_version=2, release_tag=release_tag, revision=revision,
        campaign_plan_id=campaign_plan_id, campaign_run_id=campaign_run_id,
        report_hash=hashlib.sha256(report).hexdigest(),
        source_slice_id=source_slice_id, build_id=build_id, binary_hash=binary_hash,
        required_architectures=tuple(required_architectures),
        replay_artifact_hash=replay_artifact_hash,
    )


def pointer_from_campaign_result(
    *, result: "CampaignLaneResult", release_tag: str, campaign_plan_id: str,
    architectures: tuple[str, ...], report: bytes, valid: bool,
) -> PromotionPointer:
    """RE13: the real bridge from a genuine ``execute_campaign_lane()``
    result to a promotion pointer -- every identity below is read directly
    off the campaign's own immutable artifacts, never re-derived or
    trusted from caller-supplied strings the way the old mutable-checkout
    ``tree_state`` path effectively was.

    Deliberately narrow: this builds ONE pointer from ONE already-executed
    lane result plus an already-produced release report; it does not run a
    campaign, does not decide what "validated" evidence looks like (that is
    release_validate.validate_release_claim's job), and does not persist
    anything (releases.promote() does that). Full record -> tune -> promote
    -> replay -> coverage orchestration is separate, larger work.
    """
    return make_pointer(
        release_tag=release_tag, revision=result.resolved_revision,
        campaign_plan_id=campaign_plan_id, campaign_run_id=result.run_id,
        report=report, source_slice_id=result.source_slice_id,
        build_id=result.build_plan.build_plan_id,
        binary_hash=result.binary_ref.content_hash,
        required_architectures=architectures,
        # RE13: the replay build's own runtime-bundle closure, not
        # binary_ref alone -- see PromotionPointer.replay_artifact_hash.
        replay_artifact_hash=result.runtime_bundle_ref.content_hash,
        valid=valid,
    )
