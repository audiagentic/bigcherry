"""Explicit campaign-backed release promotion pointer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

from . import ab_benchmark, provenance
from .artifacts import ArtifactError, ArtifactStore

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
    #: RE12/schema 3: the comparison-report ArtifactRef this pointer was
    #: actually built from -- "" for a schema-2 pointer (report_hash alone
    #: was the only anchor then; a schema-3 pointer additionally lets a
    #: verifier rehydrate and re-check the full report bytes, not just
    #: trust their recorded hash).
    report_artifact_id: str = ""
    #: RE12 audit fix (2026-08-18, item 6): the independent replay-validation
    #: coverage artifact this pointer's pairing was checked against.
    #: Replay arm identity is only as strong as the coverage evidence a
    #: verifier can re-fetch, so the pointer persists its own anchor to it
    #: (schema 3 only; "" for schema 2).
    coverage_artifact_id: str = ""

    def document(self) -> dict[str, object]:
        document: dict[str, object] = {
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
        if self.schema_version >= 3:
            document["report_artifact_id"] = self.report_artifact_id
            document["coverage_artifact_id"] = self.coverage_artifact_id
        return document

    @classmethod
    def from_document(cls, document: object) -> "PromotionPointer":
        """GPT-auto-agent review (RE13 follow-up, 2026-08-17): the strict
        inverse of document() -- every field document() writes is required
        and type/non-emptiness-checked here, not just ``isinstance(dict)``
        plus a revision match. A persisted release record's own claim of
        ``validated`` is only as trustworthy as this check: the review
        found a record saving ``{"schema_version": 2, "revision": "..."}``
        as its entire promotion pointer and passing the old, loose check.
        """
        if not isinstance(document, dict):
            raise PromotionError("promotion pointer document must be an object")
        schema_version = document.get("schema_version")
        if schema_version not in (2, 3):
            raise PromotionError("promotion pointer has an unsupported schema_version")

        def _str(value: object, field: str) -> str:
            if not isinstance(value, str) or not value:
                raise PromotionError(
                    f"promotion pointer field {field!r} must be a non-empty string"
                )
            return value

        release_tag = _str(document.get("release_tag"), "release_tag")
        revision = _str(document.get("revision"), "revision")
        campaign = document.get("validated_campaign")
        if not isinstance(campaign, dict):
            raise PromotionError("promotion pointer lacks validated_campaign")
        campaign_plan_id = _str(
            campaign.get("campaign_plan_id"), "validated_campaign.campaign_plan_id"
        )
        campaign_run_id = _str(
            campaign.get("campaign_run_id"), "validated_campaign.campaign_run_id"
        )
        report_hash = _str(
            campaign.get("report_hash"), "validated_campaign.report_hash"
        )
        source = document.get("promoted_source")
        if not isinstance(source, dict):
            raise PromotionError("promotion pointer lacks promoted_source")
        source_slice_id = _str(
            source.get("source_slice_id"), "promoted_source.source_slice_id"
        )
        build_id = _str(source.get("build_id"), "promoted_source.build_id")
        binary_hash = _str(source.get("binary_hash"), "promoted_source.binary_hash")
        required_architectures = document.get("required_architectures")
        if (
            not isinstance(required_architectures, list)
            or not required_architectures
            or not all(
                isinstance(arch, str) and arch for arch in required_architectures
            )
        ):
            raise PromotionError(
                "promotion pointer required_architectures must be a non-empty list of "
                "non-empty strings"
            )
        replay_artifact_hash = _str(
            document.get("replay_artifact_hash"), "replay_artifact_hash"
        )
        report_artifact_id = ""
        coverage_artifact_id = ""
        if schema_version == 3:
            report_artifact_id = _str(
                document.get("report_artifact_id"), "report_artifact_id"
            )
            coverage_artifact_id = _str(
                document.get("coverage_artifact_id"), "coverage_artifact_id"
            )
        return cls(
            schema_version=schema_version,
            release_tag=release_tag,
            revision=revision,
            campaign_plan_id=campaign_plan_id,
            campaign_run_id=campaign_run_id,
            report_hash=report_hash,
            source_slice_id=source_slice_id,
            build_id=build_id,
            binary_hash=binary_hash,
            required_architectures=tuple(required_architectures),
            replay_artifact_hash=replay_artifact_hash,
            report_artifact_id=report_artifact_id,
            coverage_artifact_id=coverage_artifact_id,
        )


def make_pointer(
    *,
    release_tag: str,
    revision: str,
    campaign_plan_id: str,
    campaign_run_id: str,
    report: bytes,
    source_slice_id: str,
    build_id: str,
    binary_hash: str,
    required_architectures: tuple[str, ...],
    replay_artifact_hash: str,
    valid: bool,
) -> PromotionPointer:
    if not valid:
        raise PromotionError("campaign report is not valid for promotion")
    if not all(
        isinstance(value, str) and value
        for value in (
            release_tag,
            revision,
            campaign_plan_id,
            campaign_run_id,
            source_slice_id,
            build_id,
            binary_hash,
            replay_artifact_hash,
        )
    ):
        raise PromotionError("promotion identities must be non-empty strings")
    if not required_architectures or not all(
        isinstance(arch, str) and arch for arch in required_architectures
    ):
        raise PromotionError(
            "promotion requires at least one non-empty required architecture"
        )
    return PromotionPointer(
        schema_version=2,
        release_tag=release_tag,
        revision=revision,
        campaign_plan_id=campaign_plan_id,
        campaign_run_id=campaign_run_id,
        report_hash=hashlib.sha256(report).hexdigest(),
        source_slice_id=source_slice_id,
        build_id=build_id,
        binary_hash=binary_hash,
        required_architectures=tuple(required_architectures),
        replay_artifact_hash=replay_artifact_hash,
    )


def pointer_from_campaign_result(
    *,
    result: "CampaignLaneResult",
    release_tag: str,
    campaign_plan_id: str,
    architectures: tuple[str, ...],
    report: bytes,
    valid: bool,
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

    ``architectures`` is cross-checked against real campaign evidence, not
    trusted from the caller unchecked (GPT-auto-agent review, RE13
    follow-up, 2026-08-17 -- both the original finding and a residual gap
    in the first fix). A build WITH a generate stage is checked against
    ``build_plan.catalog_architectures`` (the architectures generation was
    actually asked to cover). A build with NO generate stage has an empty
    ``catalog_architectures`` by construction (RE07: that field means "no
    catalog to disambiguate", not "no data") -- the first fix skipped the
    check entirely in that case, which the review correctly flagged as
    still lettting a no-generate result claim coverage for an architecture
    it was never even COMPILED for. Falls back to ``build_plan.targets``
    (the AMDGPU targets actually compiled) there instead of skipping.
    """
    catalog_architectures = result.build_plan.catalog_architectures
    covered = catalog_architectures or result.build_plan.targets
    if covered and not set(architectures) <= set(covered):
        raise PromotionError(
            f"claimed required_architectures {sorted(architectures)} exceed what "
            f"this campaign result actually covered {sorted(covered)}"
        )
    return make_pointer(
        release_tag=release_tag,
        revision=result.resolved_revision,
        campaign_plan_id=campaign_plan_id,
        campaign_run_id=result.run_id,
        report=report,
        source_slice_id=result.source_slice_id,
        build_id=result.build_plan.build_plan_id,
        binary_hash=result.binary_ref.content_hash,
        required_architectures=architectures,
        # RE13: the replay build's own runtime-bundle closure, not
        # binary_ref alone -- see PromotionPointer.replay_artifact_hash.
        replay_artifact_hash=result.runtime_bundle_ref.content_hash,
        valid=valid,
    )


def pointer_from_comparison_report(
    *,
    store: ArtifactStore,
    report_artifact_id: str,
    release_tag: str,
    replay_coverage_artifact_id: str,
    required_architectures: tuple[str, ...],
) -> PromotionPointer:
    """RE12 6.5: the release-pointer trust boundary for real balanced
    comparison evidence -- schema 3. Everything a pointer claims is read
    off the STORED, re-verified comparison-report artifact (never a
    caller-supplied campaign_plan_id/run_id/valid boolean the way the old
    pointer_from_campaign_result() call site historically trusted for
    ad-hoc reports): the report must be production-class, valid, and
    decision-grade, and its replay-side arm identity must independently
    match a real replay-validation coverage artifact this caller also
    names -- a report cannot claim replay evidence that was never actually
    run. report_hash is derived from the REHYDRATED bytes, not accepted as
    a claim inside the report itself."""
    report_ref = store.rehydrate(report_artifact_id, expected_kind="comparison-report")
    report_doc = provenance.ProvenanceV2.from_document(report_ref.provenance)
    if report_doc.project.provenance_class != "production":
        raise PromotionError(
            f"comparison report {report_artifact_id!r} is not production-class evidence "
            f"(provenance_class={report_doc.project.provenance_class!r})"
        )
    report_bytes = report_ref.path.read_bytes()
    try:
        report = json.loads(report_bytes)
    except json.JSONDecodeError as exc:
        raise PromotionError(
            f"comparison report {report_artifact_id!r} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(report, dict) or not report.get("valid"):
        raise PromotionError(f"comparison report {report_artifact_id!r} is not valid")
    if not report.get("decision_grade"):
        raise PromotionError(
            f"comparison report {report_artifact_id!r} is not decision-grade"
        )
    replay_side = report.get("replay_arm")
    if replay_side not in ("left", "right"):
        raise PromotionError(
            f"comparison report {report_artifact_id!r} has no replay arm to cross-check"
        )
    replay_arm = report.get(f"{replay_side}_arm")
    if not isinstance(replay_arm, dict):
        raise PromotionError(
            f"comparison report {report_artifact_id!r} is missing its {replay_side}_arm"
        )

    # GPT audit fix (item 3): a release pointer demands PROMOTABLE replay
    # evidence -- an ordinary rehydrate() accepted a development-class
    # coverage artifact, which the old happy-path test explicitly relied on.
    try:
        coverage_ref = store.rehydrate(
            replay_coverage_artifact_id,
            expected_kind="replay-coverage",
            require_promotable=True,
        )
    except (provenance.ProvenanceError, ArtifactError) as exc:
        raise PromotionError(
            f"replay validation artifact {replay_coverage_artifact_id!r} is not promotable: {exc}"
        ) from exc
    coverage_doc = provenance.ProvenanceV2.from_document(coverage_ref.provenance)
    reported_identity = (
        replay_arm.get("source_slice_id"),
        replay_arm.get("build_plan_id"),
        replay_arm.get("effective_build_id"),
    )
    actual_identity = (
        coverage_doc.source.source_slice_id,
        coverage_doc.build.build_plan_id,
        coverage_doc.build.effective_build_id,
    )
    if reported_identity != actual_identity:
        raise PromotionError(
            f"replay validation artifact {replay_coverage_artifact_id!r} identity "
            f"{actual_identity} does not match the comparison report's replay arm "
            f"identity {reported_identity}"
        )

    # GPT audit fix (item 4): a matching triple is NOT pairing -- a caller
    # could supply coverage from the same build run against a DIFFERENT
    # replay cache (or runtime bundle). The coverage's recorded parent
    # artifact IDs (replay validation actually consumed runtime-bundle +
    # replay-cache; execute_replay_validation_stage records both) must be
    # exactly the replay arm's own bundle/cache IDs.
    arm_bundle_id = replay_arm.get("runtime_bundle_artifact_id")
    arm_cache_id = replay_arm.get("replay_cache_artifact_id")
    if not isinstance(arm_bundle_id, str) or not arm_bundle_id:
        raise PromotionError(
            f"comparison report {report_artifact_id!r} replay arm lacks "
            f"runtime_bundle_artifact_id"
        )
    if not isinstance(arm_cache_id, str) or not arm_cache_id:
        raise PromotionError(
            f"comparison report {report_artifact_id!r} replay arm lacks "
            f"replay_cache_artifact_id -- a replay arm without a verified "
            f"cache is not release evidence"
        )
    expected_parents = {arm_bundle_id, arm_cache_id}
    actual_parents = set(coverage_doc.campaign.producer_artifact_ids)
    if actual_parents != expected_parents:
        raise PromotionError(
            f"replay validation artifact {replay_coverage_artifact_id!r} parent "
            f"artifacts {sorted(actual_parents)} are not exactly the replay arm's "
            f"runtime-bundle + replay-cache {sorted(expected_parents)} -- a matching "
            f"build identity does not prove the same cache was validated"
        )
    if (
        coverage_doc.workload.workload_id is not None
        and coverage_doc.workload.workload_id != replay_arm.get("workload_id")
    ):
        raise PromotionError(
            f"replay validation artifact {replay_coverage_artifact_id!r} workload "
            f"{coverage_doc.workload.workload_id!r} does not match the replay arm's "
            f"workload {replay_arm.get('workload_id')!r}"
        )

    # GPT audit fix (item 5): re-run the EXISTING coverage validator at the
    # final release boundary -- the promotion decision must be
    # self-contained, not inherit run-time trust.
    try:
        ab_benchmark.validate_replay_coverage(coverage_ref.path)
    except ValueError as exc:
        raise PromotionError(
            f"replay validation artifact {replay_coverage_artifact_id!r} fails "
            f"exact-replay coverage validation: {exc}"
        ) from exc

    if not required_architectures or not all(
        isinstance(arch, str) and arch for arch in required_architectures
    ):
        raise PromotionError(
            "promotion requires at least one non-empty required architecture"
        )
    replay_binary_artifact_id = report.get(f"{replay_side}_binary_artifact_id")
    if not isinstance(replay_binary_artifact_id, str) or not replay_binary_artifact_id:
        raise PromotionError(
            f"comparison report {report_artifact_id!r} is missing its {replay_side}_binary_artifact_id"
        )
    replay_binary_ref = store.rehydrate(
        replay_binary_artifact_id, expected_kind="binary"
    )
    binary_hash = replay_binary_ref.content_hash
    # GPT audit fix (item 6): replay_artifact_hash keeps its documented
    # meaning -- the content hash of the RUNTIME BUNDLE that ships, not the
    # coverage file's hash (a semantic incompatibility the schema-3
    # constructor had introduced). revision keeps the schema-2 axis too:
    # the promoted SOURCE revision from the replay arm's own binary
    # provenance, not the tooling (bigcherry) revision the report carries.
    replay_bundle_ref = store.rehydrate(arm_bundle_id, expected_kind="runtime-bundle")
    replay_binary_doc = provenance.ProvenanceV2.from_document(
        replay_binary_ref.provenance
    )

    return PromotionPointer(
        schema_version=3,
        release_tag=release_tag,
        revision=replay_binary_doc.source.upstream_revision or "",
        campaign_plan_id=report_doc.campaign.campaign_plan_id or "",
        campaign_run_id=report_doc.campaign.run_id or "",
        report_hash=hashlib.sha256(report_bytes).hexdigest(),
        source_slice_id=replay_arm.get("source_slice_id") or "",
        build_id=replay_arm.get("build_plan_id") or "",
        binary_hash=binary_hash,
        required_architectures=tuple(required_architectures),
        replay_artifact_hash=replay_bundle_ref.content_hash,
        report_artifact_id=report_artifact_id,
        coverage_artifact_id=replay_coverage_artifact_id,
    )
