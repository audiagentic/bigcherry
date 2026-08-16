"""Explicit campaign-backed release promotion pointer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


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
        }


def make_pointer(*, release_tag: str, revision: str, campaign_plan_id: str,
                 campaign_run_id: str, report: bytes, source_slice_id: str,
                 build_id: str, binary_hash: str, valid: bool) -> PromotionPointer:
    if not valid:
        raise PromotionError("campaign report is not valid for promotion")
    if not all(isinstance(value, str) and value for value in (
        release_tag, revision, campaign_plan_id, campaign_run_id,
        source_slice_id, build_id, binary_hash)):
        raise PromotionError("promotion identities must be non-empty strings")
    return PromotionPointer(
        schema_version=2, release_tag=release_tag, revision=revision,
        campaign_plan_id=campaign_plan_id, campaign_run_id=campaign_run_id,
        report_hash=hashlib.sha256(report).hexdigest(),
        source_slice_id=source_slice_id, build_id=build_id, binary_hash=binary_hash,
    )
