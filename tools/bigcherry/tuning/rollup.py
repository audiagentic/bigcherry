"""Consolidated, derived, read-only rollup across multiple tune campaigns.

WHY THIS EXISTS. HI168 produced five independently-validated campaign cells
(GPU0, GPU1, GPU2/gfx1201, GPU3/gfx1030, dual-XTX-27B). Reviewing them one
promoted.jsonl at a time is real friction, and the owner asked for a
"consolidated tune file" to make release packaging easier.

dev-gpt-agent's review (2026-09-08) is the design authority here: a single
MERGED replay cache is explicitly the wrong answer for packaging -- BigCherry's
real release model is "validated runtime-bundle + replay-cache pairs per
campaign/cell", not one binary artifact, and GPU0/GPU1 (identical hardware
digest) can genuinely disagree on a winner for the same portable key, which a
merged cache cannot represent. This module builds the SAFE consolidation
instead: a derived, read-only, per-row rollup joining every campaign's
promoted.jsonl (+ its execution-audit classification, if one exists) with the
campaign identity fields needed to tell rows apart -- deliberately NOT deduped
by dispatch digest, because a GPU0/GPU1 disagreement on an otherwise-identical
key is itself the evidence a future reconciliation decision would need.

The underlying per-campaign caches, receipts and audits remain the artifacts
of record; this file is a view over them, not a replacement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RollupRow:
    campaign_run_id: str
    model_path: str
    devices: str
    replay_source_root: str
    replay_source_slice_id: str
    replay_build_plan_id: str
    dispatch: str
    signature: str
    native_candidate: str
    replay_candidate: str
    improvement_pct: float | None
    promotion_status: str
    execution_audit_classification: str | None
    actual_launched_candidate: str | None
    actual_from_cache: bool | None

    def to_json(self) -> dict[str, Any]:
        return {
            "campaign_run_id": self.campaign_run_id,
            "model_path": self.model_path,
            "devices": self.devices,
            "replay_source_root": self.replay_source_root,
            "replay_source_slice_id": self.replay_source_slice_id,
            "replay_build_plan_id": self.replay_build_plan_id,
            "dispatch": self.dispatch,
            "signature": self.signature,
            "native_candidate": self.native_candidate,
            "replay_candidate": self.replay_candidate,
            "improvement_pct": self.improvement_pct,
            "promotion_status": self.promotion_status,
            "execution_audit_classification": self.execution_audit_classification,
            "actual_launched_candidate": self.actual_launched_candidate,
            "actual_from_cache": self.actual_from_cache,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_campaign_identity(campaign_dir: Path) -> dict[str, str]:
    """Pull the identity fields a rollup row needs from the campaign's own
    receipt. Fails closed: a campaign directory without a receipt is not
    silently rolled up with blank identity -- that would defeat the point of
    a consolidated file (telling rows apart)."""
    receipt_path = campaign_dir / "tune-campaign-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    replay = receipt.get("replay", {})
    return {
        "campaign_run_id": receipt.get("campaign_run_id", campaign_dir.name),
        "model_path": receipt.get("model_path", ""),
        "devices": receipt.get("devices", ""),
        "replay_source_root": replay.get("source_root", ""),
        "replay_source_slice_id": replay.get("source_slice_id", ""),
        "replay_build_plan_id": replay.get("build_plan_id", ""),
    }


def _load_execution_audit(campaign_dir: Path) -> dict[str, dict[str, Any]]:
    """Keyed by dispatch digest. Missing audit is legitimate (HI171 step 5
    has not run against this cache yet) -- rows still roll up, with
    execution_audit_classification=None making the gap explicit rather than
    silently treating absence as a finding."""
    audit_path = campaign_dir / "hip-tuning-execution-audit.jsonl"
    if not audit_path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(audit_path):
        dispatch = row.get("dispatch")
        if dispatch:
            out[dispatch] = row
    return out


def build_rollup(campaign_dirs: Iterable[str | Path]) -> list[RollupRow]:
    rows: list[RollupRow] = []
    for raw_dir in campaign_dirs:
        campaign_dir = Path(raw_dir)
        identity = _load_campaign_identity(campaign_dir)
        audit_by_dispatch = _load_execution_audit(campaign_dir)

        promoted_path = campaign_dir / "promoted.jsonl"
        for promoted_row in _read_jsonl(promoted_path):
            if promoted_row.get("kind") == "header":
                continue
            if promoted_row.get("promotion_status") != "promoted":
                continue
            dispatch = promoted_row.get("dispatch")
            if not dispatch or not promoted_row.get("winner"):
                continue

            audit_row = audit_by_dispatch.get(dispatch)
            rows.append(RollupRow(
                campaign_run_id=identity["campaign_run_id"],
                model_path=identity["model_path"],
                devices=identity["devices"],
                replay_source_root=identity["replay_source_root"],
                replay_source_slice_id=identity["replay_source_slice_id"],
                replay_build_plan_id=identity["replay_build_plan_id"],
                dispatch=dispatch,
                signature=promoted_row.get("signature", ""),
                native_candidate=promoted_row.get("native", ""),
                replay_candidate=promoted_row["winner"],
                improvement_pct=promoted_row.get("improvement_pct"),
                promotion_status=promoted_row["promotion_status"],
                execution_audit_classification=(
                    audit_row.get("classification") if audit_row else None
                ),
                actual_launched_candidate=(
                    audit_row.get("actual_launched_candidate") if audit_row else None
                ),
                actual_from_cache=(
                    audit_row.get("actual_from_cache") if audit_row else None
                ),
            ))
    return rows


@dataclass(frozen=True)
class RollupSummary:
    total: int
    by_campaign: dict[str, int]
    by_classification: dict[str, int]

    @property
    def unaudited(self) -> int:
        return self.by_classification.get("None", 0)


def summarize(rows: Iterable[RollupRow]) -> RollupSummary:
    rows = list(rows)
    by_campaign: dict[str, int] = {}
    by_classification: dict[str, int] = {}
    for row in rows:
        by_campaign[row.campaign_run_id] = by_campaign.get(row.campaign_run_id, 0) + 1
        cls = str(row.execution_audit_classification)
        by_classification[cls] = by_classification.get(cls, 0) + 1
    return RollupSummary(total=len(rows), by_campaign=by_campaign, by_classification=by_classification)


def write_rollup(rows: Iterable[RollupRow], output_path: str | Path) -> None:
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_json(), sort_keys=True))
            fh.write("\n")
