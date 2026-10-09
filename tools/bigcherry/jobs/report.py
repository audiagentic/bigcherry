"""Deterministic review report for one harvested scientific series.

This layer never invents a second promotion policy.  It reads only the records
named by the verified harvest manifest and uses experiment.contract's existing
session bootstrap for repeated-session lane summaries.  Contract verdicts stay
the authority for contract-specific pass/fail semantics.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from bigcherry.core import paths
from bigcherry.experiment.contract import bootstrap_session_effect
from bigcherry.patch import evidence as patch_evidence

from .store import RunStore


class ReportError(RuntimeError):
    pass


def _verified(store: RunStore, series_id: str) -> dict[str, Any]:
    value = store.read_optional(
        store.root / "series" / series_id / "verified-evidence.json"
    )
    if not value or value.get("verified") is not True:
        raise ReportError(
            f"series {series_id} has no committed verified evidence; harvest it first"
        )
    manifest = value.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("series_id") != series_id:
        raise ReportError("verified evidence manifest is malformed")
    return dict(value)


def _manifest_records(
    *, project_root: Path, manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    patch_id = str(manifest["patch_id"])
    wanted = {str(value) for value in manifest.get("record_digests", [])}
    if not wanted:
        raise ReportError("verified evidence manifest contains no record digests")
    available: dict[str, dict[str, Any]] = {}
    for row in patch_evidence.load_records(patch_id, root=paths.LLAMACPP.patches_root(project_root)):
        record = dict(row)
        digest = record.get("record_digest")
        if not isinstance(digest, str):
            continue
        if digest in wanted:
            # Recompute the evidence-layer digest before reporting it.
            if patch_evidence._record_digest(record) != digest:  # type: ignore[attr-defined]
                raise ReportError(f"harvested record {digest} failed digest verification")
            available[digest] = record
    missing = sorted(wanted - set(available))
    if missing:
        raise ReportError(
            "verified harvest names record(s) missing from canonical evidence: "
            + ", ".join(missing)
        )
    return available


def build_series_report(
    *, store: RunStore, series_id: str, project_root: Path
) -> dict[str, Any]:
    series = store.series(series_id)
    verified = _verified(store, series_id)
    manifest = dict(verified["manifest"])
    records = _manifest_records(project_root=project_root.resolve(), manifest=manifest)
    manifest_runs = manifest.get("runs")
    if not isinstance(manifest_runs, list):
        raise ReportError("verified evidence manifest has no run list")
    planned = int(series["series_material"]["planned_sessions"])
    if len(manifest_runs) != planned:
        raise ReportError(
            f"verified manifest has {len(manifest_runs)} sessions, expected {planned}"
        )

    # Session is the replication unit.  A session may produce more than one
    # evidence record, but the same (role, metric) lane must not appear twice
    # inside one session because that would ambiguously overweight it.
    lane_sessions: dict[tuple[str, str], list[tuple[int, tuple[float, ...]]]] = defaultdict(list)
    run_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    all_record_eligible = True
    for run in sorted(manifest_runs, key=lambda row: int(row["session"])):
        session = int(run["session"])
        digests = [str(value) for value in run.get("record_digests", [])]
        if not digests:
            raise ReportError(f"session {session} has no harvested evidence record")
        seen_lane: set[tuple[str, str]] = set()
        eligible: list[bool] = []
        for digest in digests:
            record = records[digest]
            eligible.append(record.get("eligible_for_validated_state") is True)
            verdicts = record.get("contract_verdicts")
            if isinstance(verdicts, dict):
                for contract_id, verdict in sorted(verdicts.items()):
                    contract_rows.append(
                        {
                            "session": session,
                            "record_digest": digest,
                            "contract_id": str(contract_id),
                            "passed": bool(
                                isinstance(verdict, dict)
                                and verdict.get("passed") is True
                            ),
                            "status": (
                                verdict.get("status")
                                if isinstance(verdict, dict)
                                else None
                            ),
                        }
                    )
            lane_effects = record.get("lane_effects")
            if not isinstance(lane_effects, list):
                raise ReportError(f"record {digest} lane_effects is malformed")
            for effect in lane_effects:
                if not isinstance(effect, dict):
                    raise ReportError(f"record {digest} contains malformed lane effect")
                role = str(effect.get("role", ""))
                metric = str(effect.get("metric", ""))
                ratios = effect.get("pair_ratios")
                if not role or not metric or not isinstance(ratios, list):
                    raise ReportError(f"record {digest} lane effect is incomplete")
                key = (role, metric)
                if key in seen_lane:
                    raise ReportError(
                        f"session {session} contains duplicate lane {role}/{metric}"
                    )
                seen_lane.add(key)
                lane_sessions[key].append(
                    (session, tuple(float(value) for value in ratios))
                )
        all_record_eligible = all_record_eligible and all(eligible)
        run_rows.append(
            {
                "run_id": str(run["run_id"]),
                "session": session,
                "attempt": int(run["attempt"]),
                "record_digests": sorted(digests),
                "records_eligible": all(eligible),
            }
        )

    aggregate_lanes: list[dict[str, Any]] = []
    for (role, metric), values in sorted(lane_sessions.items()):
        values.sort(key=lambda row: row[0])
        sessions = [ratios for _, ratios in values]
        aggregate = bootstrap_session_effect(sessions, seed=0)
        aggregate_lanes.append(
            {
                "role": role,
                "metric": metric,
                "session_numbers": [session for session, _ in values],
                "session_count": len(values),
                "complete_for_series": len(values) == planned,
                "aggregate": aggregate,
            }
        )

    contract_summary: list[dict[str, Any]] = []
    for contract_id in sorted({row["contract_id"] for row in contract_rows}):
        rows = [row for row in contract_rows if row["contract_id"] == contract_id]
        contract_summary.append(
            {
                "contract_id": contract_id,
                "verdict_records": len(rows),
                "all_persisted_verdicts_pass": all(row["passed"] for row in rows),
                "statuses": sorted({str(row["status"]) for row in rows}),
            }
        )

    return {
        "schema": "bigcherry.jobs.series-report.v1",
        "series_id": series_id,
        "patch_id": manifest["patch_id"],
        "planned_sessions": planned,
        "harvest_commit": verified.get("commit"),
        "scientific_identity_hash": manifest["scientific_identity_hash"],
        "hardware_cohort_hash": manifest["hardware_cohort_hash"],
        "evidence_record_digests": sorted(records),
        "sessions": run_rows,
        "execution_evidence_complete": len(run_rows) == planned,
        "all_records_individually_eligible": all_record_eligible,
        "contract_verdicts": contract_summary,
        "aggregate_lanes": aggregate_lanes,
        "review_ready": bool(verified.get("verified")) and len(run_rows) == planned,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# BigCherry series {report['series_id']}",
        "",
        f"- Patch: `{report['patch_id']}`",
        f"- Sessions: {len(report['sessions'])}/{report['planned_sessions']}",
        f"- Evidence commit: `{report.get('harvest_commit')}`",
        f"- Scientific identity: `{report['scientific_identity_hash']}`",
        f"- Hardware cohort: `{report['hardware_cohort_hash']}`",
        f"- Review ready: **{'yes' if report['review_ready'] else 'no'}**",
        "",
        "## Session aggregate lanes",
        "",
    ]
    if not report["aggregate_lanes"]:
        lines.append("No performance lane effects were recorded (correctness-only series).")
    for lane in report["aggregate_lanes"]:
        aggregate = lane["aggregate"]
        name = f"{lane['role']}/{lane['metric']}"
        if aggregate is None:
            lines.append(
                f"- `{name}`: not evaluable from {lane['session_count']} session(s)"
            )
        else:
            lines.append(
                f"- `{name}`: {aggregate['geometric_effect_pct']:+.4f}% "
                f"CI95 [{aggregate['ci95_low_pct']:+.4f}, "
                f"{aggregate['ci95_high_pct']:+.4f}]%, "
                f"sessions={aggregate['sessions']}, "
                f"between-session SD={aggregate['between_session_sd_pct']:.4f}%"
            )
    lines.extend(("", "## Persisted contract verdicts", ""))
    if not report["contract_verdicts"]:
        lines.append("No bound contract verdicts were recorded.")
    for row in report["contract_verdicts"]:
        lines.append(
            f"- `{row['contract_id']}`: "
            f"{'all pass' if row['all_persisted_verdicts_pass'] else 'not all pass'} "
            f"({row['verdict_records']} evidence record(s))"
        )
    lines.append("")
    return "\n".join(lines)
