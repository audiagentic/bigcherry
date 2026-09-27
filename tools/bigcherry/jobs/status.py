"""Derived operational projections for humans, agents, monitoring and future UI/API.

The filesystem run/event store remains authoritative.  These projections are
rebuildable and may be deleted at any time without losing job history.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any, Iterable

from .service import JobService


def build_snapshot(
    service: JobService,
    *,
    executors: Iterable[dict[str, object]] = (),
) -> dict[str, Any]:
    events = service.store.read_events()
    jobs = [dict(value) for value in service.list_status()]
    jobs.sort(key=lambda row: str(row["run_id"]))
    state_counts = Counter(str(row["state"]) for row in jobs)
    series = []
    for record in service.store.list_series():
        series_id = str(record["series_id"])
        review = service.review(series_id)
        series.append(
            {
                "series_id": series_id,
                "planned_sessions": int(review["planned_sessions"]),
                "completed_sessions": int(review["completed_sessions"]),
                "execution_complete": bool(review["execution_complete"]),
                "review_ready": bool(review["review_ready"]),
                "evidence_commit": review.get("evidence_commit"),
            }
        )
    series.sort(key=lambda row: row["series_id"])
    return {
        "schema": "bigcherry.jobs.status.v1",
        "generated_ns": time.time_ns(),
        "last_event_seq": events[-1].seq if events else 0,
        "paused": bool(service.store.service_control().get("paused")),
        "queue": {
            "pending_receipts": len(service.store.pending()),
            "processing_receipts": len(service.store.processing()),
        },
        "state_counts": dict(sorted(state_counts.items())),
        "jobs": jobs,
        "series": series,
        "executors": sorted(
            (dict(value) for value in executors),
            key=lambda row: str(row.get("executor_id", "")),
        ),
    }


def render_prometheus(snapshot: dict[str, Any]) -> str:
    lines = [
        "# HELP bigcherry_jobs_paused Whether BigCherry job admission is paused.",
        "# TYPE bigcherry_jobs_paused gauge",
        f"bigcherry_jobs_paused {1 if snapshot['paused'] else 0}",
        "# HELP bigcherry_jobs_last_event_seq Last durable BigCherry event sequence.",
        "# TYPE bigcherry_jobs_last_event_seq gauge",
        f"bigcherry_jobs_last_event_seq {int(snapshot['last_event_seq'])}",
        "# HELP bigcherry_jobs_queue_receipts Durable inbox receipts by state.",
        "# TYPE bigcherry_jobs_queue_receipts gauge",
        f'bigcherry_jobs_queue_receipts{{state="pending"}} {int(snapshot["queue"]["pending_receipts"])}',
        f'bigcherry_jobs_queue_receipts{{state="processing"}} {int(snapshot["queue"]["processing_receipts"])}',
        "# HELP bigcherry_jobs_runs Runs by normalized executor/domain state.",
        "# TYPE bigcherry_jobs_runs gauge",
    ]
    for state, count in sorted(snapshot["state_counts"].items()):
        safe = str(state).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'bigcherry_jobs_runs{{state="{safe}"}} {int(count)}')
    ready = sum(1 for row in snapshot["series"] if row["review_ready"])
    lines.extend(
        (
            "# HELP bigcherry_jobs_series_review_ready Series with committed verified evidence ready for review.",
            "# TYPE bigcherry_jobs_series_review_ready gauge",
            f"bigcherry_jobs_series_review_ready {ready}",
        )
    )
    return "\n".join(lines) + "\n"


def render_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        "# BigCherry jobs status",
        "",
        f"- Paused: **{'yes' if snapshot['paused'] else 'no'}**",
        f"- Last event: `{snapshot['last_event_seq']}`",
        f"- Pending receipts: {snapshot['queue']['pending_receipts']}",
        f"- Processing receipts: {snapshot['queue']['processing_receipts']}",
        "",
        "## Runs",
        "",
        "| Run | State | Attempt | Executor | Reason |",
        "|---|---|---:|---|---|",
    ]
    for row in snapshot["jobs"]:
        lines.append(
            f"| `{row['run_id']}` | {row['state']} | {row.get('attempt') or ''} | "
            f"{row.get('executor') or ''} | {row.get('reason') or ''} |"
        )
    lines.extend(("", "## Series", "", "| Series | Sessions | Execution | Review |", "|---|---:|---|---|"))
    for row in snapshot["series"]:
        lines.append(
            f"| `{row['series_id']}` | {row['completed_sessions']}/{row['planned_sessions']} | "
            f"{'complete' if row['execution_complete'] else 'incomplete'} | "
            f"{'ready' if row['review_ready'] else 'not ready'} |"
        )
    lines.append("")
    return "\n".join(lines)


def canonical_json(snapshot: dict[str, Any]) -> bytes:
    return (
        json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
