from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bigcherry.jobs.status import build_snapshot, render_markdown, render_prometheus
from bigcherry.jobs.store import RunStore


class _Service:
    def __init__(self, store: RunStore):
        self.store = store

    def list_status(self):
        return (
            {
                "run_id": "r-1",
                "series_id": "s-1",
                "attempt": 1,
                "executor": "fake",
                "native_id": "1",
                "state": "running",
                "native_state": "RUNNING",
                "reason": None,
            },
        )

    def review(self, series_id: str):
        return {
            "series_id": series_id,
            "planned_sessions": 4,
            "completed_sessions": 3,
            "execution_complete": False,
            "review_ready": False,
            "evidence_commit": None,
        }


class StatusProjectionTests(unittest.TestCase):
    def test_snapshot_metrics_and_markdown_are_derived(self):
        with tempfile.TemporaryDirectory() as temp:
            store = RunStore(Path(temp) / "jobs")
            store.create_series(
                "s-1",
                {
                    "series_id": "s-1",
                    "series_material": {"planned_sessions": 4},
                },
            )
            store.append_event(kind="job.submitted", run_id="r-1")
            snapshot = build_snapshot(
                _Service(store),
                executors=({"executor_id": "fake", "kind": "fake"},),
            )
            self.assertEqual(snapshot["last_event_seq"], 1)
            self.assertEqual(snapshot["state_counts"], {"running": 1})
            self.assertEqual(snapshot["series"][0]["completed_sessions"], 3)
            metrics = render_prometheus(snapshot)
            self.assertIn('bigcherry_jobs_runs{state="running"} 1', metrics)
            self.assertIn("bigcherry_jobs_series_review_ready 0", metrics)
            markdown = render_markdown(snapshot)
            self.assertIn("r-1", markdown)
            self.assertIn("3/4", markdown)


if __name__ == "__main__":
    unittest.main()
