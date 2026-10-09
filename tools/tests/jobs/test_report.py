from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bigcherry.jobs.report import ReportError, build_series_report, render_markdown
from bigcherry.jobs.store import RunStore
from bigcherry.patch import evidence as patch_evidence


class SeriesReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        (self.repo / "engines" / "llamacpp" / "patches" / "p").mkdir(parents=True)
        (self.repo / "engines" / "llamacpp" / "patches" / "p" / "patch.py").write_text("STATE='untested'\n")
        self.store = RunStore(self.root / "jobs")
        self.series_id = "s-report"
        self.store.create_series(
            self.series_id,
            {
                "series_id": self.series_id,
                "series_material": {
                    "planned_sessions": 4,
                    "scientific_identity_hash": "science",
                    "hardware_cohort_hash": "hardware",
                },
            },
        )
        rows = []
        run_manifest = []
        for session, effect in enumerate((1.01, 1.02, 1.015, 1.025), start=1):
            record = {
                "patch_id": "p",
                "campaign_identity_digest": f"{session:064x}",
                "patch_implementation_digest": "a" * 64,
                "gpu_architectures": ["gfx1100"],
                "eligible_for_validated_state": True,
                "contract_verdicts": {
                    "C": {"passed": True, "status": "pass", "detail": {}}
                },
                "lane_effects": [
                    {
                        "role": "positive",
                        "metric": "pp512",
                        "pair_ratios": [effect] * 10,
                    },
                    {
                        "role": "control",
                        "metric": "tg128",
                        "pair_ratios": [1.0] * 10,
                    },
                ],
            }
            record["record_digest"] = patch_evidence._record_digest(record)  # type: ignore[attr-defined]
            rows.append(record)
            run_manifest.append(
                {
                    "run_id": f"r-{session}",
                    "session": session,
                    "attempt": 1,
                    "record_digests": [record["record_digest"]],
                }
            )
        for record in rows:
            patch_evidence.write_record(record, root=self.repo / "engines" / "llamacpp" / "patches")
        verified = {
            "schema": "bigcherry.jobs.harvest-result.v1",
            "series_id": self.series_id,
            "verified": True,
            "commit": "f" * 40,
            "manifest": {
                "schema": "bigcherry.jobs.verified-evidence.v1",
                "series_id": self.series_id,
                "patch_id": "p",
                "scientific_identity_hash": "science",
                "hardware_cohort_hash": "hardware",
                "destination": "engines/llamacpp/patches/p.json",
                "runs": run_manifest,
                "record_digests": [row["record_digest"] for row in rows],
            },
        }
        path = self.store.root / "series" / self.series_id / "verified-evidence.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(verified), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_report_uses_existing_session_bootstrap(self):
        report = build_series_report(
            store=self.store, series_id=self.series_id, project_root=self.repo
        )
        self.assertTrue(report["review_ready"])
        self.assertEqual(len(report["aggregate_lanes"]), 2)
        positive = next(
            lane for lane in report["aggregate_lanes"] if lane["role"] == "positive"
        )
        self.assertEqual(positive["aggregate"]["sessions"], 4)
        self.assertGreater(positive["aggregate"]["ci95_low_pct"], 0.0)
        self.assertTrue(report["contract_verdicts"][0]["all_persisted_verdicts_pass"])
        markdown = render_markdown(report)
        self.assertIn("Session aggregate lanes", markdown)
        self.assertIn("CI95", markdown)

    def test_report_requires_verified_harvest(self):
        (self.store.root / "series" / self.series_id / "verified-evidence.json").unlink()
        with self.assertRaises(ReportError):
            build_series_report(
                store=self.store, series_id=self.series_id, project_root=self.repo
            )


if __name__ == "__main__":
    unittest.main()
