"""RSC11 B: a campaign job file describes its radiance arms as serving configurations, never as raw engine flags."""
import json
import unittest
from pathlib import Path

from bigcherry.core import paths
from bigcherry.engine_adapters import radiance_layout

JOBS = sorted((paths.REPO_ROOT / "tools/lab/radiance").glob("*.json"))
#: the acceptance speed job, whose arms differ only by a knob the engine no longer has; the owner decides what it compares now
NOT_YET = {"helper-ab.json"}


class JobFileTests(unittest.TestCase):
    def test_server_jobs_name_a_known_layout_and_no_raw_flags(self):
        layouts = set(radiance_layout.names())
        seen = 0
        for job in JOBS:
            if job.name in NOT_YET:
                continue
            doc = json.loads(job.read_text(encoding="utf-8"))
            with self.subTest(job=job.name):
                self.assertEqual(doc["server_args"], [])
                for arm in doc["arms"]:
                    self.assertNotIn("server_args", arm)
                    self.assertIn(arm["serving"]["layout"], layouts)
                    self.assertFalse([k for k in arm.get("settings", {}) if k.startswith("R3_MOE_WORKER")])
                    self.assertEqual(doc["execution_evidence"], "observe")
                    seen += 1
        self.assertGreaterEqual(seen, 8)

    def test_a_layout_with_helpers_names_its_heat_inputs(self):
        for job in JOBS:
            if job.name in NOT_YET:
                continue
            for arm in json.loads(job.read_text(encoding="utf-8"))["arms"]:
                serving = arm["serving"]
                with self.subTest(job=job.name, arm=arm["name"]):
                    self.assertEqual("heat" in serving, "histogram" in serving)
