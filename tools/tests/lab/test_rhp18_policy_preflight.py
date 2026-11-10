"""Card-free RHP18 launcher manifest policy-state preflight."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "tools/lab/radiance/lib/plan-check.sh"


def manifest(*, on: bool = True, period: int = 1, provenance: bool = True) -> dict:
    placement = {}
    if on:
        placement["live_policy"] = {
            "cadence_ns": 5_000_000_000,
            "max_heat_age_ns": 10_000_000_000,
            "margin_ppm": 54_000,
            "min_samples": 89,
            "effect_gate": False,
        }
    m = {"spec": {"handoff": {
        "placement": placement, "telemetry": {"period_s": period},
    }}}
    if provenance:
        m["provenance"] = {"/spec/handoff/placement":
                           {"source": "live policy disabled: no calibration record for key=abc"}}
    return m


class LivePlacementManifestPreflight(unittest.TestCase):
    def check(self, data: dict, *, helpers: int = 3, expected: str = "") -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manifest.json"
            path.write_text(json.dumps(data))
            env = dict(os.environ, PLAN_MANIFEST=str(path),
                       LAYOUT_HELPER_COUNT=str(helpers))
            if expected:
                env["BC_EXPECT_LIVE_POLICY"] = expected
            else:
                env.pop("BC_EXPECT_LIVE_POLICY", None)
            return subprocess.run(
                ["bash", "-c", 'source "$1"; plan_live_placement_report', "bash", str(LIB)],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=5)

    def test_on_reports_cadence_and_accepts_expectation(self):
        r = self.check(manifest(), expected="on")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("live placement: ON", r.stderr)
        self.assertIn("heat period 1 s", r.stderr)

    def test_off_with_provenance_is_explicit(self):
        r = self.check(manifest(on=False), expected="off")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("OFF -- no calibration record for key=abc", r.stderr)

    def test_mismatched_expectation_fails(self):
        r = self.check(manifest(on=False), expected="on")
        self.assertEqual(r.returncode, 2)
        self.assertIn("expected on, manifest says off", r.stderr)

    def test_bad_expected_value_fails(self):
        r = self.check(manifest(), expected="maybe")
        self.assertEqual(r.returncode, 2)
        self.assertIn("invalid BC_EXPECT_LIVE_POLICY", r.stderr)

    def test_missing_reason_fails_closed(self):
        r = self.check(manifest(on=False, provenance=False))
        self.assertEqual(r.returncode, 2)
        self.assertIn("policy absent without a manifest reason", r.stderr)

    def test_heat_feed_slower_than_cadence_fails_closed(self):
        r = self.check(manifest(period=6))
        self.assertEqual(r.returncode, 2)
        self.assertIn("histogram period 6s exceeds policy cadence", r.stderr)

    def test_bad_manifest_fails_closed(self):
        r = self.check({"not_spec": {}})
        self.assertEqual(r.returncode, 2)
        self.assertIn("invalid manifest", r.stderr)

    def test_engine_only_explicitly_not_applicable(self):
        r = self.check({"spec": {}}, helpers=0, expected="not_applicable")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NOT_APPLICABLE", r.stderr)


if __name__ == "__main__":
    unittest.main()
