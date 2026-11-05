"""The checked-in main ruleset names the required checks pin-release waits for; they must be real PR CI jobs."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _required() -> list[str]:
    document = json.loads((ROOT / ".github" / "rulesets" / "main.json").read_text(encoding="utf-8"))
    return [row["context"] for rule in document["rules"] if rule.get("type") == "required_status_checks"
            for row in rule["parameters"]["required_status_checks"]]


def _jobs(workflow: str) -> set[str]:
    text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    return set(re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", text, re.MULTILINE))


class RulesetRequiredChecks(unittest.TestCase):
    def test_every_required_check_is_a_top_level_pr_job(self):
        jobs = _jobs("pr-ci.yml") | _jobs("pr-title.yml")
        required = _required()
        self.assertTrue(required)
        for name in required:
            self.assertIn(name, jobs, f"required check {name!r} is not a job of pr-ci.yml or pr-title.yml")

    def test_the_aggregate_gate_is_required(self):
        # ci-pass is the one job that summarises every PR CI area; a ruleset without it lets pin-release merge a PR
        # GitHub then refuses ("2 of 2 required status checks have not succeeded"), as happened on the v1.17.0 release.
        self.assertIn("ci-pass", _required())


if __name__ == "__main__":
    unittest.main()
