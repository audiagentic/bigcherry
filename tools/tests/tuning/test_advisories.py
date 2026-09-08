"""Contract tests for tools/bigcherry/tuning/advisories.py.

Focused on EXECUTION_AUDIT_MISSING (HI171) and render()'s severity-sorted
output (commit 84ee106c): this is a STOP-severity gate meant to physically
block trusting a cache before another large tuning campaign, and it had no
test coverage at all before this file existed -- a future edit to the
threshold condition or the severity string could regress it silently.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import advisories as adv


class ExecutionAuditMissingAdvisoryTests(unittest.TestCase):
    def test_fires_when_promoted_and_no_audit_path(self):
        out = adv.advisories_for_campaign(promoted_count=3, execution_audit_path=None)
        tags = [a.tag for a in out]
        self.assertIn("EXECUTION_AUDIT_MISSING", tags)
        row = next(a for a in out if a.tag == "EXECUTION_AUDIT_MISSING")
        self.assertEqual(row.severity, "stop")

    def test_silent_when_audit_path_supplied(self):
        out = adv.advisories_for_campaign(promoted_count=3, execution_audit_path="somewhere.jsonl")
        tags = [a.tag for a in out]
        self.assertNotIn("EXECUTION_AUDIT_MISSING", tags)

    def test_silent_when_nothing_promoted(self):
        out = adv.advisories_for_campaign(promoted_count=0, execution_audit_path=None)
        tags = [a.tag for a in out]
        self.assertNotIn("EXECUTION_AUDIT_MISSING", tags)

    def test_silent_when_promoted_count_is_none(self):
        out = adv.advisories_for_campaign(promoted_count=None, execution_audit_path=None)
        tags = [a.tag for a in out]
        self.assertNotIn("EXECUTION_AUDIT_MISSING", tags)


class RenderSeverityOrderingTests(unittest.TestCase):
    def test_stop_severity_sorts_first_and_uses_stop_prefix(self):
        finding = adv.Advisory(tag="SOME_FINDING", headline="an informational note",
                                body=(), severity="finding")
        stop = adv.Advisory(tag="EXECUTION_AUDIT_MISSING", headline="blocking issue",
                             body=(), severity="stop")
        # Deliberately passed in finding-before-stop order -- render() must
        # reorder, not just echo input order.
        text = adv.render([finding, stop])
        stop_idx = text.index("[STOP] blocking issue")
        finding_idx = text.index("[SOME_FINDING] an informational note")
        self.assertLess(stop_idx, finding_idx)

    def test_empty_advisories_render_to_empty_string(self):
        self.assertEqual(adv.render([]), "")

    def test_non_stop_severity_keeps_its_own_tag_as_prefix(self):
        finding = adv.Advisory(tag="SOME_FINDING", headline="an informational note",
                                body=(), severity="finding")
        text = adv.render([finding])
        self.assertIn("[SOME_FINDING]", text)
        self.assertNotIn("[STOP]", text)


if __name__ == "__main__":
    unittest.main()
