"""MEN28: reviewed digest lineage is build-only; no implicit requalification."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.patch import catalog, evidence, lineage, profile_evidence


class LineageEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.package = self.root / "p"
        self.package.mkdir()
        self.old = "a" * 64
        self.middle = "b" * 64
        self.current = "c" * 64
        self.report = self.root / "rebase-recipe.json"
        self.report.write_text('{"upstream_revision":"123","patches":[]}', encoding="utf-8")
        profile_evidence.write(self.package, profile_evidence.ProfileEvidence(
            patch_id="p", pinned="old", subject_digest=self.old, default_on=False,
            models=("Model X",), checks={"mechanics": "m", "activation": "a", "identity": "i", "ab": "b"},
        ))

    def test_chained_reviewed_reanchors_qualify_only_with_lineage_opt_in(self):
        lineage.write_reanchor(
            self.package, from_digest=self.old, to_digest=self.middle,
            pin="pin-2", rebase_report=self.report,
        )
        lineage.write_reanchor(
            self.package, from_digest=self.middle, to_digest=self.current,
            pin="pin-3", rebase_report=self.report,
        )
        self.assertTrue(lineage.is_ancestor(self.package, previous=self.old, current=self.current))
        self.assertEqual(profile_evidence.verify(
            self.package, patch_id="p", pinned_ref="old",
            subject_digest=self.current, allow_lineage=True), ())
        self.assertIn("implementation changed", " ".join(profile_evidence.verify(
            self.package, patch_id="p", pinned_ref="old", subject_digest=self.current)))
        self.assertFalse(lineage.is_ancestor(self.package, previous="f"*64, current=self.current))

    def test_no_lineage_or_unrelated_lineage_refuses_changed_digest(self):
        self.assertFalse(lineage.is_ancestor(self.package, previous=self.old, current=self.current))
        lineage.write_reanchor(self.package, from_digest="f"*64, to_digest=self.current,
                               pin="pin-3", rebase_report=self.report)
        self.assertNotEqual(profile_evidence.verify(
            self.package, patch_id="p", pinned_ref="old",
            subject_digest=self.current, allow_lineage=True), ())

    def test_malformed_or_cyclic_lineage_fails_closed(self):
        path = self.package / lineage.RECORD
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "transitions": [{
            "from_digest": self.old, "to_digest": self.middle, "pin": "pin",
            "reason": "semantic rewrite", "rebase_report_sha256": "f"*64,
        }]}))
        with self.assertRaisesRegex(ValueError, "re-anchor"):
            lineage.is_ancestor(self.package, previous=self.old, current=self.middle)
        path.write_text(json.dumps({"schema": 1, "transitions": [
            {"from_digest": self.old, "to_digest": self.middle, "pin": "a",
                 "reason": "rebase re-anchor", "rebase_report_sha256": "f"*64},
            {"from_digest": self.middle, "to_digest": self.old, "pin": "b",
                 "reason": "rebase re-anchor", "rebase_report_sha256": "f"*64},
        ]}))
        with self.assertRaisesRegex(ValueError, "cyclic"):
            lineage.load(self.package)

    def test_corrupt_lineage_is_rejected_as_evidence_problem_not_an_evaluator_crash(self):
        path = self.package / lineage.RECORD
        path.parent.mkdir(exist_ok=True)
        path.write_text('{"schema":1,"transitions":"corrupt"}')
        issues = profile_evidence.verify(
            self.package, patch_id="p", pinned_ref="old",
            subject_digest=self.current, allow_lineage=True,
        )
        self.assertTrue(any("invalid patch lineage" in issue for issue in issues))
        self.assertTrue(any("implementation changed" in issue for issue in issues))

    def test_recorded_lineage_never_waives_pin_freshness(self):
        lineage.write_reanchor(self.package, from_digest=self.old,
                               to_digest=self.current, pin="pin-2", rebase_report=self.report)
        problems = profile_evidence.verify(
            self.package, patch_id="p", pinned_ref="new",
            subject_digest=self.current, allow_lineage=True)
        self.assertEqual(len(problems), 1)
        self.assertIn("current pin", problems[0])

    def test_carried_forward_uses_lineage_only_for_historical_evaluation(self):
        lineage.write_reanchor(self.package, from_digest=self.old,
                               to_digest=self.current, pin="pin-2", rebase_report=self.report)
        seen = []
        def evaluate(pid, ref, revision):
            seen.append(ref)
            problems = profile_evidence.verify(
                self.package, patch_id=pid, pinned_ref=ref,
                subject_digest=self.current, allow_lineage=ref != "new",
            )
            return evidence.EvidenceCheck("profile-evidence" if problems == () else "missing-or-stale")
        with mock.patch.object(evidence, "load_records", return_value=()):
            result = catalog._carried_forward(
                "p", evidence.EvidenceCheck("missing-or-stale"), "new", None, None,
                evaluate, engine="llamacpp", patches_root=self.root)
        self.assertEqual(result.status, "carried-forward")
        self.assertEqual(seen, ["old"])


if __name__ == "__main__":
    unittest.main()
