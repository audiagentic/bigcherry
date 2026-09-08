"""Contract tests for the per-promoted-key execution audit.

Purpose: an exact replay cache hit proves compatibility, not tuned
execution. These tests assert the classifier reaches the right conclusion
from the SAME artifact shapes the real tuning pipeline emits, and that a
missing hit log does not silently manufacture a positive finding.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import execution_audit as ea


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


PROMOTED_ROW = {
    "kind": "result",
    "dispatch": "dispatch-aaa",
    "signature": "sig-aaa",
    "native": "mmvq:native:v1",
    "winner": "mmvq:q8_0:w4:nw8:rpb1:sk0:v1",
    "promotion_status": "promoted",
    "improvement_pct": 22.89,
}


class ExecutionAuditClassificationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _promoted(self, rows):
        p = self.dir / "promoted.jsonl"
        _write_jsonl(p, rows)
        return p

    def _hits(self, rows):
        p = self.dir / "hits.jsonl"
        _write_jsonl(p, rows)
        return p

    def test_native_only_rows_are_excluded(self):
        rows = [dict(PROMOTED_ROW, promotion_status="native", winner="mmvq:native:v1")]
        audit = ea.build_audit(promoted_path=self._promoted(rows))
        self.assertEqual(audit, [])

    def test_rejected_rows_are_excluded(self):
        rows = [dict(PROMOTED_ROW, promotion_status="rejected_correctness")]
        audit = ea.build_audit(promoted_path=self._promoted(rows))
        self.assertEqual(audit, [])

    def test_no_hit_log_is_not_executed(self):
        audit = ea.build_audit(promoted_path=self._promoted([PROMOTED_ROW]))
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0].classification, ea.Classification.NOT_EXECUTED)
        self.assertIsNone(audit[0].actual_launched_candidate)
        self.assertEqual(audit[0].launch_count, 0)

    def test_hit_confirms_winner_is_different_faster_by_default(self):
        hits = [{"dispatch": "dispatch-aaa", "signature": "sig-aaa",
                 "candidate": "mmvq:q8_0:w4:nw8:rpb1:sk0:v1", "calls": 42}]
        audit = ea.build_audit(
            promoted_path=self._promoted([PROMOTED_ROW]),
            hit_log_path=self._hits(hits),
        )
        row = audit[0]
        self.assertEqual(row.classification, ea.Classification.DIFFERENT_FASTER)
        self.assertEqual(row.actual_launched_candidate, "mmvq:q8_0:w4:nw8:rpb1:sk0:v1")
        self.assertEqual(row.launch_count, 42)
        self.assertEqual(row.e2e_verdict, "not_measured")

    def test_hit_log_candidate_mismatch_is_fallback(self):
        # The resolver revalidated the exact hit and substituted something
        # else at launch time -- this is the HI160 scenario exactly.
        hits = [{"dispatch": "dispatch-aaa", "signature": "sig-aaa",
                 "candidate": "mmvq:native:v1", "calls": 7}]
        audit = ea.build_audit(
            promoted_path=self._promoted([PROMOTED_ROW]),
            hit_log_path=self._hits(hits),
        )
        self.assertEqual(audit[0].classification, ea.Classification.FALLBACK)

    def test_winner_equal_to_native_is_same_native_regardless_of_hits(self):
        row = dict(PROMOTED_ROW, winner="mmvq:native:v1")
        hits = [{"dispatch": "dispatch-aaa", "signature": "sig-aaa",
                 "candidate": "mmvq:native:v1", "calls": 5}]
        audit = ea.build_audit(
            promoted_path=self._promoted([row]),
            hit_log_path=self._hits(hits),
        )
        self.assertEqual(audit[0].classification, ea.Classification.SAME_NATIVE)

    def test_e2e_regressed_verdict_overrides_isolated_win(self):
        hits = [{"dispatch": "dispatch-aaa", "signature": "sig-aaa",
                 "candidate": "mmvq:q8_0:w4:nw8:rpb1:sk0:v1", "calls": 42}]
        e2e_path = self.dir / "e2e.json"
        e2e_path.write_text(json.dumps({"dispatch-aaa": "regressed"}), encoding="utf-8")
        audit = ea.build_audit(
            promoted_path=self._promoted([PROMOTED_ROW]),
            hit_log_path=self._hits(hits),
            e2e_verdicts_path=e2e_path,
        )
        row = audit[0]
        self.assertEqual(row.classification, ea.Classification.DIFFERENT_SLOWER)
        self.assertEqual(row.e2e_verdict, "regressed")

    def test_e2e_verdict_rejects_unknown_value(self):
        e2e_path = self.dir / "e2e.json"
        e2e_path.write_text(json.dumps({"dispatch-aaa": "amazing"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            ea.build_audit(
                promoted_path=self._promoted([PROMOTED_ROW]),
                e2e_verdicts_path=e2e_path,
            )

    def test_header_row_is_skipped(self):
        rows = [{"kind": "header", "manifest_hash": "x"}, PROMOTED_ROW]
        audit = ea.build_audit(promoted_path=self._promoted(rows))
        self.assertEqual(len(audit), 1)

    def test_isolated_improvement_pct_is_carried_through_unmodified(self):
        audit = ea.build_audit(promoted_path=self._promoted([PROMOTED_ROW]))
        self.assertEqual(audit[0].isolated_improvement_pct, 22.89)

    def test_missing_hit_log_path_is_not_an_error(self):
        audit = ea.build_audit(
            promoted_path=self._promoted([PROMOTED_ROW]),
            hit_log_path=str(self.dir / "does-not-exist.jsonl"),
        )
        self.assertEqual(audit[0].classification, ea.Classification.NOT_EXECUTED)


class ExecutionAuditSummaryTests(unittest.TestCase):
    def test_unproven_fraction_counts_not_executed_and_fallback_only(self):
        rows = [
            ea.AuditRow("d1", "s1", "native", "native", None, 0,
                        ea.Classification.SAME_NATIVE, 5.0, "not_measured", None),
            ea.AuditRow("d2", "s2", "native", "winner", None, 0,
                        ea.Classification.NOT_EXECUTED, 5.0, "not_measured", None),
            ea.AuditRow("d3", "s3", "native", "winner", "native", 3,
                        ea.Classification.FALLBACK, 5.0, "not_measured", None),
            ea.AuditRow("d4", "s4", "native", "winner", "winner", 3,
                        ea.Classification.DIFFERENT_FASTER, 5.0, "not_measured", None),
        ]
        summary = ea.summarize(rows)
        self.assertEqual(summary.total, 4)
        self.assertAlmostEqual(summary.unproven_fraction, 2 / 4)

    def test_empty_input_has_zero_unproven_fraction(self):
        summary = ea.summarize([])
        self.assertEqual(summary.total, 0)
        self.assertEqual(summary.unproven_fraction, 0.0)


class ExecutionAuditWriterTests(unittest.TestCase):
    def test_write_audit_round_trips_as_jsonl(self):
        row = ea.AuditRow("d1", "s1", "mmvq:native:v1", "mmvq:q8_0:w4:v1",
                          "mmvq:q8_0:w4:v1", 10, ea.Classification.DIFFERENT_FASTER,
                          12.5, "not_measured", None)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "audit.jsonl"
            ea.write_audit([row], out)
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["classification"], "DIFFERENT_FASTER")
            self.assertEqual(parsed["launch_count"], 10)


class AuditCoversPromotedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _promoted(self, rows):
        p = self.dir / "promoted.jsonl"
        _write_jsonl(p, rows)
        return p

    def test_missing_audit_path_does_not_cover(self):
        promoted = self._promoted([PROMOTED_ROW])
        self.assertFalse(ea.audit_covers_promoted(promoted, None))

    def test_nonexistent_audit_file_does_not_cover(self):
        promoted = self._promoted([PROMOTED_ROW])
        self.assertFalse(ea.audit_covers_promoted(promoted, self.dir / "does-not-exist.jsonl"))

    def test_stale_audit_missing_current_dispatch_does_not_cover(self):
        # The audit exists but covers a DIFFERENT promoted set -- e.g. left
        # over from an earlier run that reused this workdir/run_id.
        promoted = self._promoted([PROMOTED_ROW])
        audit = self.dir / "audit.jsonl"
        _write_jsonl(audit, [{"dispatch": "some-other-dispatch", "classification": "DIFFERENT_FASTER"}])
        self.assertFalse(ea.audit_covers_promoted(promoted, audit))

    def test_audit_covering_every_promoted_dispatch_does_cover(self):
        promoted = self._promoted([PROMOTED_ROW])
        audit = self.dir / "audit.jsonl"
        _write_jsonl(audit, [{"dispatch": "dispatch-aaa", "classification": "DIFFERENT_FASTER"}])
        self.assertTrue(ea.audit_covers_promoted(promoted, audit))

    def test_audit_covering_a_superset_still_covers(self):
        promoted = self._promoted([PROMOTED_ROW])
        audit = self.dir / "audit.jsonl"
        _write_jsonl(audit, [
            {"dispatch": "dispatch-aaa", "classification": "DIFFERENT_FASTER"},
            {"dispatch": "dispatch-from-a-prior-run", "classification": "NOT_EXECUTED"},
        ])
        self.assertTrue(ea.audit_covers_promoted(promoted, audit))

    def test_no_promoted_keys_trivially_covered(self):
        promoted = self._promoted([])
        self.assertTrue(ea.audit_covers_promoted(promoted, self.dir / "does-not-exist.jsonl"))

    def test_malformed_audit_file_does_not_cover(self):
        promoted = self._promoted([PROMOTED_ROW])
        audit = self.dir / "audit.jsonl"
        audit.write_text("not json\n", encoding="utf-8")
        self.assertFalse(ea.audit_covers_promoted(promoted, audit))


if __name__ == "__main__":
    unittest.main()
