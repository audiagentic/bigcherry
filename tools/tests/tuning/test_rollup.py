"""Contract tests for the consolidated multi-campaign tuning rollup.

Purpose: this is a derived, read-only view over multiple campaigns'
promoted.jsonl (+ execution-audit classification, if present) -- never a
binary cache merge (dev-gpt-agent review, 2026-09-08: a merged replay cache
cannot represent GPU0/GPU1 disagreeing on a winner for the same portable key,
since they share one hardware digest). These tests assert rows are never
deduped across campaigns and identity fields are carried through unmodified.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import rollup


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

RECEIPT = {
    "campaign_run_id": "hi168-9b-gpu0-20260906",
    "model_path": "/mnt/vault/llm-models/qwen3.5-9B/gguf/mtp/Qwen3.5-9B-Q6_K.gguf",
    "devices": "0",
    "replay": {
        "source_root": "/home/audumla/.cache/bigcherry/sources/abc",
        "source_slice_id": "def",
        "build_plan_id": "a8cbfbb8d40684fbf657b059c0d3d881",
    },
}


class RollupTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _campaign(self, name: str, *, promoted_rows, receipt=None, audit_rows=None) -> Path:
        campaign_dir = self.dir / name
        campaign_dir.mkdir()
        _write_jsonl(campaign_dir / "promoted.jsonl", promoted_rows)
        (campaign_dir / "tune-campaign-receipt.json").write_text(
            json.dumps(receipt or RECEIPT), encoding="utf-8",
        )
        if audit_rows is not None:
            _write_jsonl(campaign_dir / "hip-tuning-execution-audit.jsonl", audit_rows)
        return campaign_dir

    def test_single_campaign_round_trips_identity_and_promoted_fields(self):
        campaign_dir = self._campaign("c1", promoted_rows=[PROMOTED_ROW])
        rows = rollup.build_rollup([campaign_dir])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.campaign_run_id, "hi168-9b-gpu0-20260906")
        self.assertEqual(row.devices, "0")
        self.assertEqual(row.replay_build_plan_id, "a8cbfbb8d40684fbf657b059c0d3d881")
        self.assertEqual(row.dispatch, "dispatch-aaa")
        self.assertEqual(row.native_candidate, "mmvq:native:v1")
        self.assertEqual(row.replay_candidate, "mmvq:q8_0:w4:nw8:rpb1:sk0:v1")
        self.assertEqual(row.improvement_pct, 22.89)
        self.assertIsNone(row.execution_audit_classification)

    def test_native_and_rejected_rows_are_excluded(self):
        rows_in = [
            dict(PROMOTED_ROW, promotion_status="native", winner="mmvq:native:v1"),
            dict(PROMOTED_ROW, promotion_status="rejected_correctness"),
        ]
        campaign_dir = self._campaign("c1", promoted_rows=rows_in)
        self.assertEqual(rollup.build_rollup([campaign_dir]), [])

    def test_two_campaigns_with_same_dispatch_are_never_deduped(self):
        # GPU0 and GPU1 share hardware digest and can disagree on a winner
        # for the same portable key -- that disagreement is exactly the
        # evidence a future reconciliation decision would need, so both
        # rows must survive independently, not collapse into one.
        gpu0 = self._campaign(
            "gpu0", promoted_rows=[PROMOTED_ROW],
            receipt=dict(RECEIPT, campaign_run_id="gpu0-run", devices="0"),
        )
        gpu1_row = dict(PROMOTED_ROW, winner="mmvq:q8_0:w4:nw4:rpb1:sk0:v1")
        gpu1 = self._campaign(
            "gpu1", promoted_rows=[gpu1_row],
            receipt=dict(RECEIPT, campaign_run_id="gpu1-run", devices="1"),
        )
        rows = rollup.build_rollup([gpu0, gpu1])
        self.assertEqual(len(rows), 2)
        winners = {r.campaign_run_id: r.replay_candidate for r in rows}
        self.assertEqual(winners["gpu0-run"], "mmvq:q8_0:w4:nw8:rpb1:sk0:v1")
        self.assertEqual(winners["gpu1-run"], "mmvq:q8_0:w4:nw4:rpb1:sk0:v1")

    def test_execution_audit_classification_is_joined_when_present(self):
        audit_rows = [{
            "dispatch": "dispatch-aaa",
            "classification": "DIFFERENT_FASTER",
            "actual_launched_candidate": "mmvq:q8_0:w4:nw8:rpb1:sk0:v1",
            "actual_from_cache": True,
        }]
        campaign_dir = self._campaign("c1", promoted_rows=[PROMOTED_ROW], audit_rows=audit_rows)
        rows = rollup.build_rollup([campaign_dir])
        self.assertEqual(rows[0].execution_audit_classification, "DIFFERENT_FASTER")
        self.assertEqual(rows[0].actual_from_cache, True)

    def test_missing_execution_audit_is_not_an_error(self):
        campaign_dir = self._campaign("c1", promoted_rows=[PROMOTED_ROW])
        rows = rollup.build_rollup([campaign_dir])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].execution_audit_classification)

    def test_missing_receipt_raises(self):
        campaign_dir = self.dir / "broken"
        campaign_dir.mkdir()
        _write_jsonl(campaign_dir / "promoted.jsonl", [PROMOTED_ROW])
        with self.assertRaises(OSError):
            rollup.build_rollup([campaign_dir])

    def test_summarize_counts_by_campaign_and_classification(self):
        campaign_dir = self._campaign("c1", promoted_rows=[PROMOTED_ROW])
        rows = rollup.build_rollup([campaign_dir])
        summary = rollup.summarize(rows)
        self.assertEqual(summary.total, 1)
        self.assertEqual(summary.by_campaign["hi168-9b-gpu0-20260906"], 1)
        self.assertEqual(summary.unaudited, 1)

    def test_write_rollup_round_trips_as_jsonl(self):
        campaign_dir = self._campaign("c1", promoted_rows=[PROMOTED_ROW])
        rows = rollup.build_rollup([campaign_dir])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "rollup.jsonl"
            rollup.write_rollup(rows, out)
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["campaign_run_id"], "hi168-9b-gpu0-20260906")
            self.assertEqual(parsed["dispatch"], "dispatch-aaa")


if __name__ == "__main__":
    unittest.main()
