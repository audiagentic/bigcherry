"""HI80: offline tests for e2e_smoke_campaign.Campaign.s3b_correctness_evidence's
pure-Python row-selection logic -- no subprocess, no real dispatch_db, no
hardware. The paths that actually invoke the hi80 CLI/re-run promotion
(--correctness-binary set AND a genuinely blocked row with a matching
original-measurements entry) are real-hardware territory and out of scope
for this offline suite; see HI80.md."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.e2e_smoke_campaign import Campaign, CampaignError  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class S3bCorrectnessEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.model = self.root / "model.gguf"
        self.manifest = self.root / "manifest.json"
        self.tune_server = self.root / "tune-server"
        self.replay_server = self.root / "replay-server"
        self.model.write_bytes(b"model-v1")
        self.manifest.write_text("{}", encoding="utf-8")
        self.tune_server.write_bytes(b"tune-server")
        self.replay_server.write_bytes(b"replay-server")

        self.measurements = self.root / "measurements.jsonl"
        self.dispatch_db = self.root / "dispatch.sqlite"
        self.promoted = self.root / "promoted.jsonl"

    def _campaign(self, *, correctness_binary: Path | None) -> Campaign:
        return Campaign(
            model=self.model, tune_server=self.tune_server, replay_server=self.replay_server,
            manifest=self.manifest, workdir=self.root / "campaign",
            correctness_binary=correctness_binary,
        )

    def test_skips_when_no_correctness_binary_given(self):
        campaign = self._campaign(correctness_binary=None)
        _write_jsonl(self.promoted, [
            {"kind": "header"},
            {"kind": "result", "dispatch": "aa", "promotion_status": "rejected_no_correctness_evidence"},
        ])
        result = campaign.s3b_correctness_evidence(self.measurements, self.dispatch_db, self.promoted)
        self.assertEqual(result, self.promoted)

    def test_skips_when_no_row_is_blocked_purely_on_evidence(self):
        binary = self.root / "test-backend-ops"
        binary.write_bytes(b"binary")
        campaign = self._campaign(correctness_binary=binary)
        _write_jsonl(self.promoted, [
            {"kind": "header"},
            {"kind": "result", "dispatch": "aa", "promotion_status": "promoted"},
            {"kind": "result", "dispatch": "bb", "promotion_status": "rejected_bh"},
        ])
        result = campaign.s3b_correctness_evidence(self.measurements, self.dispatch_db, self.promoted)
        self.assertEqual(result, self.promoted)

    def test_raises_when_blocked_dispatch_missing_from_original_measurements(self):
        binary = self.root / "test-backend-ops"
        binary.write_bytes(b"binary")
        campaign = self._campaign(correctness_binary=binary)
        _write_jsonl(self.promoted, [
            {"kind": "header"},
            {"kind": "result", "dispatch": "aa", "promotion_status": "rejected_no_correctness_evidence"},
        ])
        _write_jsonl(self.measurements, [
            {"kind": "header"},
            {"kind": "result", "dispatch": "bb", "promotion_status": "pending_bh"},
        ])
        with self.assertRaisesRegex(CampaignError, "could not locate the blocked dispatch rows"):
            campaign.s3b_correctness_evidence(self.measurements, self.dispatch_db, self.promoted)


if __name__ == "__main__":
    unittest.main()
