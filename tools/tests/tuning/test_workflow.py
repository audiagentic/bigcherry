"""HI130: pure-Python helpers in tuning/workflow.py -- the parts testable
without a real GPU/build (the full run_tune_campaign() orchestration needs
real hardware and is validated live on Brutus, not here).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import workflow  # noqa: E402

_EVIDENCE_DIR = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "2026-08-27-dual-xtx-tune-run"
_REAL_PROMOTED = _EVIDENCE_DIR / "e2e_dualxtx_tune.promoted2.jsonl"


class CountMissingCorrectnessEvidenceTests(unittest.TestCase):
    def test_real_evidence_file_has_exactly_the_one_known_unsupported_row_missing(self):
        # This file is the SECOND promote() pass, run after correctness
        # evidence generation for every SUPPORTED candidate. Exactly one of
        # the 39 provisional winners was honestly skipped as an unsupported
        # signature domain (a non-routed/dense GLU fusion -- HI119 step 16,
        # not yet written; see the real session log: "SKIPPED (unsupported
        # signature): ... only the MoE-routed (MUL_MAT_ID-based) fused GLU
        # case is supported this slice") -- that row legitimately still has
        # no evidence and must still count as missing, not be silently
        # treated as resolved.
        if not _REAL_PROMOTED.is_file():
            self.skipTest("real evidence file not present in this checkout")
        count = workflow._count_missing_correctness_evidence(_REAL_PROMOTED)
        self.assertEqual(count, 1)

    def test_counts_rows_missing_evidence(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "promoted.jsonl"
            rows = [
                {"kind": "header"},
                {"kind": "result", "promotion_status": "promoted"},
                {"kind": "result", "promotion_status": "rejected_no_correctness_evidence"},
                {"kind": "result", "promotion_status": "rejected_no_correctness_evidence"},
                {"kind": "result", "promotion_status": "native"},
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            self.assertEqual(workflow._count_missing_correctness_evidence(path), 2)

    def test_zero_when_nothing_missing(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "promoted.jsonl"
            rows = [
                {"kind": "header"},
                {"kind": "result", "promotion_status": "promoted"},
                {"kind": "result", "promotion_status": "native"},
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            self.assertEqual(workflow._count_missing_correctness_evidence(path), 0)


class StageIdentityTests(unittest.TestCase):
    def test_uses_the_real_lane_run_id_not_a_reconstructed_one(self):
        # gpt review (2026-08-27): _stage_identity used to take a
        # SEPARATE run_id string reconstructed by the caller
        # (f"{campaign_run_id}-record") rather than the real per-lane
        # run_id run_campaign() actually used -- the two are different
        # strings, and only result.run_id matches the ArtifactStore
        # paths/provenance that build actually produced.
        from unittest.mock import MagicMock
        fake_result = MagicMock()
        fake_result.run_id = "real-lane-run-id-from-run-campaign"
        fake_result.source_slice_id = "slice1"
        fake_result.build_plan_id = "plan1"
        fake_result.manifest_ref = None
        fake_result.source_root = Path("/some/source/root")

        identity = workflow._stage_identity(fake_result)
        self.assertEqual(identity.run_id, "real-lane-run-id-from-run-campaign")


if __name__ == "__main__":
    unittest.main()
