"""MEN28: committed profile records are verifiable and survive pin bumps and patch edits."""
from __future__ import annotations

import unittest
from pathlib import Path

from bigcherry.patch import catalog, profile_evidence
from bigcherry.patch.evidence import patch_validation_subject_digest

PATCHES_ROOT = Path(__file__).resolve().parents[3] / "engines/llamacpp/patches"
RECORDED = (
    "1297_draft_vocab_trim",
    "1303_attn_kv_tensor_split",
    "1315_mtp_draft_trace",
    "1316_node_hash_trace",
    "1317_spec_round_timing",
    "1318_mtp_draft_timing",
    "1322_mtp_ahead_overlap",
    "1329_alloc_top_trace",
    "1331_alloc_peak_live",
)


class TestHistoricalPromotionEvidence(unittest.TestCase):
    def test_verified_original_pins_and_subjects(self):
        for pid in RECORDED:
            with self.subTest(pid=pid):
                package = PATCHES_ROOT / pid
                record = profile_evidence.load(package)
                self.assertIsNotNone(record)
                self.assertEqual(profile_evidence.problems(record), ())
                self.assertEqual(record.patch_id, pid)
                self.assertIn(record.pinned, ("0504396", "b11474"))
                self.assertFalse(record.default_on)
                self.assertEqual(record.subject_digest,
                                 patch_validation_subject_digest(package / "patch.py"))
                self.assertEqual(
                    profile_evidence.verify(package, patch_id=pid,
                        pinned_ref=record.pinned, subject_digest=record.subject_digest),
                    profile_evidence.Verdict((), ()),
                )

    def test_a_later_pin_or_changed_code_keeps_the_record_qualifying_with_notes(self):
        for pid in RECORDED:
            with self.subTest(pid=pid):
                record = profile_evidence.load(PATCHES_ROOT / pid)
                verdict = profile_evidence.verify(
                    PATCHES_ROOT / pid, patch_id=pid, pinned_ref="future-pin", subject_digest="f" * 64,
                )
                self.assertEqual(verdict.problems, ())
                self.assertEqual(verdict.notes, (
                    f"evidence recorded at pin {record.pinned} (now future-pin)",
                    f"evidence recorded at patch digest {record.subject_digest[:12]} (now ffffffffffff)",
                ))

    def test_edited_promoted_patches_still_qualify_at_the_live_pin(self):
        # 1355 (profile record) and 1274 (campaign record) were edited after
        # their evidence was taken; that is provenance, not a rerun trigger.
        statuses = catalog.validation_evidence_statuses(
            "llamacpp", ["1274_mmvq_kquant_f32_decode", "1355_hc_post_gate_fuse"],
        )
        for pid, check in statuses.items():
            with self.subTest(pid=pid):
                self.assertTrue(check.ok, check.problems)
                self.assertTrue(any("evidence recorded at patch digest" in note for note in check.notes),
                                check.notes)


if __name__ == "__main__":
    unittest.main()
