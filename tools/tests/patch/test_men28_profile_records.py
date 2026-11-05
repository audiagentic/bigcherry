"""MEN28: committed profile records are verifiable and survive an unrelated pin bump."""
from __future__ import annotations

import unittest
from pathlib import Path

from bigcherry.patch import catalog, evidence, profile_evidence
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
                    (),
                )

    def test_original_pin_proof_carries_forward_but_still_fails_for_changed_code(self):
        for pid in RECORDED:
            with self.subTest(pid=pid):
                record = profile_evidence.load(PATCHES_ROOT / pid)
                def evaluate(patch_id, ref, revision):
                    reasons = profile_evidence.verify(
                        PATCHES_ROOT / patch_id, patch_id=patch_id, pinned_ref=ref,
                        subject_digest=record.subject_digest,
                    )
                    return evidence.EvidenceCheck(
                        "profile-evidence" if reasons == () else "missing-or-stale"
                    )
                result = catalog._carried_forward(
                    pid, evidence.EvidenceCheck("missing-or-stale"), "future-pin",
                    None, None, evaluate, engine="llamacpp", patches_root=PATCHES_ROOT,
                )
                self.assertEqual(result.status, "carried-forward")
                stale = profile_evidence.verify(
                    PATCHES_ROOT / pid, patch_id=pid, pinned_ref=record.pinned,
                    subject_digest="f" * 64,
                )
                self.assertNotEqual(stale, ())


if __name__ == "__main__":
    unittest.main()
