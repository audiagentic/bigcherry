"""MEN28: committed profile records are verifiable and survive an unrelated pin bump."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from bigcherry.patch import catalog, evidence, lineage, profile_evidence
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

    def test_reviewed_git_reanchor_has_matching_record_and_report(self):
        for pid in ("1274_mmvq_kquant_f32_decode", "1355_hc_post_gate_fuse"):
            with self.subTest(pid=pid):
                package = PATCHES_ROOT / pid
                transitions = lineage.load(package)
                self.assertEqual(len(transitions), 1)
                entry = transitions[0]
                report_path = (PATCHES_ROOT.parents[2] /
                               f"docs/evidence/men28-llamacpp-evidence/{pid[:4]}-reanchor-audit.json")
                raw = report_path.read_bytes()
                self.assertEqual(entry["rebase_report_sha256"], hashlib.sha256(raw).hexdigest())
                report = json.loads(raw)
                self.assertEqual(report["kind"], "git-reanchor-audit-v1")
                self.assertEqual(entry["from_digest"], report["original_subject_digest"])
                self.assertEqual(entry["to_digest"], report["updated_subject_digest"])
                self.assertEqual(entry["pin"], report["pin_at_reanchor"])
                self.assertEqual(entry["to_digest"],
                                 patch_validation_subject_digest(package / "patch.py"))
                self.assertTrue(lineage.is_ancestor(
                    package, previous=entry["from_digest"], current=entry["to_digest"]))
        package = PATCHES_ROOT / "1355_hc_post_gate_fuse"
        self.assertEqual(profile_evidence.verify(
            package, patch_id="1355_hc_post_gate_fuse", pinned_ref="b11474",
            subject_digest=patch_validation_subject_digest(package / "patch.py"),
            allow_lineage=True), ())
        self.assertNotEqual(profile_evidence.verify(
            package, patch_id="1355_hc_post_gate_fuse", pinned_ref="b11474",
            subject_digest=patch_validation_subject_digest(package / "patch.py")), ())

    def test_original_pin_proof_carries_forward_but_still_fails_for_changed_code(self):
        for pid in RECORDED:
            with self.subTest(pid=pid):
                record = profile_evidence.load(PATCHES_ROOT / pid)
                def evaluate(patch_id, ref, revision, subject_digest=record.subject_digest):
                    reasons = profile_evidence.verify(
                        PATCHES_ROOT / patch_id, patch_id=patch_id, pinned_ref=ref,
                        subject_digest=subject_digest,
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
