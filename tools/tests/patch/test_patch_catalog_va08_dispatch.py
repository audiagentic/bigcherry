"""VA08: validation_evidence_statuses() status-aware dispatch (GPT session
ses_5bbee8ce5c9a4265, req_65394c5d0fdd4647) -- proves the real dispatch
routes to the right status-obligation verifier using REAL production
patch ids with REAL tracked-status entries in config/external-sources.toml,
not synthetic fixtures. Both patches used here are packaged
(patches/<id>/patch.toml) -- confirms "packaged descriptor follows same
path with no special scanner."
"""

from __future__ import annotations

import sys
import unittest
from unittest import mock

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from bigcherry.patch import catalog as patch_catalog  # noqa: E402
from bigcherry.patch import evidence as pve  # noqa: E402
from bigcherry.patch import validation_policy  # noqa: E402


class ValidationEvidenceStatusesDispatchTests(unittest.TestCase):
    def test_real_ported_benched_patch_dispatches_to_ported_benched_verifier(self) -> None:
        # 1202_rd04_bf16_flash_attn_tile: real STATE="untested" patch with a
        # real status="ported-benched" tracked entry.
        #
        # This fixture was 1233_rd73_stable_graph_cache_key until 2026-09-05,
        # when RD73 was promoted to STATE="validated". Its tracked status is
        # still "ported-benched", but the dispatcher keys on STATE, so RD73
        # now routes to verify_validated_patch and can no longer demonstrate
        # the ported-benched branch. Swapped for another real patch in that
        # state rather than pinned to a synthetic one -- the point of this
        # test is that dispatch works on the REAL catalog.
        self.assertIn(
            "ported-benched",
            validation_policy.tracked_statuses_for_patch("1202_rd04_bf16_flash_attn_tile"),
        )
        with mock.patch.object(
            pve, "verify_ported_benched_patch",
            return_value=pve.EvidenceCheck("ported-benched-evidence"),
        ) as fake_benched, mock.patch.object(
            pve, "verify_deferred_hardware_patch",
        ) as fake_deferred:
            result = patch_catalog.validation_evidence_statuses(
                ["1202_rd04_bf16_flash_attn_tile"],
            )
        fake_benched.assert_called_once()
        fake_deferred.assert_not_called()
        self.assertEqual(result["1202_rd04_bf16_flash_attn_tile"].status, "ported-benched-evidence")

    def test_real_deferred_hardware_patch_dispatches_to_deferred_hardware_verifier(self) -> None:
        # 1217_rd44_graph_opt_default_rdna35: real STATE="untested" patch
        # with a real status="deferred-hardware" tracked entry.
        self.assertIn(
            "deferred-hardware",
            validation_policy.tracked_statuses_for_patch("1217_rd44_graph_opt_default_rdna35"),
        )
        with mock.patch.object(
            pve, "verify_deferred_hardware_patch",
            return_value=pve.EvidenceCheck("deferred-hardware-evidence"),
        ) as fake_deferred, mock.patch.object(
            pve, "verify_ported_benched_patch",
        ) as fake_benched:
            result = patch_catalog.validation_evidence_statuses(
                ["1217_rd44_graph_opt_default_rdna35"],
            )
        fake_deferred.assert_called_once()
        fake_benched.assert_not_called()
        self.assertEqual(result["1217_rd44_graph_opt_default_rdna35"].status, "deferred-hardware-evidence")

    def test_untested_patch_with_no_tracked_status_is_not_required(self) -> None:
        with mock.patch.object(pve, "verify_ported_benched_patch") as fake_benched, \
             mock.patch.object(pve, "verify_deferred_hardware_patch") as fake_deferred:
            result = patch_catalog.validation_evidence_statuses(
                ["1210_rd26_bitidentical_decode_verify_standalone"],
            )
        fake_benched.assert_not_called()
        fake_deferred.assert_not_called()
        self.assertEqual(
            result["1210_rd26_bitidentical_decode_verify_standalone"].status, "not-required"
        )

    def test_validated_patch_still_uses_verify_validated_patch_unchanged(self) -> None:
        # 1000_rdna4_mmq_q2k_q6k_fix is real STATE="validated" -- must go
        # through the original, unmodified verify_validated_patch() path,
        # never the new status-obligation verifiers.
        with mock.patch.object(pve, "verify_ported_benched_patch") as fake_benched, \
             mock.patch.object(pve, "verify_deferred_hardware_patch") as fake_deferred:
            result = patch_catalog.validation_evidence_statuses(["1000_rdna4_mmq_q2k_q6k_fix"])
        fake_benched.assert_not_called()
        fake_deferred.assert_not_called()
        self.assertIn(result["1000_rdna4_mmq_q2k_q6k_fix"].status, ("validated-evidence", "missing-or-stale", "legacy-grandfathered"))


if __name__ == "__main__":
    unittest.main()
