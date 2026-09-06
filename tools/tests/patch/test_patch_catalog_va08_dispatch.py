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

    def test_validated_framework_uses_configuration_verifier_and_compiled_targets(self) -> None:
        # 0100_cmake_options is a validated, local framework package with no
        # validation-architectures.  The requested target is passed as a
        # compiled-target requirement, never as runtime gpu_architectures.
        with mock.patch.object(
            pve,
            "verify_framework_configuration_patch",
            create=True,
            return_value=pve.EvidenceCheck("framework-configuration-evidence"),
        ) as fake_framework, mock.patch.object(
            pve, "verify_validated_patch"
        ) as fake_runtime:
            result = patch_catalog.validation_evidence_statuses(
                ["0100_cmake_options"],
                default_validation_architectures=("gfx1201",),
            )
        fake_framework.assert_called_once()
        self.assertEqual(
            fake_framework.call_args.kwargs["required_compiled_targets"], ("gfx1201",)
        )
        fake_runtime.assert_not_called()
        self.assertEqual(result["0100_cmake_options"].status, "framework-configuration-evidence")

    def test_framework_target_union_deduplicates_explicit_and_requested_targets(self) -> None:
        # Exercise the explicit package declaration plus the requested target
        # fallback without changing production metadata.
        descriptor = mock.Mock(
            patch_id="framework-test",
            representation="packaged", kind="framework", origin="local",
            external_source=None, experiment_contracts=(), state="validated",
            plan_ids=(), plan_item=None,
            validation_architectures=("gfx1100", "gfx1201"),
        )
        module = mock.Mock(state="validated", patch_id="framework-test")
        with mock.patch.object(patch_catalog, "load_catalog", return_value={}), \
             mock.patch.object(patch_catalog.patchset, "catalog", return_value=[module]), \
             mock.patch.object(patch_catalog.patchset.patch_registry, "load_registry") as load_registry, \
             mock.patch.object(pve, "verify_framework_configuration_patch", create=True,
                               return_value=pve.EvidenceCheck("framework-configuration-evidence")) as fake:
            load_registry.return_value.descriptors = [descriptor]
            result = patch_catalog.validation_evidence_statuses(
                ["framework-test"], default_validation_architectures=("gfx1201", "gfx1030")
            )
        self.assertEqual(result["framework-test"].status, "framework-configuration-evidence")
        self.assertEqual(
            fake.call_args.kwargs["required_compiled_targets"], ("gfx1100", "gfx1201", "gfx1030")
        )


if __name__ == "__main__":
    unittest.main()
