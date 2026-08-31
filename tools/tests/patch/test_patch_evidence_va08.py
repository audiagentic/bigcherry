"""VA08: status-aware patch-verify-evidence -- the 'ported-benched' and
'deferred-hardware' tracked-status obligation verifiers (GPT session
ses_5bbee8ce5c9a4265, req_65394c5d0fdd4647). verify_validated_patch()'s
own STATE='validated' behavior is unchanged and already covered by
test_patch_validation_evidence.py's VerifyValidatedPatchTests -- this
file covers the two NEW status obligations only.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import evidence as pve  # noqa: E402
from bigcherry.patch import patchset  # noqa: E402
from bigcherry.patch.activation import ActivationEvidence  # noqa: E402

_HEX64 = "a" * 64
_HEX40 = "b" * 40


def _build_identity(tag: str = "") -> dict:
    return {
        "effective_build_id": f"eff{tag}", "compile_verification_id": f"cv{tag}",
        "compile_commands_digest": f"ccd{tag}", "hip_compile_commands_digest": f"hccd{tag}",
        "runtime_bundle_hash": f"rbh{tag}", "runtime_artifacts": {"llama-server": "a" * 64},
    }


class PortedBenchedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patch_path = self.root / "9999_example.py"
        self.patch_path.write_text('STATE = "untested"\n', encoding="utf-8")
        self.subject_digest = pve.patch_validation_subject_digest(self.patch_path)
        self.workdir = self.root / "campaign"
        self.workdir.mkdir()
        (self.workdir / "activation.json").write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _module(self) -> patchset.PatchModule:
        return patchset.PatchModule(
            patch_id="9999_example", path=self.patch_path, order=0, group="g", state="untested",
            upstream=None, content_hash="deadbeef" * 8,
        )

    def _real_v3_record(self, *, correctness_disposition: str = "passed") -> dict:
        """A real record built via make_record() -- the same real schema a
        real campaign produces, not a hand-authored fixture pretending to
        be one."""
        activation_evidence = ActivationEvidence(status="not_executed", mechanism="m", detail="d")
        correctness = {
            "schema_version": 1, "disposition": correctness_disposition, "mechanism": "m", "detail": "d",
        }
        return pve.make_record(
            patch_id="9999_example", patch_path=self.patch_path,
            patch_implementation_digest=_HEX64, base_ref="b10705", base_revision=_HEX40,
            framework_baseline_digest=_HEX64, patched_source_tree=_HEX40,
            gpu_architectures="gfx1100", activation_evidence=activation_evidence,
            activation_disposition="failed-activation", correctness=correctness,
            campaign_identity_digest=_HEX64,
            build_identities={"tune": _build_identity("1"), "replay": _build_identity("2"),
                               "stock": _build_identity("3")},
            validation_build_identities={"control": _build_identity("4"), "subject": _build_identity("5")},
            campaign_workdir=self.workdir,
        )

    def test_current_pin_real_benchmark_evidence_passes(self) -> None:
        # A real control/subject benchmark ran (validation_build_identities
        # populated) with no correctness failure -- ported-benched does NOT
        # require activation executed+verified or eligible_for_validated_state.
        record = self._real_v3_record()
        self.assertFalse(record["eligible_for_validated_state"])  # proves this is a REAL sub-validated record
        pve.write_record(record, root=self.root)
        result = pve.verify_ported_benched_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "ported-benched-evidence")
        self.assertTrue(result.ok)

    def test_correctness_failure_fails(self) -> None:
        record = self._real_v3_record(correctness_disposition="failed")
        pve.write_record(record, root=self.root)
        result = pve.verify_ported_benched_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)

    def test_stale_base_revision_fails(self) -> None:
        record = self._real_v3_record()
        pve.write_record(record, root=self.root)
        result = pve.verify_ported_benched_patch(
            self._module(), pinned_ref="b10705", root=self.root,
            resolved_base_revision="c" * 40,  # does not match record's base_revision (_HEX40 = "b"*40)
        )
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)

    def test_missing_benchmark_provenance_fails(self) -> None:
        record = self._real_v3_record()
        del record["validation_build_identities"]
        record["record_digest"] = pve._record_digest(record)
        pve.write_record(record, root=self.root)
        result = pve.verify_ported_benched_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)

    def test_no_evidence_at_all_fails(self) -> None:
        result = pve.verify_ported_benched_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)


class DeferredHardwareTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patch_path = self.root / "9999_example.py"
        self.patch_path.write_text('STATE = "untested"\n', encoding="utf-8")
        self.subject_digest = pve.patch_validation_subject_digest(self.patch_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _module(self) -> patchset.PatchModule:
        return patchset.PatchModule(
            patch_id="9999_example", path=self.patch_path, order=0, group="g", state="untested",
            upstream=None, content_hash="deadbeef" * 8,
        )

    def _blocked_record(self, **overrides) -> dict:
        record = {
            "patch_id": "9999_example", "patch_validation_subject_digest": self.subject_digest,
            "base_ref": "b10705", "base_revision": _HEX40,
            "patch_implementation_digest": _HEX64,
            "campaign_identity_digest": _HEX64,
            "blockers": ["required hardware (gfx1201) unavailable"],
        }
        record.update(overrides)
        return record

    def test_fresh_structured_blocked_evidence_passes(self) -> None:
        pve.write_record(self._blocked_record(), root=self.root)
        result = pve.verify_deferred_hardware_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "deferred-hardware-evidence")
        self.assertTrue(result.ok)

    def test_no_blockers_fails(self) -> None:
        pve.write_record(self._blocked_record(blockers=[]), root=self.root)
        result = pve.verify_deferred_hardware_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)

    def test_missing_evidence_fails(self) -> None:
        result = pve.verify_deferred_hardware_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)

    def test_malformed_stale_patch_identity_fails(self) -> None:
        pve.write_record(self._blocked_record(patch_validation_subject_digest="stale-digest"), root=self.root)
        result = pve.verify_deferred_hardware_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)

    def test_stale_resolved_base_revision_fails(self) -> None:
        pve.write_record(self._blocked_record(), root=self.root)
        result = pve.verify_deferred_hardware_patch(
            self._module(), pinned_ref="b10705", root=self.root, resolved_base_revision="c" * 40,
        )
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
