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

    def _real_check_results(self, *, with_artifact: bool = True) -> dict:
        """Runs the REAL bigcherry.patch.validation.evaluate_check() against
        a genuine bound performance.json artifact through the actual
        "autotune-campaign" validator (_builtin_autotune_campaign ->
        _evidence_pass()) -- the same producer real campaigns use, not a
        hand-authored ValidationResult shape guessing at what it emits.
        with_artifact=False writes NO bound artifact at all, so the real
        producer returns BLOCKED (not a hand-faked "pass with no
        artifacts", which the real producer can never actually emit)."""
        import hashlib
        import json
        from dataclasses import asdict

        from bigcherry.patch import validation as pv

        performance_evidence: dict = {}
        if with_artifact:
            perf_path = self.workdir / "performance.json"
            perf_path.write_text(json.dumps({"campaign_id": "x", "passed": True}), encoding="utf-8")
            performance_evidence = {
                "artifact": {
                    "path": "performance.json",
                    "sha256": hashlib.sha256(perf_path.read_bytes()).hexdigest(),
                }
            }
        ctx = pv.ValidationContext(
            descriptor=None, base_revision=_HEX40, control_source=None, subject_source=None,
            run_dir=self.workdir, performance_evidence=performance_evidence,
        )
        spec = pv.CheckSpec("performance", "performance", "autotune-campaign", True, {})
        result = pv.evaluate_check(spec, ctx)
        return {"performance": asdict(result)}

    def _real_v3_record(
        self, *, correctness_disposition: str = "passed", check_results: dict | None = "__default__",
    ) -> dict:
        """A real record built via make_record() -- the same real schema a
        real campaign produces, not a hand-authored fixture pretending to
        be one. check_results defaults to a real passing, bound
        performance check (what an actual benchmark run produces); pass
        check_results=None to simulate a build-only record with no
        recorded benchmark execution."""
        if check_results == "__default__":
            check_results = self._real_check_results()
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
            check_results=check_results,
        )

    def test_real_evaluate_check_pass_serializes_a_non_empty_bound_artifact(self) -> None:
        # GPT round 3 (req_021c2eb498e04bc0): the real bug -- _evidence_pass()
        # (shared by the "benchmark"/"autotune-campaign" built-in
        # validators) verified the bound artifact but never put it into
        # ValidationResult.artifacts, so a genuine PASS always serialized
        # artifacts=(). Assert the fix directly: a real evaluate_check()
        # PASS now has exactly one artifact matching the real bound path.
        check_results = self._real_check_results()
        self.assertEqual(check_results["performance"]["status"], "pass")
        artifacts = check_results["performance"]["artifacts"]
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["path"], "performance.json")
        self.assertTrue(artifacts[0]["sha256"])

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

    def test_build_and_hardware_provenance_without_a_benchmark_result_fails(self) -> None:
        # GPT round 2 (req_ecaa87b450294084): the real bug -- build/
        # hardware provenance alone is NOT proof a benchmark ran. A
        # build-only record (check_results defaults to activation+
        # correctness only, no performance/controls entry) must fail.
        record = self._real_v3_record(check_results=None)
        pve.write_record(record, root=self.root)
        result = pve.verify_ported_benched_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)

    def test_benchmark_result_without_a_bound_artifact_fails(self) -> None:
        # A "passing" performance check with no artifacts is not real
        # evidence of execution either -- an unbound claim is exactly as
        # unverifiable as no claim at all.
        record = self._real_v3_record(
            check_results=self._real_check_results(with_artifact=False)
        )
        pve.write_record(record, root=self.root)
        result = pve.verify_ported_benched_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)

    def test_forged_artifact_path_not_in_artifact_hashes_fails(self) -> None:
        # GPT round 4 (req_73faeb08760c42fd): a non-empty artifacts list is
        # not proof by itself -- it must be cross-checked against the
        # record's OWN authoritative artifact_hashes map (real files
        # _artifact_refs() actually found on disk at write time).
        check_results = self._real_check_results()
        check_results["performance"]["artifacts"] = [
            {"name": "performance", "path": "not-a-real-tracked-file.json", "sha256": "d" * 64}
        ]
        record = self._real_v3_record(check_results=check_results)
        pve.write_record(record, root=self.root)
        result = pve.verify_ported_benched_patch(self._module(), pinned_ref="b10705", root=self.root)
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)

    def test_artifact_sha_mismatch_against_artifact_hashes_fails(self) -> None:
        check_results = self._real_check_results()
        real_path = check_results["performance"]["artifacts"][0]["path"]
        check_results["performance"]["artifacts"] = [
            {"name": "performance", "path": real_path, "sha256": "e" * 64}  # wrong hash
        ]
        record = self._real_v3_record(check_results=check_results)
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
