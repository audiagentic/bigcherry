"""HI83: tracked evidence contract for patch STATE="validated"
(design: GPT, req_487497b28d444d50)."""

from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import evidence as pve # noqa: E402
from bigcherry.patch import patchset # noqa: E402
from bigcherry.patch.activation import ActivationEvidence  # noqa: E402


def _build_identity(tag: str = "") -> dict:
    return {
        "effective_build_id": f"eff{tag}", "compile_verification_id": f"cv{tag}",
        "compile_commands_digest": f"ccd{tag}", "hip_compile_commands_digest": f"hccd{tag}",
        "runtime_bundle_hash": f"rbh{tag}", "runtime_artifacts": {"llama-server": "a" * 64},
    }


_HEX64 = "a" * 64
_HEX40 = "b" * 40


class SubjectDigestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, state: str, extra: str = "") -> Path:
        path = self.root / "9999_example.py"
        path.write_text(
            f'"""doc"""\nfrom bigcherry.patcher import FilePatch\n\nGROUP = "g"\nSTATE = "{state}"\n{extra}\n',
            encoding="utf-8",
        )
        return path

    def test_state_transition_alone_does_not_change_digest(self):
        untested = self._write("untested")
        digest_untested = pve.patch_validation_subject_digest(untested)

        validated = self._write("validated")
        digest_validated = pve.patch_validation_subject_digest(validated)

        self.assertEqual(digest_untested, digest_validated)

    def test_other_byte_change_does_change_digest(self):
        base = self._write("validated")
        digest_base = pve.patch_validation_subject_digest(base)

        changed = self._write("validated", extra="EXTRA = 1")
        digest_changed = pve.patch_validation_subject_digest(changed)

        self.assertNotEqual(digest_base, digest_changed)

    def test_legacy_flat_without_state_uses_implementation_identity(self):
        path = self.root / "no_state.py"
        # write_bytes avoids platform newline translation (write_text would
        # turn \n into \r\n on Windows, desyncing this raw-byte comparison
        # from patch_validation_subject_digest's text-mode read).
        path.write_bytes(b'"""doc"""\n')
        self.assertEqual(
            pve.patch_validation_subject_digest(path),
            pve._sha256_file(path),
        )

    def test_packaged_patch_state_lives_in_manifest(self):
        package = self.root / "9999_packaged"
        package.mkdir()
        implementation = package / "patch.py"
        implementation.write_text('"""implementation"""\n', encoding="utf-8")
        (package / "patch.toml").write_text(
            'schema = 1\nid = "9999_packaged"\norder = 9999\n'
            'state = "untested"\n', encoding="utf-8",
        )
        first = pve.patch_validation_subject_digest(implementation)
        (package / "patch.toml").write_text(
            'schema = 1\nid = "9999_packaged"\norder = 9999\n'
            'state = "validated"\n', encoding="utf-8",
        )
        self.assertEqual(first, pve.patch_validation_subject_digest(implementation))


class MakeRecordTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patch_path = self.root / "9999_example.py"
        self.patch_path.write_text('STATE = "untested"\n', encoding="utf-8")
        self.workdir = self.root / "campaign"
        self.workdir.mkdir()
        (self.workdir / "activation.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _make(self, *, activation_status="executed", activation_disposition="activation-verified",
               correctness_disposition="passed", **record_kwargs):
        activation_evidence = ActivationEvidence(status=activation_status, mechanism="m", detail="d")
        correctness = {
            "schema_version": 1, "disposition": correctness_disposition, "mechanism": "m", "detail": "d",
        }
        return pve.make_record(
            patch_id="9999_example", patch_path=self.patch_path,
            patch_implementation_digest=_HEX64, base_ref="b10502", base_revision=_HEX40,
            framework_baseline_digest=_HEX64, patched_source_tree=_HEX40,
            gpu_architectures="gfx1100", activation_evidence=activation_evidence,
            activation_disposition=activation_disposition, correctness=correctness,
            campaign_identity_digest=_HEX64,
            build_identities={"tune": _build_identity("1"), "replay": _build_identity("2"),
                               "stock": _build_identity("3")},
            validation_build_identities={"control": _build_identity("4"), "subject": _build_identity("5")},
            campaign_workdir=self.workdir, lane_effects=(), **record_kwargs,
        )

    def test_fully_eligible_record(self):
        record = self._make()
        self.assertTrue(record["eligible_for_validated_state"])
        self.assertEqual(record["validation_disposition"], "validated")
        self.assertEqual(record["record_schema_version"], 4)
        self.assertEqual(record["record_digest"], pve._record_digest(record))
        self.assertIn("representation", record)
        self.assertIn("hardware", record)

    def test_v2_integrity_digest_changes_for_each_provenance_field(self):
        record = self._make()
        for field in ("representation", "validation_implementation_digest", "contracts",
                      "baseline_composition", "control_composition", "subject_composition",
                      "subject_tree", "stock_tree", "check_results", "hardware",
                      "artifact_hashes", "final_eligibility"):
            changed = dict(record)
            changed[field] = "tampered"
            self.assertNotEqual(record["record_digest"], pve._record_digest(changed), field)

    def test_contract_hash_is_written_from_authoritative_argument(self):
        # VA18: contract_id/contract_hash remain a 0/1-contract convenience
        # on make_record() -- the written field is now the plural, canonical
        # "contracts" list.
        record = self._make(contract_id="RD08-Q6K-MMVQ-VDR2", contract_hash="c" * 32)
        self.assertEqual(record["contracts"], [{"id": "RD08-Q6K-MMVQ-VDR2", "hash": "c" * 32}])

    def test_missing_correctness_is_incomplete(self):
        record = self._make(correctness_disposition="unknown")
        self.assertFalse(record["eligible_for_validated_state"])
        self.assertEqual(record["validation_disposition"], "incomplete")

    def test_gate_verified_blocked_does_not_authorize_validated(self):
        record = self._make(
            activation_status="not_applicable", activation_disposition="gate-verified-blocked",
        )
        self.assertFalse(record["eligible_for_validated_state"])

    def test_missing_build_identity_role_raises(self):
        activation_evidence = ActivationEvidence(status="executed", mechanism="m", detail="d")
        with self.assertRaises(pve.ValidationEvidenceError):
            pve.make_record(
                patch_id="9999_example", patch_path=self.patch_path,
                patch_implementation_digest=_HEX64, base_ref="b10502", base_revision=_HEX40,
                framework_baseline_digest=_HEX64, patched_source_tree=_HEX40,
                gpu_architectures="gfx1100", activation_evidence=activation_evidence,
                activation_disposition="activation-verified",
                correctness={"schema_version": 1, "disposition": "passed", "mechanism": "m", "detail": "d"},
                campaign_identity_digest=_HEX64,
                build_identities={"tune": _build_identity("1")},  # replay/stock missing
                validation_build_identities={"control": _build_identity("4"), "subject": _build_identity("5")},
                campaign_workdir=self.workdir, lane_effects=(),
            )


class WriteLoadRecordTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _record(self, digest_tag: str = "1") -> dict:
        return {
            "record_schema_version": 1, "patch_id": "9999_example",
            "campaign_identity_digest": digest_tag * 64, "some_field": "value",
        }

    def test_write_then_load_round_trips(self):
        record = self._record()
        pve.write_record(record, engine="llamacpp", patches_root=self.root, root=self.root)
        loaded = pve.load_records("9999_example", engine="llamacpp", patches_root=self.root, root=self.root)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["some_field"], "value")

    def test_rewriting_identical_record_is_idempotent(self):
        record = self._record()
        pve.write_record(record, engine="llamacpp", patches_root=self.root, root=self.root)
        pve.write_record(record, engine="llamacpp", patches_root=self.root, root=self.root)
        self.assertEqual(len(pve.load_records("9999_example", engine="llamacpp", patches_root=self.root, root=self.root)), 1)

    def test_same_build_measured_twice_records_both(self):
        # RV96, deliberately REPLACING an earlier test that asserted this
        # raised. campaign_identity_digest is a BUILD identity -- Campaign
        # derives it from built-binary content plus patch identity -- so two
        # independent measurements of the same binaries necessarily share it.
        # Rejecting the second made replication impossible, left the frozen
        # N_min/N_max extension policy unimplementable, and meant whichever
        # run was written first owned the digest for good.
        record = self._record()
        pve.write_record(record, engine="llamacpp", patches_root=self.root, root=self.root)
        second = dict(record)
        second["some_field"] = "different"
        pve.write_record(second, engine="llamacpp", patches_root=self.root, root=self.root)
        loaded = pve.load_records("9999_example", engine="llamacpp", patches_root=self.root, root=self.root)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            {row["some_field"] for row in loaded}, {"value", "different"}
        )

    def test_first_measurement_is_never_replaced_by_a_later_one(self):
        # The property that must survive: evidence is append-only. A second
        # measurement must not be able to overwrite or remove the first --
        # notably not a later PASS quietly displacing an earlier FAIL.
        pve.write_record(self._record(), engine="llamacpp", patches_root=self.root, root=self.root)
        second = dict(self._record())
        second["some_field"] = "different"
        pve.write_record(second, engine="llamacpp", patches_root=self.root, root=self.root)
        loaded = pve.load_records("9999_example", engine="llamacpp", patches_root=self.root, root=self.root)
        self.assertIn("value", {row["some_field"] for row in loaded})

    def test_records_disagreeing_on_patch_identity_still_fail_closed(self):
        # Replication is same build, different measurement. Records sharing a
        # campaign digest but disagreeing about WHAT was built mean the digest
        # is not identifying what it claims to -- corruption, not replication.
        record = dict(self._record())
        record["patch_implementation_digest"] = "a" * 64
        pve.write_record(record, engine="llamacpp", patches_root=self.root, root=self.root)
        contradictory = dict(record)
        contradictory["patch_implementation_digest"] = "b" * 64
        with self.assertRaises(pve.ValidationEvidenceError) as caught:
            pve.write_record(contradictory, engine="llamacpp", patches_root=self.root, root=self.root)
        self.assertIn("patch_implementation_digest", str(caught.exception))

    def test_multiple_measurements_are_ordered_deterministically(self):
        # A build may now hold several records; ordering must not depend on
        # write order, or the file churns on every re-write.
        for tag, value in (("1", "b"), ("1", "a"), ("1", "c")):
            row = self._record(tag)
            row["record_digest"] = value * 64
            pve.write_record(row, engine="llamacpp", patches_root=self.root, root=self.root)
        loaded = pve.load_records("9999_example", engine="llamacpp", patches_root=self.root, root=self.root)
        self.assertEqual(
            [row["record_digest"] for row in loaded],
            ["a" * 64, "b" * 64, "c" * 64],
        )

    def test_load_missing_returns_empty(self):
        self.assertEqual(pve.load_records("nonexistent", engine="llamacpp", patches_root=self.root, root=self.root), ())


class VerifyValidatedPatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patch_path = self.root / "9999_example.py"
        self.patch_path.write_text('STATE = "validated"\n', encoding="utf-8")
        self.subject_digest = pve.patch_validation_subject_digest(self.patch_path)

    def tearDown(self):
        self._tmp.cleanup()

    def _module(self, *, state="validated", content_hash="deadbeef" * 8):
        return patchset.PatchModule(engine="llamacpp", 
            patch_id="9999_example", path=self.patch_path, order=0, state=state,
            upstream=None, content_hash=content_hash, catalog_root=self.root,
        )

    def _qualifying_record(self):
        return {
            "record_schema_version": 1, "validation_contract_version": "hi83-v1",
            "patch_id": "9999_example", "patch_validation_subject_digest": self.subject_digest,
            "base_ref": "b10502", "validation_disposition": "validated",
            "eligible_for_validated_state": True,
            "activation": {"status": "executed", "disposition": "activation-verified"},
            "correctness": {"disposition": "passed"},
            "patch_implementation_digest": _HEX64, "base_revision": _HEX40,
            "framework_baseline_digest": _HEX64, "patched_source_tree": _HEX40,
            "campaign_identity_digest": _HEX64,
            "build_identities": {"tune": _build_identity("1"), "replay": _build_identity("2"),
                                  "stock": _build_identity("3")},
            "gpu_architectures": ["gfx1100"],
        }

    def test_untested_patch_is_not_required(self):
        module = self._module(state="untested")
        result = pve.verify_validated_patch(module, pinned_ref="b10502", root=self.root)
        self.assertEqual(result.status, "not-required")
        self.assertTrue(result.ok)

    def test_validated_with_no_evidence_and_no_grandfather_fails(self):
        module = self._module()
        result = pve.verify_validated_patch(
            module, pinned_ref="b10502", root=self.root, allow_legacy_grandfather=False,
        )
        self.assertEqual(result.status, "missing-or-stale")
        self.assertFalse(result.ok)

    def test_validated_with_current_qualifying_record_passes(self):
        pve.write_record(self._qualifying_record(), engine="llamacpp", patches_root=self.root, root=self.root)
        module = self._module()
        result = pve.verify_validated_patch(module, pinned_ref="b10502", root=self.root)
        self.assertEqual(result.status, "validated-evidence")
        self.assertTrue(result.ok)

    def test_editing_patch_after_evidence_still_qualifies_with_a_digest_note(self):
        pve.write_record(self._qualifying_record(), engine="llamacpp", patches_root=self.root, root=self.root)
        self.patch_path.write_text('STATE = "validated"\nEXTRA = 1\n', encoding="utf-8")
        module = self._module()
        result = pve.verify_validated_patch(
            module, pinned_ref="b10502", root=self.root, allow_legacy_grandfather=False,
        )
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.status, "validated-evidence")
        now = pve.patch_validation_subject_digest(self.patch_path)
        self.assertEqual(
            result.notes,
            (f"evidence recorded at patch digest {self.subject_digest[:12]} (now {now[:12]})",),
        )

    def test_pin_change_keeps_old_evidence_with_a_pin_note(self):
        pve.write_record(self._qualifying_record(), engine="llamacpp", patches_root=self.root, root=self.root)
        module = self._module()
        result = pve.verify_validated_patch(
            module, pinned_ref="different-pin", root=self.root, allow_legacy_grandfather=False,
        )
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.notes, ("evidence recorded at pin b10502 (now different-pin)",))

    def test_resolved_base_revision_mismatch_is_a_note_not_a_failure(self):
        pve.write_record(self._qualifying_record(), engine="llamacpp", patches_root=self.root, root=self.root)
        module = self._module()
        result = pve.verify_validated_patch(
            module, pinned_ref="b10502", root=self.root, allow_legacy_grandfather=False,
            resolved_base_revision="f" * 40,
        )
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.notes, ("evidence recorded at pin revision bbbbbbbbbbbb (now ffffffffffff)",))

    def test_resolved_base_revision_match_still_passes(self):
        record = self._qualifying_record()
        pve.write_record(record, engine="llamacpp", patches_root=self.root, root=self.root)
        module = self._module()
        result = pve.verify_validated_patch(
            module, pinned_ref="b10502", root=self.root,
            resolved_base_revision=record["base_revision"],
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.notes, ())

    def test_missing_required_architecture_fails(self):
        pve.write_record(self._qualifying_record(), engine="llamacpp", patches_root=self.root, root=self.root)
        module = self._module()
        result = pve.verify_validated_patch(
            module, pinned_ref="b10502", required_architectures=("gfx1201",), root=self.root,
        )
        self.assertFalse(result.ok)

    def test_gate_verified_blocked_record_does_not_qualify(self):
        record = self._qualifying_record()
        record["activation"] = {"status": "not_applicable", "disposition": "gate-verified-blocked"}
        record["eligible_for_validated_state"] = False
        record["validation_disposition"] = "incomplete"
        pve.write_record(record, engine="llamacpp", patches_root=self.root, root=self.root)
        module = self._module()
        result = pve.verify_validated_patch(
            module, pinned_ref="b10502", root=self.root, allow_legacy_grandfather=False,
        )
        self.assertFalse(result.ok)

    def _legacy_baseline(self):
        import json
        (self.root / "legacy-baseline.json").write_text(json.dumps({
            "schema_version": 1, "contract": "hi83-legacy-grandfather-v1",
            "patches": {"9999_example": "c" * 64},
        }), encoding="utf-8")

    def test_grandfathered_patch_id_passes(self):
        self._legacy_baseline()
        module = self._module(content_hash="c" * 64)
        result = pve.verify_validated_patch(module, pinned_ref="b10502", root=self.root)
        self.assertEqual(result.status, "legacy-grandfathered")
        self.assertTrue(result.ok)

    def test_edited_grandfathered_patch_stays_grandfathered(self):
        self._legacy_baseline()
        # The recorded hash is provenance: grandfathering is by patch id.
        module = self._module(content_hash="d" * 64)
        result = pve.verify_validated_patch(module, pinned_ref="b10502", root=self.root)
        self.assertEqual(result.status, "legacy-grandfathered")
        self.assertTrue(result.ok)

    def test_patch_absent_from_the_baseline_is_not_grandfathered(self):
        self._legacy_baseline()
        module = patchset.PatchModule(engine="llamacpp",
            patch_id="9998_other", path=self.patch_path, order=0, state="validated",
            upstream=None, content_hash="c" * 64, catalog_root=self.root,
        )
        result = pve.verify_validated_patch(module, pinned_ref="b10502", root=self.root)
        self.assertFalse(result.ok)


class EvidenceSurvivesChangeTests(unittest.TestCase):
    """Owner rule: evidence is not invalidated by code, digest or pin changes.

    A record qualifies while it is intact and says the patch passed; recorded
    digests and the pin are provenance notes. Reruns are requested by demoting
    the patch, never triggered by a change."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patches = self.root / "patches"
        self.patches.mkdir()
        self.evidence = self.root / "evidence"
        self.patch_path = self.patches / "0001_test.py"
        self._write_patch("untested")
        self.workdir = self.root / "campaign"
        self.workdir.mkdir()
        (self.workdir / "activation.json").write_text("{}", encoding="utf-8")
        self.catalog = self.root / "catalog.toml"

    def tearDown(self):
        self._tmp.cleanup()

    def _write_patch(self, state, extra=""):
        self.patch_path.write_text(
            f'GROUP = "core"\nSTATE = "{state}"\nPATCHES = []\n{extra}', encoding="utf-8",
        )

    def _write_catalog(self, state):
        self.catalog.write_text(
            'version = 1\n\n[[patch]]\nid = "0001_test"\n'
            'kind = "framework"\norigin = "local"\nbackend = "hip"\n'
            f'state = "{state}"\n', encoding="utf-8",
        )

    def _record(self, *, contracts=(), verdicts=None, base_ref="v1.16.0"):
        return pve.make_record(
            patch_id="0001_test", patch_path=self.patch_path,
            patch_implementation_digest=_HEX64, base_ref=base_ref, base_revision=_HEX40,
            framework_baseline_digest=_HEX64, patched_source_tree=_HEX40,
            gpu_architectures="gfx1100",
            activation_evidence=ActivationEvidence(status="executed", mechanism="m", detail="d"),
            activation_disposition="activation-verified",
            correctness={"schema_version": 1, "disposition": "passed", "mechanism": "m", "detail": "d"},
            campaign_identity_digest=_HEX64,
            build_identities={"tune": _build_identity("1"), "replay": _build_identity("2"),
                               "stock": _build_identity("3")},
            validation_build_identities={"control": _build_identity("4"), "subject": _build_identity("5")},
            campaign_workdir=self.workdir, lane_effects=(),
            contracts=contracts, contract_verdicts=verdicts,
            baseline_composition={"modules": []}, control_composition={"modules": []},
            subject_composition={"modules": ["0001_test"]},
            control_tree=_HEX40, subject_tree=_HEX40, stock_tree=_HEX40,
        )

    def _module(self, state="validated"):
        return patchset.PatchModule(engine="llamacpp",
            patch_id="0001_test", path=self.patch_path, order=1, state=state,
            upstream=None, content_hash="deadbeef" * 8, catalog_root=self.patches,
        )

    def _verify(self, **kwargs):
        kwargs.setdefault("pinned_ref", "v1.16.0")
        return pve.verify_validated_patch(
            self._module(), root=self.evidence, allow_legacy_grandfather=False, **kwargs,
        )

    def test_a_validated_patch_whose_code_changed_still_qualifies_with_a_digest_note(self):
        recorded = pve.patch_validation_subject_digest(self.patch_path)
        pve.write_record(self._record(), engine="llamacpp", patches_root=self.patches, root=self.evidence)
        self._write_patch("validated", extra="# a later edit\n")
        now = pve.patch_validation_subject_digest(self.patch_path)
        self.assertNotEqual(recorded, now)
        result = self._verify()
        self.assertEqual(result.status, "validated-evidence", result.problems)
        self.assertEqual(result.problems, ())
        self.assertIn(f"evidence recorded at patch digest {recorded[:12]} (now {now[:12]})", result.notes)

    def test_evidence_at_an_older_pin_qualifies_for_build_and_promote_with_a_pin_note(self):
        from types import SimpleNamespace
        from bigcherry.patch import gates
        pve.write_record(self._record(base_ref="v1.16.0"), engine="llamacpp",
                         patches_root=self.patches, root=self.evidence)
        for intent, state in ((gates.GateIntent.BUILD, "validated"), (gates.GateIntent.PROMOTE, "untested")):
            with self.subTest(intent=intent):
                self._write_patch(state)
                self._write_catalog(state)
                context = SimpleNamespace(
                    intent=intent, descriptor=SimpleNamespace(engine="llamacpp", patch_id="0001_test"),
                    catalog_path=self.catalog, patches_dir=self.patches, pinned_ref="v1.17.0",
                    evidence_root=self.evidence, allow_legacy_grandfather=False,
                    resolved_base_revision="c" * 40,
                )
                result = gates.evaluate_evidence_gate(context)
                self.assertEqual(result.status, gates.GateStatus.PASS, result.detail)
                self.assertIn(gates.EVIDENCE_NOTE_PREFIX + "evidence recorded at pin v1.16.0 (now v1.17.0)",
                              result.detail)
                self.assertIn(gates.EVIDENCE_NOTE_PREFIX
                              + "evidence recorded at pin revision bbbbbbbbbbbb (now cccccccccccc)",
                              result.detail)

    def test_a_changed_contract_hash_does_not_disqualify(self):
        contracts = ({"id": "RD05", "hash": "1" * 64},)
        pve.write_record(
            self._record(contracts=contracts, verdicts={"RD05": {"passed": True}}),
            engine="llamacpp", patches_root=self.patches, root=self.evidence,
        )
        self._write_patch("validated")
        current = (None, ({"id": "RD05", "hash": "2" * 64},))
        with unittest.mock.patch.object(pve, "_resolve_contract_identities", return_value=current):
            result = self._verify()
        self.assertEqual(result.status, "validated-evidence", result.problems)
        self.assertIn("contract RD05 recorded at hash 111111111111 (now 222222222222)", result.notes)

    def test_a_newly_bound_contract_without_a_verdict_disqualifies(self):
        contracts = ({"id": "RD05", "hash": "1" * 64},)
        pve.write_record(
            self._record(contracts=contracts, verdicts={"RD05": {"passed": True}}),
            engine="llamacpp", patches_root=self.patches, root=self.evidence,
        )
        self._write_patch("validated")
        current = (None, ({"id": "RD05", "hash": "1" * 64}, {"id": "RD06", "hash": "3" * 64}))
        with unittest.mock.patch.object(pve, "_resolve_contract_identities", return_value=current):
            result = self._verify()
        self.assertFalse(result.ok)
        joined = "; ".join(result.problems)
        self.assertIn("does not cover currently bound contract(s): ['RD06']", joined)
        self.assertIn("missing contract_verdicts for: ['RD06']", joined)

    def test_record_digest_tampering_still_disqualifies(self):
        import json
        path = pve.write_record(self._record(), engine="llamacpp", patches_root=self.patches, root=self.evidence)
        self._write_patch("validated")
        document = json.loads(path.read_text(encoding="utf-8"))
        document["records"][0]["gpu_architectures"] = ["gfx1100", "gfx1201"]
        path.write_text(json.dumps(document), encoding="utf-8")
        result = self._verify()
        self.assertFalse(result.ok)
        self.assertIn("record_digest does not match the evidence payload", "; ".join(result.problems))

    def test_a_withdrawn_record_still_does_not_qualify(self):
        import json
        record = self._record()
        path = pve.write_record(record, engine="llamacpp", patches_root=self.patches, root=self.evidence)
        self._write_patch("validated")
        (path.parent / pve.WITHDRAWALS_FILE).write_text(json.dumps({"withdrawn": [
            {"record_digest": record["record_digest"], "reason": "measured beside another GPU job"},
        ]}), encoding="utf-8")
        result = self._verify()
        self.assertFalse(result.ok)
        self.assertIn("no qualifying HI83 validation record", result.problems)


if __name__ == "__main__":
    unittest.main()
