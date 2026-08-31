"""VA07 tests for tools/bigcherry/patch/evidence.py's schema v3: the
{tune,replay,stock} campaign-build domain and {control,subject}
validation-build domain are genuinely distinct provenance domains (GPT
round-4 correction, req_82dc1c3dcc744fb2), not one mislabeled as the
other. v3 is opt-in via ``validation_build_identities`` -- omitting it
keeps producing an ordinary v2 record exactly as before.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import evidence as pve # noqa: E402
from bigcherry.patch.activation import ActivationEvidence # noqa: E402


def _build_identity(tag: str = "") -> dict:
    return {
        "effective_build_id": f"eff{tag}", "compile_verification_id": f"cv{tag}",
        "compile_commands_digest": f"ccd{tag}", "hip_compile_commands_digest": f"hccd{tag}",
        "runtime_bundle_hash": f"rbh{tag}", "runtime_artifacts": {"llama-server": "a" * 64},
    }


_HEX64 = "a" * 64
_HEX40 = "b" * 40


class SchemaV3RecordTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patch_path = self.root / "9999_example.py"
        self.patch_path.write_text('STATE = "untested"\n', encoding="utf-8")
        self.workdir = self.root / "campaign"
        self.workdir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _campaign_builds(self) -> dict:
        return {"tune": _build_identity("t"), "replay": _build_identity("r"), "stock": _build_identity("s")}

    def _validation_builds(self) -> dict:
        return {"control": _build_identity("c"), "subject": _build_identity("j")}

    def _make(self, **kwargs):
        activation_evidence = ActivationEvidence(status="executed", mechanism="m", detail="d")
        correctness = {"schema_version": 1, "disposition": "passed", "mechanism": "m", "detail": "d"}
        return pve.make_record(
            patch_id="9999_example", patch_path=self.patch_path,
            patch_implementation_digest=_HEX64, base_ref="b10692", base_revision=_HEX40,
            framework_baseline_digest=_HEX64, patched_source_tree=_HEX40,
            gpu_architectures="gfx1100", activation_evidence=activation_evidence,
            activation_disposition="activation-verified", correctness=correctness,
            campaign_identity_digest=_HEX64, build_identities=self._campaign_builds(),
            campaign_workdir=self.workdir, **kwargs,
        )

    def test_omitting_validation_builds_still_produces_v2(self):
        record = self._make()
        self.assertEqual(record["record_schema_version"], 2)
        self.assertIn("build_identities", record)
        self.assertNotIn("campaign_build_identities", record)
        self.assertNotIn("validation_build_identities", record)

    def test_supplying_validation_builds_produces_v3_with_both_domains(self):
        record = self._make(validation_build_identities=self._validation_builds())
        self.assertEqual(record["record_schema_version"], 3)
        self.assertNotIn("build_identities", record)
        self.assertIn("campaign_build_identities", record)
        self.assertIn("validation_build_identities", record)
        self.assertEqual(
            set(record["campaign_build_identities"]), {"tune", "replay", "stock"}
        )
        self.assertEqual(set(record["validation_build_identities"]), {"control", "subject"})

    def test_missing_tune_replay_stock_role_rejected(self):
        campaign = self._campaign_builds()
        del campaign["stock"]
        with self.assertRaises(pve.ValidationEvidenceError):
            pve.make_record(
                patch_id="9999_example", patch_path=self.patch_path,
                patch_implementation_digest=_HEX64, base_ref="b10692", base_revision=_HEX40,
                framework_baseline_digest=_HEX64, patched_source_tree=_HEX40,
                gpu_architectures="gfx1100",
                activation_evidence=ActivationEvidence(status="executed", mechanism="m", detail="d"),
                activation_disposition="activation-verified",
                correctness={"schema_version": 1, "disposition": "passed", "mechanism": "m", "detail": "d"},
                campaign_identity_digest=_HEX64, build_identities=campaign,
                validation_build_identities=self._validation_builds(),
                campaign_workdir=self.workdir,
            )

    def test_missing_control_subject_role_rejected(self):
        validation_builds = self._validation_builds()
        del validation_builds["control"]
        with self.assertRaises(pve.ValidationEvidenceError):
            self._make(validation_build_identities=validation_builds)

    def test_malformed_identity_in_either_domain_rejected(self):
        bad_campaign = self._campaign_builds()
        bad_campaign["tune"] = {"effective_build_id": "only-one-field"}
        with self.assertRaises(pve.ValidationEvidenceError):
            pve.make_record(
                patch_id="9999_example", patch_path=self.patch_path,
                patch_implementation_digest=_HEX64, base_ref="b10692", base_revision=_HEX40,
                framework_baseline_digest=_HEX64, patched_source_tree=_HEX40,
                gpu_architectures="gfx1100",
                activation_evidence=ActivationEvidence(status="executed", mechanism="m", detail="d"),
                activation_disposition="activation-verified",
                correctness={"schema_version": 1, "disposition": "passed", "mechanism": "m", "detail": "d"},
                campaign_identity_digest=_HEX64, build_identities=bad_campaign,
                validation_build_identities=self._validation_builds(),
                campaign_workdir=self.workdir,
            )
        bad_validation = self._validation_builds()
        bad_validation["subject"] = {"effective_build_id": "only-one-field"}
        with self.assertRaises(pve.ValidationEvidenceError):
            self._make(validation_build_identities=bad_validation)

    def test_validation_subject_equal_to_campaign_tune_is_valid(self):
        campaign = self._campaign_builds()
        record = self._make(
            validation_build_identities={"control": _build_identity("c"), "subject": campaign["tune"]}
        )
        self.assertEqual(
            record["validation_build_identities"]["subject"], record["campaign_build_identities"]["tune"]
        )

    def test_validation_subject_different_from_campaign_tune_is_also_valid(self):
        record = self._make(validation_build_identities=self._validation_builds())
        self.assertNotEqual(
            record["validation_build_identities"]["subject"], record["campaign_build_identities"]["tune"]
        )

    def test_v3_legacy_top_level_build_identities_field_rejected_by_qualifier(self):
        record = self._make(validation_build_identities=self._validation_builds())
        # Simulate a hand-corrupted record carrying BOTH the legacy field
        # and the v3 fields -- must be rejected, never silently accepted.
        tampered = dict(record)
        tampered["build_identities"] = self._campaign_builds()
        ok, problems = pve._record_qualifies(
            tampered,
            module=_FakeModule("9999_example"),
            pinned_ref="b10692",
            subject_digest=pve.patch_validation_subject_digest(self.patch_path),
        )
        self.assertFalse(ok)
        self.assertTrue(any("legacy top-level build_identities" in p for p in problems))

    def test_record_digest_covers_both_new_domains(self):
        record = self._make(validation_build_identities=self._validation_builds())
        tampered_campaign = dict(record)
        tampered_campaign["campaign_build_identities"] = dict(record["campaign_build_identities"])
        tampered_campaign["campaign_build_identities"]["tune"] = _build_identity("changed")
        self.assertNotEqual(record["record_digest"], pve._record_digest(tampered_campaign))

        tampered_validation = dict(record)
        tampered_validation["validation_build_identities"] = dict(record["validation_build_identities"])
        tampered_validation["validation_build_identities"]["subject"] = _build_identity("changed")
        self.assertNotEqual(record["record_digest"], pve._record_digest(tampered_validation))


class WriteRecordSchemaUpgradeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patch_path = self.root / "9999_example.py"
        self.patch_path.write_text('STATE = "untested"\n', encoding="utf-8")
        self.workdir = self.root / "campaign"
        self.workdir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _v1_container(self, patch_id: str) -> Path:
        path = self.root / f"{patch_id}.json"
        path.write_text(json.dumps({
            "schema_version": 1, "patch_id": patch_id,
            "records": [{"campaign_identity_digest": "b" * 64, "note": "a real historical v1 record"}],
        }), encoding="utf-8")
        return path

    def test_appending_v3_to_a_v1_container_never_touches_the_old_record(self):
        path = self._v1_container("9999_example")
        before = json.loads(path.read_text(encoding="utf-8"))
        old_record = before["records"][0]

        activation_evidence = ActivationEvidence(status="executed", mechanism="m", detail="d")
        correctness = {"schema_version": 1, "disposition": "passed", "mechanism": "m", "detail": "d"}
        new_record = pve.make_record(
            patch_id="9999_example", patch_path=self.patch_path,
            patch_implementation_digest=_HEX64, base_ref="b10692", base_revision=_HEX40,
            framework_baseline_digest=_HEX64, patched_source_tree=_HEX40,
            gpu_architectures="gfx1100", activation_evidence=activation_evidence,
            activation_disposition="activation-verified", correctness=correctness,
            campaign_identity_digest=_HEX64,
            build_identities={"tune": _build_identity("t"), "replay": _build_identity("r"), "stock": _build_identity("s")},
            validation_build_identities={"control": _build_identity("c"), "subject": _build_identity("j")},
            campaign_workdir=self.workdir,
        )
        pve.write_record(new_record, root=self.root)

        after = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(after["schema_version"], 3) # container header upgraded
        self.assertIn(old_record, after["records"]) # old record object byte-unchanged
        self.assertEqual(len(after["records"]), 2)


class _FakeModule:
    def __init__(self, patch_id: str) -> None:
        self.patch_id = patch_id


if __name__ == "__main__":
    unittest.main()
