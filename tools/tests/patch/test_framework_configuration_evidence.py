import copy
import json
import tempfile
import unittest
from pathlib import Path

from bigcherry.patch import evidence
from bigcherry.patch import patchset, registry
from bigcherry.build.generated_tree import build_manifest


HEX = "a" * 64
IDENTITY = {
    "effective_build_id": "build", "compile_verification_id": "verify",
    "compile_commands_digest": HEX, "hip_compile_commands_digest": HEX,
    "runtime_bundle_hash": HEX, "runtime_artifacts": {"binary": HEX},
}


class FrameworkConfigurationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        cls.descriptor = registry.load_registry(cls.root / "patches").get("0100_cmake_options")
        cls.module = next(m for m in patchset.catalog(cls.root / "patches") if m.patch_id == "0100_cmake_options")
        from bigcherry.core import config
        from bigcherry.campaign import resolution
        catalog = patchset.catalog(cls.root / "patches")
        lane = resolution.resolve_lane("bigcherry-native", config.load(cls.root / "config/recipes.toml"), catalog)
        resolved = patchset.resolve_exact(tuple(lane.patch_set.module_ids), directory=cls.root / "patches")
        descriptors = registry.load_registry(cls.root / "patches")
        cls.composition = tuple((member.patch_id, descriptors.get(member.patch_id).implementation_digest)
                                for member in resolved.modules)

    def _record(self, directory):
        patch = self.root / "patches/0100_cmake_options/patch.py"
        generated = Path(directory) / "generated"
        generated.mkdir()
        header = generated / "fixture.inc"
        header.write_text("#define BIGCHERRY_FIXTURE 1\n", encoding="utf-8")
        manifest = build_manifest(generated, compile_inputs=(header,))
        return evidence.make_framework_configuration_record(
            descriptor=self.descriptor, patch_path=patch, base_ref="bigcherry", base_revision="b" * 40,
            source_name="bigcherry-native", source_composition=self.composition, source_tree="c" * 40,
            source_slice_id="a" * 32, compiled_targets=("gfx1100",),
            builds={"production": IDENTITY, "diagnostic": IDENTITY},
            generated_inputs={
                role: {
                    "proof": "compiled-copy-v1", "compile_inputs_hash": manifest["compile_inputs_hash"],
                    "tree_manifest": manifest,
                    "build_identity": IDENTITY,
                } for role in ("production", "diagnostic")
            },
            check_results={k: {"status": "pass", "capability": ("apply" if k == "apply" else "build" if k == "build" else "configuration"), "check_id": k, "artifacts": ({"path": "report", "sha256": HEX},)}
                           for k in ("apply", "build", "coverage-source-selection")},
            artifact_hashes={"report": HEX}, campaign_workdir=Path(directory),
        )

    def test_constructor_is_schema5_configuration_only(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self._record(directory)
            self.assertEqual(record["record_schema_version"], 5)
            self.assertEqual(record["qualification_kind"], "framework-configuration-v1")
            self.assertFalse(record["runtime_performance_qualified"])
            self.assertFalse(record["hardware_execution_qualified"])
            self.assertTrue(record["eligible_for_validated_state"])
            self.assertEqual(record["record_digest"], evidence._record_digest(record))

    def test_real_framework_record_verifies_and_wrapper_has_distinct_status(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self._record(directory)
            ok, problems = evidence.verify_framework_configuration_record(
                record, descriptor=self.descriptor,
                patch_path=self.module.path, pinned_ref="bigcherry",
                required_compiled_targets=("gfx1100",), resolved_base_revision="b" * 40,
                source_composition=self.composition,
            )
            self.assertTrue(ok, problems)
            root = Path(directory) / "evidence"
            root.mkdir()
            (root / "0100_cmake_options.json").write_text(
                __import__("json").dumps({"schema_version": 5, "patch_id": "0100_cmake_options", "records": [record]}),
                encoding="utf-8",
            )
            status = evidence.verify_framework_configuration_patch(
                self.module, pinned_ref="bigcherry", required_compiled_targets=("gfx1100",), root=root,
                allow_legacy_grandfather=False, resolved_base_revision="b" * 40,
            )
            self.assertEqual(status.status, "framework-configuration-evidence")
            self.assertTrue(status.ok)

    def test_verifier_rejects_stale_identity_runtime_claim_and_missing_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self._record(directory)
            for field, value, expected in (
                ("patch_implementation_digest", "d" * 64, "implementation digest"),
                ("validation_digest", "e" * 64, "validation digest"),
                ("base_ref", "other", "stale base_ref"),
                ("base_revision", "f" * 40, "stale base_revision"),
                ("source_composition", [{"id": "other", "digest": HEX}], "source composition"),
            ):
                candidate = dict(base); candidate[field] = value
                candidate["record_digest"] = evidence._record_digest(candidate)
                ok, problems = evidence.verify_framework_configuration_record(
                    candidate, descriptor=self.descriptor, patch_path=self.module.path,
                    pinned_ref="bigcherry", required_compiled_targets=("gfx1100",),
                    resolved_base_revision="b" * 40,
                    source_composition=tuple((name, HEX) for name in ("0100_cmake_options", "0200_dispatch_hook")),
                )
                self.assertFalse(ok)
                self.assertTrue(any(expected in problem for problem in problems), problems)
            missing = dict(base)
            missing["check_results"] = {"apply": base["check_results"]["apply"]}
            missing["record_digest"] = evidence._record_digest(missing)
            ok, problems = evidence.verify_framework_configuration_record(
                missing, descriptor=self.descriptor, patch_path=self.module.path,
                pinned_ref="bigcherry", required_compiled_targets=("gfx1100",),
                resolved_base_revision="b" * 40,
            )
            self.assertFalse(ok)
            self.assertIn("required manifest check missing", problems)

    def test_verifier_rejects_malformed_generated_attestation_and_runtime_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self._record(directory)
            bad = dict(record)
            bad["runtime_performance_qualified"] = True
            bad["record_digest"] = evidence._record_digest(bad)
            ok, problems = evidence.verify_framework_configuration_record(
                bad, descriptor=self.descriptor, patch_path=self.module.path,
                pinned_ref="bigcherry", required_compiled_targets=("gfx1100",),
            )
            self.assertFalse(ok); self.assertIn("forbidden runtime", " ".join(problems))
            bad = dict(record)
            bad["generated_inputs"] = {"production": {"proof": "wrong"}, "diagnostic": {"proof": "wrong"}}
            bad["record_digest"] = evidence._record_digest(bad)
            ok, problems = evidence.verify_framework_configuration_record(
                bad, descriptor=self.descriptor, patch_path=self.module.path,
                pinned_ref="bigcherry", required_compiled_targets=("gfx1100",),
            )
            self.assertFalse(ok); self.assertIn("generated_inputs.production invalid", problems)


if __name__ == "__main__":
    unittest.main()

