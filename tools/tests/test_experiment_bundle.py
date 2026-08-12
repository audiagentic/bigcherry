"""Managed current experiment bundle tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import experiment_bundle  # noqa: E402


class ExperimentBundleTests(unittest.TestCase):
    def run_bundle(self, root: Path, command: list[str] | None = None) -> Path:
        model = root / "model.gguf"
        model.write_bytes(b"model")
        bundle = root / "bundle"
        status = experiment_bundle.run_managed(
            bundle, command or [sys.executable, "-c", "print('ok')"],
            source_revision="a" * 40, manifest_hash="b" * 32,
            build_descriptor_hash="c" * 32, model=model, role="test",
        )
        self.assertEqual(status, 0)
        return bundle

    def test_run_finalizes_and_validates_hashed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.run_bundle(Path(directory))
            result = experiment_bundle.validate(bundle, required_capabilities={"process_evidence"})
            self.assertTrue(result["promotable"])
            (bundle / "stdout.log").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(experiment_bundle.BundleError, "modified"):
                experiment_bundle.validate(bundle)

    def test_launch_failure_is_a_finalized_nonpromotable_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"model")
            bundle = root / "bundle"
            status = experiment_bundle.run_managed(
                bundle, [str(root / "does-not-exist")], source_revision="a" * 40,
                manifest_hash="b" * 32, build_descriptor_hash="c" * 32,
                model=model, role="test",
            )
            self.assertNotEqual(status, 0)
            result = experiment_bundle.validate(bundle)
            self.assertEqual(result["state"], "failed")
            self.assertFalse(result["promotable"])

    def test_unknown_schema_absolute_artifact_and_missing_capability_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.run_bundle(Path(directory))
            with self.assertRaisesRegex(experiment_bundle.BundleError, "capability-incomplete"):
                experiment_bundle.validate(bundle, required_capabilities={"raw_rounds"})
            document_path = bundle / "experiment.json"
            document = json.loads(document_path.read_text(encoding="ascii"))
            document["schema_version"] = 2
            experiment_bundle.write_document(document_path, document)
            with self.assertRaisesRegex(experiment_bundle.BundleError, "external_conversion"):
                experiment_bundle.validate(bundle)

    def test_parent_cycle_rejected(self):
        with self.assertRaisesRegex(experiment_bundle.BundleError, "cycle"):
            experiment_bundle.validate_parent_graph([
                {"experiment_id": "a", "parent_experiment_id": "b"},
                {"experiment_id": "b", "parent_experiment_id": "a"},
            ])

    def test_managed_run_cli_uses_only_allowlisted_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"model")
            bundle = root / "bundle"
            status = experiment_bundle.main([
                "run", str(bundle), "--source-revision", "a" * 40,
                "--manifest-hash", "b" * 32,
                "--build-descriptor-hash", "c" * 32,
                "--model", str(model), "--role", "test",
                "--env", "GGML_HIP_DISPATCH_MODE=native", "--",
                sys.executable, "-c", "print('ok')",
            ])
            self.assertEqual(status, 0)
            document = json.loads((bundle / "experiment.json").read_text(encoding="ascii"))
            self.assertEqual(document["environment"]["GGML_HIP_DISPATCH_MODE"], "native")


if __name__ == "__main__":
    unittest.main()
