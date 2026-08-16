"""RE14 parity arm loaders: legacy (plain files) and new (ArtifactStore)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.artifacts import ArtifactStore  # noqa: E402
from bigcherry.parity_loaders import load_legacy_arm, load_new_arm  # noqa: E402

_MANIFEST = {
    "source_revision": "a" * 40, "variant_set": "workload-max",
    "manifest_hash": "m1", "candidates": [
        {"stable_name": "mmq:native:v1"}, {"stable_name": "mmvq:native:v1"},
    ],
    "build_descriptor": {"descriptor_hash": "d1"},
}
_DESCRIPTOR = {"descriptor_hash": "d1"}


class LoadLegacyArmTests(unittest.TestCase):
    def test_reads_real_files_into_a_campaign_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "hip-autotune-manifest.json"
            manifest_path.write_text(json.dumps(_MANIFEST), encoding="utf-8")
            descriptor_path = root / "hip-autotune-build-descriptor.json"
            descriptor_path.write_text(json.dumps(_DESCRIPTOR), encoding="utf-8")
            binary_path = root / "llama-bench"
            binary_path.write_bytes(b"fake-binary-bytes")

            arm = load_legacy_arm(
                "legacy", manifest_path=manifest_path, descriptor_path=descriptor_path,
                binary_path=binary_path)
            self.assertEqual(arm.name, "legacy")
            self.assertEqual(arm.manifest["manifest_hash"], "m1")
            self.assertEqual(arm.descriptor["descriptor_hash"], "d1")
            self.assertEqual(arm.candidate_names, frozenset({"mmq:native:v1", "mmvq:native:v1"}))
            self.assertEqual(len(arm.binary_hash), 64)  # sha256 hex


class LoadNewArmTests(unittest.TestCase):
    def test_reads_through_artifact_store(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            store.publish_json("manifest.json", _MANIFEST)
            store.publish_bytes("llama-bench", b"fake-binary-bytes")

            arm = load_new_arm(
                "new", store=store, manifest_relative="manifest.json",
                binary_relative="llama-bench")
            self.assertEqual(arm.name, "new")
            self.assertEqual(arm.manifest["manifest_hash"], "m1")
            # descriptor comes from manifest["build_descriptor"], not a
            # separate file -- the new path embeds it, matching how
            # autotune_catalog.build_manifest() actually structures it.
            self.assertEqual(arm.descriptor["descriptor_hash"], "d1")
            self.assertEqual(arm.candidate_names, frozenset({"mmq:native:v1", "mmvq:native:v1"}))
            self.assertEqual(len(arm.binary_hash), 64)

    def test_missing_artifact_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            with self.assertRaises(Exception):
                load_new_arm(
                    "new", store=store, manifest_relative="does-not-exist.json",
                    binary_relative="llama-bench")


if __name__ == "__main__":
    unittest.main()
