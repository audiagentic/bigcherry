"""RE14 parity arm loader: real campaign-published (ArtifactStore) output.

RE23 note: this file used to also cover ``load_legacy_arm`` (plain files at
conventional legacy-checkout paths) and cross-arm (legacy vs new) integration
tests. That loader and its tests were retired in RE23 along with the legacy
build path itself; the real-file integration coverage below now compares two
independently-published new-path arms instead, which still exercises
check_parity's fail-closed behavior against real on-disk/store artifacts
(the loader-independent parts of what those tests proved)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.artifacts import ArtifactError, ArtifactStore  # noqa: E402
from bigcherry.parity import ParityError, check_parity  # noqa: E402
from bigcherry.parity_loaders import load_new_arm  # noqa: E402

_MANIFEST = {
    "source_revision": "a" * 40, "variant_set": "workload-max",
    "manifest_hash": "m1", "candidates": [
        {"stable_name": "mmq:native:v1"}, {"stable_name": "mmvq:native:v1"},
    ],
    "build_descriptor": {"descriptor_hash": "d1"},
}


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

    def test_content_hash_check_rejects_manifest_tampered_on_disk(self):
        # RE14 step 6 negative case: a published artifact modified after
        # publication. ArtifactStore.resolve() alone would not catch this --
        # it only checks the file exists, not that its bytes still match
        # what was published. Real-world equivalent: something edits
        # manifest.json directly on the filesystem (bypassing
        # ArtifactStore's own write path) between generate publishing it and
        # a parity comparison reading it back.
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            published_hash = store.publish_json("manifest.json", _MANIFEST)
            store.publish_bytes("llama-bench", b"fake-binary-bytes")

            # Tamper directly on disk, outside ArtifactStore's own API --
            # simulates corruption/modification after publication, not a
            # legitimate second publish (which publish_json would itself
            # reject if the bytes differed).
            tampered = dict(_MANIFEST)
            tampered["candidates"] = [{"stable_name": "mmq:native:v1"}]  # dropped one
            (Path(directory) / "manifest.json").write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaises(ArtifactError) as ctx:
                load_new_arm(
                    "new", store=store, manifest_relative="manifest.json",
                    binary_relative="llama-bench", manifest_content_hash=published_hash)
            self.assertIn("content_hash", str(ctx.exception))

    def test_without_content_hash_tampered_manifest_is_silently_trusted(self):
        # Documents the gap the previous test closes: omitting
        # manifest_content_hash means load_new_arm has no way to notice the
        # same tamper. Any real caller building a parity harness on this
        # loader MUST pass the manifest's ArtifactRef.content_hash -- this
        # test exists so that fact stays enforced by something other than a
        # docstring if the loader's behavior ever changes.
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            store.publish_json("manifest.json", _MANIFEST)
            store.publish_bytes("llama-bench", b"fake-binary-bytes")

            tampered = dict(_MANIFEST)
            tampered["candidates"] = [{"stable_name": "mmq:native:v1"}]
            (Path(directory) / "manifest.json").write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            arm = load_new_arm(
                "new", store=store, manifest_relative="manifest.json",
                binary_relative="llama-bench")
            self.assertEqual(arm.candidate_names, frozenset({"mmq:native:v1"}))


class ParityGateRealFileIntegrationTests(unittest.TestCase):
    """RE14 step 6: prove the loader + check_parity combination fails
    closed against real on-disk artifacts, not just in-memory CampaignArm
    objects (test_parity.py already covers the pure-function cases). Both
    arms are independently-published ArtifactStores (RE23: the legacy-arm
    loader this used to also exercise is gone) -- what matters here is that
    check_parity's fail-closed behavior holds against real files/stores, not
    which loader produced them."""

    def test_matching_real_arms_pass(self):
        with tempfile.TemporaryDirectory() as dir_a, \
             tempfile.TemporaryDirectory() as dir_b:
            store_a = ArtifactStore(Path(dir_a))
            hash_a = store_a.publish_json("manifest.json", _MANIFEST)
            store_a.publish_bytes("llama-bench", b"identical-binary-bytes")
            arm_a = load_new_arm(
                "a", store=store_a, manifest_relative="manifest.json",
                binary_relative="llama-bench", manifest_content_hash=hash_a)

            store_b = ArtifactStore(Path(dir_b))
            hash_b = store_b.publish_json("manifest.json", _MANIFEST)
            store_b.publish_bytes("llama-bench", b"identical-binary-bytes")
            arm_b = load_new_arm(
                "b", store=store_b, manifest_relative="manifest.json",
                binary_relative="llama-bench", manifest_content_hash=hash_b)

            report = check_parity(arm_a, arm_b, label="tune")
            self.assertEqual(report.missing_candidates, frozenset())
            self.assertEqual(report.extra_candidates, frozenset())

    def test_real_binary_divergence_fails_closed(self):
        # The genuinely dangerous case for a parity gate: everything about
        # the manifest/descriptor agrees, but the two sides produced a
        # different binary. This must reject even though every JSON field
        # matches -- a gate that only checked manifest/descriptor content
        # would rubber-stamp a real build divergence.
        with tempfile.TemporaryDirectory() as dir_a, \
             tempfile.TemporaryDirectory() as dir_b:
            store_a = ArtifactStore(Path(dir_a))
            hash_a = store_a.publish_json("manifest.json", _MANIFEST)
            store_a.publish_bytes("llama-bench", b"arm-a-binary-bytes")
            arm_a = load_new_arm(
                "a", store=store_a, manifest_relative="manifest.json",
                binary_relative="llama-bench", manifest_content_hash=hash_a)

            store_b = ArtifactStore(Path(dir_b))
            hash_b = store_b.publish_json("manifest.json", _MANIFEST)
            store_b.publish_bytes("llama-bench", b"arm-b-binary-bytes")
            arm_b = load_new_arm(
                "b", store=store_b, manifest_relative="manifest.json",
                binary_relative="llama-bench", manifest_content_hash=hash_b)

            with self.assertRaises(ParityError) as ctx:
                check_parity(arm_a, arm_b, label="tune")
            self.assertIn("binary_hash", str(ctx.exception))

    def test_real_descriptor_divergence_fails_closed(self):
        with tempfile.TemporaryDirectory() as dir_a, \
             tempfile.TemporaryDirectory() as dir_b:
            store_a = ArtifactStore(Path(dir_a))
            hash_a = store_a.publish_json("manifest.json", _MANIFEST)
            store_a.publish_bytes("llama-bench", b"identical-binary-bytes")
            arm_a = load_new_arm(
                "a", store=store_a, manifest_relative="manifest.json",
                binary_relative="llama-bench", manifest_content_hash=hash_a)

            divergent_manifest = dict(_MANIFEST)
            divergent_manifest["build_descriptor"] = {"descriptor_hash": "d2"}
            store_b = ArtifactStore(Path(dir_b))
            hash_b = store_b.publish_json("manifest.json", divergent_manifest)
            store_b.publish_bytes("llama-bench", b"identical-binary-bytes")
            arm_b = load_new_arm(
                "b", store=store_b, manifest_relative="manifest.json",
                binary_relative="llama-bench", manifest_content_hash=hash_b)

            with self.assertRaises(ParityError) as ctx:
                check_parity(arm_a, arm_b, label="tune")
            self.assertIn("descriptor.descriptor_hash", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
