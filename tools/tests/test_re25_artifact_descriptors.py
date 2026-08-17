"""RE25.1: ArtifactDescriptor persistence/rehydration + typed ProvenanceV2 +
kind-specific promotion contracts + sticky provenance-class taint.

GPT-auto-agent implementation plan (2026-08-17). This is the first,
additive layer -- ArtifactRef.provenance stays a plain dict everywhere
else in the codebase; ProvenanceV2.document() produces exactly that dict
shape, so nothing outside this module needs to change for these tests to
be meaningful. Real ArtifactStore, no mocking.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.artifacts import ArtifactDescriptor, ArtifactError, ArtifactStore  # noqa: E402
from bigcherry.provenance import (  # noqa: E402
    BuildProvenance, CampaignProvenance, ProjectProvenance, ProvenanceError,
    ProvenanceV2, SourceProvenance, WorkloadProvenance, derive, derived_provenance_class,
    require_promotable, validate_for_kind,
)


def _production_source() -> SourceProvenance:
    return SourceProvenance(
        upstream_revision="rev1", source_plan_id="plan1", materialization_plan_id="mat1",
        source_tree_oid="tree1", source_slice_id="slice1", git_object_format="sha1",
        patch_set_id="patchset1",
    )


def _production_build() -> BuildProvenance:
    return BuildProvenance(
        build_plan_id="build1", effective_build_id="eff1", binary_hash="binhash1",
        runtime_bundle_hash="bundlehash1", targets=("gfx1100",),
    )


def _production_provenance(*, kind_extra: dict[str, object] | None = None) -> ProvenanceV2:
    return ProvenanceV2(
        schema_version=2,
        project=ProjectProvenance(provenance_class="production", bigcherry_revision="rev-abc"),
        source=_production_source(), build=_production_build(),
        workload=WorkloadProvenance(workload_id="w1"),
        campaign=CampaignProvenance(run_id="run1", producer_stage="build"),
    )


class ProvenanceV2RoundTripTests(unittest.TestCase):
    def test_document_and_from_document_round_trip(self):
        original = _production_provenance()
        restored = ProvenanceV2.from_document(original.document())
        self.assertEqual(original, restored)

    def test_from_document_rejects_wrong_schema_version(self):
        doc = _production_provenance().document()
        doc["schema_version"] = 1
        with self.assertRaises(ProvenanceError):
            ProvenanceV2.from_document(doc)

    def test_from_document_rejects_unknown_provenance_class(self):
        doc = _production_provenance().document()
        doc["project"]["provenance_class"] = "not-a-real-class"
        with self.assertRaises(ProvenanceError):
            ProvenanceV2.from_document(doc)


class StickyTaintTests(unittest.TestCase):
    def test_derived_class_is_production_when_all_parents_are_production(self):
        parents = (_production_provenance(), _production_provenance())
        self.assertEqual(derived_provenance_class(parents), "production")

    def test_imported_legacy_parent_taints_the_derived_artifact(self):
        imported = ProvenanceV2(
            schema_version=2,
            project=ProjectProvenance(provenance_class="imported-legacy"),
            source=SourceProvenance(), build=BuildProvenance(),
            workload=WorkloadProvenance(), campaign=CampaignProvenance(run_id="run1"),
        )
        parents = (_production_provenance(), imported)
        self.assertEqual(derived_provenance_class(parents), "imported-legacy")

    def test_local_non_production_class_wins_regardless_of_parents(self):
        parents = (_production_provenance(),)
        self.assertEqual(
            derived_provenance_class(parents, local_class="development"), "development")

    def test_derive_builds_a_real_provenance_document_with_sticky_class(self):
        imported = ProvenanceV2(
            schema_version=2,
            project=ProjectProvenance(provenance_class="imported-legacy"),
            source=SourceProvenance(), build=BuildProvenance(),
            workload=WorkloadProvenance(), campaign=CampaignProvenance(run_id="run0"),
        )
        derived = derive(
            parents=(imported,), parent_artifact_ids=("parent-1",),
            project_revision="rev-abc", source=_production_source(), build=_production_build(),
            workload=WorkloadProvenance(workload_id="w1"), run_id="run1", producer_stage="build",
        )
        self.assertEqual(derived.project.provenance_class, "imported-legacy")
        self.assertEqual(derived.campaign.producer_artifact_ids, ("parent-1",))


class KindContractTests(unittest.TestCase):
    def test_production_runtime_bundle_with_full_fields_passes(self):
        doc = _production_provenance().document()
        require_promotable(doc, kind="runtime-bundle")

    def test_production_runtime_bundle_missing_a_required_field_is_rejected(self):
        provenance = _production_provenance()
        incomplete = ProvenanceV2(
            schema_version=2, project=provenance.project,
            source=SourceProvenance(),  # missing everything
            build=provenance.build, workload=provenance.workload, campaign=provenance.campaign,
        )
        with self.assertRaises(ProvenanceError):
            require_promotable(incomplete.document(), kind="runtime-bundle")

    def test_imported_legacy_with_empty_fields_validates_but_is_not_promotable(self):
        imported = ProvenanceV2(
            schema_version=2,
            project=ProjectProvenance(provenance_class="imported-legacy"),
            source=SourceProvenance(), build=BuildProvenance(),
            workload=WorkloadProvenance(), campaign=CampaignProvenance(run_id="run1"),
        )
        # Does not raise -- imported-legacy is exempt from the kind's
        # required-field contract (it never claimed a real identity).
        validate_for_kind(imported.document(), kind="runtime-bundle")
        with self.assertRaises(ProvenanceError):
            require_promotable(imported.document(), kind="runtime-bundle")

    def test_unknown_kind_has_no_registered_contract(self):
        with self.assertRaises(ProvenanceError):
            require_promotable(_production_provenance().document(), kind="not-a-real-kind")


class ArtifactDescriptorTests(unittest.TestCase):
    def test_descriptor_round_trips_through_a_fresh_store_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root)
            ref = store.publish_json_ref(
                "runs/r1/inventory.json", {"mmq_types": ["q8_0"]},
                kind="inventory", provenance=_production_provenance())
            self.assertTrue(ref.artifact_id)

            # A genuinely separate process would construct a new
            # ArtifactStore against the same root -- prove that works.
            fresh_store = ArtifactStore(root)
            rehydrated = fresh_store.rehydrate(ref.artifact_id, expected_kind="inventory")
            self.assertEqual(rehydrated.content_hash, ref.content_hash)
            self.assertEqual(rehydrated.provenance, ref.provenance)
            self.assertEqual(rehydrated.artifact_id, ref.artifact_id)

    def test_tampering_the_published_bytes_fails_rehydration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root)
            ref = store.publish_json_ref(
                "runs/r1/inventory.json", {"mmq_types": ["q8_0"]},
                kind="inventory", provenance=_production_provenance())
            ref.path.write_text('{"mmq_types": ["tampered"]}', encoding="utf-8")

            with self.assertRaises(ArtifactError):
                store.rehydrate(ref.artifact_id)

    def test_tampering_the_descriptor_content_fails_rehydration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root)
            ref = store.publish_json_ref(
                "runs/r1/inventory.json", {"mmq_types": ["q8_0"]},
                kind="inventory", provenance=_production_provenance())

            descriptor_path = root / f".descriptors/v1/{ref.artifact_id[:2]}/{ref.artifact_id}.json"
            import json as _json
            document = _json.loads(descriptor_path.read_text(encoding="utf-8"))
            # Edit the provenance INSIDE the descriptor but leave its
            # filename/claimed artifact_id alone -- the whole point of
            # recomputing artifact_id on load is to catch exactly this.
            document["provenance"]["source"]["source_slice_id"] = "forged-slice-id"
            descriptor_path.write_text(_json.dumps(document), encoding="utf-8")

            with self.assertRaises(ArtifactError):
                store.rehydrate(ref.artifact_id)

    def test_descriptor_with_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            bogus_provenance = _production_provenance()
            descriptor = ArtifactDescriptor.create(
                kind="inventory", relative_path="../../etc/passwd",
                content_hash=ArtifactStore.digest(b"whatever"), provenance=bogus_provenance)
            with self.assertRaises(ArtifactError):
                store.persist_descriptor(descriptor)

    def test_two_descriptors_can_share_one_byte_object_with_different_run_ids(self):
        # One immutable build byte object legitimately participating in
        # two different campaign runs' provenance chains -- both
        # descriptors must coexist with distinct artifact_ids.
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            data = b"shared binary bytes"
            digest = store.publish_bytes("builds/shared/binary", data)

            provenance_a = ProvenanceV2(
                schema_version=2, project=ProjectProvenance(provenance_class="production"),
                source=_production_source(), build=_production_build(),
                workload=WorkloadProvenance(), campaign=CampaignProvenance(run_id="run-a"),
            )
            provenance_b = ProvenanceV2(
                schema_version=2, project=ProjectProvenance(provenance_class="production"),
                source=_production_source(), build=_production_build(),
                workload=WorkloadProvenance(), campaign=CampaignProvenance(run_id="run-b"),
            )
            descriptor_a = ArtifactDescriptor.create(
                kind="binary", relative_path="builds/shared/binary",
                content_hash=digest, provenance=provenance_a)
            descriptor_b = ArtifactDescriptor.create(
                kind="binary", relative_path="builds/shared/binary",
                content_hash=digest, provenance=provenance_b)
            self.assertNotEqual(descriptor_a.artifact_id, descriptor_b.artifact_id)

            store.persist_descriptor(descriptor_a)
            store.persist_descriptor(descriptor_b)

            ref_a = store.rehydrate(descriptor_a.artifact_id)
            ref_b = store.rehydrate(descriptor_b.artifact_id)
            self.assertEqual(ref_a.content_hash, ref_b.content_hash)
            self.assertEqual(ref_a.provenance["campaign"]["run_id"], "run-a")
            self.assertEqual(ref_b.provenance["campaign"]["run_id"], "run-b")

    def test_rehydrate_require_promotable_rejects_imported_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            imported = ProvenanceV2(
                schema_version=2,
                project=ProjectProvenance(provenance_class="imported-legacy"),
                source=SourceProvenance(), build=BuildProvenance(),
                workload=WorkloadProvenance(), campaign=CampaignProvenance(run_id="run1"),
            )
            ref = store.publish_json_ref(
                "runs/r1/inventory.json", {"a": 1}, kind="inventory", provenance=imported)

            # Ordinary rehydrate succeeds (it's a real, verifiable artifact --
            # just not promotable evidence).
            store.rehydrate(ref.artifact_id)
            with self.assertRaises(ArtifactError):
                store.rehydrate(ref.artifact_id, require_promotable=True)

    def test_rehydrate_wrong_expected_kind_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            ref = store.publish_json_ref(
                "runs/r1/inventory.json", {"a": 1},
                kind="inventory", provenance=_production_provenance())
            with self.assertRaises(ArtifactError):
                store.rehydrate(ref.artifact_id, expected_kind="dispatch-db")


if __name__ == "__main__":
    unittest.main()
