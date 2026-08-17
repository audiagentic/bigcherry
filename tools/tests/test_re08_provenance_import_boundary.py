"""RE08 (RV48 audit fix): raw-Path lane inputs must not launder provenance.

The verified defect: _resolve_lane_inputs's raw-Path branch used to stamp
a freshly-invented provenance document claiming the CURRENT lane's
source_slice_id for a file it had no way to prove came from that source
-- an inventory genuinely produced by building source A, supplied as a
raw path while building source B, silently acquired source-B provenance.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import config as campaign_config  # noqa: E402
from bigcherry import provenance  # noqa: E402
from bigcherry.artifacts import ArtifactStore  # noqa: E402
from bigcherry.campaign_lane import _resolve_lane_inputs  # noqa: E402
from bigcherry.campaign_lane import CampaignLaneExecutionSpec  # noqa: E402


def _build(needs: frozenset[str]) -> campaign_config.Build:
    return campaign_config.Build(name="tune", options=(), variant_set="workload-max", needs=needs)


class RawPathImportDoesNotClaimUnearnedIdentityTests(unittest.TestCase):
    def test_a_plain_data_file_is_classified_imported_legacy_not_stamped_as_produced(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "store")
            inventory = Path(directory) / "inventory.json"
            # A real inventory shape -- plain data, no provenance of its own
            # -- exactly what a raw --inventory CLI import actually is.
            inventory.write_text(json.dumps({"mmq_types": ["q8_0"]}), encoding="utf-8")

            spec = CampaignLaneExecutionSpec(
                source_name="bigcherry", build_name="tune", platform_name="linux-multi",
                architectures=("gfx1100",), inputs=(("inventory", inventory),))
            resolved = _resolve_lane_inputs(
                spec, build=_build(frozenset({"inventory"})), store=store,
                source_slice_id="source-B-slice-id", run_id="run1")

            doc = resolved["inventory"].provenance
            self.assertEqual(doc["project"].get("provenance_class"), "imported-legacy")
            # The visible classification is the point -- a downstream reader
            # can now tell "this claim is unproven" instead of trusting it
            # as if the lane's own materialize/generate stage produced it.

    def test_wrong_source_inventory_is_not_silently_relabelled_as_the_current_source(self):
        # This IS the exact laundering scenario RV48 named: an inventory
        # that genuinely came from a DIFFERENT source is handed as a raw
        # path while building source-B. Assert the result does not claim
        # unqualified production-quality provenance for source-B -- it
        # must be visibly imported-legacy, not indistinguishable from a
        # real generate-stage output.
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "store")
            inventory = Path(directory) / "inventory.json"
            inventory.write_text(json.dumps({"mmq_types": ["f16"]}), encoding="utf-8")

            spec = CampaignLaneExecutionSpec(
                source_name="bigcherry", build_name="tune", platform_name="linux-multi",
                architectures=("gfx1100",), inputs=(("inventory", inventory),))
            resolved = _resolve_lane_inputs(
                spec, build=_build(frozenset({"inventory"})), store=store,
                source_slice_id="source-B-slice-id", run_id="run1")

            doc = resolved["inventory"].provenance
            self.assertEqual(doc["project"]["provenance_class"], "imported-legacy")

    def test_a_crafted_json_blob_claiming_to_be_provenance_is_not_trusted(self):
        # GPT-auto-agent review (RV48 follow-up, 2026-08-17): an earlier
        # version of this fix trusted any raw-Path bytes that merely
        # PARSED as a schema_version==2, five-namespace document as "real
        # embedded provenance" -- provenance.validate() is a structural
        # shape check only, so a hand-crafted JSON blob asserting whatever
        # source_slice_id it likes would have been accepted as real. That
        # is still a laundering route, just a smaller one. Verifying a
        # raw file's own embedded identity for real needs a chain-of-
        # custody primitive this project does not have yet (RE25b) -- until
        # then, a raw-Path input is unconditionally imported-legacy, even
        # when its bytes happen to look exactly like a provenance document.
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "store")
            spoofed_doc = provenance.make(
                project={}, source={"source_slice_id": "an-identity-the-attacker-picked"},
                build={"build_plan_id": "not-real"}, workload={}, campaign={"run_id": "not-real"})
            inventory = Path(directory) / "inventory.json"
            inventory.write_text(json.dumps(spoofed_doc), encoding="utf-8")

            spec = CampaignLaneExecutionSpec(
                source_name="bigcherry", build_name="tune", platform_name="linux-multi",
                architectures=("gfx1100",), inputs=(("inventory", inventory),))
            resolved = _resolve_lane_inputs(
                spec, build=_build(frozenset({"inventory"})), store=store,
                source_slice_id="source-B-slice-id", run_id="run1")

            doc = resolved["inventory"].provenance
            # The spoofed claim must NOT win -- the real lane identity and
            # the imported-legacy classification are what get recorded.
            self.assertEqual(doc["source"]["source_slice_id"], "source-B-slice-id")
            self.assertEqual(doc["project"]["provenance_class"], "imported-legacy")

    def test_artifactref_input_path_is_unaffected_by_the_import_boundary_fix(self):
        # An already-published ArtifactRef takes a completely different
        # branch (store.verify against its OWN recorded content_hash) --
        # this fix must not touch that path at all.
        with tempfile.TemporaryDirectory() as directory:
            from bigcherry.pipeline import ArtifactRef
            store = ArtifactStore(Path(directory) / "store")
            data = json.dumps({"mmq_types": ["q8_0"]}).encode()
            digest = store.publish_bytes("inputs/inventory/pre-published", data)
            real_doc = provenance.make(
                project={}, source={"source_slice_id": "real-producer"},
                build={}, workload={}, campaign={"run_id": "run0"})
            ref = ArtifactRef(kind="inventory", path=store.resolve("inputs/inventory/pre-published"),
                              content_hash=digest, provenance=real_doc)

            spec = CampaignLaneExecutionSpec(
                source_name="bigcherry", build_name="tune", platform_name="linux-multi",
                architectures=("gfx1100",), inputs=(("inventory", ref),))
            resolved = _resolve_lane_inputs(
                spec, build=_build(frozenset({"inventory"})), store=store,
                source_slice_id="source-B-slice-id", run_id="run1")

            self.assertIs(resolved["inventory"], ref)
            self.assertEqual(resolved["inventory"].provenance["source"]["source_slice_id"],
                             "real-producer")


if __name__ == "__main__":
    unittest.main()
