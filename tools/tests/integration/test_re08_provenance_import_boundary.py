"""RE08/RV48 + RE25.3: lane inputs must not launder provenance.

Contract layers (all verified here):

1. RE08/RV48: a raw-Path lane input used to be stamped with the CURRENT
   lane's source_slice_id -- an inventory from source A supplied while
   building source B silently acquired source-B provenance. Fixed by the
   imported-legacy classification.

2. RE25.3: the residual unearned claim is removed -- the imported-legacy
   document carries NO source_slice_id (the stage-envelope check exempts
   exactly that class; the sticky taint blocks promotion). Raw inputs are
   now descriptor-backed with a real rehydratable artifact_id.

3. RE25.3 identity handling: an ArtifactRef WITH an artifact_id has its
   claimed identity re-proven by store rehydration (kind + content hash
   must agree); an ArtifactRef WITHOUT one is legacy evidence and any
   first-party class claim is downgraded to imported-legacy so it cannot
   launder itself into production.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import config as campaign_config # noqa: E402
from bigcherry.core import provenance # noqa: E402
from bigcherry.core.artifacts import (  # noqa: E402
    ArtifactLocator,
    ArtifactStore,
)
from bigcherry.campaign.lane import (  # noqa: E402
    CampaignLaneError,
    CampaignLaneExecutionSpec,
    LaneInputValue,
    _resolve_lane_inputs,
)
from bigcherry.core.pipeline import ArtifactRef  # noqa: E402


def _build(needs: frozenset[str]) -> campaign_config.Build:
    return campaign_config.Build(
        name="tune", options=(), variant_set="workload-max", needs=needs
    )


def _spec(*inputs: tuple[str, LaneInputValue]) -> CampaignLaneExecutionSpec:
    return CampaignLaneExecutionSpec(
        source_name="bigcherry",
        build_name="tune",
        platform_name="linux-multi",
        architectures=("gfx1100",),
        inputs=inputs,
    )


def _resolve(store: ArtifactStore, *inputs: tuple[str, LaneInputValue]):
    return _resolve_lane_inputs(
        _spec(*inputs),
        build=_build(frozenset(n for n, _ in inputs)),
        store=store,
        run_id="run1",
    )


def _ns(doc: dict[str, object], name: str) -> dict[str, object]:
    """Narrow one provenance namespace (docs are dict[str, object])."""
    value = doc[name]
    assert isinstance(value, dict), f"namespace {name!r} is not a dict"
    return value


class RawPathImportDoesNotClaimUnearnedIdentityTests(unittest.TestCase):
    def test_plain_data_file_is_imported_legacy_with_no_source_claim_and_real_id(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(Path(d) / "store")
            inventory = Path(d) / "inventory.json"
            # A real inventory shape -- plain data, no provenance of its own.
            inventory.write_text(json.dumps({"mmq_types": ["q8_0"]}), encoding="utf-8")

            resolved = _resolve(store, ("inventory", inventory))
            ref = resolved["inventory"]
            doc = ref.provenance

            self.assertEqual(_ns(doc, "project")["provenance_class"], "imported-legacy")
            # RE25.3: NO source claim at all -- the previous version stamped
            # this lane's source_slice_id onto imported evidence, which was
            # itself an unearned identity (see test_wrong_source_*).
            self.assertIsNone(_ns(doc, "source").get("source_slice_id"))
            # Descriptor-backed: a real id that rehydrates in a fresh store.
            self.assertTrue(ref.artifact_id)
            fresh = ArtifactStore(Path(d) / "store")
            rehydrated = fresh.rehydrate(ref.artifact_id, expected_kind="inventory")
            self.assertEqual(rehydrated.content_hash, ref.content_hash)

    def test_wrong_source_inventory_is_not_relabelled_as_the_current_source(self):
        # THE exact laundering scenario RV48 named: an inventory from a
        # different source handed as a raw path while building source-B.
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(Path(d) / "store")
            inventory = Path(d) / "inventory.json"
            inventory.write_text(json.dumps({"mmq_types": ["f16"]}), encoding="utf-8")

            doc = _resolve(store, ("inventory", inventory))["inventory"].provenance
            self.assertEqual(_ns(doc, "project")["provenance_class"], "imported-legacy")
            # Must NOT acquire the current lane's identity (and source-B has
            # no slice id even in this fixture -- there is nothing to claim).
            self.assertIsNone(_ns(doc, "source").get("source_slice_id"))

    def test_crafted_json_blob_claiming_provenance_is_not_trusted(self):
        # A hand-crafted JSON blob that merely PARSES as a schema-v2
        # five-namespace document must not be accepted as real embedded
        # provenance -- no chain-of-custody primitive exists for raw paths,
        # so the raw-Path branch is unconditionally imported-legacy.
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(Path(d) / "store")
            spoofed = provenance.make(
                project={},
                source={"source_slice_id": "an-identity-the-attacker-picked"},
                build={"build_plan_id": "not-real"},
                workload={},
                campaign={"run_id": "not-real"},
            )
            inventory = Path(d) / "inventory.json"
            inventory.write_text(json.dumps(spoofed), encoding="utf-8")

            doc = _resolve(store, ("inventory", inventory))["inventory"].provenance
            self.assertEqual(_ns(doc, "project")["provenance_class"], "imported-legacy")
            # Neither the spoofed claim nor a fresh lane stamp may win.
            self.assertIsNone(_ns(doc, "source").get("source_slice_id"))


class ArtifactRefIdentityTests(unittest.TestCase):
    def test_ref_without_id_is_downgraded_to_imported_legacy(self):
        # Legacy evidence: store-verified bytes, but an unverified ref must
        # not carry a first-party class into the lane (no laundering).
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(Path(d) / "store")
            data = json.dumps({"mmq_types": ["q8_0"]}).encode()
            digest = store.publish_bytes("inputs/inventory/pre", data)
            doc = provenance.make(
                project={"provenance_class": "production"},
                source={"source_slice_id": "claimed-producer"},
                build={},
                workload={},
                campaign={"run_id": "run0"},
            )
            ref = ArtifactRef(
                kind="inventory",
                path=store.resolve("inputs/inventory/pre"),
                content_hash=digest,
                provenance=doc,
            )

            resolved = _resolve(store, ("inventory", ref))
            out = resolved["inventory"]
            self.assertEqual(out.content_hash, digest)  # same evidence...
            self.assertEqual(
                _ns(out.provenance, "project")["provenance_class"], "imported-legacy"
            )
            # ...but the class claim is downgraded; source fields survive so
            # a wrong-source claim still fails the envelope downstream.
            self.assertEqual(
                _ns(out.provenance, "source")["source_slice_id"], "claimed-producer"
            )
            # GPT audit fix (2026-08-18): RE25.3's locked contract is that
            # EVERY lane input comes out descriptor-backed -- this branch
            # used to leave artifact_id="" (content_hash-as-identity).
            # The persisted descriptor must rehydrate in a fresh store with
            # the downgraded class, not the original production claim.
            self.assertTrue(out.artifact_id)
            fresh = ArtifactStore(Path(d) / "store")
            rehydrated = fresh.rehydrate(out.artifact_id, expected_kind="inventory")
            self.assertEqual(rehydrated.content_hash, digest)
            self.assertEqual(
                _ns(rehydrated.provenance, "project")["provenance_class"],
                "imported-legacy",
            )

    def test_ref_with_matching_descriptor_identity_is_rehydrated(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(Path(d) / "store")
            data = json.dumps({"mmq_types": ["q8_0"]}).encode()
            published = store.publish_bytes_ref(
                "inputs/inventory/pre",
                data,
                kind="inventory",
                provenance=provenance.make(
                    project={"provenance_class": "development"},
                    source={"source_slice_id": "real-producer"},
                    build={},
                    workload={},
                    campaign={"run_id": "run0"},
                ),
            )
            # In-memory ref with the same ID but a LIEING in-memory doc:
            # the rehydrated store record must win, not the caller's copy.
            lying = ArtifactRef(
                kind="inventory",
                path=published.path,
                content_hash=published.content_hash,
                provenance=provenance.make(
                    project={},
                    source={"source_slice_id": "an-attacker-edit"},
                    build={},
                    workload={},
                    campaign={"run_id": "x"},
                ),
                artifact_id=published.artifact_id,
            )

            out = _resolve(store, ("inventory", lying))["inventory"]
            self.assertEqual(out.artifact_id, published.artifact_id)
            self.assertEqual(
                _ns(out.provenance, "project")["provenance_class"], "development"
            )
            self.assertEqual(
                _ns(out.provenance, "source")["source_slice_id"], "real-producer"
            )

    def test_ref_with_descriptor_identity_disagreeing_on_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(Path(d) / "store")
            data_a = json.dumps({"mmq_types": ["a"]}).encode()
            published = store.publish_bytes_ref(
                "inputs/inventory/pre",
                data_a,
                kind="inventory",
                provenance=provenance.make(
                    project={"provenance_class": "development"},
                    source={},
                    build={},
                    workload={},
                    campaign={},
                ),
            )
            data_b = json.dumps({"mmq_types": ["b"]}).encode()
            digest_b = store.publish_bytes("inputs/inventory/other", data_b)
            ref = ArtifactRef(
                kind="inventory",
                path=store.resolve("inputs/inventory/other"),
                content_hash=digest_b,
                provenance=provenance.make(
                    project={"provenance_class": "development"},
                    source={},
                    build={},
                    workload={},
                    campaign={},
                ),
                artifact_id=published.artifact_id,
            )
            with self.assertRaises(CampaignLaneError):
                _resolve(store, ("inventory", ref))

    def test_locator_rehydrates_and_locator_with_missing_descriptor_fails(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(Path(d) / "store")
            data = json.dumps({"mmq_types": ["q8_0"]}).encode()
            published = store.publish_bytes_ref(
                "inputs/inventory/pre",
                data,
                kind="inventory",
                provenance=provenance.make(
                    project={"provenance_class": "development"},
                    source={},
                    build={},
                    workload={},
                    campaign={},
                ),
            )

            out = _resolve(
                store, ("inventory", ArtifactLocator(published.artifact_id))
            )["inventory"]
            self.assertEqual(out.content_hash, published.content_hash)

            with self.assertRaises(CampaignLaneError):
                _resolve(store, ("inventory", ArtifactLocator("does-not-exist")))


if __name__ == "__main__":
    unittest.main()
