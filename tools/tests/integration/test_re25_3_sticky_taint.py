# Copyright (c) 2026 Audiogentic.
# SPDX-License-Identifier: MIT
"""RE25.3 step 3.5 — end-to-end sticky taint, envelope exemption, and the
publish-time kind-contract gate.

Complements test_re08_provenance_import_boundary.py (which covers the import
boundary itself: raw Path -> imported-legacy with a real rehydratable
artifact_id; unverified refs downgraded; locator rehydration) with the
campaign-level scenarios the spec calls out:

1. THE end-to-end taint chain: raw-Path inventory -> _resolve_lane_inputs
   (imported-legacy, descriptor-backed) -> the REAL build worker (faked
   compiler only) -> binary AND runtime-bundle still imported-legacy even
   though the lane's own provenance class is production;
   require_promotable() rejects the tainted runtime-bundle;
2. no degradation in either direction: production parents + production
   local class stay production end-to-end (and remain promotable); a
   development local class stays development;
3. the envelope exemption is exactly one class: imported-legacy input
   skips the stage-envelope field check, while production/development
   inputs with a mismatched (or missing) source identity are rejected
   before the stage runs;
4. a locator whose artifact_id points at a different KIND is rejected;
5. publish-time kind-contract gate: a PRODUCTION document missing a
   kind-required field fails at publication, while the same shape as
   imported-legacy publishes (RE25.3 step 3.4).

The build hop uses the faked-compiler harness pattern from
test_campaign_workers_build.py; no network, no GPU, runs in seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign import workers as campaign_workers # noqa: E402
from bigcherry.core import config as campaign_config # noqa: E402
from bigcherry.build import generated_tree # noqa: E402
from bigcherry.core import provenance # noqa: E402
from bigcherry.core.artifacts import (  # noqa: E402
    ArtifactError,
    ArtifactLocator,
    ArtifactStore,
)
from bigcherry.build.builds import BuildPlan, build_directory  # noqa: E402
from bigcherry.campaign.lane import (  # noqa: E402
    CampaignLaneError,
    CampaignLaneExecutionSpec,
    LaneInputValue,
    _resolve_lane_inputs,
)
from bigcherry.core.context import ProjectContext  # noqa: E402
from bigcherry.core.pipeline import ArtifactRef, PipelineError, PipelineService  # noqa: E402
from bigcherry.core.provenance import require_promotable  # noqa: E402

_CMAKE_CACHE = """\
CMAKE_C_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang
CMAKE_CXX_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang++
CMAKE_BUILD_TYPE:STRING=Release
AMDGPU_TARGETS:STRING=gfx1100
GGML_HIP:BOOL=ON
"""


def _ns(doc: dict[str, object], name: str) -> dict[str, object]:
    """Narrow one provenance namespace (docs are dict[str, object])."""
    value = doc[name]
    assert isinstance(value, dict), f"namespace {name!r} is not a dict"
    return value


def _class(doc: dict[str, object]) -> str:
    return str(_ns(doc, "project").get("provenance_class"))


class _Harness:
    """One make_build_worker call's worth of machinery (faked compiler),
    same pattern as test_campaign_workers_build.py."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.context = ProjectContext(
            project_root=directory,
            config_path=directory / "recipes.toml",
            artifacts_root=directory / "artifacts",
            work_root=directory / "work",
            upstream_repo=directory / "upstream",
            overlay_root=directory / "src",
            patches_root=directory / "patches",
        )
        self.store = ArtifactStore(directory / "store")
        self.run_id = "run1"
        self.source_slice_id = "s1"
        self.workload_id = "w1"
        self.build_plan = BuildPlan(
            source_slice_id=self.source_slice_id,
            phase="tune",
            platform="linux-multi",
            targets=("gfx1100",),
            variant_set="workload-max",
        )
        self.platform = campaign_config.Platform(
            name="linux-multi",
            targets=("gfx1100",),
            options=(),
            c_compiler="/opt/rocm/llvm/bin/clang",
            cxx_compiler="/opt/rocm/llvm/bin/clang++",
        )
        self.build_cfg = campaign_config.Build(
            name="tune",
            options=(),
            variant_set="workload-max",
            needs=frozenset({"inventory"}),
        )

    def generate_inputs(self) -> tuple[ArtifactRef, ...]:
        """Generate-stage outputs exactly as the real generate worker would
        publish them (production-class fixture docs)."""
        generated_root = (
            self.context.work_root / "runs" / self.run_id / "generate" / "generated"
        )
        registry = generated_root / "hip-autotune-registry.inc"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text("registry", encoding="utf-8")
        tree_document = generated_tree.build_manifest(
            generated_root, compile_inputs=(registry,)
        )
        doc = provenance.make(
            project={"provenance_class": "production"},
            source={"source_slice_id": self.source_slice_id},
            build={"build_plan_id": self.build_plan.build_plan_id},
            workload={"workload_id": self.workload_id},
            campaign={"run_id": self.run_id, "producer_stage": "generate"},
        )
        manifest_digest = self.store.publish_json(
            "gen/manifest.json", {"candidates": []}
        )
        tree_digest = self.store.publish_json("gen/generated-tree.json", tree_document)
        return (
            ArtifactRef(
                kind="manifest",
                path=self.store.resolve("gen/manifest.json"),
                content_hash=manifest_digest,
                provenance=doc,
            ),
            ArtifactRef(
                kind="generated-tree",
                path=self.store.resolve("gen/generated-tree.json"),
                content_hash=tree_digest,
                provenance=doc,
            ),
        )

    def worker(self, local_provenance_class: str, lane_inputs: dict | None = None):
        return campaign_workers.make_build_worker(
            context=self.context,
            source_root=self.directory / "source",
            run_id=self.run_id,
            build_plan=self.build_plan,
            platform=self.platform,
            build=self.build_cfg,
            store=self.store,
            binary_relative_path="bin/llama-bench",
            source_slice_id=self.source_slice_id,
            workload_id=self.workload_id,
            cmake_targets=("llama-bench",),
            lane_inputs=lane_inputs or {},
            local_provenance_class=local_provenance_class,
        )

    def fake_compiler(self):
        build_dir = build_directory(self.context, self.source_slice_id, self.build_plan)

        def run(cmd, cwd=None, check=None):
            if "--build" in cmd:
                binary = build_dir / "bin" / "llama-bench"
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b"compiled-binary-bytes")
                (build_dir / "bin" / "libggml-hip.so.0.19.0").write_bytes(
                    b"hip-dispatch-v1"
                )
            else:
                (build_dir / "CMakeCache.txt").write_text(
                    _CMAKE_CACHE, encoding="utf-8"
                )
            return subprocess.CompletedProcess(cmd, 0)

        return run


def _resolve(store: ArtifactStore, *inputs: tuple[str, LaneInputValue], build=None):
    return _resolve_lane_inputs(
        CampaignLaneExecutionSpec(
            source_name="bigcherry",
            build_name="tune",
            platform_name="linux-multi",
            architectures=("gfx1100",),
            inputs=inputs,
        ),
        build=build
        or campaign_config.Build(
            name="tune",
            options=(),
            variant_set="workload-max",
            needs=frozenset(n for n, _ in inputs),
        ),
        store=store,
        run_id="run1",
    )


class RawPathTaintEndToEndTests(unittest.TestCase):
    """(1) THE chain: raw Path -> imported-legacy -> build -> tainted bundle."""

    def test_raw_path_inventory_taints_binary_and_runtime_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            harness = _Harness(directory)

            # Hop 0 — the lane boundary (RE25.3 step 1/2): a plain file with
            # no chain-of-custody comes out imported-legacy, descriptor-backed.
            inventory = directory / "inventory.json"
            inventory.write_text(json.dumps({"mmq_types": ["q8_0"]}))
            resolved = _resolve(
                harness.store, ("inventory", inventory), build=harness.build_cfg
            )
            inv_ref = resolved["inventory"]
            self.assertEqual(_class(inv_ref.provenance), "imported-legacy")
            self.assertTrue(inv_ref.artifact_id)
            # No unearned source claim (RE25.3 step 1).
            self.assertIsNone(_ns(inv_ref.provenance, "source").get("source_slice_id"))

            # Hop 1 — the REAL build worker (compiler faked), whose OWN lane
            # class is production: the imported parent must still win.
            inputs = harness.generate_inputs()
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                refs = harness.worker(
                    local_provenance_class="production",
                    lane_inputs={"inventory": inv_ref},
                )(inputs)

            self.assertEqual([r.kind for r in refs], ["binary", "runtime-bundle"])
            binary_doc, bundle_doc = (ref.provenance for ref in refs)
            # STICKY: both derived artifacts are imported-legacy...
            self.assertEqual(_class(binary_doc), "imported-legacy")
            self.assertEqual(_class(bundle_doc), "imported-legacy")
            # ...and the lineage is intact (the inventory's real id was
            # recorded as a parent, not severed).
            self.assertIn(
                inv_ref.artifact_id,
                _ns(bundle_doc, "campaign")["producer_artifact_ids"],
            )
            # ...so neither can ever become release evidence.
            with self.assertRaises(provenance.ProvenanceError):
                require_promotable(binary_doc, kind="binary")
            with self.assertRaises(provenance.ProvenanceError) as ctx:
                require_promotable(bundle_doc, kind="runtime-bundle")
            self.assertIn("not promotable", str(ctx.exception))


class NoTaintDegradationTests(unittest.TestCase):
    """(2) production stays production (and promotable); development stays
    development — taint is one-directional, never a demotion of clean work."""

    def _production_inventory_ref(self, harness: _Harness) -> ArtifactRef:
        data = json.dumps({"mmq_types": ["q8_0"]}).encode()
        # A full production document: every inventory kind-required field.
        doc = provenance.make(
            project={"provenance_class": "production"},
            source={
                "upstream_revision": "up1",
                "source_slice_id": harness.source_slice_id,
            },
            build={
                "build_plan_id": harness.build_plan.build_plan_id,
                "effective_build_id": "eb-upstream-1",
            },
            workload={"workload_id": harness.workload_id},
            campaign={"run_id": "upstream-run", "producer_stage": "inventory"},
        )
        return harness.store.publish_bytes_ref(
            "inputs/inventory/prod", data, kind="inventory", provenance=doc
        )

    def test_production_parents_with_production_lane_stay_promotable(self):
        with tempfile.TemporaryDirectory() as td:
            harness = _Harness(Path(td))
            inv_ref = self._production_inventory_ref(harness)
            self.assertEqual(_class(inv_ref.provenance), "production")

            # Full source provenance, as the real pipeline threads it in —
            # so the production kind contract at publish time is satisfiable.
            source_provenance = provenance.SourceProvenance(
                upstream_revision="up1",
                source_plan_id="sp1",
                materialization_plan_id="mp1",
                source_tree_oid="0" * 40,
                source_slice_id=harness.source_slice_id,
                git_object_format="oid",
                patch_set_id="ps1",
            )
            worker = campaign_workers.make_build_worker(
                context=harness.context,
                source_root=harness.directory / "source",
                run_id=harness.run_id,
                build_plan=harness.build_plan,
                platform=harness.platform,
                build=harness.build_cfg,
                store=harness.store,
                binary_relative_path="bin/llama-bench",
                source_slice_id=harness.source_slice_id,
                workload_id=harness.workload_id,
                cmake_targets=("llama-bench",),
                lane_inputs={"inventory": inv_ref},
                source_provenance=source_provenance,
                local_provenance_class="production",
            )
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                refs = worker(harness.generate_inputs())

            binary_doc, bundle_doc = (ref.provenance for ref in refs)
            self.assertEqual(_class(binary_doc), "production")
            self.assertEqual(_class(bundle_doc), "production")
            # Clean work stays promotable — the gate does not demote it.
            require_promotable(binary_doc, kind="binary")
            require_promotable(bundle_doc, kind="runtime-bundle")

    def test_development_lane_class_stays_development(self):
        with tempfile.TemporaryDirectory() as td:
            harness = _Harness(Path(td))
            inv_ref = self._production_inventory_ref(harness)
            inputs = harness.generate_inputs()
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                refs = harness.worker(
                    local_provenance_class="development",
                    lane_inputs={"inventory": inv_ref},
                )(inputs)
            for ref in refs:
                self.assertEqual(_class(ref.provenance), "development")


class EnvelopeExemptionTests(unittest.TestCase):
    """(3) imported-legacy is the ONE class exempt from the stage envelope;
    every other class must match exactly, before the stage runs."""

    def _ref(self, doc: dict[str, object]) -> ArtifactRef:
        return ArtifactRef(
            kind="inventory",
            path=Path("/nonexistent-but-never-opened"),
            content_hash="sha256:" + "0" * 64,
            provenance=doc,
        )

    def test_imported_legacy_input_skips_the_envelope(self):
        # No source identity at all — the raw-Path import shape. Checking it
        # would require trusting claims we explicitly do not trust.
        doc = provenance.make(
            project={"provenance_class": "imported-legacy"},
            source={},
            build={},
            workload={},
            campaign={},
        )
        PipelineService._check_inputs(
            (self._ref(doc),), {"source.source_slice_id": "s1"}
        )

    def test_production_input_with_wrong_source_fails_before_stage(self):
        doc = provenance.make(
            project={"provenance_class": "production"},
            source={"source_slice_id": "a-different-slice"},
            build={},
            workload={},
            campaign={},
        )
        with self.assertRaises(PipelineError) as ctx:
            PipelineService._check_inputs(
                (self._ref(doc),), {"source.source_slice_id": "s1"}
            )
        self.assertIn("source.source_slice_id", str(ctx.exception))

    def test_production_input_missing_envelope_field_fails(self):
        doc = provenance.make(
            project={"provenance_class": "production"},
            source={},  # field absent entirely
            build={},
            workload={},
            campaign={},
        )
        with self.assertRaises(PipelineError):
            PipelineService._check_inputs(
                (self._ref(doc),), {"source.source_slice_id": "s1"}
            )

    def test_development_input_is_not_exempt(self):
        doc = provenance.make(
            project={"provenance_class": "development"},
            source={"source_slice_id": "a-different-slice"},
            build={},
            workload={},
            campaign={},
        )
        with self.assertRaises(PipelineError):
            PipelineService._check_inputs(
                (self._ref(doc),), {"source.source_slice_id": "s1"}
            )

    def test_matching_production_input_passes(self):
        doc = provenance.make(
            project={"provenance_class": "production"},
            source={"source_slice_id": "s1"},
            build={},
            workload={},
            campaign={},
        )
        PipelineService._check_inputs(
            (self._ref(doc),), {"source.source_slice_id": "s1"}
        )


class LocatorKindMismatchTests(unittest.TestCase):
    """(4) a locator must rehydrate as the kind the lane slot expects."""

    def test_locator_to_different_kind_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = ArtifactStore(Path(td) / "store")
            # Publish a MANIFEST artifact, then try to consume it in the
            # inventory slot. (imported-legacy class: a minimal fixture doc
            # would not satisfy the production kind contract at publish.)
            manifest_id = store.publish_bytes_ref(
                "inputs/manifest/m",
                b"{}",
                kind="manifest",
                provenance=provenance.make(
                    project={"provenance_class": "imported-legacy"},
                    source={},
                    build={},
                    workload={},
                    campaign={},
                ),
            ).artifact_id
            with self.assertRaises(CampaignLaneError) as ctx:
                _resolve(store, ("inventory", ArtifactLocator(manifest_id)))
            self.assertIn("failed rehydration", str(ctx.exception))


class PublishKindContractGateTests(unittest.TestCase):
    """(5) RE25.3 step 3.4: production documents must satisfy their kind
    contract AT PUBLICATION; imported-legacy evidence is exempt."""

    def _inventory_doc(self, *, project_class: str, **campaign_overrides) -> dict:
        campaign = {"run_id": "r1", "producer_stage": "inventory"}
        campaign.update(campaign_overrides)
        return provenance.make(
            project={"provenance_class": project_class},
            source={
                "upstream_revision": "up1",
                "source_slice_id": "s1",
            },
            build={
                "build_plan_id": "bp1",
                "effective_build_id": "eb1",
            },
            workload={"workload_id": "w1"},
            campaign=campaign,
        )

    def test_production_inventory_missing_run_id_fails_at_publish(self):
        with tempfile.TemporaryDirectory() as td:
            store = ArtifactStore(Path(td) / "store")
            doc = self._inventory_doc(
                project_class="production",
                run_id=None,  # type: ignore[arg-type]
            )
            del doc["campaign"]["run_id"]  # absent, not None
            with self.assertRaises(ArtifactError) as ctx:
                store.publish_bytes_ref(
                    "inputs/inventory/bad", b"{}", kind="inventory", provenance=doc
                )
            message = str(ctx.exception)
            self.assertIn("campaign.run_id", message)

    def test_imported_legacy_same_shape_publishes(self):
        with tempfile.TemporaryDirectory() as td:
            store = ArtifactStore(Path(td) / "store")
            doc = provenance.make(
                project={"provenance_class": "imported-legacy"},
                source={"upstream_revision": "up1", "source_slice_id": "s1"},
                build={},  # no build_plan_id/effective_build_id...
                workload={},  # ...no workload_id, no campaign at all: an honest
                campaign={},  # import never had BigCherry run fields.
            )
            ref = store.publish_bytes_ref(
                "inputs/inventory/imp", b"{}", kind="inventory", provenance=doc
            )
            self.assertTrue(ref.artifact_id)
            rehydrated = ArtifactStore(Path(td) / "store").rehydrate(
                ref.artifact_id, expected_kind="inventory"
            )
            self.assertEqual(rehydrated.content_hash, ref.content_hash)


if __name__ == "__main__":
    unittest.main()
