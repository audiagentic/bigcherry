"""RE16: execute_campaign_lane() -- the reusable, in-process production API.

Real materialize phase (a real git upstream repo, exactly like
test_campaign_build.py's MaterializeSourceTests/ExecuteBuildStageTests
fixtures) but the generate/build/smoke workers are faked at the factory
level (campaign_workers.make_generate_worker/make_build_worker/
make_smoke_worker are patched to return simple callables) -- their own
correctness is already covered by test_campaign_workers_build.py and
test_runtime_smoke.py; this file's job is campaign_lane.py's OWN
orchestration: spec-to-BuildPlan wiring, content-addressed inventory
publication, output-ref extraction by kind, and the "no automatic second
pass" contract.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import config as campaign_config # noqa: E402
from bigcherry.core import provenance # noqa: E402
from bigcherry.core.artifacts import ArtifactStore  # noqa: E402
from bigcherry.campaign.lane import (
    CampaignLaneError,
    CampaignLaneExecutionSpec,  # noqa: E402
    execute_campaign_lane,
    smoke_environment_for_hip_devices,
)
from bigcherry.core.context import ProjectContext  # noqa: E402
from bigcherry.core.pipeline import ArtifactRef  # noqa: E402
from bigcherry.campaign.smoke import RuntimeSmokeSpec  # noqa: E402
from bigcherry.campaign import workers as campaign_workers  # noqa: E402
from bigcherry.campaign.build import CampaignBuildError  # noqa: E402
from bigcherry.build.builds import BuildPlan, build_directory  # noqa: E402
from bigcherry.source.identity import (  # noqa: E402
    SourceAttestation,
    git_object_format,
    git_tree_oid,
    source_slice_id,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_upstream(directory: Path) -> tuple[Path, str]:
    repo = directory / "upstream"
    _git(directory, "init", str(repo))
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "source.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "initial")
    revision = _git(repo, "rev-parse", "HEAD")
    return repo, revision


class _Harness:
    def __init__(self, directory: Path, upstream: Path, revision: str):
        self.context = ProjectContext(
            project_root=directory,
            config_path=directory / "recipes.toml",
            artifacts_root=directory / "artifacts",
            work_root=directory / "work",
            upstream_repo=upstream,
            overlay_root=directory / "src",
            patches_root=directory / "patches",
        )
        self.store = ArtifactStore(directory / "store")
        self.cfg = campaign_config.Config(
            pinned=revision,
            patch_sets={},
            sources={
                "test-source": campaign_config.Source(
                    name="test-source", ref=revision, overlay=False, patch_sets=()
                )
            },
            builds={
                "tune": campaign_config.Build(
                    name="tune",
                    options=(),
                    variant_set="workload-max",
                    needs=frozenset({"inventory"}),
                )
            },
            platforms={
                "linux-multi": campaign_config.Platform(
                    name="linux-multi", targets=("gfx1100",), options=()
                )
            },
            experiments={},
            campaigns={},
            path=directory / "recipes.toml",
        )
        self.calls: list[str] = []

    def spec(self, **overrides) -> CampaignLaneExecutionSpec:
        inventory_path = Path(tempfile.mkdtemp()) / "inventory.json"
        inventory_path.write_text('{"mmq_types": ["q8_0"]}', encoding="utf-8")
        base = CampaignLaneExecutionSpec(
            source_name="test-source",
            build_name="tune",
            platform_name="linux-multi",
            architectures=("gfx1100",),
            inputs=(("inventory", inventory_path),),
            validation=RuntimeSmokeSpec(model_path=Path("model.gguf")),
        )
        # replace() (not a dict **spread) keeps the override keys typed --
        # spreading a heterogeneous dict into the constructor is exactly
        # what defeats static checking of this fixture.
        return base if not overrides else replace(base, **overrides)

    def fake_workers(self):
        store = self.store
        calls = self.calls

        def fake_generate_worker(**kwargs):
            def generate(inputs):
                calls.append("generate")
                assert set(inputs) == {"inventory"}
                return {}

            return generate

        def fake_build_worker(**kwargs):
            source_slice_id = kwargs["source_slice_id"]
            build_plan_id = kwargs["build_plan"].build_plan_id
            workload_id = kwargs["workload_id"]
            run_id = kwargs["run_id"]

            def build(inputs):
                calls.append("build")
                doc = provenance.make(
                    project={},
                    source={"source_slice_id": source_slice_id},
                    build={"build_plan_id": build_plan_id},
                    workload={"workload_id": workload_id},
                    campaign={"run_id": run_id},
                )
                relative = f"builds/{build_plan_id}/{run_id}/bin"
                digest = store.publish_bytes(relative, b"fake-binary")
                bundle_relative = f"builds/{build_plan_id}/{run_id}/runtime-bundle.json"
                bundle_digest = store.publish_json(
                    bundle_relative, {"entrypoint": "bin"}
                )
                return (
                    ArtifactRef(
                        kind="binary",
                        path=store.resolve(relative),
                        content_hash=digest,
                        provenance=doc,
                    ),
                    ArtifactRef(
                        kind="runtime-bundle",
                        path=store.resolve(bundle_relative),
                        content_hash=bundle_digest,
                        provenance=doc,
                    ),
                )

            return build

        def fake_smoke_worker(**kwargs):
            source_slice_id = kwargs["source_slice_id"]
            build_plan_id = kwargs["build_plan_id"]
            workload_id = kwargs["workload_id"]
            run_id = kwargs["run_id"]

            def smoke(inputs):
                calls.append("smoke")
                doc = provenance.make(
                    project={},
                    source={"source_slice_id": source_slice_id},
                    build={"build_plan_id": build_plan_id},
                    workload={"workload_id": workload_id},
                    campaign={"run_id": run_id},
                )
                relative = f"runs/{run_id}/{build_plan_id}/smoke.json"
                digest = store.publish_json(relative, {"ok": True})
                return (
                    ArtifactRef(
                        kind="smoke-result",
                        path=store.resolve(relative),
                        content_hash=digest,
                        provenance=doc,
                    ),
                )

            return smoke

        return fake_generate_worker, fake_build_worker, fake_smoke_worker


class ExecuteCampaignLaneTests(unittest.TestCase):
    def test_real_materialize_then_faked_generate_build_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            harness = _Harness(root, upstream, revision)
            fake_generate, fake_build, fake_smoke = harness.fake_workers()

            # generate stage publishes manifest+generated-tree in real code;
            # the fake stands in for that, but execute_campaign_lane still
            # needs those two ArtifactRefs to exist post-generate. Simplify
            # by having the fake generate worker itself publish them via
            # the real CampaignStageExecutor._run_generate contract: return
            # a dict with manifest/generated_tree/workload_id, exactly what
            # the real make_generate_worker returns.
            def fake_generate_worker(**kwargs):
                def generate(inputs):
                    harness.calls.append("generate")
                    assert set(inputs) == {"inventory"}
                    return {
                        "manifest": {"candidates": []},
                        "generated_tree": {
                            "schema_version": 1,
                            "files": {},
                            "compile_inputs": [],
                            "compile_inputs_hash": "h",
                        },
                    }

                return generate

            with (
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_generate_worker",
                    side_effect=fake_generate_worker,
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_build_worker",
                    side_effect=fake_build,
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_smoke_worker",
                    side_effect=fake_smoke,
                ),
            ):
                result = execute_campaign_lane(
                    harness.spec(),
                    cfg=harness.cfg,
                    context=harness.context,
                    store=harness.store,
                    run_id="run1",
                    allow_dirty_bigcherry=True,
                )

            self.assertEqual(
                harness.calls, ["generate", "build", "smoke"]
            )  # exactly once each
            self.assertEqual(result.run_id, "run1")
            self.assertEqual(result.resolved_revision, revision)
            assert result.smoke_ref is not None
            self.assertEqual(result.binary_ref.kind, "binary")
            self.assertEqual(result.smoke_ref.kind, "smoke-result")
            self.assertEqual(result.build_plan_id, result.build_plan.build_plan_id)
            self.assertIsNotNone(result.workload_id)

    def test_two_lanes_with_different_inventories_do_not_collide(self):
        # The one behavioral fix bundled into this extraction: inventory
        # publication moved off a fixed path specifically so two lanes
        # sharing one ArtifactStore don't collide when their inventories
        # differ.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            harness = _Harness(root, upstream, revision)

            def fake_generate_worker(**kwargs):
                def generate(inputs):
                    assert set(inputs) == {"inventory"}
                    return {
                        "manifest": {"candidates": []},
                        "generated_tree": {
                            "schema_version": 1,
                            "files": {},
                            "compile_inputs": [],
                            "compile_inputs_hash": "h",
                        },
                    }

                return generate

            _, fake_build, fake_smoke = harness.fake_workers()

            inventory_a = Path(tempfile.mkdtemp()) / "inv.json"
            inventory_a.write_text('{"mmq_types": ["q8_0"]}', encoding="utf-8")
            inventory_b = Path(tempfile.mkdtemp()) / "inv.json"
            inventory_b.write_text('{"mmq_types": ["f16"]}', encoding="utf-8")

            with (
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_generate_worker",
                    side_effect=fake_generate_worker,
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_build_worker",
                    side_effect=fake_build,
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_smoke_worker",
                    side_effect=fake_smoke,
                ),
            ):
                result_a = execute_campaign_lane(
                    harness.spec(inputs=(("inventory", inventory_a),)),
                    cfg=harness.cfg,
                    context=harness.context,
                    store=harness.store,
                    run_id="lane-a",
                    allow_dirty_bigcherry=True,
                )
                result_b = execute_campaign_lane(
                    harness.spec(inputs=(("inventory", inventory_b),)),
                    cfg=harness.cfg,
                    context=harness.context,
                    store=harness.store,
                    run_id="lane-b",
                    allow_dirty_bigcherry=True,
                )

            inventory_ref_a = dict(result_a.input_refs)["inventory"]
            inventory_ref_b = dict(result_b.input_refs)["inventory"]
            self.assertNotEqual(
                inventory_ref_a.content_hash, inventory_ref_b.content_hash
            )
            self.assertNotEqual(inventory_ref_a.path, inventory_ref_b.path)
            self.assertTrue(inventory_ref_a.path.is_file())
            self.assertTrue(inventory_ref_b.path.is_file())

    def test_lane_inputs_not_matching_build_needs_fails_closed(self):
        # RE17: build.needs is the sole authority for lane inputs -- an
        # exact-set-equality check, not a subset check. Providing an input
        # the build doesn't declare (or omitting one it does) must fail
        # closed before any worker is constructed.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            harness = _Harness(root, upstream, revision)
            fake_generate, fake_build, fake_smoke = harness.fake_workers()

            with (
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_generate_worker",
                    side_effect=fake_generate,
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_build_worker",
                    side_effect=fake_build,
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_smoke_worker",
                    side_effect=fake_smoke,
                ),
                self.assertRaises(CampaignLaneError),
            ):
                execute_campaign_lane(
                    harness.spec(inputs=()),
                    cfg=harness.cfg,
                    context=harness.context,
                    store=harness.store,
                    run_id="run1",
                    allow_dirty_bigcherry=True,
                )

    def test_source_backend_reaches_make_build_worker(self):
        # RE30 phase 3: execute_campaign_lane must resolve
        # cfg.sources[spec.source_name].backend and pass it to
        # make_build_worker -- the seam that used to silently default
        # every lane to "hip" regardless of the source's real backend.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            harness = _Harness(root, upstream, revision)
            harness.cfg = replace(
                harness.cfg,
                sources={
                    "test-source": replace(
                        harness.cfg.sources["test-source"], backend="vulkan"
                    )
                },
            )
            _, fake_build, fake_smoke = harness.fake_workers()
            captured: dict[str, object] = {}

            def capturing_build_worker(**kwargs):
                captured["backend"] = kwargs.get("backend")
                return fake_build(**kwargs)

            with (
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_generate_worker",
                    side_effect=Re25ProvenanceLineageTests._fake_generate_worker(),
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_build_worker",
                    side_effect=capturing_build_worker,
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_smoke_worker",
                    side_effect=fake_smoke,
                ),
            ):
                execute_campaign_lane(
                    harness.spec(),
                    cfg=harness.cfg,
                    context=harness.context,
                    store=harness.store,
                    run_id="run1",
                    allow_dirty_bigcherry=True,
                )
            self.assertEqual(captured["backend"], "vulkan")

    def test_hip_and_vulkan_produce_different_build_plan_id(self):
        # RE-backend-identity (external review, 2026-08-20) P0-1: backend
        # must be intrinsic to requested BuildPlan identity. Before this
        # fix, a HIP and a Vulkan lane sharing the same nominal source/
        # build/platform names produced the SAME build_plan_id/build_dir
        # (backend.py::build_directory keys purely off it), so a cached
        # HIP build could be silently "reused" for a Vulkan request, or
        # vice versa, with no error anywhere in the chain.
        def run_with_backend(backend: str):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                upstream, revision = _init_upstream(root)
                harness = _Harness(root, upstream, revision)
                harness.cfg = replace(
                    harness.cfg,
                    sources={
                        "test-source": replace(
                            harness.cfg.sources["test-source"], backend=backend
                        )
                    },
                )
                _, fake_build, fake_smoke = harness.fake_workers()
                with (
                    patch(
                        "bigcherry.campaign.lane.campaign_workers.make_generate_worker",
                        side_effect=Re25ProvenanceLineageTests._fake_generate_worker(),
                    ),
                    patch(
                        "bigcherry.campaign.lane.campaign_workers.make_build_worker",
                        side_effect=fake_build,
                    ),
                    patch(
                        "bigcherry.campaign.lane.campaign_workers.make_smoke_worker",
                        side_effect=fake_smoke,
                    ),
                ):
                    result = execute_campaign_lane(
                        harness.spec(),
                        cfg=harness.cfg,
                        context=harness.context,
                        store=harness.store,
                        run_id="run1",
                        allow_dirty_bigcherry=True,
                    )
                return result.build_plan

        hip_plan = run_with_backend("hip")
        vulkan_plan = run_with_backend("vulkan")

        self.assertEqual(hip_plan.backend, "hip")
        self.assertEqual(vulkan_plan.backend, "vulkan")
        self.assertNotEqual(hip_plan.cmake_options, vulkan_plan.cmake_options)
        self.assertNotEqual(hip_plan.build_plan_id, vulkan_plan.build_plan_id)
        # ... and therefore not the same on-disk build_dir either -- a
        # cached HIP build cannot be silently handed to a Vulkan request
        # sharing the same nominal source/build/platform names.
        from bigcherry.build.builds import build_directory

        context = ProjectContext(
            project_root=Path("/root"),
            config_path=Path("/root/recipes.toml"),
            artifacts_root=Path("/root/artifacts"),
            work_root=Path("/root/work"),
            upstream_repo=Path("/root/upstream"),
            overlay_root=Path("/root/src"),
            patches_root=Path("/root/patches"),
        )
        self.assertNotEqual(
            build_directory(context, "slice", hip_plan),
            build_directory(context, "slice", vulkan_plan),
        )


class Re25ProvenanceLineageTests(unittest.TestCase):
    """RE25.2: real call sites construct typed ProvenanceV2 with full lineage
    (no empty-namespace dicts), publish descriptor-backed artifacts, and the
    whole lane's provenance class honestly reflects the executing checkout.
    """

    @staticmethod
    def _fake_generate_worker():
        def fake_generate_worker(**kwargs):
            def generate(inputs):
                return {
                    "manifest": {"candidates": []},
                    "generated_tree": {
                        "schema_version": 1,
                        "files": {},
                        "compile_inputs": [],
                        "compile_inputs_hash": "h",
                    },
                }

            return generate

        return fake_generate_worker

    def test_materialize_artifact_carries_full_source_lineage_and_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            harness = _Harness(root, upstream, revision)
            _, fake_build, fake_smoke = harness.fake_workers()

            with (
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_generate_worker",
                    side_effect=self._fake_generate_worker(),
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_build_worker",
                    side_effect=fake_build,
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_smoke_worker",
                    side_effect=fake_smoke,
                ),
            ):
                result = execute_campaign_lane(
                    harness.spec(),
                    cfg=harness.cfg,
                    context=harness.context,
                    store=harness.store,
                    run_id="run1",
                    allow_dirty_bigcherry=True,
                )

            ref = result.source_metadata_ref
            # Descriptor-backed: a rehydratable identity, not just bytes +
            # an in-memory provenance dict.
            self.assertTrue(ref.artifact_id)
            project_ns = ref.provenance["project"]
            assert isinstance(project_ns, dict)
            source_ns = ref.provenance["source"]
            assert isinstance(source_ns, dict)
            # allow_dirty_bigcherry=True on a non-git temp checkout: the
            # honest class is development (never production), and no
            # revision is invented for one that doesn't exist.
            self.assertEqual(project_ns["provenance_class"], "development")
            self.assertIsNone(project_ns.get("bigcherry_revision"))
            # The FULL real source lineage from materialize_source()'s own
            # re-verified metadata -- not a bare source_slice_id string.
            self.assertEqual(source_ns["upstream_revision"], revision)
            self.assertEqual(source_ns["source_slice_id"], result.source_slice_id)
            self.assertTrue(source_ns.get("source_tree_oid"))
            self.assertTrue(source_ns.get("materialization_plan_id"))
            self.assertTrue(source_ns.get("source_plan_id"))
            # HI130: source_root is the real isolated materialization this
            # lane built from -- must be a real, existing directory (the
            # correctness-evidence workflow reads --llama-root from here).
            self.assertTrue(result.source_root.is_dir())
            self.assertTrue((result.source_root / "source.txt").is_file())
            self.assertIn(source_ns.get("git_object_format"), ("sha1", "sha256"))
            # Generate's stage outputs are descriptor-backed too.
            assert result.manifest_ref is not None
            assert result.generated_tree_ref is not None
            self.assertTrue(result.manifest_ref.artifact_id)
            self.assertTrue(result.generated_tree_ref.artifact_id)
            # The fake build worker (plain provenance.make, no effective
            # identity) leaves the property None rather than inventing one.
            self.assertIsNone(result.effective_build_id)

            # Cross-instance rehydration: a second ArtifactStore over the
            # same root recovers a byte-verified, provenance-identical ref
            # from the artifact_id alone.
            second_store = ArtifactStore(root / "store")
            rehydrated = second_store.rehydrate(
                ref.artifact_id, expected_kind="source-metadata"
            )
            self.assertEqual(rehydrated.provenance, ref.provenance)
            self.assertEqual(rehydrated.content_hash, ref.content_hash)

    def test_production_mode_fails_closed_without_a_git_checkout(self):
        # allow_dirty_bigcherry=False on a non-git project root must fail
        # closed (production provenance cannot claim an identity it can't
        # prove) -- not silently degrade.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            harness = _Harness(root, upstream, revision)
            _, fake_build, fake_smoke = harness.fake_workers()

            with (
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_generate_worker",
                    side_effect=self._fake_generate_worker(),
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_build_worker",
                    side_effect=fake_build,
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_smoke_worker",
                    side_effect=fake_smoke,
                ),
                self.assertRaises(CampaignLaneError),
            ):
                execute_campaign_lane(
                    harness.spec(),
                    cfg=harness.cfg,
                    context=harness.context,
                    store=harness.store,
                    run_id="run1",
                )

    def test_effective_build_id_property_reads_bundle_provenance(self):
        # A build worker that records a real effective_build_id in the
        # runtime bundle's provenance (what the REAL make_build_worker does)
        # surfaces it on CampaignLaneResult without re-parsing the bundle
        # manifest JSON.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            harness = _Harness(root, upstream, revision)
            store = harness.store
            _, _, fake_smoke = harness.fake_workers()

            def fake_build_worker(**kwargs):
                source_slice_id = kwargs["source_slice_id"]
                build_plan_id = kwargs["build_plan"].build_plan_id
                # The build-stage envelope (execution_expected) requires the
                # workload id on every output when the lane has one -- this
                # lane's inventory supplies it, so a doc without it fails
                # closed (which is exactly what the first run proved).
                workload_id = kwargs["workload_id"]
                run_id = kwargs["run_id"]

                def build(inputs):
                    doc = provenance.make(
                        project={},
                        source={"source_slice_id": source_slice_id},
                        build={
                            "build_plan_id": build_plan_id,
                            "effective_build_id": "eb-test-123",
                        },
                        workload={"workload_id": workload_id},
                        campaign={"run_id": run_id},
                    )
                    relative = f"builds/{build_plan_id}/{run_id}/bin"
                    digest = store.publish_bytes(relative, b"fake-binary")
                    bundle_relative = (
                        f"builds/{build_plan_id}/{run_id}/runtime-bundle.json"
                    )
                    bundle_digest = store.publish_json(
                        bundle_relative, {"entrypoint": "bin"}
                    )
                    return (
                        ArtifactRef(
                            kind="binary",
                            path=store.resolve(relative),
                            content_hash=digest,
                            provenance=doc,
                        ),
                        ArtifactRef(
                            kind="runtime-bundle",
                            path=store.resolve(bundle_relative),
                            content_hash=bundle_digest,
                            provenance=doc,
                        ),
                    )

                return build

            with (
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_generate_worker",
                    side_effect=self._fake_generate_worker(),
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_build_worker",
                    side_effect=fake_build_worker,
                ),
                patch(
                    "bigcherry.campaign.lane.campaign_workers.make_smoke_worker",
                    side_effect=fake_smoke,
                ),
            ):
                result = execute_campaign_lane(
                    harness.spec(),
                    cfg=harness.cfg,
                    context=harness.context,
                    store=harness.store,
                    run_id="run1",
                    allow_dirty_bigcherry=True,
                )

            self.assertEqual(result.effective_build_id, "eb-test-123")


class SourceAttestationWorkerTests(unittest.TestCase):
    _CMAKE_CACHE = """\
CMAKE_C_COMPILER:FILEPATH=/usr/bin/cc
CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++
CMAKE_BUILD_TYPE:STRING=Release
AMDGPU_TARGETS:STRING=gfx1100
GGML_HIP:BOOL=ON
"""

    def _fixture(self, root: Path):
        source = root / "source"
        _git(root, "init", str(source))
        _git(source, "config", "user.email", "test@example.invalid")
        _git(source, "config", "user.name", "Test")
        (source / "source.txt").write_text("one\n", encoding="utf-8")
        _git(source, "add", "source.txt")
        _git(source, "commit", "-m", "initial")

        context = ProjectContext(
            project_root=root,
            config_path=root / "recipes.toml",
            artifacts_root=root / "artifacts",
            work_root=root / "work",
            upstream_repo=root / "upstream",
            overlay_root=root / "overlay",
            patches_root=root / "patches",
        )
        store = ArtifactStore(root / "store")
        revision = _git(source, "rev-parse", "HEAD")
        object_format = git_object_format(source)
        tree_oid = git_tree_oid(source)
        slice_id = source_slice_id(
            upstream_revision=revision,
            tree_oid=tree_oid,
            object_format=object_format,
        )
        attestation = SourceAttestation(
            upstream_revision=revision,
            tree_oid=tree_oid,
            object_format=object_format,
            source_slice_id=slice_id,
        )
        plan = BuildPlan(
            source_slice_id=slice_id,
            phase="stock",
            platform="linux-multi",
            targets=("gfx1100",),
            variant_set=None,
        )
        platform = campaign_config.Platform(
            name="linux-multi", targets=("gfx1100",), options=()
        )
        build_cfg = campaign_config.Build(
            name="stock", options=(), variant_set=None, needs=frozenset()
        )
        worker = campaign_workers.make_build_worker(
            context=context,
            source_root=source,
            run_id="run1",
            build_plan=plan,
            platform=platform,
            build=build_cfg,
            store=store,
            binary_relative_path="bin/llama-bench",
            source_slice_id=slice_id,
            workload_id=None,
            has_generate_stage=False,
            local_provenance_class="development",
            source_attestation=attestation,
        )
        return source, context, store, plan, worker

    def test_tracked_source_mutation_fails_before_configure(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _, _, worker = self._fixture(Path(directory))
            (source / "source.txt").write_text("mutated\n", encoding="utf-8")
            real_run = subprocess.run

            def no_compile(command, **kwargs):
                if command[0] == "git":
                    return real_run(command, **kwargs)
                raise AssertionError("configure/build must not run")

            with patch(
                "bigcherry.campaign.workers.subprocess.run", side_effect=no_compile
            ) as run:
                with self.assertRaisesRegex(CampaignBuildError, "source attestation failed"):
                    worker(())
            self.assertFalse(any(call.args[0][0] != "git" for call in run.call_args_list))

    def test_mutation_immediately_before_configure_is_caught_before_configure_runs(self):
        """Adversarial-review follow-up: the worker-entry check and the
        post-compile check leave a real window between them -- computing
        configure_args doesn't read source_root, but a mutation landing in
        exactly that gap was previously caught only AFTER configure and
        compile had already run against the mutated tree. PA09 requires an
        attestation immediately before CMake configure specifically, not
        merely "somewhere before compile finishes"."""
        with tempfile.TemporaryDirectory() as directory:
            source, _, _, _, worker = self._fixture(Path(directory))

            real_configure_args = campaign_workers.campaign_build.cmake_configure_args
            real_run = subprocess.run

            def mutate_then_compute_args(*args, **kwargs):
                (source / "source.txt").write_text("mutated between args and configure\n",
                                                     encoding="utf-8")
                return real_configure_args(*args, **kwargs)

            def no_compile(command, **kwargs):
                if command[0] == "git":
                    return real_run(command, **kwargs)
                raise AssertionError("configure/build must not run once source is mutated")

            with patch.object(
                campaign_workers.campaign_build, "cmake_configure_args",
                side_effect=mutate_then_compute_args,
            ), patch("bigcherry.campaign.workers.subprocess.run", side_effect=no_compile):
                with self.assertRaisesRegex(CampaignBuildError, "source attestation failed"):
                    worker(())

    def test_source_mutation_during_compile_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            source, context, store, plan, worker = self._fixture(Path(directory))
            build_dir = build_directory(context, plan.source_slice_id, plan)
            calls = []
            real_run = subprocess.run

            def fake_compiler(command, cwd=None, check=None, **kwargs):
                if command[0] == "git":
                    return real_run(command, cwd=cwd, check=check, **kwargs)
                calls.append(command)
                if "--build" in command:
                    (source / "source.txt").write_text("mutated during compile\n", encoding="utf-8")
                    binary = build_dir / "bin" / "llama-bench"
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(b"compiled")
                else:
                    build_dir.mkdir(parents=True, exist_ok=True)
                    (build_dir / "CMakeCache.txt").write_text(
                        self._CMAKE_CACHE, encoding="utf-8"
                    )
                return subprocess.CompletedProcess(command, 0)

            with patch(
                "bigcherry.campaign.workers.subprocess.run", side_effect=fake_compiler
            ):
                with self.assertRaisesRegex(CampaignBuildError, "source attestation failed"):
                    worker(())

            self.assertEqual(len(calls), 2)
            self.assertFalse(
                any(path.is_file() for path in store.root.rglob("*"))
            )


class SmokeEnvironmentHelperTests(unittest.TestCase):
    def test_translates_hip_visible_devices(self):
        env = dict(smoke_environment_for_hip_devices("0,1"))
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0,1")
        self.assertIn("PATH", env)

    def test_does_not_also_set_rocr_visible_devices(self):
        # RE15 real-hardware finding: HIP_VISIBLE_DEVICES and
        # ROCR_VISIBLE_DEVICES are sequential filters, not aliases -- setting
        # both to the same raw index double-filters and breaks device
        # selection for any nonzero index (proven on real Brutus hardware,
        # device 2/gfx1201: "no ROCm-capable device is detected").
        env = dict(smoke_environment_for_hip_devices("2"))
        self.assertNotIn("ROCR_VISIBLE_DEVICES", env)


if __name__ == "__main__":
    unittest.main()
