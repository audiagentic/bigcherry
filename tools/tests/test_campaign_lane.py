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
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import config as campaign_config  # noqa: E402
from bigcherry import provenance  # noqa: E402
from bigcherry.artifacts import ArtifactStore  # noqa: E402
from bigcherry.campaign_lane import (CampaignLaneError, CampaignLaneExecutionSpec,  # noqa: E402
                                     execute_campaign_lane,
                                     smoke_environment_for_hip_devices)
from bigcherry.context import ProjectContext  # noqa: E402
from bigcherry.pipeline import ArtifactRef  # noqa: E402
from bigcherry.runtime_smoke import RuntimeSmokeSpec  # noqa: E402


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
            project_root=directory, config_path=directory / "recipes.toml",
            artifacts_root=directory / "artifacts", work_root=directory / "work",
            upstream_repo=upstream, overlay_root=directory / "src",
            patches_root=directory / "patches",
        )
        self.store = ArtifactStore(directory / "store")
        self.cfg = campaign_config.Config(
            pinned=revision, patch_sets={},
            sources={"test-source": campaign_config.Source(
                name="test-source", ref=revision, overlay=False, patch_sets=())},
            builds={"tune": campaign_config.Build(
                name="tune", options=(), variant_set="workload-max",
                needs=frozenset({"inventory"}))},
            platforms={"linux-multi": campaign_config.Platform(
                name="linux-multi", targets=("gfx1100",), options=())},
            experiments={}, campaigns={}, path=directory / "recipes.toml",
        )
        self.calls: list[str] = []

    def spec(self, **overrides) -> CampaignLaneExecutionSpec:
        inventory_path = Path(tempfile.mkdtemp()) / "inventory.json"
        inventory_path.write_text('{"mmq_types": ["q8_0"]}', encoding="utf-8")
        values = dict(
            source_name="test-source", build_name="tune", platform_name="linux-multi",
            architectures=("gfx1100",), inputs=(("inventory", inventory_path),),
            validation=RuntimeSmokeSpec(model_path=Path("model.gguf")),
        )
        values.update(overrides)
        return CampaignLaneExecutionSpec(**values)

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
                    project={}, source={"source_slice_id": source_slice_id},
                    build={"build_plan_id": build_plan_id}, workload={"workload_id": workload_id},
                    campaign={"run_id": run_id})
                relative = f"builds/{build_plan_id}/{run_id}/bin"
                digest = store.publish_bytes(relative, b"fake-binary")
                return (ArtifactRef(kind="binary", path=store.resolve(relative),
                                    content_hash=digest, provenance=doc),)
            return build

        def fake_smoke_worker(**kwargs):
            source_slice_id = kwargs["source_slice_id"]
            build_plan_id = kwargs["build_plan_id"]
            workload_id = kwargs["workload_id"]
            run_id = kwargs["run_id"]

            def smoke(inputs):
                calls.append("smoke")
                doc = provenance.make(
                    project={}, source={"source_slice_id": source_slice_id},
                    build={"build_plan_id": build_plan_id}, workload={"workload_id": workload_id},
                    campaign={"run_id": run_id})
                relative = f"runs/{run_id}/{build_plan_id}/smoke.json"
                digest = store.publish_json(relative, {"ok": True})
                return (ArtifactRef(kind="smoke-result", path=store.resolve(relative),
                                    content_hash=digest, provenance=doc),)
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
                        "generated_tree": {"schema_version": 1, "files": {}, "compile_inputs": [],
                                           "compile_inputs_hash": "h"},
                    }
                return generate

            with patch("bigcherry.campaign_lane.campaign_workers.make_generate_worker",
                       side_effect=fake_generate_worker), \
                 patch("bigcherry.campaign_lane.campaign_workers.make_build_worker",
                       side_effect=fake_build), \
                 patch("bigcherry.campaign_lane.campaign_workers.make_smoke_worker",
                       side_effect=fake_smoke):
                result = execute_campaign_lane(
                    harness.spec(), cfg=harness.cfg, context=harness.context,
                    store=harness.store, run_id="run1")

            self.assertEqual(harness.calls, ["generate", "build", "smoke"])  # exactly once each
            self.assertEqual(result.run_id, "run1")
            self.assertEqual(result.resolved_revision, revision)
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
                        "generated_tree": {"schema_version": 1, "files": {}, "compile_inputs": [],
                                           "compile_inputs_hash": "h"},
                    }
                return generate

            _, fake_build, fake_smoke = harness.fake_workers()

            inventory_a = Path(tempfile.mkdtemp()) / "inv.json"
            inventory_a.write_text('{"mmq_types": ["q8_0"]}', encoding="utf-8")
            inventory_b = Path(tempfile.mkdtemp()) / "inv.json"
            inventory_b.write_text('{"mmq_types": ["f16"]}', encoding="utf-8")

            with patch("bigcherry.campaign_lane.campaign_workers.make_generate_worker",
                       side_effect=fake_generate_worker), \
                 patch("bigcherry.campaign_lane.campaign_workers.make_build_worker",
                       side_effect=fake_build), \
                 patch("bigcherry.campaign_lane.campaign_workers.make_smoke_worker",
                       side_effect=fake_smoke):
                result_a = execute_campaign_lane(
                    harness.spec(inputs=(("inventory", inventory_a),)), cfg=harness.cfg,
                    context=harness.context, store=harness.store, run_id="lane-a")
                result_b = execute_campaign_lane(
                    harness.spec(inputs=(("inventory", inventory_b),)), cfg=harness.cfg,
                    context=harness.context, store=harness.store, run_id="lane-b")

            inventory_ref_a = dict(result_a.input_refs)["inventory"]
            inventory_ref_b = dict(result_b.input_refs)["inventory"]
            self.assertNotEqual(inventory_ref_a.content_hash, inventory_ref_b.content_hash)
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

            with patch("bigcherry.campaign_lane.campaign_workers.make_generate_worker",
                       side_effect=fake_generate), \
                 patch("bigcherry.campaign_lane.campaign_workers.make_build_worker",
                       side_effect=fake_build), \
                 patch("bigcherry.campaign_lane.campaign_workers.make_smoke_worker",
                       side_effect=fake_smoke):
                with self.assertRaises(CampaignLaneError):
                    execute_campaign_lane(
                        harness.spec(inputs=()), cfg=harness.cfg, context=harness.context,
                        store=harness.store, run_id="run1")


class SmokeEnvironmentHelperTests(unittest.TestCase):
    def test_translates_hip_visible_devices(self):
        env = dict(smoke_environment_for_hip_devices("0,1"))
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "0,1")
        self.assertIn("PATH", env)


if __name__ == "__main__":
    unittest.main()
