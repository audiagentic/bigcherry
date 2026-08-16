"""RE14 two-phase CampaignGraph construction (materialize, then build/smoke)."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.campaign_plan import (_resource_key, build_stage_graph,  # noqa: E402
                                     materialize_stage_graph)


class MaterializeStageGraphTests(unittest.TestCase):
    def test_single_node_with_no_identity_yet(self):
        graph = materialize_stage_graph(
            source_name="bigcherry", build_name="tune",
            upstream_repo_path="/work/upstream/llama.cpp.git")
        self.assertEqual(graph.order, ("bigcherry:tune:materialize",))
        node = graph.nodes["bigcherry:tune:materialize"]
        self.assertIsNone(node.source_slice_id)
        self.assertIsNone(node.build_plan_id)
        self.assertEqual(node.kind, "materialize")
        self.assertEqual(len(node.resources), 1)
        self.assertEqual(node.resources[0].kind, "upstream-worktree")

    def test_resource_key_avoids_sanitiser_collisions(self):
        # Two distinct paths that a naive sanitiser (replace non-alnum with
        # '_') could collapse to the same string must still produce
        # different resource ids once digested.
        a = _resource_key("upstream", "/work/a-b")
        b = _resource_key("upstream", "/work/a_b")
        self.assertNotEqual(a, b)


class BuildStageGraphTests(unittest.TestCase):
    def test_real_bigcherry_tune_lane_shape(self):
        graph = build_stage_graph(
            source_name="bigcherry", build_name="tune",
            source_slice_id="s" * 32, build_plan_id="b" * 32,
            workload_id="w1", gpu_resource_ids=("gpu-0", "gpu-1"))
        self.assertEqual(graph.order, (
            "bigcherry:tune:generate", "bigcherry:tune:build", "bigcherry:tune:runtime-smoke"))

        generate = graph.nodes["bigcherry:tune:generate"]
        build = graph.nodes["bigcherry:tune:build"]
        smoke = graph.nodes["bigcherry:tune:runtime-smoke"]

        for node in (generate, build, smoke):
            self.assertEqual(node.source_slice_id, "s" * 32)
            self.assertEqual(node.build_plan_id, "b" * 32)
            self.assertEqual(node.workload_id, "w1")

        self.assertEqual(generate.dependencies, ())
        self.assertEqual(build.dependencies, ("bigcherry:tune:generate",))
        self.assertEqual(smoke.dependencies, ("bigcherry:tune:build",))

        # generate and build share one exclusive build-plan lock.
        self.assertEqual(generate.resources, build.resources)
        self.assertEqual(generate.resources[0].kind, "build-plan")

        # runtime-smoke claims GPU resources, not the build-plan lock.
        self.assertEqual({r.resource_id for r in smoke.resources}, {"gpu-0", "gpu-1"})
        self.assertTrue(all(r.kind == "gpu" for r in smoke.resources))

    def test_missing_source_slice_id_rejected(self):
        with self.assertRaises(ValueError):
            build_stage_graph(
                source_name="bigcherry", build_name="tune",
                source_slice_id="", build_plan_id="b" * 32, workload_id=None)

    def test_missing_build_plan_id_rejected(self):
        with self.assertRaises(ValueError):
            build_stage_graph(
                source_name="bigcherry", build_name="tune",
                source_slice_id="s" * 32, build_plan_id="", workload_id=None)


if __name__ == "__main__":
    unittest.main()
