"""RE14 CampaignStageExecutor: PipelineService wiring and the generate-stage
narrower provenance envelope (gpt-auto-agent review, verified before
implementation -- see campaign_execution.py's module docstring)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import provenance  # noqa: E402
from bigcherry.artifacts import ArtifactStore  # noqa: E402
from bigcherry.campaign_execution import (CampaignExecutionError,  # noqa: E402
                                          CampaignStageExecutor, build_expected,
                                          source_expected, workload_expected)
from bigcherry.campaign_graph import CampaignGraph, StageNode  # noqa: E402
from bigcherry.pipeline import ArtifactRef  # noqa: E402


class ExpectedDictTests(unittest.TestCase):
    def test_source_expected_has_only_source(self):
        self.assertEqual(source_expected("s1"), {"source.source_slice_id": "s1"})

    def test_build_expected_adds_build_plan_id(self):
        self.assertEqual(
            build_expected("s1", "b1"),
            {"source.source_slice_id": "s1", "build.build_plan_id": "b1"})

    def test_workload_expected_adds_workload_id(self):
        self.assertEqual(
            workload_expected("s1", "b1", "w1"),
            {"source.source_slice_id": "s1", "build.build_plan_id": "b1",
             "workload.workload_id": "w1"})


def _inventory_artifact(
    store: ArtifactStore, *, with_workload_id: bool, source_slice_id: str = "s1"
) -> ArtifactRef:
    """An inventory artifact as it genuinely exists before generate runs --
    it has source identity (it was produced against a specific source), but
    no workload_id, because workload identity is what generate establishes,
    not something inventory already carries.
    """
    workload = {"workload_id": "premature"} if with_workload_id else {}
    doc = provenance.make(
        project={}, source={"source_slice_id": source_slice_id}, build={},
        workload=workload, campaign={"run_id": "run1"})
    relative = "inventory.json"
    digest = store.publish_json(relative, {"mmq_types": ["q8_0"]})
    return ArtifactRef(kind="inventory", path=store.resolve(relative),
                       content_hash=digest, provenance=doc)


class GenerateStageEnvelopeTests(unittest.TestCase):
    def test_generate_accepts_an_inventory_with_no_workload_id(self):
        # The exact scenario the design conversation resolved: an inventory
        # artifact genuinely has no workload_id yet, and generate must not
        # reject it for that reason.
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            inventory = _inventory_artifact(store, with_workload_id=False)
            graph = CampaignGraph(nodes=(
                StageNode("gen", "generate", "s1", "b1", None, (), ()),
            ))
            holder: list[str | None] = ["s1"]

            def fake_generate(inputs):
                self.assertEqual(len(inputs), 1)
                return {"manifest": {"candidates": []}, "workload_id": "w1"}

            executor = CampaignStageExecutor(
                graph=graph, store=store, run_id="run1",
                materialize=lambda: {}, generate=fake_generate,
                source_slice_id_holder=holder, build_plan_id="b1",
                inventory_ref=inventory,
            )
            hashes = executor("gen")
            self.assertEqual(len(hashes), 1)
            output = executor.outputs["gen"][0]
            self.assertEqual(output.provenance["workload"]["workload_id"], "w1")

    def test_generate_rejects_mismatched_source_slice_id_on_input(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            # Inventory belongs to a DIFFERENT source than what materialize
            # actually produced -- this must be rejected, not silently used.
            doc = provenance.make(
                project={}, source={"source_slice_id": "different-source"},
                build={}, workload={}, campaign={"run_id": "run1"})
            digest = store.publish_json("inventory.json", {"mmq_types": []})
            inventory = ArtifactRef(kind="inventory", path=store.resolve("inventory.json"),
                                    content_hash=digest, provenance=doc)
            graph = CampaignGraph(nodes=(
                StageNode("gen", "generate", "s1", "b1", None, (), ()),
            ))
            executor = CampaignStageExecutor(
                graph=graph, store=store, run_id="run1",
                materialize=lambda: {}, generate=lambda inputs: {"manifest": {}, "workload_id": "w1"},
                source_slice_id_holder=["s1"], build_plan_id="b1",
                inventory_ref=inventory,
            )
            with self.assertRaises(Exception):
                executor("gen")

    def test_generate_rejects_missing_workload_id_from_result(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            inventory = _inventory_artifact(store, with_workload_id=False)
            graph = CampaignGraph(nodes=(
                StageNode("gen", "generate", "s1", "b1", None, (), ()),
            ))
            executor = CampaignStageExecutor(
                graph=graph, store=store, run_id="run1",
                materialize=lambda: {}, generate=lambda inputs: {"manifest": {}},
                source_slice_id_holder=["s1"], build_plan_id="b1",
                inventory_ref=inventory,
            )
            # PipelineService.run wraps the executor's raised exception in
            # a PipelineError; the underlying CampaignExecutionError is
            # still the cause.
            from bigcherry.pipeline import PipelineError
            with self.assertRaises(PipelineError) as ctx:
                executor("gen")
            self.assertIsInstance(ctx.exception.__cause__, CampaignExecutionError)


class MaterializeStageTests(unittest.TestCase):
    def test_materialize_publishes_and_sets_holder(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            graph = CampaignGraph(nodes=(
                StageNode("mat", "materialize", None, None, None, (), ()),
            ))
            holder: list[str | None] = [None]

            def fake_materialize():
                return {"source_slice_id": "real-slice-id", "upstream_revision": "rev1"}

            executor = CampaignStageExecutor(
                graph=graph, store=store, run_id="run1",
                materialize=fake_materialize, generate=lambda inputs: {},
                source_slice_id_holder=holder,
            )
            hashes = executor("mat")
            self.assertEqual(len(hashes), 1)
            self.assertEqual(holder[0], "real-slice-id")
            output = executor.outputs["mat"][0]
            self.assertEqual(output.provenance["source"]["source_slice_id"], "real-slice-id")

    def test_materialize_without_source_slice_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            graph = CampaignGraph(nodes=(
                StageNode("mat", "materialize", None, None, None, (), ()),
            ))
            executor = CampaignStageExecutor(
                graph=graph, store=store, run_id="run1",
                materialize=lambda: {"upstream_revision": "rev1"},
                generate=lambda inputs: {}, source_slice_id_holder=[None],
            )
            with self.assertRaises(Exception):
                executor("mat")


class UnconfiguredWorkerTests(unittest.TestCase):
    def test_build_stage_without_a_worker_raises_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            graph = CampaignGraph(nodes=(
                StageNode("build", "build", "s1", "b1", "w1", (), ()),
            ))
            executor = CampaignStageExecutor(
                graph=graph, store=store, run_id="run1",
                materialize=lambda: {}, generate=lambda inputs: {},
                source_slice_id_holder=["s1"], build_plan_id="b1", workload_id="w1",
            )
            with self.assertRaises(CampaignExecutionError):
                executor("build")

    def test_unknown_stage_kind_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            graph = CampaignGraph(nodes=(
                StageNode("mystery", "something-else", "s1", "b1", "w1", (), ()),
            ))
            executor = CampaignStageExecutor(
                graph=graph, store=store, run_id="run1",
                materialize=lambda: {}, generate=lambda inputs: {},
                source_slice_id_holder=["s1"],
            )
            with self.assertRaises(CampaignExecutionError):
                executor("mystery")


class FullChainTests(unittest.TestCase):
    def test_materialize_generate_build_smoke_chain(self):
        """The whole phase A + phase B chain, with fake but self-publishing
        build/smoke workers standing in for campaign_build.execute_build_stage
        and a real llama-bench invocation respectively.
        """
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            source_slice_id_holder: list[str | None] = [None]
            build_plan_id = "b" * 32

            materialize_graph = CampaignGraph(nodes=(
                StageNode("mat", "materialize", None, None, None, (), ()),
            ))
            build_graph = CampaignGraph(nodes=(
                StageNode("gen", "generate", "pending", build_plan_id, None, (), ()),
                StageNode("build", "build", "pending", build_plan_id, "pending",
                          ("gen",), ()),
                StageNode("smoke", "runtime-smoke", "pending", build_plan_id, "pending",
                          ("build",), ()),
            ))

            def fake_build(inputs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
                self.assertEqual(len(inputs), 1)
                self.assertEqual(inputs[0].kind, "manifest")
                doc = provenance.make(
                    project={}, source={"source_slice_id": source_slice_id_holder[0]},
                    build={"build_plan_id": build_plan_id}, workload={"workload_id": "w1"},
                    campaign={"run_id": "run1"})
                digest = store.publish_bytes("binary.bin", b"fake-binary-bytes")
                return (ArtifactRef(kind="binary", path=store.resolve("binary.bin"),
                                    content_hash=digest, provenance=doc),)

            def fake_smoke(inputs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
                self.assertEqual(len(inputs), 1)
                self.assertEqual(inputs[0].kind, "binary")
                doc = provenance.make(
                    project={}, source={"source_slice_id": source_slice_id_holder[0]},
                    build={"build_plan_id": build_plan_id}, workload={"workload_id": "w1"},
                    campaign={"run_id": "run1"})
                digest = store.publish_json("smoke-result.json", {"returncode": 0})
                return (ArtifactRef(kind="smoke-result", path=store.resolve("smoke-result.json"),
                                    content_hash=digest, provenance=doc),)

            materialize_executor = CampaignStageExecutor(
                graph=materialize_graph, store=store, run_id="run1",
                materialize=lambda: {"source_slice_id": "real-slice"},
                generate=lambda inputs: {},
                source_slice_id_holder=source_slice_id_holder,
            )
            materialize_executor("mat")
            self.assertEqual(source_slice_id_holder[0], "real-slice")

            # Only now, after materialize has run, does the real
            # source_slice_id exist to build a matching inventory fixture --
            # matches how a real caller would learn it.
            inventory = _inventory_artifact(
                store, with_workload_id=False, source_slice_id="real-slice")

            build_executor = CampaignStageExecutor(
                graph=build_graph, store=store, run_id="run1",
                materialize=lambda: {}, generate=lambda inputs: {"manifest": {}, "workload_id": "w1"},
                source_slice_id_holder=source_slice_id_holder,
                build_plan_id=build_plan_id, inventory_ref=inventory,
                workload_id="w1", build=fake_build, smoke=fake_smoke,
            )
            build_executor("gen")
            build_executor("build")
            build_executor("smoke")

            self.assertEqual(len(build_executor.outputs["build"]), 1)
            self.assertEqual(len(build_executor.outputs["smoke"]), 1)
            self.assertEqual(
                build_executor.outputs["smoke"][0].provenance["workload"]["workload_id"], "w1")


if __name__ == "__main__":
    unittest.main()
