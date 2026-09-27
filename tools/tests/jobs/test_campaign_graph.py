from __future__ import annotations

import unittest

from bigcherry.jobs.executor import ResourceRequest
from bigcherry.jobs.graph import CampaignGraph, CampaignGraphError, compile_validation_graph
from bigcherry.jobs.operations import OperationSpec


def op(name: str, *deps: str) -> OperationSpec:
    return OperationSpec(
        operation_id=name,
        kind=name,
        command_semantics={"stage": name},
        environment_semantics=(),
        resources=ResourceRequest(1, None, "stage", None, 60),
        dependencies=tuple(deps),
        declared_outputs=(),
    )


class CampaignGraphTests(unittest.TestCase):
    def test_topological_order_is_deterministic_not_insertion_order(self):
        graph_a = CampaignGraph((op("report", "harvest"), op("build", "prepare"), op("harvest", "measure"), op("prepare"), op("measure", "build")))
        graph_b = CampaignGraph(tuple(reversed(graph_a.operations)))
        expected = ("prepare", "build", "measure", "harvest", "report")
        self.assertEqual(tuple(x.operation_id for x in graph_a.topological_order()), expected)
        self.assertEqual(tuple(x.operation_id for x in graph_b.topological_order()), expected)

    def test_missing_dependency_is_rejected(self):
        with self.assertRaises(CampaignGraphError):
            CampaignGraph((op("measure", "build"),))

    def test_cycle_is_rejected(self):
        with self.assertRaises(CampaignGraphError):
            CampaignGraph((op("a", "b"), op("b", "a")))

    def test_ready_returns_only_dependency_satisfied_operations(self):
        graph = CampaignGraph((op("prepare"), op("build", "prepare"), op("measure", "build"), op("ladder", "measure")))
        self.assertEqual(tuple(x.operation_id for x in graph.ready(())), ("prepare",))
        self.assertEqual(tuple(x.operation_id for x in graph.ready(("prepare",))), ("build",))
        self.assertEqual(tuple(x.operation_id for x in graph.descendants("build")), ("ladder", "measure"))

    def test_compiled_graph_keeps_timed_work_host_exclusive_by_activity_class(self):
        graph = compile_validation_graph(
            architecture="gfx1100",
            reserved_gpu_count=2,
            semantic_request={"patch": "p", "model_hash": "abc"},
            include_production_lane=True,
        )
        by_id = graph.by_id
        self.assertEqual(by_id["prepare"].resources.activity_class, "build")
        self.assertIsNone(by_id["prepare"].resources.gpu)
        self.assertEqual(by_id["correctness-activation"].resources.activity_class, "correctness")
        for name in ("timed-performance", "reference-ladder", "production-lane"):
            self.assertEqual(by_id[name].resources.activity_class, "timed-measure")
            self.assertEqual(by_id[name].resources.gpu.architecture, "gfx1100")
            self.assertEqual(by_id[name].resources.gpu.reserved_count, 2)
        self.assertEqual(by_id["harvest"].resources.activity_class, "harvest")
        self.assertIsNone(by_id["harvest"].resources.gpu)
        self.assertEqual(
            tuple(item.operation_id for item in graph.topological_order()),
            (
                "prepare",
                "correctness-activation",
                "timed-performance",
                "reference-ladder",
                "production-lane",
                "evidence-finalize",
                "harvest",
                "report",
            ),
        )


if __name__ == "__main__":
    unittest.main()
