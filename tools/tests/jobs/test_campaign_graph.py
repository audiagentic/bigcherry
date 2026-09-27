from __future__ import annotations

import unittest

from bigcherry.jobs.executor import ResourceRequest
from bigcherry.jobs.graph import CampaignGraph, CampaignGraphError
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


if __name__ == "__main__":
    unittest.main()
