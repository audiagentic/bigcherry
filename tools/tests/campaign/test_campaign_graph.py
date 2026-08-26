"""Typed DAG, resource exclusion, failure blocking, and resume tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign import CampaignRun  # noqa: E402
from bigcherry.campaign.graph import CampaignGraph, CampaignGraphError, StageNode  # noqa: E402
from bigcherry.core.resources import ResourceError, ResourceLock  # noqa: E402


class CampaignGraphTests(unittest.TestCase):
    def test_order_cycle_and_failure_blocking(self):
        graph = CampaignGraph((
            StageNode("record", "record", "s", "b", "w"),
            StageNode("tune", "tune", "s", "b", "w", ("record",)),
            StageNode("replay", "replay", "s", "b", "w", ("tune",)),
        ))
        self.assertEqual(graph.order, ("record", "tune", "replay"))
        with self.assertRaises(CampaignGraphError):
            CampaignGraph((StageNode("a", "x", None, None, None, ("b",)), StageNode("b", "x", None, None, None, ("a",))))
        run = CampaignRun(graph, Path(tempfile.mkdtemp()))
        records = run.execute(lambda stage: (_ for _ in ()).throw(RuntimeError()) if stage == "record" else ("x",), resource_root=Path(tempfile.mkdtemp()))
        self.assertEqual(records["record"].state, "failed")
        self.assertEqual(records["tune"].state, "blocked")
        self.assertEqual(records["replay"].state, "blocked")

    def test_reuse_is_repeatable_across_more_than_two_passes(self):
        # A prior version of CampaignRun.execute() only accepted
        # previous.state == "succeeded" for reuse, so after one reuse
        # (state becomes "reused") a third pass over the same unchanged
        # stage would force real re-execution for no identity reason.
        with tempfile.TemporaryDirectory() as directory:
            graph = CampaignGraph((StageNode("a", "x", "s", "b", "w"),))
            run = CampaignRun(graph, Path(directory))
            resource_root = Path(directory) / "locks"
            calls = []

            def executor(stage_id):
                calls.append(stage_id)
                return ("hash1",)

            run.execute(executor, resource_root=resource_root, reuse=lambda record: True)
            run.execute(executor, resource_root=resource_root, reuse=lambda record: True)
            run.execute(executor, resource_root=resource_root, reuse=lambda record: True)

            self.assertEqual(calls, ["a"])  # only the first pass actually ran
            self.assertEqual(run.records["a"].state, "reused")

    def test_exclusive_resource_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ResourceLock(Path(directory), "gpu:0")
            second = ResourceLock(Path(directory), "gpu:0")
            first.acquire()
            with self.assertRaises(ResourceError):
                second.acquire()
            self.assertIsNotNone(first.inspect())
            first.release()


if __name__ == "__main__":
    unittest.main()
