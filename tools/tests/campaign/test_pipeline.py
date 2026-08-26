"""Per-slice pipeline identity boundary tests."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import provenance # noqa: E402
from bigcherry.core.pipeline import ArtifactRef, PipelineError, PipelineService  # noqa: E402


def _artifact(kind: str, source: str = "s1") -> ArtifactRef:
    return ArtifactRef(
        kind=kind, path=Path(kind), content_hash="a" * 64,
        provenance=provenance.make(
            project={"bigcherry_revision": "b"},
            source={"source_slice_id": source},
            build={"build_id": "b1"},
            workload={"workload_id": "w1"},
            campaign={"campaign_run_id": "r1"},
        ),
    )


class PipelineTests(unittest.TestCase):
    def test_mismatch_fails_before_executor(self):
        calls = []
        service = PipelineService(lambda stage, inputs: calls.append(stage) or (_artifact("out"),))
        with self.assertRaises(PipelineError):
            service.tune(
                expected={"source.source_slice_id": "s1", "workload.workload_id": "w1"},
                inputs=(_artifact("inventory", source="other"),),
            )
        self.assertEqual(calls, [])

    def test_lifecycle_accepts_matching_artifacts(self):
        calls = []
        service = PipelineService(lambda stage, inputs: calls.append(stage) or (_artifact("out"),))
        output = service.build_inventory(
            expected={"source.source_slice_id": "s1", "workload.workload_id": "w1"},
            inputs=(_artifact("record"),),
        )
        self.assertEqual(output[0].kind, "out")
        self.assertEqual(calls, ["inventory"])

    def test_incompatible_output_is_rejected(self):
        service = PipelineService(lambda stage, inputs: (_artifact("out", source="wrong"),))
        with self.assertRaises(PipelineError):
            service.tune(
                expected={"source.source_slice_id": "s1"},
                inputs=(_artifact("inventory"),),
            )


if __name__ == "__main__":
    unittest.main()
