"""Pairwise comparison and explicit promotion contracts."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign.comparison import BenchmarkArm, ComparisonError, plan_pair  # noqa: E402
from bigcherry.tuning.promotion import PromotionError, make_pointer  # noqa: E402


def _arm(name: str, source: str = "s1", build: str = "b1") -> BenchmarkArm:
    return BenchmarkArm(
        name=name, runtime_bundle_artifact_id=name * 4, binary_artifact_id=name * 4,
        replay_cache_artifact_id=None, source_slice_id=source, build_plan_id=build,
        effective_build_id="eb1", workload_id="w1", environment=(), device="",
    )


class ComparisonPromotionTests(unittest.TestCase):
    def test_undeclared_difference_rejects_comparison(self):
        with self.assertRaises(ComparisonError):
            plan_pair(_arm("a"), _arm("b", source="s2"), label="bad")
        plan = plan_pair(
            _arm("a"), _arm("b", source="s2"), label="source-effect",
            allowed_differences=frozenset({
                "source_slice_id", "binary_artifact_id", "runtime_bundle_artifact_id"}),
        )
        self.assertEqual(plan.label, "source-effect")
        self.assertIn("source_slice_id", plan.allowed_differences)

    def test_promotion_requires_valid_report_and_preserves_hashes(self):
        pointer = make_pointer(
            release_tag="b10362", revision="r", campaign_plan_id="p",
            campaign_run_id="run", report=b"report", source_slice_id="s",
            build_id="b", binary_hash="bin",
            required_architectures=("gfx1100", "gfx1201"),
            replay_artifact_hash="replay-hash", valid=True,
        )
        self.assertEqual(pointer.document()["schema_version"], 2)
        self.assertEqual(pointer.document()["validated_campaign"]["report_hash"], pointer.report_hash)
        self.assertEqual(pointer.document()["required_architectures"], ["gfx1100", "gfx1201"])
        with self.assertRaises(PromotionError):
            make_pointer(
                release_tag="b", revision="r", campaign_plan_id="p", campaign_run_id="run",
                report=b"x", source_slice_id="s", build_id="b", binary_hash="h",
                required_architectures=("gfx1100",), replay_artifact_hash="h", valid=False)

    def test_promotion_rejects_missing_architecture_evidence(self):
        with self.assertRaises(PromotionError):
            make_pointer(
                release_tag="b10362", revision="r", campaign_plan_id="p",
                campaign_run_id="run", report=b"report", source_slice_id="s",
                build_id="b", binary_hash="bin", required_architectures=(),
                replay_artifact_hash="replay-hash", valid=True)


if __name__ == "__main__":
    unittest.main()
