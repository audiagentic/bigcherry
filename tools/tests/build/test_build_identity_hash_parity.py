"""RRVP02 shared hash extraction preserves existing build identities."""

from __future__ import annotations

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.build.builds import (
    BuildPlan,
    build_verification_id,
    effective_build_id,
    runtime_bundle_hash,
)


class BuildIdentityHashParityTests(unittest.TestCase):
    def test_existing_build_plan_id_is_unchanged(self):
        plan = BuildPlan(
            source_slice_id="s1",
            phase="record",
            platform="brutus",
            targets=("gfx1100",),
            cmake_options=(("GGML_HIP", "ON"),),
            variant_set="inventory",
            environment=(("CC", "clang"),),
        )
        self.assertEqual(plan.build_plan_id, "4ddcbdc25d421aa0e688a68dd72c1563")

    def test_existing_effective_build_id_is_unchanged(self):
        configure = {"source": "s1", "generator": "Ninja", "options": {"A": "B"}}
        self.assertEqual(
            effective_build_id(configure),
            "9186aa9eece0ed737a010d7fd7a7042d",
        )

    def test_existing_runtime_bundle_hash_is_unchanged(self):
        artifacts = {"llama-bench": "aaa", "libggml-hip.so.0.19.0": "bbb"}
        self.assertEqual(
            runtime_bundle_hash(artifacts),
            "74227b70676ef5860f6071732809562c",
        )

    def test_existing_build_verification_id_is_unchanged(self):
        evidence = {"command_source": "compile_commands.json", "checks": []}
        self.assertEqual(
            build_verification_id(evidence),
            "70863a6d06aef129b5070b86418f834a",
        )


if __name__ == "__main__":
    unittest.main()
