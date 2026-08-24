from __future__ import annotations

import unittest
from pathlib import Path

from bigcherry import patch_registry, patch_validation


class RD12PackageMigrationTests(unittest.TestCase):
    def test_rd12_is_packaged_and_validation_owns_trace_marker(self) -> None:
        root = Path(__file__).resolve().parents[2] / "patches"
        registry = patch_registry.load_registry(root)
        descriptor = registry.get("1205_rd12_paired_mmvq_dual_output")
        self.assertEqual(descriptor.representation, patch_registry.REPRESENTATION_PACKAGED)
        self.assertEqual(descriptor.validation_path.name, "validation.toml")
        self.assertEqual(len(patch_registry.load_implementation(descriptor, root=root)), 3)
        plan = patch_validation.build_plan_for_patch(descriptor, root=root)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            {check.validator for check in plan.checks}, {"apply", "build", "trace-marker"}
        )
        self.assertNotIn("1205_rd12_paired_mmvq_dual_output.py", {p.name for p in root.glob("*.py")})
