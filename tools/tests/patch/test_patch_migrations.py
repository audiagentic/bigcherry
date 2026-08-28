from __future__ import annotations

import unittest
from pathlib import Path

from bigcherry.patch import registry as patch_registry, validation as patch_validation


class RD12PackageMigrationTests(unittest.TestCase):
    def test_simple_patch_migration_is_packaged(self) -> None:
        root = Path(__file__).resolve().parents[3] / "patches"
        registry = patch_registry.load_registry(root)
        descriptor = registry.get("1002_hip_unsafe_math_opt_in")
        self.assertEqual(descriptor.representation, patch_registry.REPRESENTATION_PACKAGED)
        self.assertEqual(descriptor.package_root, Path("1002_hip_unsafe_math_opt_in"))
        self.assertEqual(descriptor.metadata_path, Path("1002_hip_unsafe_math_opt_in/patch.toml"))
        self.assertTrue(patch_registry.load_implementation(descriptor, root=root))
        self.assertFalse((root / "1002_hip_unsafe_math_opt_in.py").exists())
    def test_rd12_is_packaged_and_validation_owns_trace_marker(self) -> None:
        root = Path(__file__).resolve().parents[3] / "patches"
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

    def test_rd08_is_packaged_with_contract_and_vdr_marker(self) -> None:
        root = Path(__file__).resolve().parents[3] / "patches"
        registry = patch_registry.load_registry(root)
        descriptor = registry.get("1204_rd08_q6k_mmvq_vdr2")
        self.assertEqual(descriptor.representation, patch_registry.REPRESENTATION_PACKAGED)
        self.assertEqual(descriptor.experiment_contract, "RD08-Q6K-MMVQ-VDR2")
        self.assertEqual(len(patch_registry.load_implementation(descriptor, root=root)), 4)
        plan = patch_validation.build_plan_for_patch(descriptor, root=root)
        assert plan is not None
        self.assertIn("controls", plan.required_capabilities)
        trace = next(check for check in plan.checks if check.validator == "trace-marker")
        self.assertIn("1204_rd08", trace.config["marker-regex"])
        self.assertNotIn("1204_rd08_q6k_mmvq_vdr2.py", {p.name for p in root.glob("*.py")})

    def test_rd13_is_packaged_with_validation_owned_marker(self) -> None:
        root = Path(__file__).resolve().parents[3] / "patches"
        registry = patch_registry.load_registry(root)
        descriptor = registry.get("1206_rd13_mul_mat_add_view_fusion")
        self.assertEqual(descriptor.representation, patch_registry.REPRESENTATION_PACKAGED)
        self.assertEqual(len(patch_registry.load_implementation(descriptor, root=root)), 1)
        plan = patch_validation.build_plan_for_patch(descriptor, root=root)
        assert plan is not None
        trace = next(check for check in plan.checks if check.validator == "trace-marker")
        self.assertIn("1206_rd13", trace.config["marker-regex"])
        self.assertNotIn("1206_rd13_mul_mat_add_view_fusion.py", {p.name for p in root.glob("*.py")})
