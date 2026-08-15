"""Content-addressed build identity contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.builds import (BuildIdentityError, BuildPlan, binary_hash,
                              effective_build_id, validate_reuse)  # noqa: E402


class BuildIdentityTests(unittest.TestCase):
    def _plan(self, **changes):
        values = dict(
            source_slice_id="s1", phase="record", platform="brutus",
            targets=("gfx1100",), cmake_options=(("GGML_HIP", "ON"),),
            variant_set="inventory", environment=(("CC", "clang"),),
        )
        values.update(changes)
        return BuildPlan(**values)

    def test_build_plan_changes_for_every_material_input(self):
        base = self._plan().build_plan_id
        for field, value in {
            "source_slice_id": "s2", "phase": "tune", "platform": "other",
            "targets": ("gfx1201",), "cmake_options": (("GGML_HIP", "OFF"),),
            "variant_set": "full-max", "inventory_hash": "i",
            "winners_hash": "w", "resource_report_hashes": ("r",),
            "environment": (("CXXFLAGS", "-O0"),),
        }.items():
            self.assertNotEqual(base, self._plan(**{field: value}).build_plan_id, field)

    def test_reuse_validates_configure_and_binary_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "bench"
            binary.write_bytes(b"binary")
            plan = self._plan()
            configure = {"source": "s1", "generator": "Ninja", "options": {"A": "B"}}
            metadata = {
                "source_slice_id": plan.source_slice_id,
                "build_plan_id": plan.build_plan_id,
                "effective_configure": configure,
                "build_id": effective_build_id(configure),
                "binary_hash": binary_hash(binary),
            }
            validate_reuse(metadata, plan, binary=binary)
            binary.write_bytes(b"changed")
            with self.assertRaisesRegex(BuildIdentityError, "binary hash"):
                validate_reuse(metadata, plan, binary=binary)


if __name__ == "__main__":
    unittest.main()
