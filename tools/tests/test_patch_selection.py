"""Regression checks for recipe patch-state selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import patchset  # noqa: E402


class TestPatchSelection(unittest.TestCase):
    def test_release_validated_set_includes_workspace_metrics(self):
        infos = patchset.describe()
        workspace = next(
            info for info in infos if info.name == "0900_pool_workspace_metrics"
        )
        self.assertEqual(workspace.state, "validated")

        selected = patchset.load_patches(states=frozenset({"validated"}))
        paths = {patch.path for patch in selected}
        self.assertIn("ggml/src/ggml-cuda/common.cuh", paths)

    def test_release_validated_set_includes_rdna4_fix(self):
        infos = patchset.describe()
        rdna4 = next(
            info for info in infos if info.name == "1000_rdna4_mmq_q2k_q6k_fix"
        )
        self.assertEqual(rdna4.state, "validated")

    def test_tune_does_not_define_record_capability(self):
        source = Path(__file__).resolve().parents[2] / "patches" / "0100_cmake_options.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn(
            "if (GGML_HIP_AUTOTUNE_RECORD)\n"
            "        add_compile_definitions(GGML_HIP_AUTOTUNE_RECORD)",
            text,
        )
        self.assertNotIn(
            "if (GGML_HIP_AUTOTUNE OR GGML_HIP_AUTOTUNE_RECORD)", text
        )


if __name__ == "__main__":
    unittest.main()
