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


if __name__ == "__main__":
    unittest.main()
