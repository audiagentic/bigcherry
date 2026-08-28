"""Regression checks for recipe patch-state selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import patchset
from bigcherry import replay_build_audit # noqa: E402


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
        source = Path(__file__).resolve().parents[3] / "patches" / "0100_cmake_options" / "patch.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn(
            "if (GGML_HIP_AUTOTUNE_RECORD)\n"
            "        add_compile_definitions(GGML_HIP_AUTOTUNE_RECORD)",
            text,
        )
        self.assertNotIn(
            "if (GGML_HIP_AUTOTUNE OR GGML_HIP_AUTOTUNE_RECORD)", text
        )

    def test_transforms_fail_closed_without_recording_and_dispatch(self):
        source = Path(__file__).resolve().parents[3] / "patches" / "0100_cmake_options" / "patch.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn(
            "if (GGML_HIP_ROUTING_TRANSFORM AND\n"
            "        (NOT GGML_HIP_AUTOTUNE OR NOT GGML_HIP_AUTOTUNE_RECORD))",
            text,
        )
        self.assertIn(
            "GGML_HIP_ROUTING_TRANSFORM requires both GGML_HIP_AUTOTUNE and",
            text,
        )
        self.assertIn(
            "GGML_HIP_AUTOTUNE_RECORD until transform recording and dispatch",
            text,
        )

    def test_replay_source_partition_excludes_tuner_record_and_sqlite(self):
        source = Path(__file__).resolve().parents[3] / "patches" / "0100_cmake_options" / "patch.py"
        audit = replay_build_audit.audit_replay_source_partition(
            source.read_text(encoding="utf-8")
        )
        self.assertIn("hip-autotune-replay.cpp", audit.replay_only)
        self.assertNotIn("hip-autotune-record.cpp", audit.replay_sources)
        self.assertNotIn("hip-autotune-tuner.cu", audit.replay_sources)
        self.assertNotIn("hip-autotune-journal.cpp", audit.replay_sources)
        self.assertNotIn("hip-autotune-smi.cpp", audit.replay_sources)

    def test_replay_partition_retains_strict_loader_and_coverage(self):
        source = Path(__file__).resolve().parents[3] / "patches" / "0100_cmake_options" / "patch.py"
        audit = replay_build_audit.audit_replay_source_partition(
            source.read_text(encoding="utf-8")
        )
        self.assertTrue(
            replay_build_audit.REPLAY_REQUIRED <= audit.replay_sources
        )
        self.assertTrue(any("No SQLite anywhere" in line for line in audit.sqlite_mentions))

    def test_replay_build_option_is_exclusive_from_tuning(self):
        source = (Path(__file__).resolve().parents[3] / "patches" /
                  "0100_cmake_options" / "patch.py").read_text(encoding="utf-8")
        self.assertIn(
            "GGML_HIP_DISPATCH_REPLAY and GGML_HIP_AUTOTUNE are mutually",
            source,
        )


if __name__ == "__main__":
    unittest.main()
