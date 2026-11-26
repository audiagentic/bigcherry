"""The step-phase trace: an owner-selected diagnostic that costs nothing when it is not selected."""
from __future__ import annotations

import re
import sys
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vtp_promotion_support import assert_promoted_recipe, load_patch_files

PACKAGE = "rad0055_step_phase_trace"
PATCH_DIR = REPO / "engines/radiance/patches" / PACKAGE


class StepPhaseTraceTests(unittest.TestCase):
    def test_package_is_in_the_production_set_behind_the_serving_manifest(self):
        meta = tomllib.loads((PATCH_DIR / "patch.toml").read_text())
        self.assertEqual(meta["requires"], ["rad0070_managed_native_entry"])
        self.assertEqual(meta["plan-ids"], ["RUM09"])
        assert_promoted_recipe(self, PACKAGE)

    def test_the_trace_has_no_environment_input_and_one_owner_path(self):
        text = (PATCH_DIR / "patch.py").read_text()
        for forbidden in ("getenv", "secure_getenv", "environ"):
            self.assertNotIn(forbidden, text)
        # The directory is the sealed owner's, read from the serving component, nowhere else.
        self.assertEqual(len(re.findall(r"bc_serving_step_phase_dir\(\)", text)), 3)

    def test_the_untraced_paths_are_the_upstream_calls(self):
        edits = {e.id: e.text for f in load_patch_files(PACKAGE) for e in f.edits}
        # Untraced tape replay is the upstream call itself, with no lane change around it.
        tape = edits["phase-tape-segment"]
        self.assertIn("if (!phase_tracing_) {\n                rc = tape_play(t.dev, st.a, st.b);", tape)
        # Off, no trace event is started: phase_kv_begin refuses before any device call.
        collect = edits["phase-collect"]
        begin = collect[collect.index("size_t Ctx::phase_kv_begin"):]
        self.assertLess(begin.index("if (!phase_wall_start_ns_ || rec_) return SIZE_MAX;"),
                        begin.index("rad_event_record"))
        # The recorded tape keeps its op map only for a traced start; profiling is not forced on.
        self.assertIn("if (phase_tracing_) {\n        out->trace_ops.reserve", edits["phase-store-ops"])
        self.assertIn("phase_tracing_ = bc_serving_step_phase_dir() != nullptr;", edits["phase-init"])
        self.assertIn("if (profiling_ || phase_tracing_) profile_collect();", edits["phase-run-collect"])


if __name__ == "__main__":
    unittest.main()
