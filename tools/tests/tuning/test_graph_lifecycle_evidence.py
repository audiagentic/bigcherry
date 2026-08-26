"""HI14: graph_lifecycle_evidence.py's marker parsing -- no real HIP hardware
or a compiled binary needed for this layer's own correctness (the C++
instrumentation and its exact anchor placement were verified separately by
materializing an isolated worktree for patches/1231 this session)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import graph_lifecycle_evidence as gle  # noqa: E402


def _marker(stage: str) -> str:
    return f"BIGCHERRY_GRAPH_LIFECYCLE stage={stage}\n"


class ParseGraphLifecycleMarkersTests(unittest.TestCase):
    def test_all_four_stages_observed(self):
        text = "".join(_marker(s) for s in ("capture_begin", "capture_end", "instantiate", "replay"))
        observed = gle.parse_graph_lifecycle_markers(text)
        self.assertEqual(observed, {"capture_begin", "capture_end", "instantiate", "replay"})

    def test_no_markers_present(self):
        text = "some ordinary server log noise\nnothing relevant here\n"
        self.assertEqual(gle.parse_graph_lifecycle_markers(text), set())

    def test_repeated_marker_deduplicates(self):
        # The C++ side is emit-once via std::atomic_flag, but the parser
        # should be robust to a log containing the same marker more than
        # once anyway (e.g. concatenated logs from more than one process).
        text = _marker("replay") * 5
        self.assertEqual(gle.parse_graph_lifecycle_markers(text), {"replay"})

    def test_unrecognised_stage_name_is_ignored(self):
        text = _marker("capture_begin") + _marker("some_future_stage")
        observed = gle.parse_graph_lifecycle_markers(text)
        self.assertEqual(observed, {"capture_begin"})

    def test_markers_interleaved_with_unrelated_log_lines(self):
        text = (
            "INFO: server started\n"
            + _marker("capture_begin")
            + "some other diagnostic line\n"
            + _marker("capture_end")
            + _marker("instantiate")
            + "handling request...\n"
            + _marker("replay")
        )
        observed = gle.parse_graph_lifecycle_markers(text)
        self.assertEqual(observed, {"capture_begin", "capture_end", "instantiate", "replay"})


class CaptureLifecycleFromLogTests(unittest.TestCase):
    def test_full_lifecycle_produces_all_true(self):
        text = "".join(_marker(s) for s in ("capture_begin", "capture_end", "instantiate", "replay"))
        lifecycle = gle.capture_lifecycle_from_log(text)
        self.assertEqual(
            lifecycle,
            {"capture_begin": True, "capture_end": True, "instantiate": True, "replay": True},
        )

    def test_partial_lifecycle_produces_mixed_booleans(self):
        # e.g. a run that captured but crashed before ever launching --
        # exactly the kind of partial evidence this schema must reject.
        text = _marker("capture_begin") + _marker("capture_end")
        lifecycle = gle.capture_lifecycle_from_log(text)
        self.assertEqual(
            lifecycle,
            {"capture_begin": True, "capture_end": True, "instantiate": False, "replay": False},
        )

    def test_empty_log_produces_all_false(self):
        lifecycle = gle.capture_lifecycle_from_log("")
        self.assertEqual(
            lifecycle,
            {"capture_begin": False, "capture_end": False, "instantiate": False, "replay": False},
        )

    def test_output_shape_matches_multi_gpu_validate_contract(self):
        # Direct integration check: feed this module's output straight into
        # the real validator's enabled-graph-mode lifecycle check.
        from bigcherry.tuning import multi_gpu as mgv

        text = "".join(_marker(s) for s in ("capture_begin", "capture_end", "instantiate", "replay"))
        lifecycle = gle.capture_lifecycle_from_log(text)
        evidence = {
            "topology": {"device_count": 1, "ordinals": [0],
                         "devices": [{"ordinal": 0, "identity": "gfx1100"}]},
            "graph": {"mode": "enabled", "capture_observed": True, "capture_lifecycle": lifecycle},
            "per_device": [{"ordinal": 0, "dispatches": 10, "replay_dispatches": 10,
                             "signatures": 3, "winners": ["mmq:fb1"], "identity": "gfx1100"}],
        }
        mgv.validate_multi_gpu_evidence(evidence)  # must not raise


if __name__ == "__main__":
    unittest.main()
