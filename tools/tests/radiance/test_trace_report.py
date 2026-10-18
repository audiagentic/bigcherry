"""RAD10-02: the helper TRACE report bucketed by the rounds of a stream-bench record (synthetic lines, no GPU)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[3] / "tools/lab/radiance/heat-placement"
sys.path.insert(0, str(HERE))

import trace_report as tr  # noqa: E402


def line(lane, seq, layer, rows, enter, wall, wait=20.0, gpu=100.0):
    return (f"TRACE,lane,{lane},seq,{seq},layer,{layer},rows,{rows},routes,6,local_us,50.00,wait_us,{wait:.2f},"
            f"helper_gpu_us,{gpu:.2f},enter,{enter},wall_ns,{wall}")


class TraceWindows(unittest.TestCase):
    def test_steps_are_bucketed_by_round_and_small_lane_share_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            h0, h1 = Path(td, "h0.log"), Path(td, "h1.log")
            rows0, rows1 = [], []
            # round A (2 streams): 8-row steps, round B (4 streams): 16-row steps, with a wait twice as long
            for i in range(10):
                rows0.append(line(0, i, 3, 8, 1000 + i, 1_000_000_000 + i * 1000, wait=10, gpu=80))
                rows1.append(line(1, i, 3, 8, 1000 + i, 1_000_000_000 + i * 1000, wait=12, gpu=90))
            for i in range(10):
                rows0.append(line(0, 100 + i, 3, 16, 2000 + i, 5_000_000_000 + i * 1000, wait=20, gpu=150))
                rows1.append(line(1, 100 + i, 3, 16, 2000 + i, 5_000_000_000 + i * 1000, wait=24, gpu=160))
            h0.write_text("\n".join(rows0) + "\nnoise line\n")
            h1.write_text("\n".join(rows1) + "\n")
            steps = tr.read_steps([str(h0), str(h1)], 8)
            rounds = [{"streams": 2, "wall_start_ns": 900_000_000, "wall_end_ns": 1_100_000_000},
                      {"streams": 4, "wall_start_ns": 4_900_000_000, "wall_end_ns": 5_100_000_000}]
            summary = tr.window_summary(steps, rounds, 300_000_000)
            self.assertEqual(sorted(summary), [2, 4])
            self.assertEqual(summary[2]["layer_steps"], 10)
            self.assertEqual(summary[2]["small_lane_share"], 1.0)             # 8 rows is the small lane
            self.assertEqual(summary[4]["small_lane_share"], 0.0)             # 16 rows is the large lane
            self.assertAlmostEqual(summary[2]["exposed_wait_us_p50"], 12.0)   # the larger of the two helpers' waits
            self.assertAlmostEqual(summary[4]["exposed_wait_us_p50"], 24.0)
            self.assertAlmostEqual(summary[4]["helper_imbalance_p50"], (160 - 150) / 160)

    def test_lines_without_a_wall_clock_have_no_window(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td, "h.log")
            p.write_text("TRACE,lane,0,seq,1,layer,3,rows,8,routes,6,local_us,5.00,wait_us,1.00,helper_gpu_us,9.00,enter,77\n")
            steps = tr.read_steps([str(p)], 8)
            self.assertIsNone(steps[0]["wall_ns"])
            self.assertEqual(tr.window_summary(steps, [{"streams": 2, "wall_start_ns": 0, "wall_end_ns": 10**18}], 0)[2]["layer_steps"], 0)


if __name__ == "__main__":
    unittest.main()
