"""RHP05 stage A: heat-drift.py decides keep / re-rank from a live histogram against the active heat file (synthetic, no GPU)."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[3] / "tools/lab/radiance/heat-placement/heat-drift.py"
L, E = 128, 512


def hist(hot_experts: range, layers=range(4)) -> np.ndarray:
    """Counts with experts in `hot_experts` hot (1000 routes) and the rest cold (1), on `layers` only."""
    h = np.zeros((L, E), np.uint32)
    for l in layers:
        h[l, :] = 1
        h[l, list(hot_experts)] = 1000
    return h.reshape(-1)


def run(active_counts: np.ndarray, live: np.ndarray, *extra: str) -> tuple[int, str, Path]:
    td = Path(tempfile.mkdtemp())
    (active_counts.max() - active_counts).astype(np.uint32).tofile(td / "active.hot-first")
    live.astype(np.uint32).tofile(td / "live.rank0")
    r = subprocess.run([sys.executable, str(TOOL), str(td / "active.hot-first"), str(td / "live.rank0"),
                        "--tail", "400", "--heads", "1600", *extra], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr, td


class HeatDrift(unittest.TestCase):
    def test_matching_heat_is_kept(self):
        h = hist(range(0, 400))
        rc, out, _ = run(h, h)
        self.assertEqual(rc, 0, out)
        self.assertIn("keep the active file", out)

    def test_moved_heat_advises_a_rerank_and_writes_the_file(self):
        active, live = hist(range(0, 400)), hist(range(112, 512))   # the hot set moved onto what the active file calls cold
        new = Path(tempfile.mkdtemp()) / "new.hot-first"
        rc, out, _ = run(active, live, "--write", str(new))
        self.assertEqual(rc, 3, out)
        self.assertIn("re-rank advised", out)
        written = np.fromfile(new, dtype=np.uint32)
        np.testing.assert_array_equal(written, live.max() - live)   # hot-first: max - count of the live counts
        rc2, out2, _ = run(live, live)
        self.assertEqual(rc2, 0, out2)   # the live heat itself is a good file

    def test_a_small_tail_share_is_kept_even_if_a_rerank_would_shrink_it(self):
        # hot set moved by a little: the tail's share grows but stays under --min-share
        active, live = hist(range(0, 400)), hist(range(4, 404))
        rc, out, _ = run(active, live, "--min-share", "60")
        self.assertEqual(rc, 0, out)
        rc, out, _ = run(active, live, "--min-share", "0")
        self.assertEqual(rc, 3, out)

    def test_empty_live_histogram_is_refused(self):
        rc, out, _ = run(hist(range(0, 400)), np.zeros(L * E, np.uint32))
        self.assertEqual(rc, 1)
        self.assertIn("empty", out)


if __name__ == "__main__":
    unittest.main()
