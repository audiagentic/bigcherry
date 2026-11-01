"""RHP05 stage A: heat-drift.py decides keep / re-rank from a live histogram against the active heat file (synthetic, no GPU)."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import array

TOOL = Path(__file__).resolve().parents[3] / "tools/lab/radiance/heat-placement/heat-drift.py"
L, E = 128, 512


def hist(hot_experts: range, layers=range(4)) -> list:
    """Counts with experts in `hot_experts` hot (1000 routes) and the rest cold (1), on `layers` only."""
    h = [0] * (L * E)
    for l in layers:
        for e in range(E):
            h[l * E + e] = 1000 if e in hot_experts else 1
    return h


def write(path: Path, values) -> None:
    path.write_bytes(array.array("I", values).tobytes())


def run(active_counts: list, live: list, *extra: str) -> tuple:
    td = Path(tempfile.mkdtemp())
    top = max(active_counts)
    write(td / "active.hot-first", [top - v for v in active_counts])
    write(td / "live.rank0", live)
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
        written = array.array("I"); written.frombytes(new.read_bytes())
        hi = max(live)
        self.assertEqual(list(written), [hi - v for v in live])   # hot-first: max - count of the live counts
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
        rc, out, _ = run(hist(range(0, 400)), [0] * (L * E))
        self.assertEqual(rc, 1)
        self.assertIn("empty", out)


if __name__ == "__main__":
    unittest.main()
