"""RHP05 stage A: heat-watch.py --once advises a re-rank (and writes the file) or keeps the active heat (synthetic, no GPU)."""

import array
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "lab/radiance/heat-placement/heat-watch.py"
L, E = 128, 512


def hist(hot: range) -> list:
    h = [0] * (L * E)
    for l in range(4):
        for e in range(E):
            h[l * E + e] = 1000 if e in hot else 1
    return h


def write(p: Path, v) -> None:
    p.write_bytes(array.array("I", v).tobytes())


class HeatWatch(unittest.TestCase):
    def run_once(self, active, live):
        td = Path(tempfile.mkdtemp())
        top = max(active)
        write(td / "active", [top - v for v in active])
        write(td / "live.rank0", live)
        r = subprocess.run([sys.executable, str(TOOL), "--heat", str(td / "active"), "--live", str(td / "live.rank0"),
                            "--out", str(td / "new"), "--once", "--min-share", "1"], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr, td

    def test_moved_heat_is_advised_with_the_restart_command(self):
        rc, out, td = self.run_once(hist(range(0, 400)), hist(range(112, 512)))
        self.assertEqual(rc, 3, out)
        self.assertIn("RE-RANK ADVISED", out)
        self.assertIn("--heat " + str(td / "new"), out)
        self.assertTrue((td / "new").exists())

    def test_matching_heat_is_kept(self):
        h = hist(range(0, 400))
        rc, out, _ = self.run_once(h, h)
        self.assertEqual(rc, 0, out)
        self.assertIn("keep", out)


if __name__ == "__main__":
    unittest.main()
