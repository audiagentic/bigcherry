"""Ensure the standalone RHP design mock remains executable in normal unittest runs."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

REPO = Path(__file__).resolve().parents[3]
MOCK = REPO / "tools/lab/radiance/heat-placement/heat_mock.mjs"


class HeatPlacementMockSmoke(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the optional RHP mock")
    def test_cpu_mock_regressions(self):
        proc = subprocess.run(
            ["node", str(MOCK)], text=True, capture_output=True, timeout=15, check=False
        )
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        record = json.loads(proc.stdout)
        self.assertEqual(record["tests"], 23)
        self.assertEqual(record["failures"], 0)
        self.assertIn("NOT GPU", record["scope"])


if __name__ == "__main__":
    unittest.main()
