"""Card-free checks of the Gate E runner (tools/lab/radiance/gate-e.sh): syntax and argument handling only."""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "lab" / "radiance" / "gate-e.sh"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=60)


class GateERunnerTests(unittest.TestCase):
    def test_script_parses(self) -> None:
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)

    def test_no_tag_prints_usage(self) -> None:
        r = run()
        self.assertEqual(r.returncode, 2)
        self.assertIn("gate-e.sh TAG", r.stderr)

    def test_unknown_option_and_bad_timeout_are_refused(self) -> None:
        self.assertEqual(run("2020.11.2", "--bogus").returncode, 2)
        r = run("2020.11.2", "--wait", "--wait-timeout", "0")
        self.assertEqual(r.returncode, 2)
        self.assertIn("positive integer", r.stderr)

    def test_script_deletes_only_its_own_temp_directory(self) -> None:
        text = SCRIPT.read_text()
        self.assertEqual(text.count("rm -rf"), 1)
        self.assertIn('rm -rf -- "${TMP:?}"', text)


if __name__ == "__main__":
    unittest.main()
