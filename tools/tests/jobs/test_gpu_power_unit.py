"""PWR04: the boot unit that applies each card's configured power settings."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path, PurePosixPath

from admin.install_bigcherry_gpu_power import UNIT, main, unit_text


class GpuPowerUnitTests(unittest.TestCase):
    def test_unit_applies_at_start_and_restores_at_stop_from_the_given_checkout(self):
        text = unit_text(project_root=PurePosixPath("/srv/bc"), environment_file=PurePosixPath("/srv/env.toml"),
                         python=PurePosixPath("/usr/bin/python3"))
        self.assertIn("ExecStart=/usr/bin/python3 -m bigcherry gpu-power apply", text)
        self.assertIn("ExecStop=/usr/bin/python3 -m bigcherry gpu-power restore", text)
        self.assertIn("BIGCHERRY_ENVIRONMENT=/srv/env.toml", text)
        self.assertIn("PYTHONPATH=/srv/bc/tools", text)
        self.assertIn("RemainAfterExit=yes", text)

    def test_dest_root_writes_without_installing(self):
        with tempfile.TemporaryDirectory() as d:
            rc = main(["--project-root", "/srv/bc", "--environment-file", "/srv/env.toml", "--dest-root", d])
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "etc/systemd/system" / UNIT).is_file())


if __name__ == "__main__":
    unittest.main()
