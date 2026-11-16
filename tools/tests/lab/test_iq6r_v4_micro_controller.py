"""Card-free monotonic receipt and fail-closed micro-controller tests."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[2] / "lab/radiance"
sys.path.insert(0, str(TOOLS))
PATH = TOOLS / "iq6r_v4_micro_controller.py"
spec = importlib.util.spec_from_file_location("iq6r_v4_micro_controller", PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class MicroControllerTests(unittest.TestCase):
    def test_starts_at_stage_two_without_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:  # noqa: SIM117
            with (
                patch.object(mod, "MICRO_ROOT", Path(tmp)),
                patch.object(mod, "predecessor", return_value=(None, {})) as pred,
            ):
                self.assertEqual(mod.next_stage([b"r"] * 289), 2)
                pred.assert_called_once()

    def test_contiguous_receipts_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stage in (2, 3, 4):
                (root / f"micro-{stage:04d}.json").write_text(json.dumps({"stage": stage}))
            with (
                patch.object(mod, "MICRO_ROOT", root),
                patch.object(mod, "predecessor", return_value=(root, {})) as pred,
            ):
                self.assertEqual(mod.next_stage([b"r"] * 289), 5)
                pred.assert_called_once()

    def test_missing_receipt_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stage in (2, 4):
                (root / f"micro-{stage:04d}.json").write_text(json.dumps({"stage": stage}))
            with (
                patch.object(mod, "MICRO_ROOT", root),
                self.assertRaisesRegex(RuntimeError, "not contiguous"),
            ):
                mod.next_stage([b"r"] * 289)

    def test_stage_failure_retains_status(self):
        with (
            patch.object(mod, "stage_environment", return_value={}),
            patch.object(mod.subprocess, "run", return_value=type("R", (), {"returncode": 124})()) as runner,
        ):
            self.assertEqual(mod.run(2, 10800), 124)
            cmd = runner.call_args.args[0]
            self.assertIn("--wait", cmd)
            self.assertEqual(cmd[cmd.index("--stage") + 1], "2")
            self.assertEqual(cmd[cmd.index("--stage-timeout") + 1], "10800")


if __name__ == "__main__":
    unittest.main()
