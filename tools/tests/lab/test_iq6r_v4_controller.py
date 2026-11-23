"""Offline controller tests: receipt continuity, leases, reuse and fail-close."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / "lab/radiance/iq6r_v4_controller.py"
sys.path.insert(0, str(PATH.parent))
spec = importlib.util.spec_from_file_location("iq6r_v4_controller", PATH)
assert spec and spec.loader
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def test_stage_environment_includes_numpy(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            (Path(tmp) / "pydeps/numpy").mkdir(parents=True)
            with patch.object(controller, "WORK", Path(tmp)):
                value = controller.stage_environment()
            self.assertEqual(value["PYTHONPATH"].split(controller.os.pathsep)[0],
                             str(Path(tmp) / "pydeps"))

    def test_lease_command_uses_stage_limit_and_wait(self):
        command = controller.lease_command(2, 10800)
        self.assertEqual(command[:4], ["agent-lease", "acquire", "gpu", "--wait"])
        self.assertEqual(command[command.index("--stage-limit") + 1], "2")
        self.assertEqual(command[command.index("--stage-timeout") + 1], "10800")
        self.assertIn("IQ6R staged R9700 stage 02", command)

    def test_fail_missing_stage1_receipt(self):
        with (tempfile.TemporaryDirectory() as tmp,
              patch.object(controller, "STAGED", Path(tmp)),
              self.assertRaisesRegex(RuntimeError, "stage 1 missing")):
                    controller.check_stages({
                        "stages": 16, "chunk": 20,
                        "full_recipe_sha256": controller.PINNED_RECIPE_SHA256,
                        "release": "bc-radiance-1170.26.0",
                        "production_promotion": False,
                        "done": [{"stage": 1}],
                    })

    def test_verified_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root / "stage-01.rad"
            recipe = root / "stage-01.recipe"
            stage.write_bytes(b"sealed RAD2 fixture")
            recipe.write_text("fixture recipe")
            digest = hashlib.sha256(stage.read_bytes()).hexdigest()
            item = {
                "stage": 1, "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
                "bytes": stage.stat().st_size, "sha256": digest,
            }
            (root / "stage-01.json").write_text(json.dumps(item))
            data = {
                "stages": 16, "chunk": 20,
                "full_recipe_sha256": controller.PINNED_RECIPE_SHA256,
                "release": "bc-radiance-1170.26.0",
                "production_promotion": False, "done": [item],
            }
            with patch.object(controller, "STAGED", root):
                self.assertEqual(controller.check_stages(data), 1)
                stage.write_bytes(b"modified data")
                with self.assertRaisesRegex(RuntimeError, "missing or modified"):
                    controller.check_stages(data)

    def test_rejects_missing_proof_of_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stage-02-convert-stderr.log").write_text("conversion succeeded\n")
            with (patch.object(controller, "STAGED", root),
              self.assertRaisesRegex(RuntimeError, "lacks evidence")):
                    controller.validate_reuse(2)

    def test_rejects_requantized_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stage-02-convert-stderr.log").write_text(
                "libquant: GPTQ quantises 'output.k6'\n"
                "I --reuse: 1029 weight(s), 25 GiB, copied\n"
            )
            with (patch.object(controller, "STAGED", root),
              self.assertRaisesRegex(RuntimeError, "recomputed")):
                    controller.validate_reuse(2)

    def test_positive_reuse_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stage-02-convert-stderr.log").write_text(
                "I --reuse: 1015 weight(s), 24 GiB, copied\n"
            )
            with patch.object(controller, "STAGED", root):
                self.assertEqual(controller.validate_reuse(2)["copied_weights"], 1015)


if __name__ == "__main__":
    unittest.main()
