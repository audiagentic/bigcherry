"""Card-free tests of single-tensor IQ6R Stage-1 migration and fail-closed resume."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[2] / "lab/radiance"
sys.path.insert(0, str(TOOLS))
PATH = TOOLS / "iq6r_v4_micro_convert.py"
spec = importlib.util.spec_from_file_location("iq6r_v4_micro_convert", PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class MicroConvertTests(unittest.TestCase):
    def test_stage_sequence_one_new_trunk_per_stage(self):
        data = mod.micro_recipes()
        self.assertEqual(len(data), 289)
        self.assertEqual([x.count(b" gptq ") for x in data[:5]], [4, 5, 6, 7, 8])
        self.assertEqual(data[-1], (mod.PINNED / "data/recipes/q38-27b-iq6r-v4.recipe").read_bytes())

    def test_micro_stage_one_is_original_recipe(self):
        data = mod.micro_recipes()
        self.assertEqual(data[0], mod.ORIGINAL_STAGE_ONE_RECIPE.read_bytes())

    def test_stage_plan_boundaries(self):
        sources = [b"x"] * 289
        original = {"sha256": "sealed"}
        self.assertEqual(mod.planned_stage(2, sources, original)["gptq_after"], 5)
        self.assertEqual(mod.planned_stage(289, sources, original)["gptq_after"], 292)
        with self.assertRaises(ValueError):
            mod.planned_stage(1, sources, original)
        with self.assertRaises(ValueError):
            mod.planned_stage(290, sources, original)

    def test_stage_one_validation_detects_corrupt_or_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            artifact = d / "stage-01.rad"
            recipe = d / "stage-01.recipe"
            status = d / "status.json"
            artifact.write_bytes(b"sealed-data")
            recipe.write_bytes(b"pinned-recipe")
            record = {
                "stage": 1, "bytes": artifact.stat().st_size,
                "sha256": mod.sha(artifact), "recipe_sha256": digest(recipe.read_bytes()),
            }
            status.write_text(json.dumps({
                "release": "bc-radiance-1170.26.0", "chunk": 20,
                "done": [record],
            }))
            with (patch.object(mod, "ORIGINAL_STAGE_ONE", artifact),
                  patch.object(mod, "ORIGINAL_STAGE_ONE_RECIPE", recipe),
                  patch.object(mod, "STATUS", status)):
                self.assertEqual(mod.stage_one_record(recipe.read_bytes()), record)
                with self.assertRaisesRegex(RuntimeError, "recipe differs"):
                    mod.stage_one_record(b"modified")
                artifact.write_bytes(b"bad-bytes!!")
                with self.assertRaisesRegex(RuntimeError, "hash or recipe"):
                    mod.stage_one_record(recipe.read_bytes())

    def test_micro_predecessor_rejects_tampered_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "micro-0002.rad"
            r = root / "micro-0002.json"
            a.write_bytes(b"artifact")
            rec = {
                "stage": 2, "sha256": digest(a.read_bytes()),
                "bytes": a.stat().st_size, "recipe_sha256": digest(b"recipe2"),
            }
            r.write_text(json.dumps(rec))
            with patch.object(mod, "MICRO_ROOT", root):
                self.assertEqual(mod.predecessor(3, [b"recipe1", b"recipe2", b"recipe3"]), (a, rec))
                a.write_bytes(b"tampered")
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    mod.predecessor(3, [b"recipe1", b"recipe2", b"recipe3"])

    def test_preflight_no_gpu_lease_required(self):
        import contextlib
        from io import StringIO

        original = {"sha256": "sealed"}
        sources = [b"recipe"] * 289
        with (patch.object(mod, "micro_recipes", return_value=sources),
              patch.object(mod, "stage_one_record", return_value=original),
              patch.object(sys, "argv", [str(PATH), "--stage", "2", "--preflight"])):
            with contextlib.redirect_stdout(StringIO()) as captured:
                self.assertEqual(mod.main(), 0)
            self.assertEqual(json.loads(captured.getvalue())["gptq_after"], 5)


if __name__ == "__main__":
    unittest.main()
