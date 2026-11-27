"""Offline regression tests for IQ6R-V4 inventory (no GPU or converter)."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "lab/radiance/iq6r_v4_offline.py"
spec = importlib.util.spec_from_file_location("iq6r_v4_offline", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root / "checkpoint"
        self.src.mkdir()
        (self.src / "config.json").write_text(
            json.dumps({"text_config": {"num_hidden_layers": 64}})
        )
        (self.src / "model-00001-of-00018.safetensors").write_bytes(b"")
        (self.src / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": "model-00001-of-00018.safetensors"}
        }))
        self.recipe = self.root / "test.recipe"
        weights = module.expected_weights()
        self.recipe.write_text("".join(
            name + " " + ("rtn codes=u6 group=128 scale=bf16"
                           if name == "token_embd.weight" else
                           "gptq codes=u6 transform=fwht128 calib=$CALIB")
            + "\n" for name in sorted(weights)
        ))
        self.vhead = self.root / "class.cls"
        self.vhead.write_bytes(b"classes")

    def test_recipe_coverage_math(self):
        weights = module.expected_weights()
        self.assertEqual(len(weights), 293)
        self.assertIn("blk.63.attn_output.weight", weights)
        self.assertNotIn("blk.62.attn_output.weight", weights)

    def test_missing_checkpoint_shards_fails(self):
        result = module.inspect(self.src, self.recipe, self.vhead, None)
        self.assertEqual(result["status"], "offline_inventory_failed")
        self.assertIn("checkpoint shard count or index coverage mismatch", result["errors"])
        self.assertFalse(result["calibration"]["observed_all_64_layer_indices"])

    def test_missing_tensor_fails(self):
        s = self.recipe.read_text()
        self.recipe.write_text(s.replace(
            "blk.63.ffn_down.weight gptq codes=u6 transform=fwht128 calib=$CALIB\n", ""
        ))
        result = module.inspect(self.src, self.recipe, self.vhead, None)
        self.assertIn("blk.63.ffn_down.weight", result["recipe"]["missing"])

    def test_bad_activation_transform_fails(self):
        s = self.recipe.read_text().replace(
            "blk.0.ffn_down.weight gptq codes=u6 transform=fwht128",
            "blk.0.ffn_down.weight gptq codes=u6 transform=none"
        )
        self.recipe.write_text(s)
        result = module.inspect(self.src, self.recipe, self.vhead, None)
        self.assertIn("blk.0.ffn_down.weight", result["recipe"]["invalid_trunk"])

    def test_calibration_detects_only_observed_layers(self):
        d = self.root / "calibration"
        d.mkdir()
        (d / "model.language_model.layers.0.input.safetensors").write_bytes(b"x")
        (d / "model.language_model.layers.63.input.safetensors").write_bytes(b"y")
        result = module.inspect(self.src, self.recipe, self.vhead, d)
        self.assertEqual(result["calibration"]["observed_layers"], [0, 63])
        self.assertFalse(result["calibration"]["observed_all_64_layer_indices"])


if __name__ == "__main__":
    unittest.main()
