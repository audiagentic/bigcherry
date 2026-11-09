"""Card-free 1M-context KV math and speculative stage ownership gates."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lab.radiance import xtx27_kv_budget as kb


def shape() -> dict:
    return kb.model_shape({
        "text_config": {
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
        }
    })


class KVBudgetTests(unittest.TestCase):
    def test_logical_fp8_total_and_tp2_at_one_million(self):
        s = shape()
        self.assertEqual(len(s["attention_indices"]), 16)
        self.assertEqual(
            kb.kv_bytes_per_token(full_layers=16, heads=4, head_dim=256,
                                  dtype_bytes=1, tp=2) * 1048576, 16 * kb.GIB,
        )
        dual = kb.scenario(s, cut=64, tokens=1048576)
        self.assertEqual(dual["stages"][0]["logical_attention_kv_gib_per_card"], 16)
        self.assertEqual(dual["all_card_logical_kv_gib"], 32)
        self.assertFalse(dual["illustrative_memory_margin_nonnegative"])
        self.assertTrue(dual["exceeds_model_native_context"])

    def test_full_layer_stage_owns_its_own_kv_not_duplicated_across_xtx(self):
        s = shape()
        cases = [(32, 8, 16), (40, 10, 12), (48, 12, 8)]
        for cut, xtx_gib, r9700_gib in cases:
            with self.subTest(cut=cut):
                r = kb.scenario(s, cut=cut, tokens=1048576)
                self.assertEqual(len(r["stages"]), 2)
                self.assertEqual(r["stages"][0]["logical_attention_kv_gib_per_card"], xtx_gib)
                self.assertEqual(r["stages"][1]["logical_attention_kv_gib_per_card"], r9700_gib)
                self.assertEqual(r["stages"][0]["tensor_parallel"], 2)
                self.assertEqual(r["stages"][1]["tensor_parallel"], 1)
                self.assertEqual(r["all_card_logical_kv_gib"], 32)
                self.assertTrue(r["illustrative_memory_margin_nonnegative"])

    def test_fp16_doubles_and_current_context_not_extrapolated(self):
        s = shape()
        cur = kb.scenario(s, cut=64, tokens=262144, dtype_bytes=1)
        self.assertEqual(cur["stages"][0]["logical_attention_kv_gib_per_card"], 4)
        self.assertFalse(cur["exceeds_model_native_context"])
        full = kb.scenario(s, cut=64, tokens=1048576, dtype_bytes=2)
        self.assertEqual(full["stages"][0]["logical_attention_kv_gib_per_card"], 32)

    def test_invalid_shapes_and_non_head_shardable_tp_fail(self):
        with self.assertRaises(ValueError):
            kb.model_shape({"layer_types": ["full_attention"], "num_hidden_layers": 2})
        with self.assertRaises(ValueError):
            kb.kv_bytes_per_token(full_layers=4, heads=5, head_dim=256,
                                  dtype_bytes=1, tp=2)
        for cut in (0, 65, -1):
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                kb.scenario(shape(), cut=cut, tokens=1048576)

    def test_config_cli_dry_run_never_allocates_gpu(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            conf = root / "config.json"
            conf.write_text(json.dumps({
                "text_config": {"layer_types": shape()["layers"], "num_hidden_layers": 64,
                                "num_key_value_heads": 4, "head_dim": 256,
                                "max_position_embeddings": 262144}
            }))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(kb.main(["--config", str(conf), "--cut", "40"]), 0)
            record = json.loads(output.getvalue())
            self.assertEqual(record["scenarios"][0]["stages"][1]["full_attention_layers"], 6)
            self.assertIn("NOT implemented", " ".join(record["limitations"]))


if __name__ == "__main__":
    unittest.main()
