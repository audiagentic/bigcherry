"""VTP26 card-free original 64-layer TP3 VRAM lower-bound and plan tests."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(ROOT))
import test_xtx27_vtp13_ownership as fixture
import xtx27_vtp26_full_model_inventory as model


def sample() -> tuple[dict, dict]:
    config, shapes = fixture.fake_checkpoint()
    config["text_config"]["max_position_embeddings"] = 262144
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}."
        shapes[prefix + "input_layernorm.weight"] = [5120]
        shapes[prefix + "post_attention_layernorm.weight"] = [5120]
    shapes["model.language_model.norm.weight"] = [5120]
    shapes["model.visual.stub.weight"] = [12, 16]
    shapes["mtp.layers.0.stub.weight"] = [7, 5]
    return config, shapes


class FullModelInventoryTests(unittest.TestCase):
    def setUp(self):
        self.config, self.shapes = sample()

    def plan(self, context=8192, streams=4):
        return model.physical_ledger(self.config, self.shapes,
                                     {"config_sha256": "test-only"}, context, streams)

    def test_exact_48_gdn_16_attention_64_ffn_source_coverage(self):
        d = self.plan()
        self.assertEqual(d["count"], 64)
        self.assertEqual(d["gdn_layers"], 48)
        self.assertEqual(d["full_attention_layers"], 16)
        self.assertEqual(d["sharded_source_tensors"], 640)
        self.assertEqual(d["layer_schedule"][:8], [
            "gdn", "gdn", "gdn", "full_attention",
            "gdn", "gdn", "gdn", "full_attention",
        ])
        self.assertEqual(d["excluded_optional_tensor_count"], 2)
        self.assertEqual(
            d["host_embedding_head_final_norm_bytes"],
            2 * (248320 * 5120 * 2) + 5120 * 2,
        )
        self.assertFalse(d["gpu_admission_approved"])
        self.assertFalse(d["full_model_execution_supported"])
        self.assertEqual(d["original_bf16_weight_provenance"],
                         "HEADERS_ONLY_NOT_FULL_PAYLOAD_HASHED")

    def test_per_physical_kv_gdn_state_resident_counts_scale_with_tokens(self):
        d = self.plan(8192, 4)
        for idx, role in enumerate(model.own.ROLES):
            record = d["per_gpu_minimum"][role]
            kv = 4 * 8192 * 16 * model.own.FULL_KV_HEADS[idx] * 256 * 4
            gdn = 4 * 48 * model.own.GDN_V_HEADS[idx] * 128 * 128 * 4
            conv = (4 * 48 * 3 * (2 * model.own.GDN_K_HEADS[idx]
                                    + model.own.GDN_V_HEADS[idx]) * 128 * 2)
            self.assertEqual(record["kv_cache_bytes"], kv)
            self.assertEqual(record["gdn_recurrent_state_bytes"], gdn)
            self.assertEqual(record["gdn_conv_history_bytes"], conv)
            self.assertEqual(record["minimum_live_state_bytes"], kv + gdn + conv)
            self.assertEqual(
                record["minimum_GPU_bytes_before_scratch_and_reserve"],
                record["model_weight_plus_replicated_norm_bytes"] + kv + gdn + conv,
            )
        self.assertEqual(d["per_gpu_minimum"]["r9700"]["kv_cache_bytes"],
                         2 * d["per_gpu_minimum"]["xtx0"]["kv_cache_bytes"])

    def test_mandatory_exclusion_only_vision_and_mtp(self):
        self.shapes["model.language_model.layers.0.unhandled.weight"] = [17]
        with self.assertRaisesRegex(ValueError, "unowned original text tensor"):
            self.plan()
        del self.shapes["model.language_model.layers.0.unhandled.weight"]
        del self.shapes["model.language_model.layers.31.post_attention_layernorm.weight"]
        with self.assertRaisesRegex(ValueError, "post_attention_layernorm"):
            self.plan()

    def test_context_and_stream_bounds_are_not_auto_admission(self):
        for context, streams in (
            (0, 4), (262145, 1), (1024, 0), (8192, 33),
            (8192.5, 1), (8192, True),
        ):
            with self.subTest(context=context, streams=streams), self.assertRaises(ValueError):
                self.plan(context, streams)
        longest = self.plan(262144, 1)
        self.assertEqual(longest["total_retained_tokens"], 262144)
        self.assertFalse(longest["gpu_admission_approved"])
        self.assertIn("HIP_scratch", longest["unmeasured_nonzero_resources"])
        self.assertIn("live_available_VRAM", longest["unmeasured_nonzero_resources"])

    def test_output_is_fail_closed_immutable_and_card_free(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "existing.json"
            path.write_text("{}")
            with (
                patch.object(sys, "argv", [
                    "model", "--checkpoint", temp, "--output", str(path),
                ]),
                patch.object(model.own, "load_original_checkpoint",
                             side_effect=AssertionError("should not read model")),
                self.assertRaises(SystemExit),
            ):
                model.main()
            self.assertEqual(json.loads(path.read_text()), {})
        self.assertEqual(model.SCHEMA,
                         "vtp26-original-qwen38-27b-64layer-physical-tp3-lower-bound-v1")


if __name__ == "__main__":
    unittest.main()
