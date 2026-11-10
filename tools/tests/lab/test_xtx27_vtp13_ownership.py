"""VTP13 card-free exact BF16 tensor-source ownership and fail-closed tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lab" / "radiance"))
import xtx27_vtp13_ownership as o

PREFIX = "model.language_model.layers."


def fake_checkpoint() -> tuple[dict, dict]:
    cfg = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 64, "hidden_size": 5120,
            "intermediate_size": 17408, "num_attention_heads": 24,
            "num_key_value_heads": 4, "head_dim": 256,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "attn_output_gate": True,
            "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
            "layer_types": [("full_attention" if i % 4 == 3 else "linear_attention")
                            for i in range(64)],
        },
    }
    shapes = {
        "model.language_model.embed_tokens.weight": [248320, 5120],
        "lm_head.weight": [248320, 5120],
    }
    for i in range(64):
        root = f"{PREFIX}{i}."
        for name in ("gate_proj", "up_proj"):
            shapes[root + f"mlp.{name}.weight"] = [17408, 5120]
        shapes[root + "mlp.down_proj.weight"] = [5120, 17408]
        if i % 4 == 3:
            shapes.update({
                root + "self_attn.q_proj.weight": [12288, 5120],
                root + "self_attn.k_proj.weight": [1024, 5120],
                root + "self_attn.v_proj.weight": [1024, 5120],
                root + "self_attn.o_proj.weight": [5120, 6144],
                root + "self_attn.q_norm.weight": [256],
                root + "self_attn.k_norm.weight": [256],
            })
        else:
            g = root + "linear_attn."
            shapes.update({
                g + "in_proj_qkv.weight": [10240, 5120],
                g + "conv1d.weight": [10240, 1, 4],
                g + "in_proj_a.weight": [48, 5120],
                g + "in_proj_b.weight": [48, 5120],
                g + "in_proj_z.weight": [6144, 5120],
                g + "A_log": [48], g + "dt_bias": [48],
                g + "out_proj.weight": [5120, 6144],
                g + "norm.weight": [128],
            })
    return cfg, shapes


class PhysicalOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.shapes = fake_checkpoint()

    def test_complete_64_layer_census_and_noncontiguous_gdn(self) -> None:
        result = o.make_map(self.config, self.shapes)
        self.assertEqual(result["sharded_source_tensors"], 640)
        self.assertEqual(len(result["noncontiguous_source_tensors"]), 96)
        self.assertEqual(result["gdn_layers"], 48)
        self.assertEqual(result["full_attention_layers"], 16)
        self.assertEqual(result["physical_ffn_intermediate_widths"], [5120, 4864, 7424])
        self.assertEqual(result["status"], "STRUCTURAL_OWNERSHIP_PROVEN_NOT_ENGINE_SUPPORTED")

    def test_attention_q_and_gate_are_kept_per_head(self) -> None:
        d = o.make_map(self.config, self.shapes)["tensors"]
        root = PREFIX + "3.self_attn."
        self.assertEqual(d[root + "q_proj.weight"]["roles"], {
            "xtx0": [{"offset": 0, "length": 3072}],
            "xtx1": [{"offset": 3072, "length": 3072}],
            "r9700": [{"offset": 6144, "length": 6144}],
        })
        self.assertEqual(d[root + "k_proj.weight"]["roles"]["r9700"],
                         [{"offset": 512, "length": 512}])
        self.assertEqual(d[root + "o_proj.weight"]["roles"]["r9700"],
                         [{"offset": 3072, "length": 3072}])
        self.assertEqual(d[PREFIX + "3.mlp.down_proj.weight"]["roles"]["r9700"],
                         [{"offset": 9984, "length": 7424}])

    def test_gdn_qkv_conv_packed_offsets_and_recurrent_head_mapping(self) -> None:
        d = o.make_map(self.config, self.shapes)["tensors"]
        root = PREFIX + "0.linear_attn."
        expected = [
            {"offset": 1024, "length": 1024},
            {"offset": 3072, "length": 1024},
            {"offset": 7168, "length": 3072},
        ]
        self.assertEqual(d[root + "in_proj_qkv.weight"]["roles"]["r9700"], expected)
        self.assertEqual(d[root + "conv1d.weight"]["roles"]["r9700"], expected)
        self.assertTrue(d[root + "conv1d.weight"]["noncontiguous_source_requires_pack"])
        self.assertEqual(d[root + "A_log"]["roles"]["r9700"],
                         [{"offset": 24, "length": 24}])
        self.assertEqual(d[root + "in_proj_z.weight"]["roles"]["r9700"],
                         [{"offset": 3072, "length": 3072}])
        self.assertEqual(d[root + "out_proj.weight"]["roles"]["r9700"],
                         [{"offset": 3072, "length": 3072}])

    def test_overlapping_missing_and_wrong_axis_rejected(self) -> None:
        for shape, axis, parts in (
            ([8, 3], 0, [[(0, 3)], [(2, 3)], [(5, 3)]]),
            ([8, 3], 0, [[(0, 2)], [(3, 2)], [(5, 3)]]),
            ([8, 3], 0, [[(0, 2)], [(2, 2)], [(4, 2)]]),
            ([8, 3], 2, [[(0, 2)], [(2, 2)], [(4, 4)]]),
        ):
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                o.validate_parts(shape, axis, parts)

    def test_malformed_ffn_widths_rejected(self) -> None:
        for widths in ((5120, 4864, 7168), (5120, 5000, 7288),
                       (5120, 0, 12288), (5120, 5120, 7169)):
            with self.subTest(widths=widths), self.assertRaisesRegex(ValueError, "FFN"):
                o.make_map(self.config, self.shapes, widths)

    def test_corrupt_packed_gdn_tensor_detected(self) -> None:
        self.shapes[PREFIX + "15.self_attn.q_proj.weight"] = [6144, 5120]
        with self.assertRaisesRegex(ValueError, "q_proj"):
            o.make_map(self.config, self.shapes)

    def test_missing_gdn_coeff_or_conv_rejected(self) -> None:
        del self.shapes[PREFIX + "0.linear_attn.A_log"]
        with self.assertRaisesRegex(ValueError, "A_log"):
            o.make_map(self.config, self.shapes)

    def test_wrong_gdn_norm_rejected(self) -> None:
        self.shapes[PREFIX + "0.linear_attn.norm.weight"] = [48]
        with self.assertRaisesRegex(ValueError, "norm"):
            o.make_map(self.config, self.shapes)

    def test_unexpected_layer_type_or_q_gate_rejected(self) -> None:
        self.config["text_config"]["layer_types"][0] = "full_attention"
        with self.assertRaisesRegex(ValueError, "layer-type"):
            o.make_map(self.config, self.shapes)
        self.config, self.shapes = fake_checkpoint()
        self.config["text_config"]["attn_output_gate"] = False
        with self.assertRaisesRegex(ValueError, "attn_output_gate"):
            o.make_map(self.config, self.shapes)

    def test_no_gpu_access_in_layout_proof(self) -> None:
        with patch.object(o.checkpoint, "shard_metadata",
                          side_effect=AssertionError("GPU or weight I/O")):
            d = o.make_map(self.config, self.shapes)
        self.assertNotIn("runtime", d["status"].lower())

    def test_main_refuses_overwrite_before_source_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "existing.json"
            target.write_text("{}")
            with (
                patch.object(sys, "argv", ["runner", "--checkpoint", tmp,
                                           "--output", str(target)]),
                patch.object(o, "load_original_checkpoint", side_effect=AssertionError),
                self.assertRaises(SystemExit),
            ):
                o.main()
            self.assertEqual(json.loads(target.read_text()), {})


if __name__ == "__main__":
    unittest.main()
