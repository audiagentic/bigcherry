"""Card-free VTP08 W8A8 RAD2 FFN/FP8-KV impossibility-bound tests."""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

BASE = pathlib.Path(__file__).resolve().parents[2] / "lab/radiance"
sys.path.insert(0, str(BASE))
SPEC = importlib.util.spec_from_file_location("xtx27_rad_vram_floor", BASE / "xtx27_rad_vram_floor.py")
assert SPEC and SPEC.loader
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def fake_rad2() -> str:
    lines = [
        "format        RAD2, 29.04 GiB, little-endian",
        "source quant  w8a8_fwht128",
        "n_layers 64", "n_embd 5120", "n_ff 17408",
        "n_head_kv 4", "head_dim 256", "head_dim 256",
        "n_ctx_train 262144",
    ]
    for layer in range(64):
        lines.extend((
            f"      blk.{layer}.ffn_gate_up.weight  {q.ENCODING} 34816x5120   172.66 MiB  @0",
            f"      blk.{layer}.ffn_down.weight  {q.ENCODING} 5120x17408   86.33 MiB  @0",
        ))
    return "\n".join(lines) + "\n"


class VramFloorTests(unittest.TestCase):
    def test_valid_manifest_and_exact_quant_encoding(self):
        source = q.parse_rad_info(fake_rad2())
        self.assertEqual(source["confirmed_fused_ffn_projections"], 128)
        self.assertEqual(source["ffn_encoder"], "i8*bf16[1x128]/fwht128")
        self.assertEqual(
            q.encoded_ffn_bytes(q.INTERMEDIATE),
            17380147200,
        )

    def test_invalid_metadata_or_encoding_rejected(self):
        original = fake_rad2()
        modifications = (
            original.replace("source quant  w8a8_fwht128", "source quant  fp8"),
            original.replace("n_head_kv 4", "n_head_kv 8"),
            original.replace("head_dim 256", "head_dim 128", 1),
            original.replace("ffn_down.weight", "ffn_unknown.weight", 1),
            original.replace("i8*bf16[1x128]/fwht128", "f16", 1),
            original.replace("34816x5120", "34816x4096", 1),
            original + "      blk.0.ffn_down.weight i8*bf16[1x128]/fwht128 5120x17408 90 MiB @0\n",
        )
        for text in modifications:
            with self.subTest(text=text[:85]), self.assertRaises(ValueError):
                q.parse_rad_info(text)

    def test_at_4x256k_projections_not_admission(self):
        result = q.vram_floor(candidate="capacity", streams=4, per_stream_tokens=262144)
        cards = result["cards"]
        self.assertEqual(result["logical_ffn_widths"], [5120, 5120, 3584, 3584])
        self.assertEqual([card["kv_heads"] for card in cards], [1, 1, 2])
        self.assertEqual(
            [card["fp8_attention_kv_bytes"] // q.GIB for card in cards],
            [8, 8, 16],
        )
        self.assertFalse(result["definite_no_go"])
        self.assertTrue(result["positive_floor_headroom_is_not_admission"])
        self.assertLess(cards[2]["headroom_before_all_nonffn_and_runtime_bytes"],
                        10 * q.GIB)

    def test_8x256k_is_hard_no_go_even_if_every_other_weight_free(self):
        result = q.vram_floor(candidate="capacity", streams=8, per_stream_tokens=262144)
        cards = result["cards"]
        self.assertEqual(cards[2]["fp8_attention_kv_bytes"], 32 * q.GIB)
        self.assertTrue(cards[2]["physically_impossible_even_without_other_weights"])
        self.assertTrue(result["definite_no_go"])

    def test_equal_tp2_dual_xtx_4x256k_is_hard_no_go(self):
        control = q.tp2_xtx_control_floor(streams=4, per_stream_tokens=262144)
        self.assertEqual(control["ffn_width_each"], 8704)
        self.assertEqual(control["kv_bytes_each"], 16 * q.GIB)
        self.assertGreater(control["hard_floor_bytes_each"], 24 * q.GIB)
        self.assertTrue(control["definite_no_go"])
        self.assertFalse(q.tp2_xtx_control_floor(4, 131072)["definite_no_go"])

    def test_8x128k_does_not_exceed_lower_bound(self):
        result = q.vram_floor(candidate="capacity", streams=8, per_stream_tokens=131072)
        self.assertFalse(result["definite_no_go"])
        self.assertEqual(result["cards"][2]["fp8_attention_kv_bytes"], 16 * q.GIB)

    def test_all_candidates_legal_and_ffn_bytes_conserved(self):
        for name, _ in q.CANDIDATES:
            with self.subTest(name=name):
                result = q.vram_floor(candidate=name, streams=4, per_stream_tokens=8192)
                self.assertEqual(sum(result["logical_ffn_widths"]), 17408)
                self.assertEqual(
                    result["total_ffn_codes_and_scales_bytes"],
                    q.encoded_ffn_bytes(17408),
                )
                self.assertEqual(len(result["cards"]), 3)

    def test_bad_requests_not_normalized_to_native_limit(self):
        for streams, tokens in ((0, 8192), (4, 0), (4, 262145)):
            with self.assertRaises(ValueError):
                q.vram_floor(candidate="capacity", streams=streams, per_stream_tokens=tokens)
        with self.assertRaises(ValueError):
            q.vram_floor(candidate="made-up", streams=4, per_stream_tokens=8192)

    def test_safe_main_no_gpu_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            model = root / "model.rad"
            model.write_bytes(b"RAD2fake")
            info = root / "rad-info"
            info.write_bytes(b"fake")
            output = root / "receipt.json"
            spy = Mock(returncode=0, stdout=fake_rad2(), stderr="")
            with patch.object(q.subprocess, "run", return_value=spy) as run:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(q.main([
                        "--rad-info", str(info), "--model", str(model),
                        "--receipt", str(output),
                    ]), 0)
                self.assertEqual(run.call_count, 1)
            doc = json.loads(output.read_text())
            self.assertFalse(doc["gpu_used"])
            self.assertEqual(doc["serving_admission"], "NOT_TESTED")
            self.assertTrue(doc["hypothetical_equal_tp2_xtx_control"][0]["definite_no_go"])
            with (
                patch.object(q.subprocess, "run", side_effect=AssertionError("ran with existing receipt")),
                self.assertRaises(FileExistsError),
            ):
                q.main(["--rad-info", str(info), "--model", str(model),
                        "--receipt", str(output)])


if __name__ == "__main__":
    unittest.main()
