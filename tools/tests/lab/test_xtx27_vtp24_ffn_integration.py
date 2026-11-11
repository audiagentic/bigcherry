"""VTP24/25 card-free FFN original-checkpoint CPU/GPU integration tests."""
from __future__ import annotations

import io
import json
import pathlib
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

LAB = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(LAB))

import xtx27_vtp24_ffn_oracle as oracle
import xtx27_vtp24_ffn_pack as pack
import xtx27_vtp25_ffn_gpu as gpu


class OriginalBF16FFNTests(unittest.TestCase):
    def test_unequal_physical_ffn_owner_rows_and_down_cols(self):
        self.assertEqual(pack.WIDTHS, (5120, 4864, 7424))
        self.assertEqual(sum(pack.WIDTHS), 17408)
        self.assertEqual(pack.ROLES, ("xtx0", "xtx1", "r9700"))
        self.assertEqual(pack.SPECS["gate"][1], [17408, 5120])
        self.assertEqual(pack.SPECS["up"][1], [17408, 5120])
        self.assertEqual(pack.SPECS["down"][1], [5120, 17408])
        self.assertEqual(pack.SPECS["post_norm"][0], "post_attention_layernorm.weight")

    def test_weighted_original_post_attention_rmsnorm(self):
        values = struct.pack(f"<{pack.HIDDEN}H", *([0x3F80] * pack.HIDDEN))
        gain = struct.pack(f"<{pack.HIDDEN}H", *([0] * pack.HIDDEN))
        result = pack.weighted_norm(values, gain, 1)
        self.assertEqual(len(result), len(values))
        self.assertEqual(result, values)  # variance=1; eps below BF16 significance
        zero = struct.pack(f"<{pack.HIDDEN}H", *([0] * pack.HIDDEN))
        self.assertEqual(pack.weighted_norm(zero, gain, 1), zero)
        with self.assertRaises(ValueError):
            pack.weighted_norm(values[:-2], gain, 1)
        with self.assertRaises(ValueError):
            pack.weighted_norm(values, gain[:-2], 1)

    def test_fail_closed_incorrect_checkpoint_layer_and_parent(self):
        for layer in (-1, 64, 2.5, None, "0"):
            with self.assertRaises(ValueError):
                pack.source_info(pathlib.Path("/does/not/exist"), layer)
        with tempfile.TemporaryDirectory() as temp:
            bogus = pathlib.Path(temp) / "ref.json"
            bogus.write_text(json.dumps({"schema": pack.gdn_oracle.SCHEMA,
                                        "status": "INCOMPLETE", "rows": 1}))
            with self.assertRaises(ValueError):
                pack.validate_gdn_reference(bogus)
            with self.assertRaises(ValueError):
                gpu.validate(bogus, bogus)

    def test_cardfree_entry_requires_opt_in_and_no_gpu(self):
        with (patch.object(pack, "source_info",
                           return_value={"layer": 0, "source": "synthetic-mock"}),
              patch.object(pack, "pack", side_effect=AssertionError("would pack")),
              redirect_stdout(io.StringIO()) as stdout):
            self.assertEqual(pack.main(["--checkpoint", "/unused"]), 0)
        self.assertFalse(json.loads(stdout.getvalue())["gpu_required"])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            pack.main(["--checkpoint", "/unused", "--phase", "pack"])
        with (patch.object(gpu.subprocess, "run",
                           side_effect=AssertionError("GPU command not allowed")),
              redirect_stdout(io.StringIO()) as stdout):
            self.assertEqual(gpu.main([]), 0)
        self.assertFalse(json.loads(stdout.getvalue())["full_model"])
        self.assertIsNone(json.loads(stdout.getvalue())["input"])
        for args in (["--phase", "gpu"], ["--phase", "compile"],
                     ["--phase", "gpu", "--run"], ["--iterations", "0"],
                     ["--iterations", "31"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                gpu.main(args)

    def test_real_ffn_cpu_source_produces_full_layer_hidden(self):
        src = oracle.SOURCE.read_text()
        for token in (
            "constexpr std::array<int,3> WIDTHS{5120,4864,7424}",
            "const auto gate=project(wgate,normed,rows,width,H)",
            "const auto up=project(wup,normed,rows,width,H)",
            "const auto mid=swiglu(gate,up)",
            "const auto output=down_project(wdown,mid,rows,width)",
            "value+=partials[1][i]",
            "value+=partials[2][i]",
            "post[i]=bfround(b+fb16(residual[i]))",
            "original-layer0-post-ffn.bf16",
        ):
            self.assertIn(token, src)
        self.assertNotIn("hipMalloc", src)

    def test_original_ffn_gpu_kernel_operator_chain(self):
        src = gpu.SOURCE.read_text()
        for token in (
            "constexpr std::array<int,3> WIDTHS{5120,4864,7424}",
            "original_ffn_bf16_projection<<<",
            "original_ffn_swiglu<<<",
            "original_ffn_bf16_down<<<",
            "for(auto& r:ranks)launch(r)",
            "value+=ranks[1].pinned_partial[i]",
            "value+=ranks[2].pinned_partial[i]",
            "hipHostMallocPortable",
            "next-layer GPU residual broadcast mismatch",
            "VTP25 original FFN next-layer BF16 residual outside tolerance",
        ):
            self.assertIn(token, src)
        self.assertNotIn("hipMemcpyPeer", src)

    def test_ffn_cpu_reference_gate_requires_real_checkpoint(self):
        self.assertTrue(oracle.SOURCE.is_file())
        self.assertTrue(gpu.SOURCE.is_file())
        self.assertTrue(gpu.RECEIPT_TYPE.startswith("VTP25_"))
        with tempfile.TemporaryDirectory() as temp:
            packet = pathlib.Path(temp) / "invalid.json"
            packet.write_text("{}")
            with self.assertRaises(ValueError):
                pack.validate(packet)
            with self.assertRaises(ValueError):
                gpu.validate(packet, packet)


if __name__ == "__main__":
    unittest.main()
