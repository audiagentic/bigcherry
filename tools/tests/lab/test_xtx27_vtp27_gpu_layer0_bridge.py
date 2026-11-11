"""VTP27 original BF16 GPU GDN-to-FFN boundary provenance tests.

No GPU access. Verifies exact model activation contracts and
fail-closed behavior before accepting any live source.
"""
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

import xtx27_vtp27_gpu_layer0_bridge as bridge


class RealGPUFirstLayerBridgeTests(unittest.TestCase):
    def test_checked_original_bf16_handoff_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            expected = root / "expected.bf16"
            gpu = root / "gpu.bf16"
            same = struct.pack("<5120H", *([0x3F80] * 5120))
            expected.write_bytes(same)
            gpu.write_bytes(same)
            result = bridge.exact_bf16(expected, gpu)
            self.assertTrue(result["exact"])
            self.assertEqual(result["exact_values"], 5120)
            self.assertEqual(result["expected_sha256"], result["GPU_export_sha256"])
            changed = bytearray(same)
            changed[2] = 1
            gpu.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "not exactly match"):
                bridge.exact_bf16(expected, gpu)
            gpu.write_bytes(same[:-2])
            with self.assertRaisesRegex(ValueError, "size invalid"):
                bridge.exact_bf16(expected, gpu)

    def test_full_ffn_output_requires_finite_tolerance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            expected = root / "ref.bf16"
            gpu = root / "gpu.bf16"
            good = struct.pack("<5120H", *([0x3F80] * 5120))
            expected.write_bytes(good)
            gpu.write_bytes(good)
            result = bridge.bf16_reference_difference(expected, gpu)
            self.assertTrue(result["bitwise_exact"])
            self.assertEqual(result["exact_count"], 5120)
            modified = bytearray(good)
            modified[:2] = struct.pack("<H", 0x3F81)
            gpu.write_bytes(modified)
            result = bridge.bf16_reference_difference(expected, gpu)
            self.assertEqual(result["nonexact_count"], 1)
            self.assertTrue(result["tolerance_pass"])
            modified[:2] = struct.pack("<H", 0x7F80)
            gpu.write_bytes(modified)
            with self.assertRaisesRegex(ValueError, "exceeds original FFN reference"):
                bridge.bf16_reference_difference(expected, gpu)

    def test_original_gdn_ffn_parent_same_source_required(self):
        first = {
            "rows": 1,
            "original_output_reference_sha256": "gdn_source_sha",
            "original_output_packet": "/tmp/model/gdn.json",
            "original_output_packet_sha256": "packet_sha",
        }
        last = {"rows": 1}
        original = {"layer": 0, "input": {
            "receipt_sha256": "different_sha",
            "gdn_packet": "/tmp/model/gdn.json",
        }}
        with (
            patch.object(bridge.gdn, "validate", return_value=first),
            patch.object(bridge.ffn, "validate", return_value=last),
            patch.object(bridge.pack, "validate", return_value=original),
            self.assertRaisesRegex(ValueError, "incompatible ancestry"),
        ):
            bridge.checked_sources(*([pathlib.Path("/unused")] * 8))

    def test_cardfree_default_and_missing_complete_receipts(self):
        with (patch.object(bridge.subprocess, "run",
                           side_effect=AssertionError("no external command")),
              redirect_stdout(io.StringIO()) as output):
            self.assertEqual(bridge.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["schema"], bridge.SCHEMA)
        self.assertEqual(result["phase"], "card_free")
        self.assertTrue(result["still_not_full_model"])
        self.assertTrue(result["not_resident_or_native"])
        self.assertIsNone(result["original_model_sources"])
        for args in (
            ["--phase", "gpu"], ["--phase", "compile"],
            ["--phase", "gpu", "--run"],
            ["--iterations", "0"], ["--iterations", "21"],
        ):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                bridge.main(args)

    def test_immutable_gdn_export_ffn_import_gpu_source(self):
        for source in (bridge.SOURCE_GDN, bridge.SOURCE_FFN):
            self.assertTrue(source.is_file())
            s = source.read_text()
            self.assertIn('vtp27_opt(argc,argv,"--export-bf16")', s)
            self.assertIn("VTP27 refuses existing BF16 output path", s)
        gdn = bridge.SOURCE_GDN.read_text()
        ffn = bridge.SOURCE_FFN.read_text()
        self.assertIn("VTP20 experimental one-execution-chain", gdn)
        self.assertIn("VTP27 original GDN BF16 import", ffn)
        self.assertIn('vtp27_opt(argc,argv,"--import-bf16")', ffn)
        self.assertIn("VTP27 original GDN->FFN boundary is not BF16-bit-exact", ffn)
        self.assertIn("original_ffn_bf16_projection<<<", ffn)
        self.assertIn("gdn_rank_to_out_bf16<<<", gdn)
        self.assertIn("gpu-after-GDN-residual.bf16", pathlib.Path(
            bridge.__file__).read_text())
        self.assertNotIn("hipMemcpyPeer", gdn)
        self.assertNotIn("hipMemcpyPeer", ffn)


if __name__ == "__main__":
    unittest.main()
