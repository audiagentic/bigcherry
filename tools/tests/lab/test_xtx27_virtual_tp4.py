"""VTP02 card-free checks; no GPU/ROCm dependency."""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from unittest.mock import patch

MODULE = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance" / "xtx27_virtual_tp4.py"
SPEC = importlib.util.spec_from_file_location("xtx27_virtual_tp4", MODULE)
assert SPEC and SPEC.loader
vtp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vtp)


class VirtualTp4CapacityTests(unittest.TestCase):
    def test_weight_split_matches_physical_capacity(self):
        values = vtp.parse_ratios(("0.30", "0.30", "0.20", "0.20"))
        result = vtp.calculate(values, tokens=1048576, kv="fp8",
                               model_shardable_gib=Decimal(30),
                               overhead_gib=(Decimal(2), Decimal(2), Decimal(3)),
                               num_streams=4)
        self.assertEqual([x["projected_shardable_weights_gib"] for x in result["cards"]], [9, 9, 12])
        self.assertEqual([x["logical_attention_kv_gib"] for x in result["cards"]], [8, 8, 16])
        self.assertEqual([x["illustrative_total_gib"] for x in result["cards"]], [19, 19, 31])
        self.assertEqual(result["physical_kv_heads"], [1, 1, 2])
        self.assertTrue(result["native_possible_even_split_per_stream"])

    def test_compressed_kv_reduces_total_but_not_weight_share(self):
        result = vtp.calculate(
            vtp.parse_ratios(("0.3", "0.3", "0.2", "0.2")),
            tokens=1048576, kv="k6_v4", num_streams=4,
            model_shardable_gib=Decimal(30),
            overhead_gib=(Decimal(2), Decimal(2), Decimal(3)),
        )
        self.assertEqual([x["logical_attention_kv_gib"] for x in result["cards"]], [5.5, 5.5, 11])
        self.assertEqual(result["cards"][2]["illustrative_total_gib"], 26)
        self.assertFalse(result["kv_format_hardware_qualified"])

    def test_fp8_v4_kv_size(self):
        self.assertEqual(vtp.KV_BPT["fp8_v4"], Decimal(25600))
        self.assertEqual(vtp.KV_BPT["k6_v4"], Decimal(22528))
        self.assertEqual(vtp.KV_BPT["fp8"], Decimal(32768))

    def test_ffn_tile_rounding_all_candidates(self):
        expect = ((41, 41, 27, 27), (48, 48, 20, 20), (51, 51, 17, 17))
        for (_, raw), counts in zip(vtp.CANDIDATES, expect):
            with self.subTest(raw=raw):
                self.assertEqual(tuple(vtp.tile_counts(vtp.parse_ratios(raw))), counts)
                self.assertEqual(sum(counts) * 128, 17408)

    def test_capacity_and_roofline_differ(self):
        result = vtp.calculate(
            vtp.parse_ratios(("0.30", "0.30", "0.20", "0.20")),
            tokens=1048576, kv="fp8", num_streams=1,
            model_shardable_gib=Decimal(30),
            overhead_gib=(Decimal(2), Decimal(2), Decimal(3)),
        )
        self.assertEqual(result["bandwidth_only_relative_slowest_to_xtx0"], 2)
        self.assertFalse(result["native_possible_even_split_per_stream"])
        self.assertTrue(result["single_request_over_native_262144"])

    def test_invalid_ratios_or_budget_refused(self):
        for bad in (("0.3", "0.3", "0.2", "0.1"), ("0.25", "0.25", "0.3", "0.2"),
                    ("0.3", "0.3", "NaN", "0.4")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                vtp.parse_ratios(bad)
        with self.assertRaises(ValueError):
            vtp.tile_counts(vtp.parse_ratios(vtp.CANDIDATES[0][1]), width=17409)
        with self.assertRaises(ValueError):
            vtp.calculate(vtp.parse_ratios(vtp.CANDIDATES[0][1]), tokens=0, kv="fp8",
                          num_streams=4, model_shardable_gib=Decimal(30),
                          overhead_gib=(Decimal(2), Decimal(2), Decimal(3)))

    def test_cli_is_projection_not_admission(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = vtp.main(["--demo", "--candidate", "capacity", "--kv-format", "k6_v4"])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["input_class"], "synthetic")
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["label"], "CARD_FREE_PROJECTION_NOT_RUNTIME_ADMISSION")

    def test_cli_needs_explicit_weights_and_overhead(self):
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as caught:
            vtp.main([])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
