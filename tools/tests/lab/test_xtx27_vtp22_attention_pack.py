"""Card-free exact-original BF16 Qwen3.8 full-attention physical TP3 pack tests."""
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

import xtx27_vtp22_attention_pack as attention


class FullAttentionSourcePackTests(unittest.TestCase):
    def test_query_gate_is_interleaved_and_rank_complete(self):
        plan = attention.ownership_map()
        self.assertEqual(attention.QUERIES, (6, 6, 12))
        self.assertEqual(attention.KVS, (1, 1, 2))
        self.assertEqual(plan["xtx0"]["q"], (0, 3072))
        self.assertEqual(plan["xtx1"]["q"], (3072, 3072))
        self.assertEqual(plan["r9700"]["q"], (6144, 6144))
        self.assertEqual(plan["r9700"]["o"], (3072, 3072))
        self.assertEqual(plan["r9700"]["k"], (512, 512))
        for role in attention.ROLES:
            self.assertEqual(plan[role]["q"][1], 512 * plan[role]["query_heads"])
            self.assertEqual(plan[role]["o"][1], 256 * plan[role]["query_heads"])
            self.assertEqual(plan[role]["k"][1], 256 * plan[role]["kv_heads"])
            self.assertEqual(plan[role]["query_heads"], 6 * plan[role]["kv_heads"])
            self.assertTrue(plan[role]["query_gate_interleaved"])

    def test_three_original_byte_exact_row_shards(self):
        values = [i & 0xFFFF for i in range(36)]
        original = struct.pack("<36H", *values)
        fragments = attention.rows_shards(original, 6, (2, 1, 3))
        self.assertEqual([len(f) for f in fragments], [24, 12, 36])
        self.assertEqual(b"".join(fragments), original)
        with self.assertRaises(ValueError):
            attention.rows_shards(original, 6, (1, 1, 2))
        with self.assertRaises(ValueError):
            attention.rows_shards(original + b"\0", 6, (2, 1, 3))

    def test_three_original_byte_exact_input_column_shards(self):
        rows, width, counts = 4, 9, (2, 3, 4)
        values = [i & 0xFFFF for i in range(rows * width)]
        original = struct.pack("<36H", *values)
        fragments = attention.col_shards(original, rows, width, counts)
        for rank, (offset, span) in enumerate(((0, 2), (2, 3), (5, 4))):
            expected = [values[row * width + offset + d]
                        for row in range(rows) for d in range(span)]
            self.assertEqual(fragments[rank],
                             struct.pack(f"<{len(expected)}H", *expected))
        self.assertEqual(sum(len(x) for x in fragments), len(original))
        with self.assertRaises(ValueError):
            attention.col_shards(original, rows, width, (2, 2, 4))
        with self.assertRaises(ValueError):
            attention.col_shards(original[:-2], rows, width, counts)

    def test_wrong_attention_layer_rejected_before_checkpoint_access(self):
        for layer in (-1, 0, 1, 2, 4, 6, 64, 65):
            with self.assertRaises(ValueError):
                attention.sources(pathlib.Path("/does/not/exist"), layer)
        for layer in (3.0, "3", None):
            with self.assertRaises(ValueError):
                attention.sources(pathlib.Path("/does/not/exist"), layer)

    def test_immutable_shard_writer_checks_shape_and_exists(self):
        with tempfile.TemporaryDirectory() as d:
            filename = pathlib.Path(d) / "small.bf16"
            item = attention._output_record(filename, b"\1\0\2\0", [2])
            self.assertEqual(item["sha256"], attention.sha(b"\1\0\2\0"))
            self.assertEqual(item["bytes"], 4)
            with self.assertRaises(ValueError):
                attention._output_record(filename, b"\1\0\2\0", [2])
            with self.assertRaises(ValueError):
                attention._output_record(pathlib.Path(d) / "wrong.bf16",
                                         b"\1\0\2\0", [3])

    def test_cli_plan_and_run_are_fail_closed(self):
        with (redirect_stdout(io.StringIO()) as output,
              patch.object(attention, "sources", return_value={
                  "layer": 3, "sources": {}, "ownership": attention.ownership_map(),
              }),
              patch.object(attention, "pack", side_effect=AssertionError("must not pack"))):
            self.assertEqual(attention.main(["--checkpoint", "not-loaded"]), 0)
        self.assertFalse(json.loads(output.getvalue())["real_weight_payloads_read"])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            attention.main(["--checkpoint", "not-loaded", "--phase", "pack"])

    def test_incomplete_receipt_fails_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            receipt = pathlib.Path(d) / "summary.json"
            receipt.write_text(json.dumps({
                "schema": attention.SCHEMA, "status": "INCOMPLETE",
                "writer_sha256": attention.projection.sha(pathlib.Path(attention.__file__)),
            }))
            with self.assertRaises(ValueError):
                attention.validate(receipt)


if __name__ == "__main__":
    unittest.main()
