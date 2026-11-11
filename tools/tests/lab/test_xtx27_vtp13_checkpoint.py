"""Card-free checks for original BF16 safetensors tensor descriptors."""
from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

LAB = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(LAB))
import xtx27_vtp13_checkpoint as reader

SHAPES = reader.OPS
PREFIX = "model.language_model.layers.0.mlp."


class CheckpointSourceTests(unittest.TestCase):
    @staticmethod
    def create_checkpoint(root: pathlib.Path, *, wrong_shape: bool = False) -> None:
        raw_map = {}
        entries = {}
        cursor = 0
        file = "model-00001-of-00018.safetensors"
        for op, (kind, expected_shape) in SHAPES.items():
            shape = ([17408, 5119] if wrong_shape and op == "gate"
                     else expected_shape)
            nbytes = shape[0] * shape[1] * 2
            name = PREFIX + kind + ".weight"
            raw_map[name] = file
            entries[name] = {
                "dtype": "BF16", "shape": shape,
                "data_offsets": [cursor, cursor + nbytes],
            }
            cursor += nbytes
        header = json.dumps(entries, separators=(",", ":")).encode()
        with (root / file).open("wb") as stream:
            stream.write(struct.pack("<Q", len(header)))
            stream.write(header)
            stream.truncate(8 + len(header) + cursor)  # Sparse, no 535 MB payload written
        (root / "config.json").write_text("{}")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": raw_map}))

    def test_exact_header_views_are_sane_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.create_checkpoint(root)
            with patch.object(reader.ownership, "expected_config"):
                doc = reader.descriptors(root, 0)
            self.assertEqual(doc["schema"], "vtp13-checkpoint-bf16-ffn-sources-v1")
            self.assertFalse(doc["payloads_fingerprinted"])
            self.assertEqual(sum(x["bytes"] for x in doc["tensors"].values()), 534773760)
            args = reader.hip_args(doc)
            self.assertEqual(len(args), 12)
            self.assertEqual(args[0], "--gate-file")
            self.assertIn("--down-byte", args)
            ranges = sorted((x["offset"], x["offset"] + x["bytes"])
                            for x in doc["tensors"].values())
            self.assertEqual(ranges[0][1], ranges[1][0])
            self.assertEqual(ranges[1][1], ranges[2][0])
            for layer in (-1, 64, "0", True):
                with self.subTest(layer=layer), self.assertRaises(ValueError):
                    reader.descriptors(root, layer)

    def test_changed_original_shape_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.create_checkpoint(root, wrong_shape=True)
            with (patch.object(reader.ownership, "expected_config"),
                  self.assertRaisesRegex(ValueError, "changed original BF16")):
                reader.descriptors(root, 0)

    def test_symlink_escape_and_missing_source_refused(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = pathlib.Path(tmp)
            self.create_checkpoint(root)
            other = pathlib.Path(outside) / "elsewhere"
            other.write_text("not a checkpoint")
            name = "model-00001-of-00018.safetensors"
            (root / name).unlink()
            (root / name).symlink_to(other)
            with (patch.object(reader.ownership, "expected_config"),
                  self.assertRaisesRegex(ValueError, "escaping checkpoint")):
                reader.descriptors(root, 0)
            (root / name).unlink()
            with patch.object(reader.ownership, "expected_config"), self.assertRaises(FileNotFoundError):
                reader.descriptors(root, 0)

    def test_payload_hashes_are_exact_and_truncated_bytes_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / "raw.bf16"
            source.write_bytes(b"prefix" + b"DATA" + b"suffix")
            self.assertEqual(reader.tensor_sha256(source, 6, 4),
                             hashlib.sha256(b"DATA").hexdigest())
            with self.assertRaisesRegex(ValueError, "truncated"):
                reader.tensor_sha256(source, 7, 100)


if __name__ == "__main__":
    unittest.main()
