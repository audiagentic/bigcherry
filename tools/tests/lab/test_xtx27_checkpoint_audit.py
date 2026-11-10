"""VTP06: source BF16 checkpoint audit, no real tensor allocation or GPU."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest
from unittest.mock import patch

SRC = pathlib.Path(__file__).resolve().parents[2] / "lab/radiance/xtx27_checkpoint_audit.py"
SPEC = importlib.util.spec_from_file_location("xtx27_checkpoint_audit", SRC)
assert SPEC and SPEC.loader
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def shard(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    entries = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        offset = len(payload)
        payload.extend(data)
        entries[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, len(payload)],
        }
    encoded = json.dumps(entries).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def fixture(root: pathlib.Path) -> None:
    shard(root / "a.safetensors", {
        "layer.a": ("BF16", [2, 3], bytes(range(12))),
        "layer.b": ("BF16", [3], bytes(range(6))),
    })
    shard(root / "b.safetensors", {
        "layer.c": ("BF16", [], bytes(range(2))),
    })
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "text_config": {"num_hidden_layers": 64},
    }))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 20},
        "weight_map": {
            "layer.a": "a.safetensors", "layer.b": "a.safetensors",
            "layer.c": "b.safetensors",
        },
    }))


class SourceAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        fixture(self.root)

    def _audit(self, *, fingerprint_payload: bool = False) -> dict:
        with patch.object(q, "check_dense_ffn", return_value={"ffn_projection_tensors": 192}):
            return q.audit(self.root, fingerprint_payload=fingerprint_payload)

    def test_valid_index_shards_and_bounded_read(self):
        result = self._audit()
        self.assertEqual(result["status"],
                         "STRUCTURALLY_VALID_HEADER_ONLY_PAYLOAD_NOT_HASHED")
        self.assertEqual(result["source_tensors"], 3)
        self.assertEqual(result["shard_count"], 2)
        self.assertEqual(result["total_tensor_payload_bytes"], 20)
        self.assertNotIn("full_shard_sha256", result["shards"]["a.safetensors"])
        self.assertEqual(result["model"]["ffn_projection_tensors"], 192)
        self.assertIn("tokenizer.json", result["source_file_sha256"])
        self.assertEqual(result["tensor_census"]["other"],
                         {"tensors": 3, "bf16_bytes": 20})

    def test_full_payload_fingerprint_only_when_selected(self):
        result = self._audit(fingerprint_payload=True)
        self.assertEqual(
            result["shards"]["a.safetensors"]["full_shard_sha256"],
            hashlib.sha256((self.root / "a.safetensors").read_bytes()).hexdigest(),
        )
        self.assertIn("NOT_EXTERNALLY_VERIFIED", result["payload_integrity"])

    def test_index_extra_missing_bad_path_or_total_rejected(self):
        index_path = self.root / "model.safetensors.index.json"
        original = json.loads(index_path.read_text())
        changes = [
            ("path traversal", lambda d: d["weight_map"].update({"layer.a": "../a.safetensors"})),
            ("missing name", lambda d: d["weight_map"].pop("layer.a")),
            ("wrong shard", lambda d: d["weight_map"].update({"layer.a": "b.safetensors"})),
            ("incorrect total", lambda d: d["metadata"].update({"total_size": 21})),
        ]
        for name, change in changes:
            with self.subTest(name=name):
                doc = json.loads(json.dumps(original))
                change(doc)
                index_path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    self._audit()
        index_path.write_text(json.dumps(original))

    def test_duplicate_json_keys_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            json.loads('{"a":1,"a":2}', object_pairs_hook=q.unique_keys)

    def test_bad_dtype_layout_and_payload_length(self):
        path = self.root / "a.safetensors"
        orig = path.read_bytes()
        self.addCleanup(lambda: path.write_bytes(orig))
        for name, mutation, error in (
            ("dtype", lambda v: v["layer.a"].update({"dtype": "F32"}), "original BF16"),
            ("offset gap", lambda v: v["layer.b"].update({"data_offsets": [13, 19]}),
             "noncontiguous"),
            ("wrong length", lambda v: v["layer.a"].update({"shape": [3, 3]}),
             "byte-count"),
            ("overlap", lambda v: v["layer.b"].update({"data_offsets": [0, 6]}),
             "noncontiguous"),
        ):
            with self.subTest(name=name):
                length = struct.unpack("<Q", orig[:8])[0]
                j = json.loads(orig[8:8 + length])
                mutation(j)
                h = json.dumps(j).encode()
                path.write_bytes(struct.pack("<Q", len(h)) + h + orig[8 + length:])
                with self.assertRaisesRegex(ValueError, error):
                    q.shard_metadata(path)
        path.write_bytes(orig)

    def test_truncated_header_and_trailing_payload_rejected(self):
        path = self.root / "b.safetensors"
        original = path.read_bytes()
        path.write_bytes(struct.pack("<Q", q.MAX_HEADER_BYTES + 1) + original[8:])
        with self.assertRaisesRegex(ValueError, "invalid header length"):
            q.shard_metadata(path)
        path.write_bytes(original + b"\x00")
        with self.assertRaisesRegex(ValueError, "final tensor data end"):
            q.shard_metadata(path)

    def test_tensor_category_does_not_miscount_vision_mlp(self):
        self.assertEqual(q.tensor_group("model.visual.blocks.0.mlp.up_proj.weight"),
                         "vision")
        self.assertEqual(q.tensor_group(
            "model.language_model.layers.0.mlp.up_proj.weight"), "text_ffn")
        self.assertEqual(q.tensor_group("mtp.layers.0.weight"), "mtp")

    def test_27b_ffn_geometry_requirements(self):
        config = {"model_type": "qwen3_5", "text_config": {
            "num_hidden_layers": 64, "hidden_size": 5120,
            "intermediate_size": 17408, "max_position_embeddings": 262144,
            "layer_types": ["full_attention"] * 16 + ["linear_attention"] * 48,
        }}
        shapes = {}
        for layer in range(64):
            prefix = f"model.language_model.layers.{layer}.mlp."
            shapes[prefix + "gate_proj.weight"] = [17408, 5120]
            shapes[prefix + "up_proj.weight"] = [17408, 5120]
            shapes[prefix + "down_proj.weight"] = [5120, 17408]
        self.assertEqual(q.check_dense_ffn(config, shapes)["ffn_projection_tensors"], 192)
        shapes["model.language_model.layers.63.mlp.down_proj.weight"] = [5120, 17407]
        with self.assertRaisesRegex(ValueError, "FFN tensor geometry"):
            q.check_dense_ffn(config, shapes)

    def test_receipt_creation_refuses_overwrite(self):
        receipt = self.root / "receipt.json"
        with patch.object(q, "check_dense_ffn", return_value={}):
            self.assertEqual(q.main(["--checkpoint", str(self.root),
                                     "--receipt", str(receipt)]), 0)
        before = receipt.read_bytes()
        with (
            patch.object(q, "audit", side_effect=AssertionError("must not read 55 GiB before checking receipt")),
            self.assertRaises(FileExistsError),
        ):
            q.main(["--checkpoint", str(self.root), "--receipt", str(receipt)])
        self.assertEqual(receipt.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
