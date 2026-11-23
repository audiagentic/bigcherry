"""Card-free checkpoint-view tests; no source tensors are modified."""
from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


P = Path(__file__).resolve().parents[2] / "lab/radiance/iq6r_v4_head_view.py"
S = importlib.util.spec_from_file_location("iq6r_head_view", P)
assert S and S.loader
mod = importlib.util.module_from_spec(S)
S.loader.exec_module(mod)


class HeadViewTests(unittest.TestCase):
    def setup(self, root: Path, include_head: bool = True):
        orig, classified = root / "original", root / "classified"
        orig.mkdir()
        classified.mkdir()
        shard = orig / "part.safetensors"
        payload = {"lm_head.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}} if include_head else {}
        head = json.dumps(payload).encode()
        shard.write_bytes(struct.pack("<Q", len(head)) + head + b"\x00\x00")
        (classified / "part.safetensors").symlink_to(shard)
        (orig / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"lm_head.weight": "part.safetensors"}}))
        (classified / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"output.k4.weight": "part.safetensors"}}))
        return orig, classified, shard

    def test_creates_linked_view_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orig, classified, shard = self.setup(root)
            out = root / "view"
            first = mod.expose_target_head(classified, orig, out, root)
            second = mod.expose_target_head(classified, orig, out, root)
            self.assertEqual(first, second)
            self.assertEqual(first["mapped_weights"], 2)
            self.assertTrue((out / "part.safetensors").samefile(shard))
            idx = json.loads((out / "model.safetensors.index.json").read_text())
            self.assertEqual(idx["weight_map"]["lm_head.weight"], "part.safetensors")
            self.assertNotIn("lm_head.weight",
                             json.loads((classified / "model.safetensors.index.json").read_text())["weight_map"])

    def test_refuses_missing_physical_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orig, classified, _ = self.setup(root, include_head=False)
            with self.assertRaisesRegex(ValueError, "physically contain"):
                mod.expose_target_head(classified, orig, root / "view", root)

    def test_refuses_to_write_into_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orig, classified, _ = self.setup(root)
            with self.assertRaisesRegex(ValueError, "new directory"):
                mod.expose_target_head(classified, orig, classified, root)

    def test_refuses_conflicting_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orig, classified, _ = self.setup(root)
            out = root / "view"
            out.mkdir()
            (out / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"bogus": "part.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "index differs"):
                mod.expose_target_head(classified, orig, out, root)


if __name__ == "__main__":
    unittest.main()
