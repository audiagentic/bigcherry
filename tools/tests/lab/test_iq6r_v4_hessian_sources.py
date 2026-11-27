"""Offline tests for the pinned IQ6R-V4 public Hessian import gates."""
from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "lab/radiance"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = load("iq6r_v4_fetch_hessians")
validate = load("iq6r_v4_validate_hessians")


class HessianSourcesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_header(self, metadata: dict, dtype: str = "F32") -> Path:
        path = self.root / "h.safetensors"
        h = {
            "__metadata__": metadata,
            "hessian_upper": {
                "dtype": dtype,
                "shape": [3],
                "data_offsets": [0, 12],
            },
        }
        payload = json.dumps(h).encode()
        path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"\x00" * 12)
        return path

    def test_header_valid(self):
        path = self.write_header({
            "k": "2", "rows": "7",
            "linears": "example",
            "layout": "upper triangle, row-major, packed",
        })
        self.assertEqual(validate.inspect_header(path)["rows"], 7)

    def test_header_wrong_dtype_fails(self):
        path = self.write_header({
            "k": "2", "rows": "7",
            "linears": "example",
            "layout": "upper triangle, row-major, packed",
        }, dtype="F16")
        with self.assertRaisesRegex(ValueError, "invalid RADGRAM2"):
            validate.inspect_header(path)

    def test_expected_257_sites(self):
        kinds = ["full_attention" if i % 4 == 3 else "linear_attention"
                 for i in range(64)]
        names = validate.expected_names({"text_config": {"layer_types": kinds}})
        self.assertEqual(len(names), 257)
        self.assertIn("lm_head.input.safetensors", names)
        self.assertIn("model.language_model.layers.63.self_attn.o_proj.input.safetensors", names)
        self.assertNotIn("model.language_model.layers.62.self_attn.o_proj.input.safetensors", names)

    def test_reject_untrusted_path(self):
        sample = {
            "path": "hessians/calib/../checkpoint.safetensors",
            "lfs": {"oid": "a" * 64},
            "size": 50,
        }
        with self.assertRaisesRegex(ValueError, "unexpected upstream path"):
            fetch.manifest_entries([sample])

    def test_reject_incomplete_upstream_set(self):
        sample = {
            "path": "hessians/calib/lm_head.input.safetensors",
            "lfs": {"oid": "a" * 64},
            "size": 50,
        }
        with self.assertRaisesRegex(ValueError, "expected 257"):
            fetch.manifest_entries([sample])


if __name__ == "__main__":
    unittest.main()
