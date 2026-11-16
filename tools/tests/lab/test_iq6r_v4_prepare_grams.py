"""Card-free tests of the packed-F32 to full RADGRAM2 adapter."""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "lab/radiance"
PYDEPS = Path(os.environ.get("BC_WORK_ROOT", ".")) / "iq6r-v4-qualification/pydeps"
if PYDEPS.is_dir():
    sys.path.insert(0, str(PYDEPS))
sys.path.insert(0, str(ROOT))

try:
    import iq6r_v4_prepare_grams as adapter
    import numpy as np
except ModuleNotFoundError:
    np = None
    adapter = None


@unittest.skipIf(np is None, "NumPy is required for the offline expansion test")
class GramAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, k: int, matrix: np.ndarray) -> Path:
        upper = matrix[np.triu_indices(k)].astype("<f4")
        meta = {
            "__metadata__": {
                "k": str(k), "rows": str(adapter.ROWS),
                "linears": "example", "layout": "upper triangle, row-major, packed",
            },
            "hessian_upper": {
                "dtype": "F32", "shape": [len(upper)],
                "data_offsets": [0, len(upper) * 4],
            },
        }
        text = json.dumps(meta).encode()
        src = self.root / "input.safetensors"
        src.write_bytes(struct.pack("<Q", len(text)) + text + upper.tobytes())
        return src

    def test_expand_and_alias(self):
        k = 4
        matrix = np.array([[11, 12, 13, 14],
                           [12, 22, 23, 24],
                           [13, 23, 33, 34],
                           [14, 24, 34, 44]], dtype="<f4")
        src = self.source(k, matrix)
        path = self.root / "gram.r0.blk.3.attn_qg.bin"
        adapter.convert_one(src, path, ("blk.3.attn_k", "blk.3.attn_v"))
        with path.open("rb") as stream:
            self.assertEqual(adapter.HEADER.unpack(stream.read(32)),
                             (b"RADGRAM2", 4, 0, adapter.ROWS, 0, 0))
            result = np.frombuffer(stream.read(), dtype="<f4").reshape((k, k))
        np.testing.assert_array_equal(result, matrix)
        self.assertTrue(path.samefile(self.root / "gram.r0.blk.3.attn_k.bin"))
        self.assertTrue(path.samefile(self.root / "gram.r0.blk.3.attn_v.bin"))

    def test_refuses_existing_destination(self):
        path = self.root / "gram.r0.output.bin"
        path.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            adapter.convert_one(self.source(2, np.eye(2, dtype="<f4")), path, ())

    def test_site_mapping(self):
        self.assertEqual(adapter.names_for_site(
            "model.language_model.layers.3.input.safetensors"),
            ("blk.3.attn_qg", "blk.3.attn_k", "blk.3.attn_v"))
        self.assertEqual(adapter.names_for_site(
            "model.language_model.layers.0.input.safetensors"),
            ("blk.0.ssm_inz",))
        self.assertEqual(adapter.names_for_site("lm_head.input.safetensors"), ("output",))
        with self.assertRaises(ValueError):
            adapter.names_for_site("../wrong.safetensors")


if __name__ == "__main__":
    unittest.main()
