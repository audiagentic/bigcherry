"""Card-free numerical/byte-layout tests for VTP13 GDN rank packer."""
from __future__ import annotations

import json
import pathlib
import struct
import sys
import tempfile
import unittest

LAB = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(LAB))
import xtx27_vtp13_gdn_pack as gdn


class GdnPackTests(unittest.TestCase):
    def test_qkv_noncontiguous_ranges_cover_exact_channels(self):
        mapping = gdn.layout()
        self.assertEqual(mapping["xtx0"], [(0, 512), (2048, 512), (4096, 1536)])
        self.assertEqual(mapping["xtx1"], [(512, 512), (2560, 512), (5632, 1536)])
        self.assertEqual(mapping["r9700"], [(1024, 1024), (3072, 1024), (7168, 3072)])
        self.assertEqual([sum(n for _, n in mapping[r]) for r in gdn.ROLES],
                         [2560, 2560, 5120])
        ordered = sorted((start, length)
                         for segments in mapping.values() for start, length in segments)
        self.assertEqual(sum(width for _, width in ordered), 10240)

    @staticmethod
    def source_fixture(tmp: pathlib.Path) -> dict:
        entries = {}
        for op, row_width in (("in_proj_qkv", 1), ("conv1d", 4)):
            src = tmp / f"source-{op}.bf16"
            # Every original row is distinguishable; fixed little-endian
            # elements retain bit identity through the physical Q|K|V pack.
            with src.open("wb") as output:
                for row in range(10240):
                    output.write(struct.pack("<H", row & 0xFFFF) * row_width)
            entries[op] = {
                "path": str(src), "offset": 0,
                "bytes": 10240 * row_width * 2,
                "row_width": row_width,
            }
        return {"schema": gdn.SCHEMA, "layer": 0,
                "sources": entries, "layout": gdn.layout()}

    def test_full_bytes_reconstruct_original_source(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = pathlib.Path(temp)
            desc = self.source_fixture(tmp)
            output = tmp / "output"
            receipt = gdn.pack(desc, output)
            self.assertEqual(receipt["status"], "EXACT_ORIGINAL_BF16_GDN_QKV_CONV_PACK_PASS_NOT_RUNTIME")
            self.assertEqual(set(receipt["outputs"]), {"in_proj_qkv", "conv1d"})
            for op, result in receipt["outputs"].items():
                self.assertTrue(result["complete_byte_equivalence"])
                self.assertEqual(result["reassembled_source_sha256"],
                                 result["source_payload_sha256"])
                for role, expected_rows in (("xtx0", 2560), ("xtx1", 2560),
                                            ("r9700", 5120)):
                    self.assertEqual(result["ranks"][role]["bytes"],
                                     expected_rows * desc["sources"][op]["row_width"] * 2)
            persisted = json.loads((output / "summary.json").read_text())
            self.assertEqual(persisted["status"], receipt["status"])
            with self.assertRaises(FileExistsError):
                gdn.pack(desc, output)

    def test_malformed_map_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = pathlib.Path(temp)
            desc = self.source_fixture(tmp)
            desc["layout"]["r9700"][0] = (1023, 1024)
            output = tmp / "bad"
            with self.assertRaisesRegex(ValueError, "gap/overlap"):
                gdn.pack(desc, output)
            persisted = json.loads((output / "summary.json").read_text())
            self.assertEqual(persisted["status"], "FAILED_INCOMPLETE")

    def test_input_gate_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = pathlib.Path(temp)
            desc = self.source_fixture(tmp)
            with self.assertRaises(ValueError):
                gdn.pack(desc, gdn.ROOT / "forbidden")
            self.assertFalse((gdn.ROOT / "forbidden").exists())


if __name__ == "__main__":
    unittest.main()
