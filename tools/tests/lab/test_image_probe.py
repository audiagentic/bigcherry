"""image_probe draws a valid PNG and judges answers and refusals (no server needed)."""

import importlib.util
import struct
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("image_probe", ROOT / "tools/lab/radiance/image_probe.py")
image_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_probe)


class ImageProbeTests(unittest.TestCase):
    def test_the_picture_is_a_valid_png_with_the_two_shapes(self) -> None:
        data = image_probe.png(64, 64)
        self.assertEqual(b"\x89PNG\r\n\x1a\n", data[:8])
        offset, chunks = 8, {}
        while offset < len(data):
            (length,) = struct.unpack(">I", data[offset:offset + 4])
            kind, body = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + length]
            (crc,) = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])
            self.assertEqual(zlib.crc32(kind + body) & 0xFFFFFFFF, crc)
            chunks[kind] = body
            offset += 12 + length
        self.assertEqual((64, 64), struct.unpack(">II", chunks[b"IHDR"][:8]))
        raw = zlib.decompress(chunks[b"IDAT"])
        self.assertEqual(64 * (1 + 64 * 3), len(raw))
        pixel = lambda x, y: tuple(raw[y * (1 + 64 * 3) + 1 + x * 3:][:3])
        self.assertEqual((220, 30, 30), pixel(16, 32))   # the disc at the left quarter
        self.assertEqual((30, 190, 40), pixel(48, 32))   # the square at the right quarter
        self.assertEqual((20, 40, 150), pixel(2, 2))     # the field

    def test_the_bmp_is_well_formed_and_holds_the_shapes(self) -> None:
        data = image_probe.bmp(64, 64)
        self.assertEqual(b"BM", data[:2])
        size, offset = struct.unpack("<I4xI", data[2:14])
        self.assertEqual(len(data), size)
        row = 64 * 3
        pixel = lambda x, y: tuple(data[offset + (63 - y) * row + x * 3:][:3][::-1])   # rows run bottom-up, BGR
        self.assertEqual((220, 30, 30), pixel(16, 32))
        self.assertEqual((30, 190, 40), pixel(48, 32))
        self.assertEqual((20, 40, 150), pixel(2, 2))

    def test_answers_refusals_and_wrong_answers_are_told_apart(self) -> None:
        ok = {"choices": [{"message": {"content": "The round shape is red and the square is green."}}]}
        wrong = {"choices": [{"message": {"content": "I see two shapes."}}]}
        self.assertEqual("answered", image_probe.judge(200, ok)[0])
        self.assertEqual("wrong", image_probe.judge(200, wrong)[0])
        verdict, text = image_probe.judge(400, {"error": {"message": "the engine has no media decoder"}})
        self.assertEqual("refused", verdict)
        self.assertIn("no media decoder", text)


if __name__ == "__main__":
    unittest.main()
