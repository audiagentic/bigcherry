"""HI121 M1a: tests for the generic backend-neutral CapabilityMask128."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning.capabilities import CapabilityMask128, CapabilityMaskError  # noqa: E402


class ConstructionTests(unittest.TestCase):
    def test_direct_value(self):
        self.assertEqual(CapabilityMask128(0x1f).value, 0x1f)

    def test_from_words(self):
        mask = CapabilityMask128.from_words(lo=0x1f, hi=0)
        self.assertEqual(mask.value, 0x1f)

    def test_from_words_hi_nonzero(self):
        mask = CapabilityMask128.from_words(lo=0, hi=1)
        self.assertEqual(mask.value, 1 << 64)

    def test_from_bytes(self):
        data = (0x1f).to_bytes(16, "big")
        self.assertEqual(CapabilityMask128.from_bytes(data).value, 0x1f)

    def test_from_hex(self):
        mask = CapabilityMask128.from_hex("0000000000000000000000000000001f")
        self.assertEqual(mask.value, 0x1f)


class RoundTripTests(unittest.TestCase):
    def test_hex_bytes_words_roundtrip(self):
        mask = CapabilityMask128.from_hex("0000000000000000000000000000001f")
        self.assertEqual(CapabilityMask128.from_bytes(mask.to_bytes()), mask)
        lo, hi = mask.to_words()
        self.assertEqual(CapabilityMask128.from_words(lo, hi), mask)
        self.assertEqual(mask.to_hex(), "0000000000000000000000000000001f")

    def test_hi_word_roundtrip(self):
        mask = CapabilityMask128.from_words(lo=0x123, hi=0x456)
        lo, hi = mask.to_words()
        self.assertEqual((lo, hi), (0x123, 0x456))
        self.assertEqual(CapabilityMask128.from_bytes(mask.to_bytes()), mask)


class ContainsTests(unittest.TestCase):
    def test_contains_subset(self):
        full = CapabilityMask128(0b1111)
        subset = CapabilityMask128(0b0101)
        self.assertTrue(full.contains(subset))

    def test_does_not_contain_missing_bit(self):
        full = CapabilityMask128(0b0101)
        required = CapabilityMask128(0b1111)
        self.assertFalse(full.contains(required))

    def test_contains_self(self):
        mask = CapabilityMask128(0x1f)
        self.assertTrue(mask.contains(mask))

    def test_contains_zero_always_true(self):
        mask = CapabilityMask128(0)
        self.assertTrue(CapabilityMask128(0x1f).contains(mask))

    def test_contains_rejects_non_mask(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128(0x1f).contains(5)  # type: ignore[arg-type]


class ValidationTests(unittest.TestCase):
    def test_negative_value_rejected(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128(-1)

    def test_too_large_value_rejected(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128(1 << 128)

    def test_bool_rejected(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128(True)  # type: ignore[arg-type]

    def test_non_int_rejected(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128("0x1f")  # type: ignore[arg-type]

    def test_from_words_lo_out_of_range(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128.from_words(lo=1 << 64, hi=0)

    def test_from_words_hi_out_of_range(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128.from_words(lo=0, hi=1 << 64)

    def test_from_words_negative_rejected(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128.from_words(lo=-1, hi=0)

    def test_from_bytes_wrong_length_short(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128.from_bytes(b"\x00" * 15)

    def test_from_bytes_wrong_length_long(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128.from_bytes(b"\x00" * 17)

    def test_from_bytes_rejects_non_bytes(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128.from_bytes("00" * 16)  # type: ignore[arg-type]

    def test_from_hex_wrong_length(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128.from_hex("1f")

    def test_from_hex_uppercase_rejected(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128.from_hex("0000000000000000000000000000001F")

    def test_from_hex_non_hex_characters(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128.from_hex("g" * 32)

    def test_from_hex_rejects_non_str(self):
        with self.assertRaises(CapabilityMaskError):
            CapabilityMask128.from_hex(31)  # type: ignore[arg-type]


class OrTests(unittest.TestCase):
    def test_or_combines_bits(self):
        combined = CapabilityMask128(0b0001) | CapabilityMask128(0b0010)
        self.assertEqual(combined.value, 0b0011)


if __name__ == "__main__":
    unittest.main()
