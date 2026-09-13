"""Canonical JSON and domain-separated hashing contracts."""

from __future__ import annotations

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core.canonical import canonical_json_bytes, domain_blake2b_128


class CanonicalTests(unittest.TestCase):
    def test_mapping_order_does_not_change_bytes_or_digest(self):
        left = {"b": 2, "a": {"y": 2, "x": 1}}
        right = {"a": {"x": 1, "y": 2}, "b": 2}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(
            domain_blake2b_128("domain", left),
            domain_blake2b_128("domain", right),
        )

    def test_json_is_compact_sorted_utf8(self):
        self.assertEqual(canonical_json_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}')
        self.assertEqual(canonical_json_bytes({"value": "caf\u00e9"}), b'{"value":"caf\\u00e9"}')

    def test_domain_separates_same_payload(self):
        payload = {"a": 1}
        self.assertNotEqual(
            domain_blake2b_128("one", payload),
            domain_blake2b_128("two", payload),
        )

    def test_digest_shape(self):
        digest = domain_blake2b_128("domain", {"a": 1})
        self.assertRegex(digest, r"^[0-9a-f]{32}$")


if __name__ == "__main__":
    unittest.main()
