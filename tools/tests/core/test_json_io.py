"""Byte and failure contracts for the shared JSON persistence helpers."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.core.json_io import read_jsonl
from bigcherry.hardware.power_receipt import write_json_atomic


class JsonPersistenceTests(unittest.TestCase):
    def test_writer_preserves_both_existing_encodings(self):
        document = {"z": ["é", {"b": 2, "a": None}], "a": "雪"}
        for ensure_ascii in (True, False):
            with self.subTest(ensure_ascii=ensure_ascii), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "nested" / "record.json"
                write_json_atomic(
                    path, document, ensure_ascii=ensure_ascii,
                    sync_directory=False, newline="\n",
                )
                expected = """{
  "a": "雪",
  "z": [
    "é",
    {
      "a": null,
      "b": 2
    }
  ]
}
"""
                if ensure_ascii:
                    expected = expected.replace("雪", r"\u96ea").replace("é", r"\u00e9")
                self.assertEqual(path.read_bytes(), expected.encode("utf-8"))
                self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_serialization_preserves_destination_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_bytes(b"previous\n")
            with self.assertRaises(TypeError):
                write_json_atomic(path, {"bad": object()})
            self.assertEqual(path.read_bytes(), b"previous\n")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_replace_preserves_destination_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_bytes(b"previous\n")
            with patch("bigcherry.hardware.power_receipt.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    write_json_atomic(path, {"new": True})
            self.assertEqual(path.read_bytes(), b"previous\n")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_only_durable_receipts_sync_the_directory(self):
        for sync_directory in (True, False):
            with self.subTest(sync_directory=sync_directory), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "record.json"
                with patch("bigcherry.hardware.power_receipt.os.fsync") as fsync:
                    write_json_atomic(path, {}, sync_directory=sync_directory)
                self.assertEqual(fsync.call_count, 1 + int(sync_directory and hasattr(os, "O_DIRECTORY")))

    def test_jsonl_preserves_order_and_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("\n  {\"b\": 2}  \n\t\n{\"a\": \"é\"}\n", encoding="utf-8")
            self.assertEqual(read_jsonl(path), [{"b": 2}, {"a": "é"}])

    def test_jsonl_propagates_missing_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            with self.assertRaises(FileNotFoundError):
                read_jsonl(path)
            path.write_text("{}\ninvalid\n", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                read_jsonl(path)
