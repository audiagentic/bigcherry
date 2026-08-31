"""bigcherry.analysis.jsonl_io -- shared strict JSONL reader."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.analysis import jsonl_io  # noqa: E402


class ReadRowsTests(unittest.TestCase):
    def test_truncated_final_line_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "a"}) + "\n")
                handle.write('{"kind":"b","x":')  # torn, no trailing newline
            rows = jsonl_io.read_rows(path)
        self.assertEqual(rows, [{"kind": "a"}])

    def test_newline_terminated_corrupt_final_line_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "a"}) + "\n")
                handle.write('{"kind":"b","x":\n')  # fully written, still corrupt
            with self.assertRaises(jsonl_io.JsonlReadError):
                jsonl_io.read_rows(path)

    def test_interior_corruption_raises_and_does_not_drop_later_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "a"}) + "\n")
                handle.write('{"kind":"corrupt\n')
                handle.write(json.dumps({"kind": "c"}) + "\n")
            with self.assertRaises(jsonl_io.JsonlReadError):
                jsonl_io.read_rows(path)

    def test_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            path.write_text(
                json.dumps({"kind": "a"}) + "\n\n" + json.dumps({"kind": "b"}) + "\n",
                encoding="utf-8",
            )
            rows = jsonl_io.read_rows(path)
        self.assertEqual(rows, [{"kind": "a"}, {"kind": "b"}])

    def test_empty_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            path.write_text("", encoding="utf-8")
            self.assertEqual(jsonl_io.read_rows(path), [])


class ReadResultRecordsTests(unittest.TestCase):
    def test_filters_to_kind_result_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            path.write_text(
                json.dumps({"kind": "header"}) + "\n"
                + json.dumps({"kind": "result", "signature": "a"}) + "\n"
                + json.dumps({"kind": "observation", "signature": "b"}) + "\n",
                encoding="utf-8",
            )
            results = jsonl_io.read_result_records(path)
        self.assertEqual(results, [{"kind": "result", "signature": "a"}])


if __name__ == "__main__":
    unittest.main()
