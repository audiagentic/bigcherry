from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import paths


class EvidenceDirTests(TestCase):
    def test_rejects_path_separators(self) -> None:
        with self.assertRaises(ValueError):
            paths.evidence_dir("HI65/../escape")

    def test_rejects_leading_dot(self) -> None:
        with self.assertRaises(ValueError):
            paths.evidence_dir(".hidden")

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            paths.evidence_dir("")

    def test_accepts_plan_item_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(paths, "ARTIFACTS", root / "artifacts"):
                result = paths.evidence_dir("HI65")
                self.assertEqual(result, root / "artifacts" / "HI65")
                self.assertTrue(result.is_dir())

    def test_accepts_dated_run_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(paths, "ARTIFACTS", root / "artifacts"):
                result = paths.evidence_dir("2026-08-21-HI35-HI36-27b-r9700")
                self.assertTrue(result.is_dir())

    def test_create_false_does_not_make_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(paths, "ARTIFACTS", root / "artifacts"):
                result = paths.evidence_dir("HI65", create=False)
                self.assertFalse(result.exists())


if __name__ == "__main__":
    import unittest

    unittest.main()
