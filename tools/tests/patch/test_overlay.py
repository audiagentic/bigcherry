"""Focused I/O and rollback tests for source-overlay materialization."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import paths  # noqa: E402
from bigcherry.patch import overlay  # noqa: E402


class TestOverlayIO(unittest.TestCase):
    def test_changed_target_is_read_once_and_original_bytes_are_restorable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_root = tmp_path / "src"
            checkout = tmp_path / "checkout"
            source = source_root / "nested" / "file.txt"
            target = checkout / "nested" / "file.txt"
            source.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            source.write_bytes(b"new\n")
            target.write_bytes(b"old\r\n")

            path_type = type(target)
            real_read_bytes = path_type.read_bytes
            real_read_text = path_type.read_text
            target_reads = {"bytes": 0, "text": 0}

            def tracked_read_bytes(path):
                if path == target:
                    target_reads["bytes"] += 1
                return real_read_bytes(path)

            def tracked_read_text(path, *args, **kwargs):
                if path == target:
                    target_reads["text"] += 1
                return real_read_text(path, *args, **kwargs)

            backup: dict[str, str | None] = {}
            simulated: dict[str, str] = {}
            with (
                mock.patch.object(paths, "SRC_OVERLAY", source_root),
                mock.patch.object(path_type, "read_bytes", new=tracked_read_bytes),
                mock.patch.object(path_type, "read_text", new=tracked_read_text),
            ):
                changed = overlay.copy_overlay(
                    checkout,
                    dry_run=False,
                    backup=backup,
                    sim_texts=simulated,
                )

            self.assertEqual(changed, ["nested/file.txt"])
            self.assertEqual(target_reads, {"bytes": 1, "text": 0})
            self.assertEqual(backup, {"nested/file.txt": "old\r\n"})
            self.assertEqual(simulated, {"nested/file.txt": "new\n"})
            self.assertEqual(target.read_bytes(), b"new\n")

            overlay.restore_overlay(checkout, backup)
            self.assertEqual(target.read_bytes(), b"old\r\n")

    def test_unchanged_target_keeps_fast_path_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_root = tmp_path / "src"
            checkout = tmp_path / "checkout"
            source_root.mkdir()
            checkout.mkdir()
            (source_root / "same.txt").write_bytes(b"same\n")
            (checkout / "same.txt").write_bytes(b"same\n")
            backup: dict[str, str | None] = {}
            simulated: dict[str, str] = {}

            with mock.patch.object(paths, "SRC_OVERLAY", source_root):
                changed = overlay.copy_overlay(
                    checkout,
                    dry_run=False,
                    backup=backup,
                    sim_texts=simulated,
                )

            self.assertEqual(changed, [])
            self.assertEqual(backup, {})
            self.assertEqual(simulated, {})
            self.assertEqual((checkout / "same.txt").read_bytes(), b"same\n")

    def test_dry_run_captures_overlay_without_writing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_root = tmp_path / "src"
            checkout = tmp_path / "checkout"
            source_root.mkdir()
            checkout.mkdir()
            (source_root / "file.txt").write_bytes(b"new\n")
            target = checkout / "file.txt"
            target.write_bytes(b"old\n")
            backup: dict[str, str | None] = {}
            simulated: dict[str, str] = {}

            with mock.patch.object(paths, "SRC_OVERLAY", source_root):
                changed = overlay.copy_overlay(
                    checkout,
                    dry_run=True,
                    backup=backup,
                    sim_texts=simulated,
                )

            self.assertEqual(changed, ["file.txt"])
            self.assertEqual(backup, {"file.txt": "old\n"})
            self.assertEqual(simulated, {"file.txt": "new\n"})
            self.assertEqual(target.read_bytes(), b"old\n")


if __name__ == "__main__":
    unittest.main()
