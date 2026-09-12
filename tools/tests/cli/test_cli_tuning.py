from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli import tuning  # noqa: E402


class TuningCliRecordLookupTests(unittest.TestCase):
    @staticmethod
    def _args(*, force: bool = False, dry_run: bool = False, llama_root: str | None = None) -> Namespace:
        return Namespace(
            llama_root=llama_root,
            force=force,
            variant_set="default",
            arch="gfx1100",
            inventory=None,
            winners=None,
            generated_root=None,
            dry_run=dry_run,
        )

    def test_generate_uses_release_records_and_refuses_unpatched_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = mock.Mock(stage="pulled", revision="abc123")
            args = self._args(llama_root=str(root))
            stderr = StringIO()
            with mock.patch.object(
                tuning.paths, "llama_root", return_value=root,
            ), mock.patch.object(
                tuning.releases, "record_for_checkout", return_value=record,
            ) as lookup, mock.patch(
                "bigcherry.tuning.catalog.main",
            ) as catalog, redirect_stderr(stderr):
                status = tuning.cmd_generate(args)

            self.assertEqual(status, 2)
            lookup.assert_called_once_with(root)
            catalog.assert_not_called()
            self.assertIn("unpatched tree", stderr.getvalue())

    def test_generate_force_bypasses_stage_but_does_not_save_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = mock.Mock(stage="pulled", revision="abc123")
            args = self._args(force=True, llama_root=str(root))
            with mock.patch.object(
                tuning.paths, "llama_root", return_value=root,
            ), mock.patch.object(
                tuning.releases, "record_for_checkout", return_value=record,
            ), mock.patch(
                "bigcherry.tuning.catalog.main", return_value=1,
            ) as catalog:
                status = tuning.cmd_generate(args)

            self.assertEqual(status, 1)
            catalog.assert_called_once()
            record.advance_to.assert_not_called()
            record.save.assert_not_called()

    def test_generate_dry_run_does_not_persist_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = mock.Mock(stage="patched", revision="abc123")
            args = self._args(dry_run=True, llama_root=str(root))
            with mock.patch.object(
                tuning.paths, "llama_root", return_value=root,
            ), mock.patch.object(
                tuning.releases, "record_for_checkout", return_value=record,
            ), mock.patch(
                "bigcherry.tuning.catalog.main", return_value=0,
            ) as catalog:
                status = tuning.cmd_generate(args)

            self.assertEqual(status, 0)
            forwarded = catalog.call_args.args[0]
            self.assertIn("--dry-run", forwarded)
            record.advance_to.assert_not_called()
            record.save.assert_not_called()

    def test_generate_success_advances_and_saves_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "hip-autotune-manifest.json").write_text(
                '{"manifest_hash": "manifest-hash"}\n', encoding="utf-8",
            )
            record = mock.Mock(stage="patched", revision="abc123")
            args = self._args(llama_root=str(root))
            with mock.patch.object(
                tuning.paths, "llama_root", return_value=root,
            ), mock.patch.object(
                tuning.paths, "artifact_dir", return_value=artifacts,
            ), mock.patch.object(
                tuning.releases, "record_for_checkout", return_value=record,
            ), mock.patch(
                "bigcherry.tuning.catalog.main", return_value=0,
            ) as catalog:
                status = tuning.cmd_generate(args)

            self.assertEqual(status, 0)
            self.assertEqual(record.manifest_hash, "manifest-hash")
            record.advance_to.assert_called_once_with("generated")
            record.save.assert_called_once_with()
            self.assertEqual(catalog.call_args.args[0][:4], [
                "--variant-set", "default", "--arch", "gfx1100",
            ])


if __name__ == "__main__":
    unittest.main()
