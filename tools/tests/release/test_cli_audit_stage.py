"""Repeated audit must not demote a later release stage."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli import source as cli_source
from bigcherry.release import records as releases # noqa: E402


class AuditStageTests(unittest.TestCase):
    def test_later_stage_is_preserved_after_successful_reaudit(self):
        record = releases.ReleaseRecord(revision="abc123", stage="patched")
        args = mock.Mock()
        args.llama_root = "."
        args.strict = True
        args.verbose = False
        # cmd_audit writes its report under paths.ARTIFACTS; keep it out of the repo.
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        with mock.patch.object(cli_source.paths, "ARTIFACTS", Path(scratch.name)),              mock.patch.object(cli_source.releases, "record_for_checkout", return_value=record), \
             mock.patch.object(cli_source.source_audit, "audit", return_value={"source_revision": "abc123", "source_dirty": False, "summary": {}, "checks": []}), \
             mock.patch.object(cli_source.source_audit, "passed", return_value=True), \
             mock.patch.object(cli_source.source_audit, "format_report", return_value=""), \
             mock.patch.object(record, "save"):
            self.assertEqual(cli_source.cmd_audit(args), 0)
        self.assertEqual(record.stage, "patched")


class PullStageTests(unittest.TestCase):
    def test_repull_preserves_a_later_stage(self):
        record = releases.ReleaseRecord(revision="abc123", stage="generated")
        checkout = tempfile.TemporaryDirectory()
        self.addCleanup(checkout.cleanup)
        (Path(checkout.name) / ".git").mkdir()
        args = mock.Mock()
        args.llama_root = checkout.name
        args.ref = None
        args.source = None
        args.full = False
        with mock.patch.object(cli_source, "_run"), \
             mock.patch.object(cli_source.upstream, "clear_stale_locks", return_value=[]), \
             mock.patch.object(cli_source.releases, "record_for_checkout", return_value=record), \
             mock.patch.object(record, "save"):
            self.assertEqual(cli_source.cmd_pull(args), 0)
        self.assertEqual(record.stage, "generated")


if __name__ == "__main__":
    unittest.main()
