"""Repeated audit must not demote a later release stage."""

from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import __main__ as main
from bigcherry.release import records as releases # noqa: E402


class AuditStageTests(unittest.TestCase):
    def test_later_stage_is_preserved_after_successful_reaudit(self):
        record = releases.ReleaseRecord(revision="abc123", stage="patched")
        args = mock.Mock()
        args.llama_root = "."
        args.strict = True
        args.verbose = False
        with mock.patch.object(main, "_record_for", return_value=record), \
             mock.patch.object(main.source_audit, "audit", return_value={"source_revision": "abc123", "source_dirty": False, "summary": {}, "checks": []}), \
             mock.patch.object(main.source_audit, "passed", return_value=True), \
             mock.patch.object(main.source_audit, "format_report", return_value=""), \
             mock.patch.object(record, "save"):
            self.assertEqual(main.cmd_audit(args), 0)
        self.assertEqual(record.stage, "patched")


if __name__ == "__main__":
    unittest.main()
