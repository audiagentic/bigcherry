"""Withdrawn evidence records stay on file but are never pooled."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bigcherry.patch import evidence


def _write(root: Path, records: list[dict], withdrawn: list[dict] | None) -> None:
    (root / "p.json").write_text(json.dumps({
        "schema_version": sorted(evidence.READABLE_SCHEMA_VERSIONS)[-1],
        "patch_id": "p", "records": records,
    }), encoding="utf-8")
    if withdrawn is not None:
        (root / evidence.WITHDRAWALS_FILE).write_text(json.dumps({"withdrawn": withdrawn}), encoding="utf-8")


class WithdrawalTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_withdrawn_records_are_kept_but_not_pooled(self) -> None:
        _write(self.root, [{"record_digest": "a"}, {"record_digest": "b"}],
               [{"record_digest": "b", "reason": "measured alongside a concurrent GPU job"}])
        self.assertEqual(len(evidence.load_records("p", root=self.root)), 2)
        self.assertEqual([r["record_digest"] for r in evidence.poolable_records("p", root=self.root)], ["a"])

    def test_no_withdrawals_file_pools_everything(self) -> None:
        _write(self.root, [{"record_digest": "a"}], None)
        self.assertEqual(len(evidence.poolable_records("p", root=self.root)), 1)

    def test_withdrawal_without_reason_is_rejected(self) -> None:
        _write(self.root, [{"record_digest": "a"}], [{"record_digest": "a"}])
        with self.assertRaises(evidence.ValidationEvidenceError):
            evidence.poolable_records("p", root=self.root)


if __name__ == "__main__":
    unittest.main()
