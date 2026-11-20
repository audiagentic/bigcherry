"""Exhaustive, non-disclosing hashes for legacy recovered work-root evidence."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "docs/evidence/2026-11-17-work-root-results"


class RecoveryBundleIntegrityTests(unittest.TestCase):
    def test_hashed_path_index_covers_all_retained_bytes(self) -> None:
        index_bytes = (BUNDLE / "integrity-digests.tsv").read_bytes()
        meta = json.loads((BUNDLE / "integrity-index.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["kind"], "legacy-recovered-work-root-results")
        self.assertEqual(meta["integrity_digests_sha256"], hashlib.sha256(index_bytes).hexdigest())

        entries: dict[str, str] = {}
        for line in index_bytes.decode("utf-8").splitlines():
            path_digest, separator, content_digest = line.partition("\t")
            self.assertEqual(separator, "\t")
            self.assertRegex(path_digest, r"^[0-9a-f]{64}$")
            self.assertRegex(content_digest, r"^[0-9a-f]{64}$")
            self.assertNotIn(path_digest, entries)
            entries[path_digest] = content_digest
        self.assertEqual(list(entries), sorted(entries))

        actual: dict[str, str] = {}
        for p in BUNDLE.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(BUNDLE).as_posix()
            if rel in {"integrity-digests.tsv", "integrity-index.json"}:
                continue
            self.assertFalse(p.is_symlink())
            path_digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()
            self.assertNotIn(path_digest, actual)
            actual[path_digest] = hashlib.sha256(p.read_bytes()).hexdigest()

        self.assertEqual(entries, actual)
        self.assertEqual(meta["file_count"], len(actual))


if __name__ == "__main__":
    unittest.main()
