"""WSL04: builds/sources are visible in a read-only, protect-all inventory."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

from bigcherry.cli import workspace_cmds
from bigcherry.core import retention


class CacheInventory(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_build_and_source_sidecars_never_authorize_gc(self):
        src = self.root / "sources" / "source-key"
        src.mkdir(parents=True)
        (src / "foo.txt").write_text("source")
        (src.parent / "source-key.metadata.json").write_text('{"source": "known"}')
        b = self.root / "builds" / "recipe-key" / "build-key"
        b.mkdir(parents=True)
        (b / "binary").write_bytes(b"binary")
        (b / "bigcherry-build-metadata-radiance.json").write_text('{"build": "known"}')

        result = retention.inspect_cache_roots(self.root)
        self.assertEqual(len(result), 2)
        self.assertEqual([x["kind"] for x in result], ["build", "source"])
        self.assertTrue(all(x["action"] == "protected" for x in result))
        self.assertTrue(all(x["metadata"] == "present-unverified" for x in result))
        self.assertTrue(all("verified" in x["reasons"][0] for x in result))
        self.assertGreater(result[0]["size_bytes"], 0)

    def test_orphan_and_malformed_metadata_are_protected(self):
        source_root = self.root / "sources"
        source_root.mkdir()
        (source_root / "orphan.metadata.json").write_text("{}")
        src = source_root / "unverified"
        src.mkdir()
        (source_root / "unverified.metadata.json").write_text("{broken")
        entries = retention.inspect_cache_roots(self.root)
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(e["action"] == "protected" for e in entries))
        self.assertEqual({e["metadata"] for e in entries}, {"invalid", "untrusted"})

    @unittest.skipIf(sys.platform == "win32", "symlink privileges vary")
    def test_build_symlink_is_reported_without_traversal(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "large").write_text("do-not-follow" * 20)
        source_dir = self.root / "builds" / "recipe"
        source_dir.mkdir(parents=True)
        (source_dir / "borrowed").symlink_to(outside, target_is_directory=True)
        entries = retention.inspect_cache_roots(self.root)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], "protected")
        self.assertEqual(entries[0]["metadata"], "untrusted")
        self.assertIsNone(entries[0]["size_bytes"])

    def test_cli_builds_json_is_read_only_and_apply_still_refused(self):
        p = self.root / "builds" / "recipe" / "build"
        p.mkdir(parents=True)
        args = Namespace(work_root=str(self.root), apply=False, builds=True, json=True, no_size=True,
                         min_age_days=7, keep_newest=3, verbose=False)
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(workspace_cmds.cmd_clean(args), 0)
        data = json.loads(captured.getvalue())
        self.assertTrue(data["read_only"])
        self.assertEqual(len(data["entries"]), 1)
        self.assertEqual(data["entries"][0]["action"], "protected")
        self.assertIsNone(data["entries"][0]["size_bytes"])
        args.apply = True
        with redirect_stdout(io.StringIO()):
            self.assertEqual(workspace_cmds.cmd_clean(args), 2)
        self.assertTrue(p.exists())


if __name__ == "__main__":
    unittest.main()
