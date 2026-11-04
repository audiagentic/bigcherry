"""BU01 ccache artifact verification and safe extraction (offline, no ROCm required)."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "ci_ccache", ROOT / "tools/lab/radiance/ci_ccache.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.patch = mock.patch.object(mod, "fingerprint",
                                       return_value={"schema": mod.SCHEMA, "cc": "matching"})
        self.patch.start()
        workspace = ROOT / "work" / "bu01-offline-tests"
        workspace.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=workspace)
        self.base = Path(self.temporary.name)
        self.cache = self.base / "cache"
        (self.cache / "a").mkdir(parents=True)
        (self.cache / "a" / "object").write_bytes(b"compiled object")
        self.out = self.base / "artifact"
        self.out.mkdir()
        (self.out / "compiler.json").write_text(
            json.dumps({"schema": mod.SCHEMA, "cc": "matching"}))
        mod.pack(self.cache, self.out)
        self.work = self.base / "lab"

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def test_valid_import(self):
        mod.import_cache(self.out, self.work)
        self.assertEqual((self.work / ".ccache/a/object").read_bytes(), b"compiled object")

    def test_compiler_mismatch_refuses_before_copy(self):
        (self.out / "compiler.json").write_text(json.dumps({"cc": "different"}))
        with self.assertRaisesRegex(mod.CacheError, "mismatch"):
            mod.import_cache(self.out, self.work)
        self.assertFalse((self.work / ".ccache").exists())

    def test_corrupt_archive_refuses_before_copy(self):
        with (self.out / mod.ARCHIVE).open("ab") as stream:
            stream.write(b"garbage")
        with self.assertRaisesRegex(mod.CacheError, "checksum mismatch"):
            mod.import_cache(self.out, self.work)
        self.assertFalse((self.work / ".ccache").exists())

    def test_path_traversal_refuses_before_copy(self):
        archive = self.out / mod.ARCHIVE
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo(".ccache/../../poison")
            member.size = 6
            tar.addfile(member, io.BytesIO(b"poison"))
        (self.out / "SHA256SUMS").write_text(f"{mod.sha256(archive)}  {mod.ARCHIVE}\n")
        with self.assertRaisesRegex(mod.CacheError, "unsafe"):
            mod.import_cache(self.out, self.work)
        self.assertFalse((self.base / "poison").exists())

    def test_link_member_refuses_before_copy(self):
        archive = self.out / mod.ARCHIVE
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo(".ccache/evil")
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            tar.addfile(member)
        (self.out / "SHA256SUMS").write_text(f"{mod.sha256(archive)}  {mod.ARCHIVE}\n")
        with self.assertRaisesRegex(mod.CacheError, "unsafe"):
            mod.import_cache(self.out, self.work)

    def test_fetch_rejects_pr_before_download(self):
        payload = {"headBranch": "evil", "conclusion": "success",
                   "workflowName": "Radiance build", "event": "pull_request"}
        with mock.patch.object(mod, "gh", return_value=json.dumps(payload)) as gh:
            with self.assertRaisesRegex(mod.CacheError, "successful main"):
                mod.fetch(self.work, "123")
            gh.assert_called_once()


if __name__ == "__main__":
    unittest.main()
