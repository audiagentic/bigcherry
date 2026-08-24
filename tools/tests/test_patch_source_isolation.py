"""RS05 tests for tools/bigcherry/patch_source_isolation.py (patch-system PA02).

Covers the runbook RS05 test list (section 58): legacy materialization,
package materialization, tampered tree rejection, missing-manifest
non-trust, wrong-manifest-identity rejection, base-pin-movement identity,
and legacy/package migration tree equivalence. Real git fixtures throughout
(same convention as test_re04_materialization_safety.py).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import patch_source_isolation as psi  # noqa: E402

FLAT_PATCH_PY = """\
from bigcherry.patcher import Edit, FilePatch
GROUP = 'core'
STATE = 'untested'
PATCHES = [FilePatch(path='a.txt', edits=(Edit(id='e1', anchor='one', text='MARKER'),))]
"""

PKG_PATCH_PY = """\
from bigcherry.patcher import Edit, FilePatch
PATCHES = [FilePatch(path='a.txt', edits=(Edit(id='e1', anchor='one', text='MARKER'),))]
"""

PKG_TOML = """\
schema = 1
id = "1202_dual"
order = 1202
group = "core"
state = "untested"
kind = "framework"
origin = "local"
backend = "hip"
"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_upstream(directory: Path) -> tuple[Path, str]:
    repo = directory / "upstream"
    _git(directory, "init", str(repo))
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-m", "initial")
    return repo, _git(repo, "rev-parse", "HEAD")


class MaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-iso-")
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.upstream, self.base_rev = _init_upstream(self.base)
        self.worktrees = self.base / "worktrees"
        self.patches_root = self.base / "patches"
        self.patches_root.mkdir()
        self._old_patches_root = psi.PATCHES_ROOT
        psi.PATCHES_ROOT = self.patches_root

    def tearDown(self) -> None:
        # Remove the worktree registrations from the upstream repo before the
        # temp dir goes away (mirrors psi.remove_worktree's hygiene).
        _git(self.upstream, "worktree", "prune")
        psi.PATCHES_ROOT = self._old_patches_root

    def _flat(self, patch_id: str = "0001_flat") -> None:
        (self.patches_root / f"{patch_id}.py").write_text(FLAT_PATCH_PY, encoding="utf-8")

    def _package(self, patch_id: str = "1202_dual") -> None:
        package = self.patches_root / "rd" / patch_id
        package.mkdir(parents=True)
        (package / "patch.toml").write_text(
            PKG_TOML.replace('"1202_dual"', f'"{patch_id}"'), encoding="utf-8"
        )
        (package / "patch.py").write_text(PKG_PATCH_PY, encoding="utf-8")

    def _materialize(self, patch_id: str) -> Path:
        return psi.materialize_source(
            base_repo=self.upstream, worktree_root=self.worktrees,
            patch_module=patch_id, base_revision=self.base_rev,
        )

    def test_legacy_materialization(self) -> None:
        self._flat()
        source = self._materialize("0001_flat")
        content = (source / "a.txt").read_text(encoding="utf-8")
        self.assertIn("MARKER", content)
        manifest = json.loads(psi._manifest_path(source).read_text(encoding="utf-8"))
        self.assertEqual(manifest["patch_name"], "0001_flat")
        # Cache reuse: same identity -> same directory, no error.
        again = self._materialize("0001_flat")
        self.assertEqual(again, source)

    def test_package_materialization(self) -> None:
        self._package()
        source = self._materialize("1202_dual")
        content = (source / "a.txt").read_text(encoding="utf-8")
        self.assertIn("MARKER", content)
        manifest = json.loads(psi._manifest_path(source).read_text(encoding="utf-8"))
        self.assertEqual(manifest["patch_name"], "1202_dual")

    def test_tampered_tree_rejected(self) -> None:
        self._flat()
        source = self._materialize("0001_flat")
        (source / "a.txt").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(psi.PatchSourceIsolationError, "modified after materialization"):
            self._materialize("0001_flat")

    def test_missing_manifest_not_trusted(self) -> None:
        self._flat()
        source = self._materialize("0001_flat")
        psi._manifest_path(source).unlink()
        # Not trusted -> rebuilt from scratch, still correct.
        again = self._materialize("0001_flat")
        self.assertEqual(again, source)
        self.assertIn("MARKER", (again / "a.txt").read_text(encoding="utf-8"))
        self.assertTrue(psi._manifest_path(again).is_file())

    def test_wrong_manifest_identity_rejected(self) -> None:
        self._flat()
        source = self._materialize("0001_flat")
        manifest_path = psi._manifest_path(source)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Tamper an IDENTITY field (the _verify_reuse key loop) -- a wrong
        # identity is a provenance mismatch, raised fail-closed.
        manifest["patch_name"] = "1999_forged"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(psi.PatchSourceIsolationError, "provenance mismatch"):
            self._materialize("0001_flat")

    def test_base_pin_movement_changes_identity(self) -> None:
        self._flat()
        first = self._materialize("0001_flat")
        # Move the base revision: a new upstream commit.
        (self.upstream / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        _git(self.upstream, "add", "a.txt")
        _git(self.upstream, "commit", "-m", "move base")
        new_rev = _git(self.upstream, "rev-parse", "HEAD")
        self.assertNotEqual(new_rev, self.base_rev)
        second = psi.materialize_source(
            base_repo=self.upstream, worktree_root=self.worktrees,
            patch_module="0001_flat", base_revision=new_rev,
        )
        self.assertNotEqual(first, second)
        # The moved base is visible in the new worktree AND the old one is
        # untouched (still at the old base's content + marker).
        self.assertIn("three", (second / "a.txt").read_text(encoding="utf-8"))
        self.assertNotIn("three", (first / "a.txt").read_text(encoding="utf-8"))

    def test_legacy_package_migration_tree_equivalence(self) -> None:
        # Same edit expressed as a flat module and as a package: the
        # materialized CONTENT trees must be byte-identical (migration
        # criterion -- representation changes, tree does not).
        self._flat("0002_dual")
        self._package("1202_dual")
        legacy_source = self._materialize("0002_dual")
        packaged_source = self._materialize("1202_dual")
        self.assertNotEqual(legacy_source, packaged_source,
                            "different patch names -> different source keys")
        legacy_tree = psi.git_worktree_tree(legacy_source)
        packaged_tree = psi.git_worktree_tree(packaged_source)
        self.assertEqual(legacy_tree, packaged_tree)


if __name__ == "__main__":
    unittest.main()
