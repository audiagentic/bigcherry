"""Tests for tools/bigcherry/patch_source_isolation.py (patch-system PA02).

Covers the RS05 list (runbook section 58) updated for the RV80/B2 v2
identity, plus the 12 RV80 acceptance cases (runbook 14.3 / reviews/PA02/
RV80): topological (not global-sort) ordering, overlay/patch-change
identity, packaged STATE independence, resolved-SHA anchoring, exact
composition validation, digest-mismatch and path/import safety.
Real git fixtures throughout (same convention as
test_re04_materialization_safety.py).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import registry as patch_registry, patchset # noqa: E402
from bigcherry.patch import source as psi # noqa: E402

FLAT_PATCH_PY = """\
from bigcherry.patcher import Edit, FilePatch
GROUP = 'core'
STATE = 'untested'
PATCHES = [FilePatch(path='a.txt', edits=(Edit(id='e1', anchor='one', text='MARKER'),))]
"""

FLAT_PATCH_PY_VALIDATED = FLAT_PATCH_PY.replace(
    "STATE = 'untested'", "STATE = 'validated'"
)

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
        _git(self.upstream, "worktree", "prune")
        psi.PATCHES_ROOT = self._old_patches_root

    def _flat(self, patch_id: str = "0001_flat", state: str = "untested") -> None:
        text = FLAT_PATCH_PY if state == "untested" else FLAT_PATCH_PY_VALIDATED
        (self.patches_root / f"{patch_id}.py").write_text(text, encoding="utf-8")

    def _package(self, patch_id: str = "1202_dual", state: str = "untested") -> None:
        package = self.patches_root / "rd" / patch_id
        package.mkdir(parents=True)
        (package / "patch.toml").write_text(
            PKG_TOML.replace('"1202_dual"', f'"{patch_id}"')
            .replace('"untested"', f'"{state}"'),
            encoding="utf-8",
        )
        (package / "patch.py").write_text(PKG_PATCH_PY, encoding="utf-8")

    def _composition(self, *patch_ids: str) -> tuple[tuple[str, str], ...]:
        registry = psi._patch_registry().load_registry(self.patches_root)
        return tuple(
            (
                patch_id,
                hashlib.sha256(
                    (registry.root / registry.get(patch_id).implementation_path)
                    .read_bytes()
                ).hexdigest(),
            )
            for patch_id in patch_ids
        )

    def _materialize(self, *patch_ids: str, resolved: str | None = None) -> Path:
        return psi.materialize_composition(
            base_repo=self.upstream, worktree_root=self.worktrees,
            resolved_revision=resolved or self.base_rev,
            composition=self._composition(*patch_ids),
        )

    def test_legacy_materialization(self) -> None:
        self._flat()
        source = self._materialize("0001_flat")
        content = (source / "a.txt").read_text(encoding="utf-8")
        self.assertIn("MARKER", content)
        manifest = json.loads(psi._manifest_path(source).read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "bigcherry-patch-source-v2")
        self.assertEqual(manifest["composition"][0][0], "0001_flat")
        self.assertEqual(manifest["resolved_revision"], self.base_rev)
        self.assertEqual(manifest["materialization_plan_id"], manifest["source_key"])
        self.assertEqual(manifest["source_tree_oid"], manifest["patched_tree"])
        self.assertTrue(manifest["source_slice_id"])
        # Cache reuse: same identity -> same directory, no error.
        again = self._materialize("0001_flat")
        self.assertEqual(again, source)

    def test_package_materialization(self) -> None:
        self._package()
        source = self._materialize("1202_dual")
        content = (source / "a.txt").read_text(encoding="utf-8")
        self.assertIn("MARKER", content)
        manifest = json.loads(psi._manifest_path(source).read_text(encoding="utf-8"))
        self.assertEqual(manifest["composition"][0][0], "1202_dual")

    def test_tampered_tree_rejected(self) -> None:
        self._flat()
        source = self._materialize("0001_flat")
        (source / "a.txt").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(psi.PatchSourceIsolationError, "modified after materialization"):
            self._materialize("0001_flat")

    def test_tampered_tree_and_manifest_tree_rejected(self) -> None:
        self._flat()
        source = self._materialize("0001_flat")
        (source / "a.txt").write_text("tampered\n", encoding="utf-8")
        manifest_path = psi._manifest_path(source)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["patched_tree"] = psi.git_worktree_tree(source)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(psi.PatchSourceIsolationError):
            self._materialize("0001_flat")

    def test_tampered_tree_and_source_ids_rejected(self) -> None:
        self._flat()
        source = self._materialize("0001_flat")
        (source / "a.txt").write_text("tampered\n", encoding="utf-8")
        manifest_path = psi._manifest_path(source)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tree = psi.git_worktree_tree(source)
        manifest["source_tree_oid"] = tree
        manifest["source_slice_id"] = psi._source_slice_id(
            source_dir=source, upstream_revision=self.base_rev, tree_oid=tree
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(psi.PatchSourceIsolationError):
            self._materialize("0001_flat")

    def test_ignored_file_rejected_on_reuse(self) -> None:
        self._flat()
        source = self._materialize("0001_flat")
        (source / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
        (source / "ignored.txt").write_text("must not be ignored\n", encoding="utf-8")
        with self.assertRaisesRegex(psi.PatchSourceIsolationError, "ignored"):
            self._materialize("0001_flat")

    def test_manifest_schema_must_be_exact_current_version(self) -> None:
        self._flat()
        for value in (None, 0, 2):
            source = self._materialize("0001_flat")
            manifest_path = psi._manifest_path(source)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if value is None:
                del manifest["manifest_schema_version"]
            else:
                manifest["manifest_schema_version"] = value
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(psi.PatchSourceIsolationError, "schema"):
                self._materialize("0001_flat")
            manifest["manifest_schema_version"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(self._materialize("0001_flat"), source)

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
        manifest["resolved_revision"] = "0" * 40
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
        second = self._materialize("0001_flat", resolved=new_rev)
        self.assertNotEqual(first, second)
        self.assertIn("three", (second / "a.txt").read_text(encoding="utf-8"))
        self.assertNotIn("three", (first / "a.txt").read_text(encoding="utf-8"))

    def test_semantics_version_change_changes_plan_identity(self) -> None:
        """PA07 (L1.2): the same patch bytes + same composition must give
        the same source_key only when patch_application_semantics_version
        also matches -- a bump there must change identity even though no
        patch.py digest changed."""
        self._flat()
        selection = patchset.resolve_exact(("0001_flat",), directory=self.patches_root)
        composition = tuple(
            (module.patch_id, module.content_hash) for module in selection.modules
        )
        same_semantics_a = psi._make_source_identity_v2(
            resolved_revision=self.base_rev, composition=composition, overlay_root=None,
        )
        same_semantics_b = psi._make_source_identity_v2(
            resolved_revision=self.base_rev, composition=composition, overlay_root=None,
        )
        self.assertEqual(same_semantics_a["source_key"], same_semantics_b["source_key"])

        import bigcherry.patch.apply as apply_module
        with mock.patch.object(apply_module, "PATCH_APPLICATION_SEMANTICS_VERSION", 2), \
             mock.patch.object(psi, "PATCH_APPLICATION_SEMANTICS_VERSION", 2):
            bumped = psi._make_source_identity_v2(
                resolved_revision=self.base_rev, composition=composition, overlay_root=None,
            )
        self.assertNotEqual(same_semantics_a["source_key"], bumped["source_key"])

    def test_legacy_package_migration_tree_equivalence(self) -> None:
        # Same edit expressed as a flat module and as a package: the
        # materialized CONTENT trees must be byte-identical (migration
        # criterion -- representation changes, tree does not).
        self._flat("0002_dual")
        self._package("1202_dual")
        legacy_source = self._materialize("0002_dual")
        packaged_source = self._materialize("1202_dual")
        self.assertNotEqual(legacy_source, packaged_source,
                            "different compositions -> different source keys")
        legacy_tree = psi.git_worktree_tree(legacy_source)
        packaged_tree = psi.git_worktree_tree(packaged_source)
        self.assertEqual(legacy_tree, packaged_tree)
        legacy_manifest = json.loads(psi._manifest_path(legacy_source).read_text(encoding="utf-8"))
        packaged_manifest = json.loads(psi._manifest_path(packaged_source).read_text(encoding="utf-8"))
        self.assertNotEqual(legacy_manifest["materialization_plan_id"], packaged_manifest["materialization_plan_id"])
        self.assertEqual(legacy_manifest["source_slice_id"], packaged_manifest["source_slice_id"])


class Rv80AcceptanceTests(MaterializationTests):
    """The 12 RV80 acceptance cases (runbook 14.3 / RV80 review)."""

    # 1. Topological order is NOT a global (order, patch_id) sort: a
    #    numerically-earlier patch that REQUIRES a numerically-later one
    #    must still apply AFTER its dependency.
    def test_rv80_topological_order_not_global_sort(self) -> None:
        def _mod(patch_id: str, order: int, requires=()) -> patchset.PatchModule:
            return patchset.PatchModule(
                patch_id=patch_id, path=Path("/nonexistent") / f"{patch_id}.py",
                order=order, group="core", state="validated", upstream=None,
                content_hash="ab" * 32, requires=tuple(requires),
            )

        parent = _mod("0200_parent", 200)
        child = _mod("0100_child", 100, requires=("0200_parent",))
        modules = {m.patch_id: m for m in (parent, child)}
        ordered = patchset.topological_order(("0100_child", "0200_parent"), modules=modules)
        self.assertEqual(ordered, ("0200_parent", "0100_child"))
        # And a global sort would have been wrong:
        self.assertEqual(
            sorted(ordered, key=lambda pid: (modules[pid].order, pid)),
            ["0100_child", "0200_parent"],
        )

    # 2. Overlay file change -> new identity (overlay_digest in payload).
    def test_rv80_overlay_change_changes_identity(self) -> None:
        self._flat()
        overlay = self.base / "overlay"
        (overlay / "src" / "extra").mkdir(parents=True)
        (overlay / "src" / "extra" / "new.cpp").write_text("v1\n", encoding="utf-8")
        first = psi.materialize_composition(
            base_repo=self.upstream, worktree_root=self.worktrees,
            resolved_revision=self.base_rev, composition=self._composition("0001_flat"),
            overlay_root=overlay,
        )
        self.assertEqual((first / "src" / "extra" / "new.cpp").read_text(encoding="utf-8"), "v1\n")
        (overlay / "src" / "extra" / "new.cpp").write_text("v2\n", encoding="utf-8")
        second = psi.materialize_composition(
            base_repo=self.upstream, worktree_root=self.worktrees,
            resolved_revision=self.base_rev, composition=self._composition("0001_flat"),
            overlay_root=overlay,
        )
        self.assertNotEqual(first, second)
        self.assertEqual((second / "src" / "extra" / "new.cpp").read_text(encoding="utf-8"), "v2\n")

    # 3. Patch implementation change -> new identity (composition entry
    #    digest in payload).
    def test_rv80_patch_change_changes_identity(self) -> None:
        self._flat()
        first = self._materialize("0001_flat")
        (self.patches_root / "0001_flat.py").write_text(
            FLAT_PATCH_PY.replace("MARKER", "MARKER2"), encoding="utf-8"
        )
        second = self._materialize("0001_flat")
        self.assertNotEqual(first, second)
        self.assertIn("MARKER2", (second / "a.txt").read_text(encoding="utf-8"))

    # 4. Packaged lifecycle STATE flip (patch.toml) leaves patch.py bytes
    #    unchanged -> SAME identity (state independence; the B6 core).
    def test_rv80_state_independence_packaged(self) -> None:
        self._package(state="untested")
        first = self._materialize("1202_dual")
        # Flip ONLY the lifecycle state in patch.toml; patch.py untouched.
        (self.patches_root / "rd" / "1202_dual" / "patch.toml").write_text(
            PKG_TOML.replace('"1202_dual"', '"1202_dual"').replace('"untested"', '"validated"'),
            encoding="utf-8",
        )
        again = self._materialize("1202_dual")
        self.assertEqual(again, first, "state flip must not change the source identity")
        manifest = json.loads(psi._manifest_path(first).read_text(encoding="utf-8"))
        self.assertNotIn("framework_baseline_digest", manifest)

    # 5. A MOVED REF is a new identity: the resolved SHA (not the ref name)
    #    is in the hashed payload; annotated tags peel to commits.
    def test_rv80_ref_movement_changes_identity(self) -> None:
        _git(self.upstream, "tag", "-a", "pin", "-m", "pin v1")
        sha_v1 = psi.resolve_base_revision("pin", repo=self.upstream)
        self.assertEqual(sha_v1, self.base_rev)
        (self.upstream / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        _git(self.upstream, "add", "a.txt")
        _git(self.upstream, "commit", "-m", "move")
        _git(self.upstream, "tag", "-d", "pin")
        _git(self.upstream, "tag", "-a", "pin", "-m", "pin v2")
        sha_v2 = psi.resolve_base_revision("pin", repo=self.upstream)
        self.assertNotEqual(sha_v1, sha_v2)
        self._flat()
        first = psi.materialize_composition(
            base_repo=self.upstream, worktree_root=self.worktrees,
            resolved_revision=sha_v1, composition=self._composition("0001_flat"),
            requested_revision="pin",
        )
        second = psi.materialize_composition(
            base_repo=self.upstream, worktree_root=self.worktrees,
            resolved_revision=sha_v2, composition=self._composition("0001_flat"),
            requested_revision="pin",
        )
        self.assertNotEqual(first, second)
        # The requested ref is manifest-info only, NOT the hashed payload:
        # identical resolved SHA + composition + overlay -> identical key
        # regardless of how the ref was spelled.
        third = psi.materialize_composition(
            base_repo=self.upstream, worktree_root=self.worktrees,
            resolved_revision=sha_v1, composition=self._composition("0001_flat"),
            requested_revision="HEAD",
        )
        self.assertEqual(third, first)

    # 6. Stale composition entry (patch changed after identity computation)
    #    is rejected before application.
    def test_rv80_stale_composition_digest_rejected(self) -> None:
        self._flat()
        composition = self._composition("0001_flat")
        source = psi.materialize_composition(
            base_repo=self.upstream, worktree_root=self.worktrees,
            resolved_revision=self.base_rev, composition=composition,
        )
        (self.patches_root / "0001_flat.py").write_text(
            FLAT_PATCH_PY.replace("MARKER", "MARKER3"), encoding="utf-8"
        )
        # Reuse the SAME directory (same key) with the stale digest:
        # _apply_composition re-hashes and must fail closed.
        with self.assertRaisesRegex(psi.PatchSourceIsolationError, "changed between identity"):
            psi._apply_composition(source, composition, overlay_root=None, root=self.patches_root)

    # 7. Exact-composition violations fail closed (resolve_exact is the
    #    authoritative validator): unknown ID, missing explicit requires,
    #    rejected member. (The focal-already-in-base guard is #7b below.)
    def test_rv80_exact_composition_failures(self) -> None:
        self._flat()
        self._flat("0200_parent")
        with self.assertRaisesRegex(ValueError, "unknown patch module"):
            patchset.resolve_exact(("9999_nope",), directory=self.patches_root)
        # Missing explicit requires:
        child = FLAT_PATCH_PY.replace("MARKER", "CHILD").replace(
            "STATE = 'untested'", "STATE = 'untested'\nREQUIRES = ['0200_parent']"
        )
        (self.patches_root / "0101_child.py").write_text(child, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "requires explicitly selected"):
            patchset.resolve_exact(("0101_child",), directory=self.patches_root)
        # Rejected member:
        (self.patches_root / "0103_rejected.py").write_text(
            FLAT_PATCH_PY.replace("MARKER", "REJ")
            .replace("STATE = 'untested'", "STATE = 'rejected'"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "rejected patch requires"):
            patchset.resolve_exact(("0103_rejected",), directory=self.patches_root)

    # 7b. Focal-already-in-baseline is rejected by resolve_source_composition.
    def test_rv80_focal_guard(self) -> None:
        # The focal/duplicate guard in resolve_source_composition raises on a
        # focal that is already in the resolved base. Use a minimal in-repo
        # recipes fixture so the full path is exercised without the real
        # config.
        recipes = self.base / "recipes.toml"
        recipes.write_text(
            """
version = 2
pinned = "HEAD"
[patch-set.framework]
required-state = "untested"
patches = ["0001_flat"]
[source.bigcherry]
ref = "HEAD"
overlay = false
patch-sets = ["framework"]
""",
            encoding="utf-8",
        )
        self._flat()
        with self.assertRaisesRegex(psi.PatchSourceIsolationError, "already in source"):
            psi.resolve_source_composition(
                "bigcherry", focal="0001_flat", base_ref="HEAD",
                base_repo=self.upstream, recipes=recipes, patches_root=self.patches_root,
            )
        sha, composition = psi.resolve_source_composition(
            "bigcherry", base_ref="HEAD", base_repo=self.upstream,
            recipes=recipes, patches_root=self.patches_root,
        )
        self.assertEqual(sha, self.base_rev)
        self.assertEqual([pid for pid, _ in composition], ["0001_flat"])

    # 8. Loader digest re-check (B3): the registry loader re-hashes the
    #    executed bytes against the descriptor digest.
    def test_rv80_loader_digest_recheck(self) -> None:
        self._flat()
        registry = psi._patch_registry().load_registry(self.patches_root)
        descriptor = registry.get("0001_flat")
        # Tamper the descriptor digest -> byte-compile load must fail closed.
        tampered = patch_registry.PatchDescriptor(
            **{**descriptor.__dict__, "implementation_digest": "0" * 64}
        )
        with self.assertRaises(patch_registry.PatchRegistryError):
            psi._patch_registry().load_implementation(tampered, root=self.patches_root)

    # 9. Packaged patch.py with a relative import is rejected before exec
    #    (B5 import restriction) -> materialization fails closed.
    def test_rv80_packaged_relative_import_rejected(self) -> None:
        package = self.patches_root / "rd" / "1204_evil"
        package.mkdir(parents=True)
        (package / "patch.toml").write_text(
            PKG_TOML.replace('"1202_dual"', '"1204_evil"').replace("order = 1202", "order = 1204"), encoding="utf-8"
        )
        (package / "patch.py").write_text(
            "from . import helper\nPATCHES = ()\n", encoding="utf-8"
        )
        with self.assertRaises(patch_registry.PatchRegistryError):
            self._materialize("1204_evil")

    # 10. Symlink escape out of the patches root is rejected by discovery.
    def test_rv80_symlink_escape_rejected(self) -> None:
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "9999_evil.py").write_text(FLAT_PATCH_PY, encoding="utf-8")
        link = self.patches_root / "9999_evil.py"
        try:
            link.symlink_to(outside / "9999_evil.py")
        except OSError as exc:
            self.skipTest(
                "symlink escape test requires symlink creation privilege; "
                f"unavailable in this Windows environment: {exc}"
            )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "escapes"):
            psi._patch_registry().load_registry(self.patches_root)

    # 11. Variant identity: same composition, different variant digest ->
    #     different keys; same variant -> reuse.
    def test_rv80_variant_identity(self) -> None:
        self._flat()
        composition = self._composition("0001_flat")
        a = psi.materialize_source_variant(
            base_repo=self.upstream, worktree_root=self.worktrees,
            resolved_revision=self.base_rev, composition=composition,
            variant_name="v1", variant_digest="digest-a",
        )
        b = psi.materialize_source_variant(
            base_repo=self.upstream, worktree_root=self.worktrees,
            resolved_revision=self.base_rev, composition=composition,
            variant_name="v1", variant_digest="digest-b",
        )
        self.assertNotEqual(a, b)
        again = psi.materialize_source_variant(
            base_repo=self.upstream, worktree_root=self.worktrees,
            resolved_revision=self.base_rev, composition=composition,
            variant_name="v1", variant_digest="digest-a",
        )
        self.assertEqual(again, a)
        manifest = json.loads(psi._manifest_path(a).read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "bigcherry-patch-source-variant-v2")

    # 12. Stock = empty composition + no overlay: distinct key, reusable,
    #     pristine content.
    def test_rv80_stock_identity(self) -> None:
        first = psi.materialize_stock_source(
            base_repo=self.upstream, worktree_root=self.worktrees, base_revision="HEAD",
        )
        manifest = json.loads(psi._manifest_path(first).read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "bigcherry-patch-source-v2")
        self.assertEqual(manifest["composition"], [])
        self.assertEqual(manifest["requested_revision"], "HEAD")
        self.assertEqual(manifest["resolved_revision"], self.base_rev)
        self.assertEqual((first / "a.txt").read_text(encoding="utf-8"), "one\ntwo\n")
        again = psi.materialize_stock_source(
            base_repo=self.upstream, worktree_root=self.worktrees, base_revision="HEAD",
        )
        self.assertEqual(again, first)


if __name__ == "__main__":
    unittest.main()
