"""RE04 (RV48 audit fix): source materialisation content-safety and the
dirty-BigCherry-tree check, negative/falsification tests.

Real git fixtures throughout (same convention as test_campaign_build.py's
MaterializeSourceTests) -- these are exactly the cases the audit found
missing: an in-place edit under an unchanged canonical patch ID or overlay
file, a tampered cached worktree, and a dirty-tree bypass via cache reuse.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.campaign_build import CampaignBuildError, materialize_source  # noqa: E402
from bigcherry.context import ProjectContext  # noqa: E402
from bigcherry.workspace import SourcePlan, WorkspaceError  # noqa: E402


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_upstream(directory: Path) -> tuple[Path, str]:
    repo = directory / "upstream"
    _git(directory, "init", str(repo))
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "source.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "initial")
    revision = _git(repo, "rev-parse", "HEAD")
    return repo, revision


def _init_project(directory: Path) -> None:
    # require_clean_bigcherry runs `git status --porcelain` against
    # context.project_root -- it must be a real, clean git repo for the
    # allow_dirty_bigcherry=False path to be exercised meaningfully.
    _git(directory, "init", str(directory))
    _git(directory, "config", "user.email", "test@example.invalid")
    (directory / ".gitkeep").write_text("", encoding="utf-8")
    _git(directory, "add", ".gitkeep")
    _git(directory, "config", "user.name", "Test")
    _git(directory, "commit", "-m", "initial")


def _context(root: Path, upstream: Path) -> ProjectContext:
    # project_root is its own sibling directory, never containing work_root
    # -- materialize_source's own writes into work_root must not dirty a
    # project_root git repo used for the dirty-tree check tests below.
    project = root / "project"
    project.mkdir(parents=True, exist_ok=True)
    return ProjectContext(
        project_root=project, config_path=root / "recipes.toml",
        artifacts_root=root / "artifacts", work_root=root / "work",
        upstream_repo=upstream, overlay_root=root / "src",
        patches_root=root / "patches",
    )


def _write_marker_patch(patches_root: Path, *, marker_text: str) -> None:
    """One real, applicable patch module at a FIXED canonical id
    (0001_marker) whose EFFECT (and therefore content_hash) varies with
    ``marker_text`` -- the exact "edit a patch under its unchanged ID"
    scenario RV48 flagged.
    """
    patches_root.mkdir(parents=True, exist_ok=True)
    (patches_root / "0001_marker.py").write_text(
        "from bigcherry.patcher import Edit, FilePatch\n"
        "GROUP = 'core'\n"
        "STATE = 'validated'\n"
        "PATCH = FilePatch(\n"
        "    path='source.txt',\n"
        "    edits=(Edit(\n"
        "        id='marker',\n"
        "        anchor=r'one',\n"
        f"        text='{marker_text}',\n"
        "        mode='insert_after',\n"
        "    ),),\n"
        ")\n",
        encoding="utf-8",
    )


class PatchSetIdentityPersistedTests(unittest.TestCase):
    """RE03 (RV48 audit): patch_set_id/classification, computed by
    campaign_resolution.resolve_lane, must reach the materialised source's
    own persisted metadata -- verifiable without recomputing patch
    resolution -- not be silently discarded on the way to SourcePlan."""

    def test_source_plan_for_real_lane_persists_patch_set_id_into_metadata(self):
        import json as _json
        from bigcherry import campaign_resolution
        from bigcherry import config as campaign_config
        from bigcherry import patchset as bc_patchset
        from bigcherry.campaign_build import _source_metadata_path
        from bigcherry.campaign_source import (materialization_plan_id,
                                               resolve_materialization_identity,
                                               source_plan_for)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            _write_marker_patch(context.patches_root, marker_text="marker-v1")

            cfg = campaign_config.Config(
                pinned=revision,
                patch_sets={"core": campaign_config.PatchSet(
                    name="core", patches=("0001_marker",), required_state="validated")},
                sources={"test-source": campaign_config.Source(
                    name="test-source", ref=revision, overlay=False,
                    patch_sets=("core",))},
                builds={}, platforms={}, experiments={}, campaigns={},
                path=root / "recipes.toml",
            )
            catalog = bc_patchset.catalog(directory=context.patches_root)
            # resolve_lane() cross-checks the supplied catalog against
            # patchset.catalog() with NO directory override, which defaults
            # to paths.PATCHES (the real project's patches/), not
            # context.patches_root -- see RE24's own notes on this same
            # architectural gap. Patch the shared default for this isolated
            # fixture's catalog to be visible to that check.
            with mock.patch("bigcherry.paths.PATCHES", context.patches_root):
                plan = source_plan_for(cfg, "test-source", catalog=catalog)
                lane = campaign_resolution.resolve_lane("test-source", cfg, catalog)

            record = materialize_source(context, plan, allow_dirty_bigcherry=True)

            self.assertEqual(record["plan"]["patch_set_id"], lane.patch_set.patch_set_id)
            self.assertTrue(record["plan"]["patch_set_id"])
            self.assertEqual(record["plan"]["classification"], lane.patch_set.classification)

            # Verifiable WITHOUT recomputation: reading the persisted
            # .metadata.json file back off disk carries the same identity.
            identity = resolve_materialization_identity(context, plan)
            destination = context.work_root / "sources" / materialization_plan_id(identity)
            persisted = _json.loads(
                _source_metadata_path(destination).read_text(encoding="utf-8"))
            self.assertEqual(persisted["plan"]["patch_set_id"], lane.patch_set.patch_set_id)


class PatchContentEditNotReusedTests(unittest.TestCase):
    def test_editing_a_patch_module_under_its_unchanged_id_does_not_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            _write_marker_patch(context.patches_root, marker_text="marker-v1")
            plan = SourcePlan(revision, False, ("0001_marker",), None)

            first = materialize_source(context, plan, allow_dirty_bigcherry=True)

            # Edit the SAME patch module (same canonical id, same file path)
            # to a different effect -- content_hash changes, patch_ids does
            # not.
            _write_marker_patch(context.patches_root, marker_text="marker-v2")
            second = materialize_source(context, plan, allow_dirty_bigcherry=True)

            self.assertNotEqual(
                first["source_slice_id"], second["source_slice_id"],
                "editing a patch module's content under its unchanged "
                "canonical id must not reuse the old materialisation")


class OverlayContentEditNotReusedTests(unittest.TestCase):
    def test_editing_an_overlay_files_content_does_not_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            context.overlay_root.mkdir(parents=True, exist_ok=True)
            (context.overlay_root / "extra.txt").write_text("v1\n", encoding="utf-8")
            plan = SourcePlan(revision, True, (), None)

            first = materialize_source(context, plan, allow_dirty_bigcherry=True)

            (context.overlay_root / "extra.txt").write_text("v2\n", encoding="utf-8")
            second = materialize_source(context, plan, allow_dirty_bigcherry=True)

            self.assertNotEqual(
                first["source_slice_id"], second["source_slice_id"],
                "editing an overlay file's content must not reuse the old "
                "materialisation")


class CachedWorktreeTamperFailsClosedTests(unittest.TestCase):
    def test_modifying_the_cached_worktree_after_materialisation_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            plan = SourcePlan(revision, False, (), None)

            metadata = materialize_source(context, plan, allow_dirty_bigcherry=True)

            from bigcherry import campaign_source
            plan_id = campaign_source.materialization_plan_id(
                campaign_source.resolve_materialization_identity(context, plan))
            destination = context.work_root / "sources" / plan_id
            # Tamper with the cached worktree's bytes directly -- nothing
            # about the STORED metadata changes, only what's actually on
            # disk. A reuse that trusts the metadata alone would silently
            # compile these tampered bytes.
            (destination / "source.txt").write_text("tampered\n", encoding="utf-8")

            with self.assertRaises(CampaignBuildError):
                materialize_source(context, plan, allow_dirty_bigcherry=True)
            self.assertTrue(metadata["source_slice_id"])  # sanity: first call succeeded


class DirtyBigCherryRejectedOnBothPathsTests(unittest.TestCase):
    def test_dirty_bigcherry_rejects_fresh_materialisation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            _init_project(context.project_root)
            (context.project_root / "dirty.txt").write_text("uncommitted", encoding="utf-8")
            plan = SourcePlan(revision, False, (), None)

            with self.assertRaises(WorkspaceError):
                materialize_source(context, plan, allow_dirty_bigcherry=False)

    def test_dirty_bigcherry_rejects_cache_reuse_too(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            _init_project(context.project_root)
            plan = SourcePlan(revision, False, (), None)

            # First materialisation with a clean project succeeds and
            # populates the cache.
            materialize_source(context, plan, allow_dirty_bigcherry=False)

            # Now dirty the project and try again -- RV48's finding: a cache
            # hit used to return before the dirty check ever ran.
            (context.project_root / "dirty.txt").write_text("uncommitted", encoding="utf-8")
            with self.assertRaises(WorkspaceError):
                materialize_source(context, plan, allow_dirty_bigcherry=False)

    def test_explicit_override_allows_dirty_bigcherry_on_both_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            _init_project(context.project_root)
            plan = SourcePlan(revision, False, (), None)

            materialize_source(context, plan, allow_dirty_bigcherry=False)
            (context.project_root / "dirty.txt").write_text("uncommitted", encoding="utf-8")
            # Explicit override -- must succeed on the cache-hit path too.
            result = materialize_source(context, plan, allow_dirty_bigcherry=True)
            self.assertTrue(result["source_slice_id"])


if __name__ == "__main__":
    unittest.main()
