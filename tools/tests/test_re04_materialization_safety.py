"""RE04 (RV48 audit fix): source materialisation content-safety and the
dirty-BigCherry-tree check, negative/falsification tests.

Real git fixtures throughout (same convention as test_campaign_build.py's
MaterializeSourceTests) -- these are exactly the cases the audit found
missing: an in-place edit under an unchanged canonical patch ID or overlay
file, a tampered cached worktree, and a dirty-tree bypass via cache reuse.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
            # GPT-auto-agent review follow-up (2026-08-17): resolve_patch_set()
            # now re-derives its cross-check directory from the supplied
            # catalog itself, rather than defaulting to paths.PATCHES (the
            # real project's patches/) -- no monkeypatch needed any more.
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

    def test_two_sources_with_identical_bytes_but_different_logical_composition_do_not_alias(self):
        # GPT-auto-agent review (RE03/RE05 follow-up, 2026-08-17): the
        # exact bug found in the standard campaign profile --
        # bigcherry-native resolves patch-sets=[framework] and bigcherry
        # resolves patch-sets=[framework, validated-enhancements], and
        # while validated-enhancements is empty the two sources produce
        # byte-identical trees but genuinely different logical patch_set_id/
        # classification. Before this fix, resolve_materialization_identity()
        # omitted patch_set_id/classification, so both requests shared one
        # materialisation directory and the cache-hit path returned the
        # FIRST request's persisted logical provenance verbatim for the
        # SECOND -- bigcherry-native runs first in campaign.standard, so
        # bigcherry's own materialisation would have silently reported
        # bigcherry-native's patch_set_id/classification.
        from bigcherry import campaign_resolution
        from bigcherry import config as campaign_config
        from bigcherry import patchset as bc_patchset
        from bigcherry.campaign_source import source_plan_for

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            _write_marker_patch(context.patches_root, marker_text="marker-v1")

            cfg = campaign_config.Config(
                pinned=revision,
                patch_sets={
                    "framework": campaign_config.PatchSet(
                        name="framework", patches=("0001_marker",), required_state="validated"),
                    "validated-enhancements": campaign_config.PatchSet(
                        name="validated-enhancements", patches=(), required_state="validated"),
                },
                sources={
                    "bigcherry-native": campaign_config.Source(
                        name="bigcherry-native", ref=revision, overlay=False,
                        patch_sets=("framework",)),
                    "bigcherry": campaign_config.Source(
                        name="bigcherry", ref=revision, overlay=False,
                        patch_sets=("framework", "validated-enhancements")),
                },
                builds={}, platforms={}, experiments={}, campaigns={},
                path=root / "recipes.toml",
            )
            catalog = bc_patchset.catalog(directory=context.patches_root)

            native_plan = source_plan_for(
                cfg, "bigcherry-native", catalog=catalog, catalog_directory=context.patches_root)
            full_plan = source_plan_for(
                cfg, "bigcherry", catalog=catalog, catalog_directory=context.patches_root)
            native_lane = campaign_resolution.resolve_lane(
                "bigcherry-native", cfg, catalog, catalog_directory=context.patches_root)
            full_lane = campaign_resolution.resolve_lane(
                "bigcherry", cfg, catalog, catalog_directory=context.patches_root)

            # Same byte-producing patch selection...
            self.assertEqual(native_plan.patch_ids, full_plan.patch_ids)
            # ...but genuinely different logical composition identity.
            self.assertNotEqual(native_lane.patch_set.patch_set_id, full_lane.patch_set.patch_set_id)

            native_record = materialize_source(context, native_plan, allow_dirty_bigcherry=True)
            full_record = materialize_source(context, full_plan, allow_dirty_bigcherry=True)

            # The bug: these used to be equal (aliased). Now each request
            # gets its own materialisation and its own correct provenance.
            self.assertNotEqual(
                native_record["materialization_plan_id"], full_record["materialization_plan_id"])
            self.assertEqual(native_record["plan"]["patch_set_id"], native_lane.patch_set.patch_set_id)
            self.assertEqual(full_record["plan"]["patch_set_id"], full_lane.patch_set.patch_set_id)
            self.assertNotEqual(
                native_record["plan"]["patch_set_id"], full_record["plan"]["patch_set_id"])


class ThreeSourceIdentitiesPersistedTests(unittest.TestCase):
    """RE05 (RV48 audit): source provenance must retain all three source
    identities explicitly -- source_plan_id ("what request did we ask
    for"), materialization_plan_id/source_tree_oid ("what did that
    request actually produce"), source_slice_id ("BigCherry's durable
    content-domain identity") -- not leave any of them implicit in facts
    a reader has to already know (e.g. the destination directory's own
    name)."""

    def test_all_three_identities_are_explicit_in_the_persisted_record(self):
        from bigcherry import campaign_source

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            plan = SourcePlan(revision, False, (), None)

            record = materialize_source(context, plan, allow_dirty_bigcherry=True)

            self.assertEqual(record["source_plan_id"], campaign_source.source_plan_id(plan))
            identity = campaign_source.resolve_materialization_identity(context, plan)
            self.assertEqual(record["materialization_plan_id"],
                             campaign_source.materialization_plan_id(identity))
            self.assertTrue(record["source_tree_oid"])
            self.assertTrue(record["source_slice_id"])
            # Four genuinely distinct values for a non-trivial request --
            # collapsing any pair together would mean one identity secretly
            # stood in for another rather than each answering its own
            # question.
            values = {record["source_plan_id"], record["materialization_plan_id"],
                     record["source_tree_oid"], record["source_slice_id"]}
            self.assertEqual(len(values), 4)


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

    def test_forging_the_persisted_source_slice_id_fails_closed(self):
        # GPT-auto-agent review (RE03/RE04/RE05 comprehensive follow-up,
        # 2026-08-17): a real gap the worktree-tamper test above does not
        # cover -- the sibling .metadata.json file lives BESIDE the
        # worktree, not inside it, so git_tree_oid() never sees an edit to
        # it. A directly forged source_slice_id in that file used to be
        # returned to the caller verbatim on cache-hit and trusted for the
        # rest of campaign execution, even though the worktree bytes
        # themselves were never touched.
        from bigcherry import campaign_source
        from bigcherry.campaign_build import _source_metadata_path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            plan = SourcePlan(revision, False, (), None)

            materialize_source(context, plan, allow_dirty_bigcherry=True)

            plan_id = campaign_source.materialization_plan_id(
                campaign_source.resolve_materialization_identity(context, plan))
            destination = context.work_root / "sources" / plan_id
            metadata_path = _source_metadata_path(destination)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_slice_id"] = "forged-slice-id-not-actually-derived"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaises(CampaignBuildError):
                materialize_source(context, plan, allow_dirty_bigcherry=True)

    def test_forging_the_persisted_source_plan_id_fails_closed(self):
        from bigcherry import campaign_source
        from bigcherry.campaign_build import _source_metadata_path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            plan = SourcePlan(revision, False, (), None)

            materialize_source(context, plan, allow_dirty_bigcherry=True)

            plan_id = campaign_source.materialization_plan_id(
                campaign_source.resolve_materialization_identity(context, plan))
            destination = context.work_root / "sources" / plan_id
            metadata_path = _source_metadata_path(destination)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_plan_id"] = "forged-plan-id"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaises(CampaignBuildError):
                materialize_source(context, plan, allow_dirty_bigcherry=True)


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
