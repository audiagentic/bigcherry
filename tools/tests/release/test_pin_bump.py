"""HI153: bigcherry pin-bump orchestrator -- unit tests for the composable
phase functions. Full end-to-end `run()` testing against a real git
fixture is tracked as follow-up (see HI153's plan item) -- these tests
cover the state machine's pure logic: state persistence, the structured
failure envelope, the narrow overlay self-heal decision, bad-rebase-status
stops, and the coverage gate delegation.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.release import pin_bump  # noqa: E402
from bigcherry.patch import disposition as patch_disposition  # noqa: E402


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _init_repo(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "file.txt").write_text("hello\n", encoding="utf-8")
    _git(root, "add", "file.txt")
    _git(root, "commit", "-m", "initial")


class PinBumpStateTests(unittest.TestCase):
    def test_round_trips_through_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            state = pin_bump.PinBumpState(
                schema_version=1, run_id="run-1", from_ref="b10502", from_sha="a" * 40,
                to_ref="b10680", to_sha="b" * 40, transition_commit="c" * 40,
                tree_name="local", tree_path="/some/path",
                completed_phases=["preflight", "declare"], next_phase="pull",
            )
            state.save(state_dir)
            loaded = pin_bump.PinBumpState.load(state_dir)
            self.assertEqual(loaded, state)


class FailureEnvelopeTests(unittest.TestCase):
    def test_envelope_has_the_documented_shape(self):
        exc = pin_bump.PinBumpStop(
            "coverage", "PATCH_QUARANTINED", "patch_b became invalid",
            evidence={"patch_id": "patch_b", "status": "QUARANTINED"},
            recommended_actions=["reconcile", "rerun with --resume"],
        )
        envelope = pin_bump.failure_envelope(
            "run-1", {"from_ref": "b10502", "to_ref": "b10680"}, "c" * 40,
            {"name": "local", "path": "/x"}, exc,
        )
        self.assertEqual(envelope["schema_version"], 1)
        self.assertEqual(envelope["operation"], "pin-bump")
        self.assertEqual(envelope["status"], "STOPPED")
        self.assertEqual(envelope["phase"], "coverage")
        self.assertEqual(envelope["failure"]["code"], "PATCH_QUARANTINED")
        self.assertTrue(envelope["failure"]["human_required"])
        self.assertEqual(envelope["failure"]["evidence"]["patch_id"], "patch_b")
        self.assertEqual(envelope["failure"]["recommended_actions"], ["reconcile", "rerun with --resume"])


class OverlaySelfHealTests(unittest.TestCase):
    def test_not_safe_when_other_checks_also_failed(self):
        report = {"checks": [
            {"id": "overlay.vendor_sync", "ok": False, "actual": ["a.cpp"]},
            {"id": "mmq.types", "ok": False},
        ]}
        safe, drifted = pin_bump.check_overlay_self_heal(report)
        self.assertFalse(safe)
        self.assertEqual(drifted, [])

    def test_safe_when_overlay_vendor_sync_is_the_only_failure(self):
        report = {"checks": [
            {"id": "overlay.vendor_sync", "ok": False, "actual": ["a.cpp", "b.cu"]},
            {"id": "mmq.types", "ok": True},
        ]}
        safe, drifted = pin_bump.check_overlay_self_heal(report)
        self.assertTrue(safe)
        self.assertEqual(drifted, ["a.cpp", "b.cu"])


class SyncCampaignMirrorBestEffortTests(unittest.TestCase):
    """Found live TWICE (b10680->b10687 and b10687->b10692): the separate
    campaign-build mirror repo doesn't learn about a new tag just because
    vendor/llama.cpp did, breaking the very next `bigcherry build`. This
    helper must be best-effort -- never raise -- since it's a build
    convenience, not a bump-correctness requirement."""

    def test_never_raises_when_no_mirror_exists(self):
        from unittest import mock
        from bigcherry.core.context import ProjectContext

        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            fake_context = ProjectContext(
                project_root=Path(project), config_path=Path(project) / "config" / "recipes.toml",
                artifacts_root=Path(project) / "artifacts", work_root=Path(work),
                upstream_repo=Path(work) / "upstream" / "llama.cpp.git",  # never created
                overlay_root=Path(project) / "src", patches_root=Path(project) / "patches",
            )
            with mock.patch.object(ProjectContext, "resolve", return_value=fake_context):
                pin_bump._sync_campaign_mirror_best_effort(target_ref="b99999", revision="a" * 40)
                # must not raise -- that is the entire test

    def test_never_raises_on_a_broken_mirror(self):
        from unittest import mock
        from bigcherry.core.context import ProjectContext

        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            mirror = Path(work) / "upstream" / "llama.cpp.git"
            mirror.mkdir(parents=True)
            (mirror / "HEAD").write_text("not a real git dir\n", encoding="utf-8")
            fake_context = ProjectContext(
                project_root=Path(project), config_path=Path(project) / "config" / "recipes.toml",
                artifacts_root=Path(project) / "artifacts", work_root=Path(work),
                upstream_repo=mirror, overlay_root=Path(project) / "src",
                patches_root=Path(project) / "patches",
            )
            with mock.patch.object(ProjectContext, "resolve", return_value=fake_context):
                pin_bump._sync_campaign_mirror_best_effort(target_ref="b99999", revision="a" * 40)


class WriteReleaseDocBestEffortTests(unittest.TestCase):
    """The release doc is a documentation convenience (real patch-doc
    template), not a bump-correctness requirement -- must never raise even
    when the recipe/vendor tree can't be resolved."""

    def test_never_raises_on_a_bogus_recipe(self):
        pin_bump._write_release_doc_best_effort(
            repo_root=Path("H:/development/projects/bigcherry"),
            vendor_root=Path("does-not-exist"),
            recipe_name="not-a-real-recipe-name",
            target_ref="b99999",
        )  # must not raise -- that is the entire test


class AcquireMaintenanceLockTests(unittest.TestCase):
    """Found live on pin-bump's first real invocation: acquire_maintenance_lock()
    used to call .acquire() itself AND get used as `with acquire_maintenance_lock(...)`,
    whose __enter__ also calls .acquire() -- a double-acquire in one process
    that tripped its own "already held" check before any real work happened."""

    def test_returned_lock_is_not_pre_acquired(self):
        from bigcherry.core.context import ProjectContext
        from unittest import mock

        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            fake_context = ProjectContext(
                project_root=Path(project), config_path=Path(project) / "config" / "recipes.toml",
                artifacts_root=Path(project) / "artifacts", work_root=Path(work),
                upstream_repo=Path(work) / "upstream", overlay_root=Path(project) / "src",
                patches_root=Path(project) / "patches",
            )
            with mock.patch.object(ProjectContext, "resolve", return_value=fake_context):
                lock = pin_bump.acquire_maintenance_lock(Path(project))
                self.assertFalse(lock.path.is_dir())
                with lock:  # must not raise -- __enter__ does the one real acquire
                    self.assertTrue(lock.path.is_dir())
                self.assertFalse(lock.path.is_dir())


class RequireCleanControllerCheckoutTests(unittest.TestCase):
    def test_passes_on_a_clean_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _init_repo(root)
            pin_bump.require_clean_controller_checkout(root)  # must not raise

    def test_stops_on_a_dirty_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _init_repo(root)
            (root / "file.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                pin_bump.require_clean_controller_checkout(root)
            self.assertEqual(ctx.exception.code, "CONTROLLER_DIRTY")
            self.assertEqual(ctx.exception.phase, "preflight")


class StopOnBadRebaseStatusTests(unittest.TestCase):
    def test_failed_needs_reconciliation_maps_to_the_right_code(self):
        with self.assertRaises(pin_bump.PinBumpStop) as ctx:
            pin_bump.stop_on_bad_rebase_status(
                phase="coverage", report={}, patch_id="0300_x",
                entry={"status": "FAILED", "requires": ()},
            )
        self.assertEqual(ctx.exception.code, "PATCH_FAILED_NEEDS_RECONCILIATION")

    def test_quarantined_maps_to_the_right_code(self):
        with self.assertRaises(pin_bump.PinBumpStop) as ctx:
            pin_bump.stop_on_bad_rebase_status(
                phase="coverage", report={}, patch_id="0400_y",
                entry={"status": "QUARANTINED", "requires": ("0300_x",)},
            )
        self.assertEqual(ctx.exception.code, "PATCH_QUARANTINED")
        self.assertEqual(ctx.exception.evidence["requires"], ("0300_x",))

    def test_blocked_by_dependency_maps_to_the_right_code(self):
        with self.assertRaises(pin_bump.PinBumpStop) as ctx:
            pin_bump.stop_on_bad_rebase_status(
                phase="coverage", report={}, patch_id="0400_y",
                entry={"status": "BLOCKED_BY_DEPENDENCY", "requires": ()},
            )
        self.assertEqual(ctx.exception.code, "PATCH_BLOCKED_BY_DEPENDENCY")


class CoverageGateDelegationTests(unittest.TestCase):
    def test_stops_with_coverage_incomplete_when_uncovered(self):
        with tempfile.TemporaryDirectory() as directory:
            dispositions_dir = Path(directory)
            with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                pin_bump.enforce_all_patches_clean_or_dispositioned(
                    all_report={"patches": [
                        {"patch_id": "1206_x", "status": "FAILED", "implementation_digest": "d1"},
                    ]},
                    recipe_report={"patches": []},
                    catalog_states={"1206_x": "untested"},
                    dispositions_dir=dispositions_dir,
                    target_revision="rev-a",
                )
            self.assertEqual(ctx.exception.code, "COVERAGE_INCOMPLETE")
            self.assertIn("1206_x", ctx.exception.evidence["uncovered_patch_ids"])

    def test_passes_through_when_a_matching_disposition_covers_it(self):
        with tempfile.TemporaryDirectory() as directory:
            dispositions_dir = Path(directory)
            patch_disposition.save_disposition(dispositions_dir, patch_disposition.Disposition(
                patch_id="1206_x", target_revision="rev-a", patch_digest="d1",
                disposition="known_broken", failure_status="FAILED_NEEDS_RECONCILIATION",
                reason="upstream removed anchor", owner="rd", tracking_item="RD13",
            ))
            result = pin_bump.enforce_all_patches_clean_or_dispositioned(
                all_report={"patches": [
                    {"patch_id": "1206_x", "status": "FAILED", "implementation_digest": "d1"},
                ]},
                recipe_report={"patches": []},
                catalog_states={"1206_x": "untested"},
                dispositions_dir=dispositions_dir,
                target_revision="rev-a",
            )
            self.assertTrue(result["complete"])


if __name__ == "__main__":
    unittest.main()
