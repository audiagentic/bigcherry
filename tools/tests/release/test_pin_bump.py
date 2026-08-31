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


class SchemaTwoRoundTripTests(unittest.TestCase):
    def test_selector_fields_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            state = pin_bump.PinBumpState(
                schema_version=2, run_id="run-1", from_ref="b10502", from_sha="a" * 40,
                to_ref="b10680", to_sha="b" * 40, transition_commit="c" * 40,
                tree_name="local", tree_path="/some/path",
                completed_phases=["preflight", "declare"], next_phase="pull",
                selector_kind="recipe", selector_name="bigcherry",
                selector_patch_ids=("0100_x", "0200_y"),
                coverage_report_sha256="deadbeef",
            )
            state.save(state_dir)
            loaded = pin_bump.PinBumpState.load(state_dir)
            self.assertEqual(loaded, state)

    def test_schema_one_state_loads_with_empty_selector_not_a_crash(self):
        # A real schema-1 state.json on disk (written before this plan)
        # has no "selector" key at all -- load() must not KeyError.
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            state = pin_bump.PinBumpState(
                schema_version=1, run_id="run-1", from_ref="b10502", from_sha="a" * 40,
                to_ref="b10680", to_sha="b" * 40, transition_commit="c" * 40,
                tree_name="local", tree_path="/some/path",
                completed_phases=["preflight"], next_phase="declare",
            )
            state.save(state_dir)
            loaded = pin_bump.PinBumpState.load(state_dir)
            self.assertEqual(loaded.selector_kind, "")
            self.assertEqual(loaded.selector_patch_ids, ())


class ValidateResumeTests(unittest.TestCase):
    """compat.recipe removal plan (gpt-dev-agent reviewed, session
    ses_5307d9c58ec645cb): resume-time identity checks that did not exist
    before this plan -- a --resume with a different target/tree, or an
    in-flight run whose state predates selector binding, must fail
    closed, not silently continue."""

    def _state(self, **overrides) -> pin_bump.PinBumpState:
        base = dict(
            schema_version=2, run_id="run-1", from_ref="b10502", from_sha="a" * 40,
            to_ref="b10680", to_sha="b" * 40, transition_commit="c" * 40,
            tree_name="local", tree_path=str(Path("/some/path")),
            completed_phases=["preflight"], next_phase="declare",
            selector_kind="recipe", selector_name="bigcherry",
            selector_patch_ids=("0100_x",),
        )
        base.update(overrides)
        return pin_bump.PinBumpState(**base)

    def test_schema_one_resume_fails_closed(self):
        state = self._state(schema_version=1)
        with self.assertRaises(pin_bump.PinBumpStop) as ctx:
            pin_bump._validate_resume(
                state, target_ref="b10680", vendor_root=Path("/some/path"),
            )
        self.assertEqual(ctx.exception.code, "LEGACY_STATE_SELECTOR_UNBOUND")

    def test_target_mismatch_fails_closed(self):
        state = self._state()
        with self.assertRaises(pin_bump.PinBumpStop) as ctx:
            pin_bump._validate_resume(
                state, target_ref="b99999", vendor_root=Path("/some/path"),
            )
        self.assertEqual(ctx.exception.code, "RESUME_TARGET_MISMATCH")

    def test_tree_mismatch_fails_closed(self):
        state = self._state()
        with self.assertRaises(pin_bump.PinBumpStop) as ctx:
            pin_bump._validate_resume(
                state, target_ref="b10680", vendor_root=Path("/different/path"),
            )
        self.assertEqual(ctx.exception.code, "RESUME_TREE_MISMATCH")

    def test_matching_target_and_tree_passes(self):
        state = self._state()
        pin_bump._validate_resume(
            state, target_ref="b10680", vendor_root=Path("/some/path"),
        )  # no raise


class ResumeSelectorTests(unittest.TestCase):
    def _state(self, **overrides) -> pin_bump.PinBumpState:
        base = dict(
            schema_version=2, run_id="run-1", from_ref="b10502", from_sha="a" * 40,
            to_ref="b10680", to_sha="b" * 40, transition_commit="c" * 40,
            tree_name="local", tree_path="/some/path",
            completed_phases=["preflight"], next_phase="declare",
            selector_kind="recipe", selector_name="bigcherry",
            selector_patch_ids=("0100_x",),
        )
        base.update(overrides)
        return pin_bump.PinBumpState(**base)

    def test_no_cli_selector_reuses_persisted_selector(self):
        state = self._state()
        kind, name = pin_bump._resume_selector(state, recipe_name=None)
        self.assertEqual((kind, name), ("recipe", "bigcherry"))

    def test_matching_cli_recipe_is_accepted(self):
        state = self._state()
        kind, name = pin_bump._resume_selector(state, recipe_name="bigcherry")
        self.assertEqual((kind, name), ("recipe", "bigcherry"))

    def test_mismatched_cli_recipe_name_fails_closed(self):
        state = self._state()
        with self.assertRaises(pin_bump.PinBumpStop) as ctx:
            pin_bump._resume_selector(state, recipe_name="release")
        self.assertEqual(ctx.exception.code, "RESUME_SELECTOR_MISMATCH")


class RequireSelectorMembershipUnchangedTests(unittest.TestCase):
    def test_unchanged_membership_passes(self):
        state = pin_bump.PinBumpState(
            schema_version=2, run_id="r", from_ref="a", from_sha="a" * 40,
            to_ref="b", to_sha="b" * 40, transition_commit="c" * 40,
            tree_name="local", tree_path="/p", completed_phases=[], next_phase="coverage",
            selector_kind="recipe", selector_name="bigcherry-native",
            selector_patch_ids=tuple(sorted(
                pin_bump.patch_rebase._selection_patch_ids(
                    recipe_name="bigcherry-native", all_patches=False,
                )
            )),
        )
        pin_bump._require_selector_membership_unchanged(
            state, recipe_name="bigcherry-native",
        )  # no raise -- real catalog, unchanged since state was built above

    def test_drifted_membership_fails_closed(self):
        state = pin_bump.PinBumpState(
            schema_version=2, run_id="r", from_ref="a", from_sha="a" * 40,
            to_ref="b", to_sha="b" * 40, transition_commit="c" * 40,
            tree_name="local", tree_path="/p", completed_phases=[], next_phase="coverage",
            selector_kind="recipe", selector_name="bigcherry-native",
            selector_patch_ids=("this_patch_id_does_not_exist_anymore",),
        )
        with self.assertRaises(pin_bump.PinBumpStop) as ctx:
            pin_bump._require_selector_membership_unchanged(
                state, recipe_name="bigcherry-native",
            )
        self.assertEqual(ctx.exception.code, "RESUME_SELECTION_CHANGED")


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


class RequireCoverageReportTests(unittest.TestCase):
    """gpt-dev-agent review of c236acc (P1, session ses_5307d9c58ec645cb):
    the coverage-report digest gate must be unconditional and run every
    time apply is about to happen, not only on --resume."""

    def _state(self, **overrides) -> pin_bump.PinBumpState:
        base = dict(
            schema_version=2, run_id="r", from_ref="a", from_sha="a" * 40,
            to_ref="b", to_sha="b" * 40, transition_commit="c" * 40,
            tree_name="local", tree_path="/p", completed_phases=[], next_phase="apply",
            selector_kind="recipe", selector_name="bigcherry",
            selector_patch_ids=("0100_x",), coverage_report_sha256="",
        )
        base.update(overrides)
        return pin_bump.PinBumpState(**base)

    def test_unbound_digest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "rebase-recipe.json"
            report_path.write_text("{}", encoding="utf-8")
            state = self._state(coverage_report_sha256="")
            with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                pin_bump._require_coverage_report(state, report_path)
            self.assertEqual(ctx.exception.code, "COVERAGE_REPORT_UNBOUND")

    def test_missing_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "rebase-recipe.json"  # never written
            state = self._state(coverage_report_sha256="deadbeef")
            with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                pin_bump._require_coverage_report(state, report_path)
            self.assertEqual(ctx.exception.code, "COVERAGE_REPORT_MISSING")

    def test_modified_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "rebase-recipe.json"
            report_path.write_text('{"a": 1}', encoding="utf-8")
            recorded = pin_bump._sha256_file(report_path)
            report_path.write_text('{"a": 2}', encoding="utf-8")  # modified after
            state = self._state(coverage_report_sha256=recorded)
            with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                pin_bump._require_coverage_report(state, report_path)
            self.assertEqual(ctx.exception.code, "COVERAGE_REPORT_MODIFIED")

    def test_matching_digest_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "rebase-recipe.json"
            report_path.write_text('{"a": 1}', encoding="utf-8")
            recorded = pin_bump._sha256_file(report_path)
            state = self._state(coverage_report_sha256=recorded)
            pin_bump._require_coverage_report(state, report_path)  # no raise


class LoadStateOrStopTests(unittest.TestCase):
    """gpt-dev-agent review (session ses_5307d9c58ec645cb, second pass on
    e0a5b34): PinBumpState.load() itself can raise JSONDecodeError/KeyError/
    OSError on a corrupt state.json, violating pin_bump's own "PinBumpStop,
    never a bare exception" contract. _load_state_or_stop() closes that."""

    def test_malformed_json_fails_closed_with_structured_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            (state_dir / "state.json").write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                pin_bump._load_state_or_stop(state_dir)
            self.assertEqual(ctx.exception.code, "RESUME_STATE_INVALID")

    def test_missing_required_field_fails_closed_with_structured_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            (state_dir / "state.json").write_text(
                '{"schema_version": 2}', encoding="utf-8",  # missing run_id, target, etc.
            )
            with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                pin_bump._load_state_or_stop(state_dir)
            self.assertEqual(ctx.exception.code, "RESUME_STATE_INVALID")

    def test_valid_state_loads_normally(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            state = pin_bump.PinBumpState(
                schema_version=2, run_id="r", from_ref="a", from_sha="a" * 40,
                to_ref="b", to_sha="b" * 40, transition_commit="c" * 40,
                tree_name="local", tree_path="/p",
                completed_phases=[], next_phase="declare",
                selector_kind="recipe", selector_name="bigcherry",
                selector_patch_ids=("0100_x",),
            )
            state.save(state_dir)
            loaded = pin_bump._load_state_or_stop(state_dir)
            self.assertEqual(loaded, state)


class RequireCoverageReportIOErrorTests(unittest.TestCase):
    """gpt-dev-agent review (second pass on e0a5b34): _sha256_file() can
    raise OSError/FileNotFoundError if the report becomes unreadable or
    disappears between the is_file() check and the read -- must become a
    structured PinBumpStop, not a bare exception."""

    def _state(self, **overrides) -> pin_bump.PinBumpState:
        base = dict(
            schema_version=2, run_id="r", from_ref="a", from_sha="a" * 40,
            to_ref="b", to_sha="b" * 40, transition_commit="c" * 40,
            tree_name="local", tree_path="/p", completed_phases=[], next_phase="apply",
            selector_kind="recipe", selector_name="bigcherry",
            selector_patch_ids=("0100_x",), coverage_report_sha256="deadbeef",
        )
        base.update(overrides)
        return pin_bump.PinBumpState(**base)

    def test_disappearing_report_between_check_and_read_fails_closed(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "rebase-recipe.json"
            report_path.write_text("{}", encoding="utf-8")
            state = self._state()
            with mock.patch.object(
                pin_bump, "_sha256_file", side_effect=FileNotFoundError("gone"),
            ):
                with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                    pin_bump._require_coverage_report(state, report_path)
            self.assertEqual(ctx.exception.code, "COVERAGE_REPORT_MISSING")

    def test_unreadable_report_fails_closed(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "rebase-recipe.json"
            report_path.write_text("{}", encoding="utf-8")
            state = self._state()
            with mock.patch.object(
                pin_bump, "_sha256_file",
                side_effect=PermissionError("denied"),
            ):
                with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                    pin_bump._require_coverage_report(state, report_path)
            self.assertEqual(ctx.exception.code, "COVERAGE_REPORT_UNREADABLE")


class ContextRecoveryNeverMasksOriginalStopTests(unittest.TestCase):
    """gpt-dev-agent review (second pass): the except block's best-effort
    disk re-read (when `state` is None) must never itself raise and
    replace the original PinBumpStop with an unrelated one."""

    def test_corrupt_state_on_disk_during_context_recovery_does_not_mask_original(self):
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "resume-b99999"
            report_dir.mkdir(parents=True)
            # A state.json exists but is corrupt -- run() never gets far
            # enough to bind `state` locally (RESUME_STATE_MISSING path
            # isn't hit since the file DOES exist; _load_state_or_stop
            # itself raises RESUME_STATE_INVALID, and `state` stays None
            # per its unconditional pre-try initialization).
            (report_dir / "state.json").write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                pin_bump.run(target_ref="b99999", resume=True, report_dir=report_dir)
            # The ORIGINAL failure must survive -- not get replaced by a
            # second failure from the except block's own disk re-read.
            self.assertEqual(ctx.exception.code, "RESUME_STATE_INVALID")


class RunResumeStateMissingTests(unittest.TestCase):
    """gpt-dev-agent review of c236acc (P1): --resume must never silently
    reinterpret a missing state.json as "start a fresh run"."""

    def test_resume_with_no_state_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "resume-b99999"  # never created
            with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                pin_bump.run(
                    target_ref="b99999", resume=True, report_dir=report_dir,
                )
            self.assertEqual(ctx.exception.code, "RESUME_STATE_MISSING")
            # gpt-dev-agent review P2: even this early failure must carry
            # real context, not fall through as "unresolved" -- here there
            # genuinely is no state yet, so run_id=="unresolved" IS correct;
            # this asserts the placeholder path itself still populates all
            # three fields rather than leaving any unset/None.
            self.assertEqual(ctx.exception.run_id, "unresolved")
            self.assertEqual(ctx.exception.target, {"from_ref": "?", "to_ref": "b99999"})
            self.assertIsNotNone(ctx.exception.tree)

    def test_resume_with_mismatched_target_retains_loaded_state_context(self):
        # gpt-dev-agent review P2: a PinBumpStop raised by resume
        # VALIDATION (state WAS loaded) must attach that state's real
        # run_id/target/tree -- not the "unresolved" placeholder -- since
        # moving _validate_resume() inside run()'s try is what this test
        # actually exercises.
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "resume-b10680"
            state = pin_bump.PinBumpState(
                schema_version=2, run_id="real-run-id", from_ref="b10502",
                from_sha="a" * 40, to_ref="b10680", to_sha="b" * 40,
                transition_commit="c" * 40, tree_name="local",
                tree_path=str(Path("/some/path")),
                completed_phases=["preflight"], next_phase="declare",
                selector_kind="recipe", selector_name="bigcherry",
                selector_patch_ids=("0100_x",),
            )
            state.save(report_dir)
            with self.assertRaises(pin_bump.PinBumpStop) as ctx:
                pin_bump.run(
                    target_ref="b99999-WRONG",  # deliberate mismatch
                    resume=True, report_dir=report_dir,
                    root=Path("/some/path"),
                )
            self.assertEqual(ctx.exception.code, "RESUME_TARGET_MISMATCH")
            self.assertEqual(ctx.exception.run_id, "real-run-id")
            self.assertEqual(ctx.exception.target, {"from_ref": "b10502", "to_ref": "b10680"})


class WriteReleaseDocReportBindingTests(unittest.TestCase):
    """gpt-dev-agent review of c236acc (P1): with report_dir supplied, a
    missing/invalid report must skip the doc, never silently fall back to
    a fresh (TOCTOU-prone) selector resolution."""

    def test_missing_report_skips_doc_without_fresh_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)  # no rebase-recipe.json in it
            from unittest import mock

            with mock.patch(
                "bigcherry.patch.rebase._selection_patch_ids",
            ) as fresh_resolve:
                pin_bump._write_release_doc_best_effort(
                    repo_root=Path("H:/development/projects/bigcherry"),
                    vendor_root=Path("does-not-exist"),
                    recipe_name="bigcherry", target_ref="b99999",
                    report_dir=report_dir,
                )  # must not raise
            fresh_resolve.assert_not_called()


class CommitReleaseRecordsTests(unittest.TestCase):
    """gpt-dev-agent review, 2026-08-31: a successful bump used to leave its
    own releases/<tag>.json etc uncommitted, blocking the very next
    mandatory step (build+smoke, which refuses a dirty controller tree).
    _commit_release_records() closes that gap -- narrowly, by exact
    pathspec, never touching pin-transition.json or anything else."""

    def test_noop_when_nothing_owned_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _init_repo(root)
            head_before = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            pin_bump._commit_release_records(repo_root=root, target_ref="b99999")  # must not raise
            head_after = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            self.assertEqual(head_before, head_after)  # no commit made

    def test_commits_only_owned_paths_not_an_unrelated_dirty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _init_repo(root)
            (root / "releases").mkdir()
            (root / "releases" / "b99999.json").write_text('{"x": 1}\n', encoding="utf-8")
            (root / "releases" / "index.json").write_text("[]\n", encoding="utf-8")
            (root / "releases" / "b99999-patches.md").write_text("# doc\n", encoding="utf-8")
            # Simulate a concurrent, unrelated uncommitted change on this
            # shared working tree -- must survive untouched and unstaged.
            (root / "unrelated.txt").write_text("someone else's work\n", encoding="utf-8")

            pin_bump._commit_release_records(repo_root=root, target_ref="b99999")

            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True, capture_output=True, text=True,
            ).stdout
            self.assertIn("?? unrelated.txt", status)
            self.assertNotIn("releases", status)  # the owned paths are now committed, not staged

            log = subprocess.run(
                ["git", "-C", str(root), "show", "--stat", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout
            self.assertIn("b99999.json", log)
            self.assertIn("index.json", log)
            self.assertIn("b99999-patches.md", log)
            self.assertNotIn("unrelated.txt", log)

    def test_idempotent_on_a_second_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _init_repo(root)
            (root / "releases").mkdir()
            (root / "releases" / "b99999.json").write_text('{"x": 1}\n', encoding="utf-8")
            (root / "releases" / "index.json").write_text("[]\n", encoding="utf-8")

            pin_bump._commit_release_records(repo_root=root, target_ref="b99999")
            head_after_first = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()

            pin_bump._commit_release_records(repo_root=root, target_ref="b99999")  # must not raise
            head_after_second = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            self.assertEqual(head_after_first, head_after_second)  # no empty second commit

    def test_never_touches_pin_transition_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _init_repo(root)
            (root / "releases").mkdir()
            (root / "releases" / "b99999.json").write_text('{"x": 1}\n', encoding="utf-8")
            (root / "releases" / "index.json").write_text("[]\n", encoding="utf-8")
            (root / "releases" / "pin-transition.json").write_text('{"tag": "b99999"}\n', encoding="utf-8")

            pin_bump._commit_release_records(repo_root=root, target_ref="b99999")

            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True, capture_output=True, text=True,
            ).stdout
            self.assertIn("pin-transition.json", status)  # still uncommitted -- untouched


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
