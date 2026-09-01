from __future__ import annotations

import json
import math
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.release import records as releases # noqa: E402
from bigcherry import __main__ as bigcherry_main  # noqa: E402
from bigcherry.tuning.promotion import PromotionError  # noqa: E402


class AtomicJsonWriteTests(unittest.TestCase):
    def test_replaces_existing_file_only_after_validating_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "record.json"
            target.write_text('{"old": true}\n', encoding="utf-8")

            releases._atomic_write_json(target, {"new": True})

            self.assertEqual(json.loads(target.read_text(encoding="utf-8")),
                             {"new": True})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_invalid_payload_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "index.json"
            original = '{"old": true}\n'
            target.write_text(original, encoding="utf-8")

            with self.assertRaises(ValueError):
                releases._atomic_write_json(target, {"bad": math.nan})

            self.assertEqual(target.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class ReleasePublicationTests(unittest.TestCase):
    def test_save_publishes_record_and_index_through_atomic_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(releases, "RELEASES_DIR", root), \
                    mock.patch.object(releases, "INDEX_PATH", root / "index.json"), \
                    mock.patch.object(releases, "_atomic_write_json", wraps=releases._atomic_write_json) as write:
                from bigcherry.tuning.promotion import make_pointer
                pointer = make_pointer(
                    release_tag="b10362", revision="abc123", campaign_plan_id="plan1",
                    campaign_run_id="run1", report=b"report-bytes",
                    source_slice_id="slice1", build_id="build1", binary_hash="bin-hash",
                    required_architectures=("gfx1100",), replay_artifact_hash="replay-hash",
                    valid=True)
                record = releases.ReleaseRecord(
                    revision="abc123", release_tag="b10362", stage="validated",
                    promotion=pointer.document())
                path = record.save()

            self.assertEqual(path, root / "b10362.json")
            self.assertEqual(write.call_count, 2)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["stage"],
                             "validated")
            self.assertEqual(len(json.loads((root / "index.json").read_text(
                encoding="utf-8"))["releases"]), 1)

    def test_resaving_unchanged_content_does_not_dirty_the_files(self):
        # HI130: save()/_rebuild_index() used to unconditionally rewrite
        # updated_at/generated_at even when nothing substantive changed --
        # an idempotent audit/apply/generate re-save would still dirty
        # tracked releases/*.json, tripping require_clean_bigcherry() on the
        # campaign build engine for no real reason.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(releases, "RELEASES_DIR", root), \
                    mock.patch.object(releases, "INDEX_PATH", root / "index.json"):
                record = releases.ReleaseRecord(revision="abc123", release_tag="b10362")
                path = record.save()
                first_bytes = path.read_bytes()
                first_index_bytes = (root / "index.json").read_bytes()
                first_updated_at = record.updated_at

                # A second save() of an equivalent record (same substantive
                # content, fresh object) must not touch either file's bytes.
                record2 = releases.ReleaseRecord(revision="abc123", release_tag="b10362")
                with mock.patch.object(
                    releases, "_atomic_write_json", wraps=releases._atomic_write_json,
                ) as write:
                    record2.save()
                self.assertEqual(write.call_count, 0)
                self.assertEqual(path.read_bytes(), first_bytes)
                self.assertEqual((root / "index.json").read_bytes(), first_index_bytes)
                self.assertEqual(record2.updated_at, first_updated_at)

                # A real content change still writes for real.
                record3 = releases.ReleaseRecord(
                    revision="abc123", release_tag="b10362", stage="broken",
                )
                with mock.patch.object(
                    releases, "_atomic_write_json", wraps=releases._atomic_write_json,
                ) as write:
                    record3.save()
                self.assertEqual(write.call_count, 2)
                self.assertNotEqual(path.read_bytes(), first_bytes)

    def test_all_records_ignores_pin_transition_marker(self):
        # Found during a real b10502->b10680 bump: releases/pin-transition.json
        # lives in the same directory as the per-revision *.json records but
        # has no `revision` field, so all_records() (called from save() via
        # _rebuild_index()) crashed with a TypeError on every save while a
        # transition marker was present.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pin-transition.json").write_text(
                json.dumps({
                    "schema_version": 1, "from_sha": "a" * 40, "to_sha": "b" * 40,
                    "tag": "b10680", "declaring_commit": "c" * 40,
                    "set_at": "2026-08-29T00:00:00+00:00",
                }),
                encoding="utf-8",
            )
            with mock.patch.object(releases, "RELEASES_DIR", root), \
                    mock.patch.object(releases, "INDEX_PATH", root / "index.json"):
                record = releases.ReleaseRecord(revision="abc123", release_tag="b10362")
                record.save()  # must not raise
                self.assertEqual(len(releases.all_records()), 1)

    def test_all_records_ignores_patch_rebase_report(self):
        # Same real bump: PIN_BUMP.md's own documented command
        # (`patch-rebase-check --recipe bigcherry --json releases/patch-rebase.json`)
        # writes a non-ReleaseRecord report straight into releases/, which
        # also has no `revision` field -- must not crash all_records() either.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "patch-rebase.json").write_text(
                json.dumps({"upstream_revision": "d" * 40, "patches": []}),
                encoding="utf-8",
            )
            with mock.patch.object(releases, "RELEASES_DIR", root), \
                    mock.patch.object(releases, "INDEX_PATH", root / "index.json"):
                record = releases.ReleaseRecord(revision="abc123", release_tag="b10362")
                record.save()  # must not raise
                self.assertEqual(len(releases.all_records()), 1)


class PromotionWiringTests(unittest.TestCase):
    """RE13: `validated` can only be reached through a real, campaign-backed
    PromotionPointer -- releases.promote(), not a bare stage assignment."""

    def _pointer(self, revision="abc123", release_tag="b10362"):
        from bigcherry.tuning.promotion import make_pointer
        return make_pointer(
            release_tag=release_tag, revision=revision, campaign_plan_id="plan1",
            campaign_run_id="run1", report=b"report-bytes",
            source_slice_id="slice1", build_id="build1", binary_hash="bin-hash",
            required_architectures=("gfx1100", "gfx1201"),
            replay_artifact_hash="replay-hash", valid=True)

    def test_advance_to_validated_is_always_rejected_directly(self):
        # RE13 (GPT-auto-agent review): even WITH a promotion pointer
        # already set, advance_to("validated") must still reject -- only
        # promote() may perform this transition.
        record = releases.ReleaseRecord(revision="abc123", stage="tested")
        with self.assertRaisesRegex(ValueError, "releases.promote"):
            record.advance_to("validated")
        record.promotion = self._pointer().document()
        with self.assertRaisesRegex(ValueError, "releases.promote"):
            record.advance_to("validated")

    def test_promote_sets_promotion_and_advances_stage(self):
        record = releases.ReleaseRecord(
            revision="abc123", release_tag="b10362", stage="tested")
        releases.promote(record, self._pointer())
        self.assertEqual(record.stage, "validated")
        self.assertEqual(record.promotion["revision"], "abc123")
        self.assertEqual(record.promotion["required_architectures"],
                         ["gfx1100", "gfx1201"])

    def test_promote_rejects_a_pointer_for_a_different_revision(self):
        record = releases.ReleaseRecord(
            revision="abc123", release_tag="b10362", stage="tested")
        with self.assertRaisesRegex(ValueError, "does not match"):
            releases.promote(record, self._pointer(revision="other-revision"))
        self.assertEqual(record.stage, "tested")
        self.assertIsNone(record.promotion)

    def test_promote_rejects_a_pointer_for_a_different_release_tag(self):
        record = releases.ReleaseRecord(
            revision="abc123", release_tag="b10362", stage="tested")
        with self.assertRaisesRegex(ValueError, "release_tag"):
            releases.promote(record, self._pointer(release_tag="b99999"))
        self.assertEqual(record.stage, "tested")
        self.assertIsNone(record.promotion)

    def test_promote_accepts_revision_as_release_tag_for_an_untagged_release(self):
        # ReleaseRecord.release_tag is legitimately "" for an untagged
        # revision; the pointer's own (required non-empty) release_tag
        # must then equal the revision itself, matching slug()'s own
        # fallback convention.
        record = releases.ReleaseRecord(revision="abc123", stage="tested")
        releases.promote(record, self._pointer(release_tag="abc123"))
        self.assertEqual(record.stage, "validated")

    def test_promote_rejects_a_hand_tampered_pointer_object(self):
        # GPT-auto-agent review (RE13 follow-up, 2026-08-17): promote()
        # must re-validate through PromotionPointer.from_document(), not
        # trust an already-constructed object -- frozen dataclasses do not
        # prevent object.__setattr__, so a caller-tampered pointer could
        # otherwise bypass make_pointer()'s own validation entirely.
        pointer = self._pointer()
        object.__setattr__(pointer, "source_slice_id", "")
        record = releases.ReleaseRecord(
            revision="abc123", release_tag="b10362", stage="tested")
        with self.assertRaises(PromotionError):
            releases.promote(record, pointer)
        self.assertEqual(record.stage, "tested")
        self.assertIsNone(record.promotion)

    def test_validate_rejects_a_validated_record_loaded_with_no_pointer(self):
        record = releases.ReleaseRecord(revision="abc123", stage="validated")
        with self.assertRaisesRegex(ValueError, "invalid promotion pointer"):
            record.validate()

    def test_validate_rejects_a_minimal_incomplete_promotion_pointer(self):
        # GPT-auto-agent review: this exact minimal shape used to pass the
        # old loose check.
        record = releases.ReleaseRecord(
            revision="abc123", stage="validated",
            promotion={"schema_version": 2, "revision": "abc123"})
        with self.assertRaisesRegex(ValueError, "invalid promotion pointer"):
            record.validate()

    def test_validate_rejects_a_promotion_pointer_for_the_wrong_revision(self):
        record = releases.ReleaseRecord(
            revision="abc123", release_tag="b10362", stage="validated",
            promotion=self._pointer(revision="not-abc123").document())
        with self.assertRaisesRegex(ValueError, "disagrees"):
            record.validate()

    def test_validate_rejects_a_promotion_pointer_for_the_wrong_release_tag(self):
        record = releases.ReleaseRecord(
            revision="abc123", release_tag="b10362", stage="validated",
            promotion=self._pointer(release_tag="b99999").document())
        with self.assertRaisesRegex(ValueError, "disagrees"):
            record.validate()


class ReleaseLifecycleTests(unittest.TestCase):
    def test_stage_transitions_are_monotonic(self):
        record = releases.ReleaseRecord(revision="abc123", stage="tested")
        with self.assertRaisesRegex(ValueError, "move backwards"):
            record.advance_to("patched")
        self.assertEqual(record.stage, "tested")

    def test_broken_state_can_be_explicitly_recovered(self):
        record = releases.ReleaseRecord(revision="abc123", stage="broken")
        record.advance_to("pulled")
        record.advance_to("audited")
        self.assertEqual(record.stage, "audited")

    def test_invalid_stage_is_fail_closed(self):
        record = releases.ReleaseRecord(revision="abc123", stage="mystery")
        with self.assertRaisesRegex(ValueError, "unknown release stage"):
            record.validate()

    def test_successful_reapply_preserves_later_evidence(self):
        for stage in ("generated", "built", "tested", "tuned", "validated"):
            with self.subTest(stage=stage):
                record = releases.ReleaseRecord(revision="abc123", stage=stage)
                releases.record_apply_result(record, True)
                self.assertEqual(record.stage, stage)

    def test_mutating_reapply_invalidates_later_evidence(self):
        record = releases.ReleaseRecord(
            revision="abc123", stage="validated", manifest_hash="dead" * 8)
        releases.record_apply_result(record, True, mutated=True)
        self.assertEqual(record.stage, "patched")
        self.assertEqual(record.manifest_hash, "")

    def test_mutating_reapply_clears_stale_promotion_pointer(self):
        # GPT-auto-agent review (RE13 follow-up, 2026-08-17): the exact
        # stale-evidence bypass the review found -- a validated record's
        # promotion pointer must not survive an evidence-invalidating
        # mutation, even though advance_to("validated") itself is now
        # unconditionally blocked as a second, independent guard.
        from bigcherry.tuning.promotion import make_pointer
        pointer = make_pointer(
            release_tag="b10362", revision="abc123", campaign_plan_id="plan1",
            campaign_run_id="run1", report=b"report-bytes",
            source_slice_id="slice1", build_id="build1", binary_hash="bin-hash",
            required_architectures=("gfx1100",), replay_artifact_hash="replay-hash",
            valid=True)
        record = releases.ReleaseRecord(
            revision="abc123", release_tag="b10362", stage="validated",
            manifest_hash="dead" * 8, promotion=pointer.document())
        releases.record_apply_result(record, True, mutated=True)
        self.assertEqual(record.stage, "patched")
        self.assertIsNone(record.promotion)

    def test_failed_apply_also_clears_stale_promotion_pointer(self):
        from bigcherry.tuning.promotion import make_pointer
        pointer = make_pointer(
            release_tag="b10362", revision="abc123", campaign_plan_id="plan1",
            campaign_run_id="run1", report=b"report-bytes",
            source_slice_id="slice1", build_id="build1", binary_hash="bin-hash",
            required_architectures=("gfx1100",), replay_artifact_hash="replay-hash",
            valid=True)
        record = releases.ReleaseRecord(
            revision="abc123", release_tag="b10362", stage="validated",
            promotion=pointer.document())
        releases.record_apply_result(record, False)
        self.assertEqual(record.stage, "broken")
        self.assertIsNone(record.promotion)

    def test_selection_change_is_a_mutating_reapply(self):
        record = releases.ReleaseRecord(revision="abc123", stage="tested")
        releases.record_apply_result(record, True, mutated=True)
        self.assertEqual(record.stage, "patched")

    def test_apply_exact_selection_verifies_live_revision_first(self):
        # A moved pin (or any HEAD mismatch) must refuse to mutate, before
        # anything else is even attempted.
        from bigcherry.patch.selection import CliPatchSelection

        selection = CliPatchSelection(
            label="", source_name="bigcherry",
            source_ref="deadbeef" * 5, patch_set_id="psid",
            patch_ids=("0100_x",), overlay=True, overlay_digest="digest",
        )
        with mock.patch.object(
            bigcherry_main.patch_rebase_module, "_git",
            side_effect=lambda root, *a: (
                "deadbeef" * 5 if a == ("rev-parse", "deadbeef" * 5 + "^{commit}")
                else "cafef00d" * 5
            ),
        ):
            ok = bigcherry_main._apply_exact_selection(
                Path("/tmp/bigcherry-test-tree"), selection,
            )
        self.assertFalse(ok)

    def test_apply_exact_selection_rejects_toctou_composition_drift(self):
        from bigcherry.patch.selection import CliPatchSelection

        selection = CliPatchSelection(
            label="", source_name="bigcherry",
            source_ref="deadbeef" * 5, patch_set_id="psid-old",
            patch_ids=("0100_x",), overlay=True, overlay_digest="digest",
        )
        drifted = CliPatchSelection(
            label="", source_name="bigcherry",
            source_ref="deadbeef" * 5, patch_set_id="psid-NEW",
            patch_ids=("0100_x", "0200_y"), overlay=True, overlay_digest="digest",
        )
        with mock.patch.object(
            bigcherry_main.patch_rebase_module, "_git", return_value="deadbeef" * 5,
        ), mock.patch(
            "bigcherry.patch.selection._resolve_exact_selection", return_value=drifted,
        ):
            ok = bigcherry_main._apply_exact_selection(
                Path("/tmp/bigcherry-test-tree"), selection,
            )
        self.assertFalse(ok)

    def test_apply_exact_selection_fails_closed_on_unresolved_overlay(self):
        from bigcherry.patch.selection import CliPatchSelection

        selection = CliPatchSelection(
            label="", source_name="bigcherry",
            source_ref="deadbeef" * 5, patch_set_id="psid",
            patch_ids=("0100_x",), overlay=None,
        )
        with mock.patch.object(
            bigcherry_main.patch_rebase_module, "_git", return_value="deadbeef" * 5,
        ), mock.patch(
            "bigcherry.patch.selection._resolve_exact_selection", return_value=selection,
        ):
            ok = bigcherry_main._apply_exact_selection(
                Path("/tmp/bigcherry-test-tree"), selection,
            )
        self.assertFalse(ok)

    def test_apply_exact_selection_skips_overlay_when_source_overlay_false(self):
        from bigcherry.patch.selection import CliPatchSelection

        selection = CliPatchSelection(
            label="", source_name="llama-native",
            source_ref="deadbeef" * 5, patch_set_id="psid",
            patch_ids=(), overlay=False, overlay_digest=None,
        )
        record = releases.ReleaseRecord(
            revision="abc123", release_tag="b1234", stage="validated",
            tree_state="", manifest_hash="dead" * 8, audit={"passed": True},
        )
        with mock.patch.object(
            bigcherry_main.patch_rebase_module, "_git", return_value="deadbeef" * 5,
        ), mock.patch(
            "bigcherry.patch.selection._resolve_exact_selection", return_value=selection,
        ), mock.patch.object(
            bigcherry_main, "_record_for", return_value=record,
        ), mock.patch.object(
            bigcherry_main, "_copy_overlay",
        ) as copy_overlay, mock.patch.object(
            bigcherry_main.patchset, "resolve_exact", return_value=object(),
        ), mock.patch.object(
            bigcherry_main.patchset, "load_resolved", return_value=[],
        ), mock.patch.object(
            bigcherry_main.patcher, "apply_all", return_value=[],
        ), mock.patch.object(record, "save"):
            ok = bigcherry_main._apply_exact_selection(
                Path("/tmp/bigcherry-test-tree"), selection,
            )
        self.assertTrue(ok)
        copy_overlay.assert_not_called()

    def test_apply_exact_selection_happy_path_updates_record(self):
        from bigcherry.patch.selection import CliPatchSelection

        selection = CliPatchSelection(
            label="", source_name="bigcherry",
            source_ref="deadbeef" * 5, patch_set_id="psid",
            patch_ids=("0100_x",), overlay=True, overlay_digest="digest",
        )
        record = releases.ReleaseRecord(
            revision="abc123", release_tag="b1234", stage="validated",
            tree_state="", manifest_hash="dead" * 8, audit={"passed": True},
        )
        patch_result = SimpleNamespace(
            path="src/example.cpp", results=[], failed=[], changed=True, ok=True)
        with mock.patch.object(
            bigcherry_main.patch_rebase_module, "_git", return_value="deadbeef" * 5,
        ), mock.patch(
            "bigcherry.patch.selection._resolve_exact_selection", return_value=selection,
        ), mock.patch.object(
            bigcherry_main, "_record_for", return_value=record,
        ), mock.patch.object(
            bigcherry_main, "_copy_overlay", return_value=[],
        ), mock.patch.object(
            bigcherry_main.patchset, "resolve_exact", return_value=object(),
        ), mock.patch.object(
            bigcherry_main.patchset, "load_resolved", return_value=[object()],
        ), mock.patch.object(
            bigcherry_main.patcher, "apply_all", return_value=[patch_result],
        ), mock.patch.object(record, "save"):
            ok = bigcherry_main._apply_exact_selection(
                Path("/tmp/bigcherry-test-tree"), selection,
            )
        self.assertTrue(ok)
        self.assertEqual(record.stage, "patched")
        self.assertEqual(
            record.tree_state, selection.tree_state_key("deadbeef" * 5)
        )

    def test_successful_reapply_recovers_explicit_broken_state(self):
        record = releases.ReleaseRecord(revision="abc123", stage="broken")
        releases.record_apply_result(record, True)
        self.assertEqual(record.stage, "patched")

    def test_failed_apply_poisoning_remains_explicit(self):
        record = releases.ReleaseRecord(revision="abc123", stage="generated")
        releases.record_apply_result(record, False)
        self.assertEqual(record.stage, "broken")

    def test_index_must_match_source_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = releases.ReleaseRecord(
                revision="abc123", stage="tested", updated_at="now")
            record_path = root / "abc123.json"
            record_path.write_text(json.dumps({**releases.asdict(record)}),
                                   encoding="utf-8")
            index_path = root / "index.json"
            index_path.write_text(json.dumps({"releases": [{
                "slug": "abc123", "revision": "wrong", "release_tag": "",
                "stage": "tested", "manifest_hash": "",
                "audit_passed": False, "updated_at": "now",
            }]}), encoding="utf-8")
            with mock.patch.object(releases, "RELEASES_DIR", root):
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    releases.validate_index_consistency(index_path=index_path)


if __name__ == "__main__":
    unittest.main()
