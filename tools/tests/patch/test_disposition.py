"""HI152: revision-bound patch coverage/disposition schema + gate."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import disposition as disp  # noqa: E402


def _record(**overrides) -> disp.Disposition:
    fields = dict(
        patch_id="1206_rd13_mul_mat_add_view_fusion", target_revision="rev-a",
        patch_digest="digest-a", disposition="known_broken",
        failure_status="FAILED_NEEDS_RECONCILIATION", reason="upstream removed anchor",
        owner="rdna-boost-experiments", tracking_item="RD13",
    )
    fields.update(overrides)
    return disp.Disposition(**fields)


class DispositionStorageTests(unittest.TestCase):
    def test_round_trips_through_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _record()
            disp.save_disposition(root, record)
            loaded = disp.load_disposition(root, record.patch_id)
            self.assertEqual(loaded, record)

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(disp.load_disposition(Path(directory), "nonexistent"))

    def test_clear_removes_and_returns_true_only_if_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            disp.save_disposition(root, _record())
            self.assertTrue(disp.clear_disposition(root, "1206_rd13_mul_mat_add_view_fusion"))
            self.assertFalse(disp.clear_disposition(root, "1206_rd13_mul_mat_add_view_fusion"))

    def test_rejects_unsupported_disposition_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(disp.DispositionError):
                disp.save_disposition(Path(directory), _record(disposition="ignore_forever"))

    def test_list_dispositions_keyed_by_patch_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            disp.save_disposition(root, _record())
            disp.save_disposition(root, _record(patch_id="other_patch"))
            listing = disp.list_dispositions(root)
            self.assertEqual(set(listing), {"1206_rd13_mul_mat_add_view_fusion", "other_patch"})


class ApplicabilityTests(unittest.TestCase):
    def test_applies_only_to_the_exact_revision_and_digest(self):
        record = _record()
        self.assertTrue(record.applies_to(target_revision="rev-a", patch_digest="digest-a"))
        self.assertFalse(record.applies_to(target_revision="rev-b", patch_digest="digest-a"))
        self.assertFalse(record.applies_to(target_revision="rev-a", patch_digest="digest-b"))


def _all_report(*entries: dict) -> dict:
    return {"patches": list(entries)}


class ComputeCoverageTests(unittest.TestCase):
    def test_complete_when_everything_is_clean(self):
        result = disp.compute_coverage(
            catalog_states={"0100_a": "validated", "0200_b": "validated"},
            all_report=_all_report(
                {"patch_id": "0100_a", "status": "CLEAN", "implementation_digest": "d1"},
                {"patch_id": "0200_b", "status": "CLEAN_NOOP", "implementation_digest": "d2"},
            ),
            recipe_patch_ids=frozenset({"0100_a"}),
            dispositions={},
            target_revision="rev-a",
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.uncovered_patch_ids, ())

    def test_recipe_selected_patch_has_no_escape_hatch(self):
        # A disposition exists and applies, but the patch is in the recipe
        # selection -- must still be uncovered.
        d = _record(patch_id="0100_a", target_revision="rev-a", patch_digest="d1")
        result = disp.compute_coverage(
            catalog_states={"0100_a": "validated"},
            all_report=_all_report(
                {"patch_id": "0100_a", "status": "FAILED", "implementation_digest": "d1"},
            ),
            recipe_patch_ids=frozenset({"0100_a"}),
            dispositions={"0100_a": d},
            target_revision="rev-a",
        )
        self.assertFalse(result.complete)
        self.assertEqual(result.uncovered_patch_ids, ("0100_a",))

    def test_non_selected_broken_patch_covered_by_matching_disposition(self):
        d = _record(patch_id="1206_rd13_mul_mat_add_view_fusion", target_revision="rev-a", patch_digest="d9")
        result = disp.compute_coverage(
            catalog_states={"1206_rd13_mul_mat_add_view_fusion": "untested"},
            all_report=_all_report(
                {"patch_id": "1206_rd13_mul_mat_add_view_fusion", "status": "FAILED",
                 "implementation_digest": "d9"},
            ),
            recipe_patch_ids=frozenset(),
            dispositions={"1206_rd13_mul_mat_add_view_fusion": d},
            target_revision="rev-a",
        )
        self.assertTrue(result.complete)

    def test_disposition_invalidated_by_revision_change_leaves_it_uncovered(self):
        d = _record(patch_id="1206_rd13_mul_mat_add_view_fusion", target_revision="rev-OLD", patch_digest="d9")
        result = disp.compute_coverage(
            catalog_states={"1206_rd13_mul_mat_add_view_fusion": "untested"},
            all_report=_all_report(
                {"patch_id": "1206_rd13_mul_mat_add_view_fusion", "status": "FAILED",
                 "implementation_digest": "d9"},
            ),
            recipe_patch_ids=frozenset(),
            dispositions={"1206_rd13_mul_mat_add_view_fusion": d},
            target_revision="rev-NEW",
        )
        self.assertFalse(result.complete)
        self.assertIn("1206_rd13_mul_mat_add_view_fusion", result.uncovered_patch_ids)

    def test_disposition_invalidated_by_digest_change_leaves_it_uncovered(self):
        d = _record(patch_id="1206_rd13_mul_mat_add_view_fusion", target_revision="rev-a", patch_digest="d-old")
        result = disp.compute_coverage(
            catalog_states={"1206_rd13_mul_mat_add_view_fusion": "untested"},
            all_report=_all_report(
                {"patch_id": "1206_rd13_mul_mat_add_view_fusion", "status": "FAILED",
                 "implementation_digest": "d-new"},
            ),
            recipe_patch_ids=frozenset(),
            dispositions={"1206_rd13_mul_mat_add_view_fusion": d},
            target_revision="rev-a",
        )
        self.assertFalse(result.complete)

    def test_rejected_patches_are_excluded_not_uncovered(self):
        result = disp.compute_coverage(
            catalog_states={"0100_a": "validated", "0300_old": "rejected"},
            all_report=_all_report(
                {"patch_id": "0100_a", "status": "CLEAN", "implementation_digest": "d1"},
            ),
            recipe_patch_ids=frozenset({"0100_a"}),
            dispositions={},
            target_revision="rev-a",
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.excluded, ({"patch_id": "0300_old", "reason": "state=rejected"},))
        self.assertNotIn("0300_old", result.uncovered_patch_ids)

    def test_a_registry_patch_missing_from_the_all_report_entirely_is_uncovered(self):
        # e.g. --all was run before a brand new patch was added to the catalog.
        result = disp.compute_coverage(
            catalog_states={"0100_a": "validated", "0900_new": "untested"},
            all_report=_all_report(
                {"patch_id": "0100_a", "status": "CLEAN", "implementation_digest": "d1"},
            ),
            recipe_patch_ids=frozenset({"0100_a"}),
            dispositions={},
            target_revision="rev-a",
        )
        self.assertFalse(result.complete)
        self.assertIn("0900_new", result.uncovered_patch_ids)


if __name__ == "__main__":
    unittest.main()
