"""`bigcherry patches --kind/--backend/--origin` filters against the
packaged patch.toml metadata. catalog.toml remains available for compatibility
fixtures, but production patches are package directories."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import __main__ as cli  # noqa: E402
from bigcherry.patch import catalog as patch_catalog  # noqa: E402
from bigcherry.patch import patchset  # noqa: E402


def _run(argv: list[str]) -> tuple[int, str, str]:
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = args.func(args)
    return code, out.getvalue(), err.getvalue()


def _real_catalog_total() -> int:
    """The real, current patch count -- read fresh at test time rather than
    hardcoded, so this test does not silently drift every time a patch is
    added to or removed from the real catalog (CO01 closure audit,
    2026-09-08: three assertions here hardcoded a stale total and failed on
    every unrelated catalog growth, independent of whatever change was
    actually under test)."""
    return len(patchset.catalog())


class PatchesCatalogFilterTests(unittest.TestCase):
    def test_no_filter_shows_every_patch_unchanged(self):
        code, out, _ = _run(["patches"])
        self.assertEqual(code, 0)
        self.assertNotIn("catalog:", out)
        total = _real_catalog_total()
        self.assertIn(f"{total} of {total} shown selected", out)

    def test_kind_framework_shows_only_framework_patches(self):
        code, out, _ = _run(
            [
                "patches",
                "--kind",
                "framework"
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("catalog:   kind=framework backend=any origin=any", out)
        self.assertIn("0100_cmake_options", out)
        self.assertNotIn("1200_rd19_single_gpu_meta_bypass", out)
        self.assertIn(f"({_real_catalog_total()} total in catalog)", out)

    def test_backend_vulkan_currently_matches_nothing(self):
        # Real state of the catalog today: zero Vulkan patches exist (RE30
        # phase 3 hasn't started). The filter must say so plainly, not
        # silently print an empty table indistinguishable from an error.
        code, out, _ = _run(
            [
                "patches",
                "--backend",
                "vulkan"
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "no patches match the given --kind/--backend/--origin filter", out
        )

    def test_backend_hip_matches_every_current_patch(self):
        code, out, _ = _run(
            [
                "patches",
                "--backend",
                "hip"
            ]
        )
        self.assertEqual(code, 0)
        # cli/patch.py's own --backend filter is an EXACT match
        # (entry.backend != args.backend), not patch_catalog.
        # patches_for_backend()'s "hip or agnostic" semantics -- the two
        # disagree today (a real, separate inconsistency worth its own
        # look, out of scope here). This test verifies the CLI's actual
        # behavior, so it must count entries the same way the CLI does.
        snapshot = patch_catalog.build_snapshot()
        hip_matching = sum(
            1 for entry in snapshot.metadata.values() if entry.backend == "hip")
        total = _real_catalog_total()
        self.assertIn(
            f"{hip_matching} of {hip_matching} shown selected ({total} total in catalog)", out)

    def test_origin_external_fork_matches_only_rdna_boost_patches(self):
        code, out, _ = _run(
            [
                "patches",
                "--origin",
                "external-fork"
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("1221_rd50_gdn_chunked_recurrence", out)
        self.assertNotIn("0100_cmake_options", out)

    def test_combined_filters_are_conjunctive(self):
        code, out, _ = _run(
            [
                "patches",
                "--kind",
                "enhancement",
                "--backend",
                "hip",
                "--origin",
                "external-fork"
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("1221_rd50_gdn_chunked_recurrence", out)
        self.assertNotIn("0100_cmake_options", out)

    def test_invalid_kind_choice_rejected_by_argparse(self):
        parser = cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["patches", "--kind", "not-a-real-kind"])

    def test_catalog_label_column_shows_kind_and_backend(self):
        code, out, _ = _run(
            [
                "patches",
                "--kind",
                "framework"
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("framework/hip", out)


if __name__ == "__main__":
    unittest.main()
