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


def _run(argv: list[str]) -> tuple[int, str, str]:
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = args.func(args)
    return code, out.getvalue(), err.getvalue()


class PatchesCatalogFilterTests(unittest.TestCase):
    def test_no_filter_shows_every_patch_unchanged(self):
        code, out, _ = _run(["patches"])
        self.assertEqual(code, 0)
        self.assertNotIn("catalog:", out)
        self.assertIn("53 of 53 shown selected", out)

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
        self.assertIn("(53 total in catalog)", out)

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
        self.assertIn("53 of 53 shown selected (53 total in catalog)", out)

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
