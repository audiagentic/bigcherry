"""RE43: `bigcherry patch-explain <id>` / `patch-graph` -- real CLI tests
against the actual catalog, matching the style of test_cli_patches_catalog_filter.py."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import __main__ as cli  # noqa: E402
from bigcherry import patch_catalog  # noqa: E402


def _run(argv: list[str]) -> tuple[int, str, str]:
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = args.func(args)
    return code, out.getvalue(), err.getvalue()


class PatchExplainCliTests(unittest.TestCase):
    def test_explains_a_real_patch_with_requires(self):
        code, out, _ = _run(["patch-explain", "1217_rd44_graph_opt_default_rdna35"])
        self.assertEqual(code, 0)
        self.assertIn("kind:           enhancement", out)
        self.assertIn("1215_rd394041_amd_stream_moe_overlap", out)
        self.assertIn("1216_rd43_concurrent_join_fusion_guard", out)
        self.assertIn("plan item:      RD44", out)
        self.assertIn("selected by experiments: rd44-only", out)

    def test_unknown_patch_id_fails_closed(self):
        code, out, err = _run(["patch-explain", "not-a-real-patch-id"])
        self.assertEqual(code, 1)
        self.assertIn("not-a-real-patch-id", err)

    def test_framework_patch_has_no_provenance_but_still_explains(self):
        code, out, _ = _run(["patch-explain", "0100_cmake_options"])
        self.assertEqual(code, 0)
        self.assertIn("kind:           framework", out)
        self.assertIn("origin:         local", out)


class PatchGraphCliTests(unittest.TestCase):
    def test_graph_with_roots_shows_real_dependency_closure(self):
        code, out, _ = _run(["patch-graph", "--roots", "1217_rd44_graph_opt_default_rdna35"])
        self.assertEqual(code, 0)
        self.assertIn("1215_rd394041_amd_stream_moe_overlap", out)
        self.assertIn("1216_rd43_concurrent_join_fusion_guard", out)
        self.assertIn("requires -> 1215_rd394041_amd_stream_moe_overlap", out)

    def test_graph_no_roots_shows_the_known_conflict_pair(self):
        code, out, _ = _run(["patch-graph"])
        self.assertEqual(code, 0)
        self.assertIn("conflicts x 1207_rd17_moe_topk_down_fold", out)
        self.assertIn("conflicts x 1205_rd12_paired_mmvq_dual_output", out)

    def test_graph_omits_isolated_patches_with_no_edges(self):
        code, out, _ = _run(["patch-graph"])
        self.assertEqual(code, 0)
        # 0100_cmake_options has no requires/conflicts of its own and
        # nothing else references it -- must not appear as a bare node.
        self.assertNotIn("0100_cmake_options", out)


class PatchCatalogExplainUnitTests(unittest.TestCase):
    def test_explain_raises_keyerror_for_unknown_id(self):
        snapshot = patch_catalog.build_snapshot()
        with self.assertRaises(KeyError):
            patch_catalog.explain("does-not-exist", snapshot, cfg=None)

    def test_explain_without_cfg_still_works(self):
        snapshot = patch_catalog.build_snapshot()
        info = patch_catalog.explain(
            "1217_rd44_graph_opt_default_rdna35", snapshot, cfg=None)
        self.assertEqual(info.selected_by_patch_sets, ())
        self.assertEqual(info.selected_by_experiments, ())
        self.assertIn("1215_rd394041_amd_stream_moe_overlap", info.requires)


if __name__ == "__main__":
    unittest.main()
