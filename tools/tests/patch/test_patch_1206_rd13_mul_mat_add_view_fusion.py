"""Offline safety checks for RD13's MUL_MAT -> RESHAPE -> ADD matcher."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import csource  # noqa: E402
from bigcherry.patcher import apply_patch  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "patches" / "rd" / "1206_rd13_mul_mat_add_view_fusion" / "patch.py"
VENDOR_SOURCE = ROOT / "vendor" / "llama.cpp" / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("rd13_patch", PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RD13 = _load_patch_module()


class Rd13PatchApplicationTests(unittest.TestCase):
    def test_patch_dry_run_rewrites_only_the_exact_pre_change_block(self) -> None:
        """The package's anchor applies to the intended old block once."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / RD13.PATCH.path
            target.parent.mkdir(parents=True)
            target.write_text(RD13._OLD, encoding="utf-8", newline="")

            result = apply_patch(RD13.PATCH, root, dry_run=True)

        self.assertTrue(result.ok, result.results)
        self.assertEqual(result.results[0].status, "applied")

    def test_current_pinned_source_has_the_expected_fusion_site(self) -> None:
        """The current pin still has the RD13 site and its structural guards."""
        source = VENDOR_SOURCE.read_text(encoding="utf-8")
        self.assertIn("const bool has_view = i + 1 < cgraph->n_nodes", source)
        self.assertIn("cgraph->nodes[i + 1]->op == GGML_OP_RESHAPE", source)
        self.assertIn("const ggml_op ops[3] = { op, GGML_OP_RESHAPE, bias_op }", source)
        self.assertIn("cgraph->nodes[i + 1]->src[0] != cgraph->nodes[i]", source)
        self.assertIn("bias_node = cgraph->nodes[has_view ? i + 2 : i + 1]", source)
        self.assertIn("fused_node_count  = has_view ? 3 : 2", source)


class Rd13NearMissGuardTests(unittest.TestCase):
    def test_view_path_requires_exact_view_and_add_wiring(self) -> None:
        """A view path accepts only MUL_MAT -> RESHAPE -> ADD(src[0]=view)."""
        text = RD13._NEW
        self.assertIn(
            "} else if (!has_view && bias_node->src[1] == mm_or_view) {", text
        )
        self.assertIn("if (bias_tensor == nullptr) {", text)

        # The view-specific branch cannot accept reversed ADD operands.
        view_branch = text[text.index("if (bias_op == GGML_OP_ADD) {") :]
        self.assertNotIn(
            "else if (bias_node->src[1] == mm_or_view)",
            view_branch.split("const ggml_tensor * src0", 1)[0],
        )

    def test_all_requested_near_misses_are_rejected_by_static_matcher(self) -> None:
        """Model the matcher predicates independently of CUDA/hardware."""
        class Node:
            def __init__(self, op: str, src0=None, src1=None, uses: int = 1):
                self.op = op
                self.src0 = src0
                self.src1 = src1
                self.uses = uses

        def matches(nodes: list[Node]) -> bool:
            # RD13's new path: exact three-node sequence and one-use chain.
            if len(nodes) != 3 or [node.op for node in nodes] != ["MUL_MAT", "RESHAPE", "ADD"]:
                return False
            mm, view, add = nodes
            if mm.uses != 1 or view.uses != 1:
                return False
            if view.src0 is not mm or add.src0 is not view or add.src1 is None:
                return False
            return True

        mm = Node("MUL_MAT")
        view = Node("RESHAPE", src0=mm)
        addend = Node("INPUT")

        self.assertTrue(matches([mm, view, Node("ADD", src0=view, src1=addend)]))
        # Direct ADD: no RESHAPE node, so RD13's new matcher does not match.
        self.assertFalse(matches([mm, Node("ADD", src0=mm, src1=addend), addend]))
        # Wrong/reversed view wiring: ADD.src[0] is not the view.
        self.assertFalse(matches([mm, view, Node("ADD", src0=addend, src1=view)]))
        # Intermediate has an extra consumer and cannot be silently elided.
        view.uses = 2
        self.assertFalse(matches([mm, view, Node("ADD", src0=view, src1=addend)]))
        view.uses = 1
        # Non-VIEW node in the intermediate/src[0] position.
        non_view = Node("MUL")
        self.assertFalse(matches([mm, non_view, Node("ADD", src0=non_view, src1=addend)]))
        # Null addend is explicitly rejected.
        self.assertFalse(matches([mm, view, Node("ADD", src0=view)]))


class Rd13AnchorNoiseTests(unittest.TestCase):
    def test_anchor_matches_pre_change_code_after_comment_stripping(self) -> None:
        stripped = csource.strip_noise(RD13._OLD)
        self.assertEqual(len(re.findall(RD13.PATCH.edits[0].anchor, stripped)), 1)
        self.assertEqual(
            len(re.findall(RD13.PATCH.edits[0].anchor, RD13._NEW)), 0,
            "the replacement must not be mistaken for a second pre-change anchor",
        )


if __name__ == "__main__":
    unittest.main()
