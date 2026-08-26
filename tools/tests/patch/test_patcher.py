"""Regression tests for the anchored patch engine.

Every case here corresponds to a way an edit can land in the wrong place while
still *reporting success*. That is the failure mode worth testing: a patch that
cannot find its anchor says so loudly, but a patch that finds the wrong anchor
produces a tree that configures, compiles, and behaves incorrectly.

Run with: python -m unittest discover -s tools/tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import csource # noqa: E402
from bigcherry.patcher import (  # noqa: E402
    Edit, FilePatch, apply_all, apply_patch,
)


class TempTree:
    """A throwaway checkout root holding one file."""

    def __init__(self, relative: str, content: str):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")
        self.relative = relative

    def read(self) -> str:
        return (self.root / self.relative).read_text(encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._dir.cleanup()


CMAKE = """\
option(GGML_HIP                "ggml: use HIP"          OFF)
option(GGML_HIP_EXPORT_METRICS "ggml: metrics"          OFF)
option(GGML_MUSA_GRAPHS        "ggml: use MUSA graph"   OFF)
option(GGML_VULKAN             "ggml: use Vulkan"       OFF)
option(GGML_WEBGPU             "ggml: use WebGPU"       OFF)
"""

SOURCE = """\
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvq.cuh"
#include <vector>

// A decoy in a comment: #include "ggml-cuda/mmvq.cuh"
static void thing() {
    const char * s = "option(GGML_HIP_EXPORT_METRICS decoy)";
}
"""


class TestGreedyAnchors(unittest.TestCase):
    """An anchor must not swallow the rest of the file.

    `^option\\(FOO.*$` reads as "that one line". Under re.DOTALL it matches
    from FOO to the end of the file, and the edit lands hundreds of lines away
    while still reporting success.
    """

    def test_dot_does_not_cross_lines(self):
        with TempTree("CMakeLists.txt", CMAKE) as tree:
            patch = FilePatch(
                path="CMakeLists.txt",
                edits=(Edit(
                    id="opts",
                    anchor=r"^option\(GGML_HIP_EXPORT_METRICS.*$",
                    text="\nINSERTED\n",
                    guard=r"^INSERTED$",
                ),),
            )
            result = apply_patch(patch, tree.root)
            self.assertTrue(result.ok, result.results)

            lines = tree.read().splitlines()
            self.assertEqual(lines[2], "INSERTED",
                             "edit must land on the line after its anchor, "
                             f"got: {lines}")

    def test_runaway_anchor_is_rejected(self):
        """A deliberately cross-line anchor beyond the span limit must fail."""
        with TempTree("CMakeLists.txt", CMAKE) as tree:
            patch = FilePatch(
                path="CMakeLists.txt",
                edits=(Edit(
                    id="runaway",
                    anchor=r"^option\(GGML_HIP\b[\s\S]*WEBGPU.*$",
                    text="\nINSERTED\n",
                    guard=r"^INSERTED$",
                    max_span_lines=2,
                ),),
            )
            result = apply_patch(patch, tree.root)
            self.assertFalse(result.ok)
            self.assertIn("greedy", result.failed[0].detail)


class TestNoiseStripping(unittest.TestCase):
    """Anchors match code, not comments -- but include paths *are* code."""

    def test_include_path_is_anchorable(self):
        with TempTree("a.cu", SOURCE) as tree:
            patch = FilePatch(
                path="a.cu",
                edits=(Edit(
                    id="include",
                    anchor=r'^#include "ggml-cuda/mmvq\.cuh"$',
                    text="\n#include \"bigcherry.cuh\"\n",
                    guard=r'#include "bigcherry\.cuh"',
                ),),
            )
            result = apply_patch(patch, tree.root)
            self.assertTrue(result.ok, result.results)
            lines = tree.read().splitlines()
            self.assertEqual(lines[2], '#include "bigcherry.cuh"')

    def test_commented_out_include_is_not_an_anchor(self):
        """The decoy in a comment must not count towards the match total."""
        stripped = csource.strip_noise(SOURCE)
        self.assertEqual(stripped.count('#include "ggml-cuda/mmvq.cuh"'), 1,
                         "the commented-out include must be blanked")

    def test_string_literal_is_not_an_anchor(self):
        stripped = csource.strip_noise(SOURCE)
        self.assertNotIn("option(GGML_HIP_EXPORT_METRICS decoy)", stripped)

    def test_cmake_keeps_its_strings(self):
        """CMake's interesting content lives inside quotes, so it must survive."""
        text = 'file(GLOB SRCS "../ggml-cuda/*.cu")\n# "commented"\n'
        stripped = csource.strip_noise(text, "cmake")
        self.assertIn('"../ggml-cuda/*.cu"', stripped)
        self.assertNotIn("commented", stripped)


class TestIdempotence(unittest.TestCase):
    def test_second_apply_is_a_no_op(self):
        with TempTree("CMakeLists.txt", CMAKE) as tree:
            patch = FilePatch(
                path="CMakeLists.txt",
                edits=(Edit(
                    id="opts",
                    anchor=r"^option\(GGML_HIP_EXPORT_METRICS.*$",
                    text="\nINSERTED\n",
                    guard=r"^INSERTED$",
                ),),
            )
            apply_patch(patch, tree.root)
            first = tree.read()
            second_result = apply_patch(patch, tree.root)

            self.assertEqual(second_result.results[0].status, "already-applied")
            self.assertEqual(tree.read(), first)


class TestAmbiguityAndAtomicity(unittest.TestCase):
    def test_ambiguous_anchor_fails(self):
        """Two matches is a bug in the anchor, not licence to pick the first."""
        with TempTree("CMakeLists.txt", CMAKE) as tree:
            patch = FilePatch(
                path="CMakeLists.txt",
                edits=(Edit(
                    id="ambiguous",
                    anchor=r"^option\(GGML_\w+ .*$",
                    text="\nINSERTED\n",
                    guard=r"^INSERTED$",
                ),),
            )
            result = apply_patch(patch, tree.root)
            self.assertFalse(result.ok)
            self.assertIn("matched 5 time(s)", result.failed[0].detail)

    def test_a_failing_edit_leaves_the_file_untouched(self):
        """No half-patched trees: a later failure must undo nothing, because
        nothing was written."""
        with TempTree("CMakeLists.txt", CMAKE) as tree:
            before = tree.read()
            patch = FilePatch(
                path="CMakeLists.txt",
                edits=(
                    Edit(id="good",
                         anchor=r"^option\(GGML_HIP_EXPORT_METRICS.*$",
                         text="\nINSERTED\n", guard=r"^INSERTED$"),
                    Edit(id="bad", anchor=r"^this anchor does not exist$",
                         text="\nNOPE\n", guard=r"^NOPE$"),
                ),
            )
            results = apply_all([patch], tree.root)
            self.assertFalse(results[0].ok)
            self.assertEqual(tree.read(), before)


class TestPatchDependencies(unittest.TestCase):
    """A patch may anchor on an earlier patch's output.

    This is not exotic: the coverage hook attaches to a parameter the
    forced-variant patch adds. An implementation that validated each patch
    against the on-disk file made such a patch impossible to place during the
    trial pass, however correct it was -- and the failure named a missing
    anchor, pointing at the patch rather than at the validation.
    """

    def test_second_patch_sees_first_patch_output(self):
        with TempTree("a.cu", "void f(int a) {\n    body();\n}\n") as tree:
            first = FilePatch(
                path="a.cu",
                edits=(Edit(
                    id="add-param",
                    anchor=r"^void f\(int a\) \{$",
                    mode="replace",
                    text="void f(int a, int b) {",
                    guard=r"void f\(int a, int b\) \{",
                ),),
            )
            # Anchors on text that only exists after `first` has applied.
            second = FilePatch(
                path="a.cu",
                edits=(Edit(
                    id="use-param",
                    anchor=r"^void f\(int a, int b\) \{$",
                    text="\n    use(b);",
                    guard=r"use\(b\);",
                ),),
            )
            results = apply_all([first, second], tree.root)
            self.assertTrue(all(r.ok for r in results),
                            [e.detail for r in results for e in r.failed])

            text = tree.read()
            self.assertIn("void f(int a, int b) {", text)
            self.assertIn("use(b);", text)

    def test_dependent_failure_still_writes_nothing(self):
        with TempTree("a.cu", "void f(int a) {\n    body();\n}\n") as tree:
            before = tree.read()
            first = FilePatch(
                path="a.cu",
                edits=(Edit(
                    id="add-param",
                    anchor=r"^void f\(int a\) \{$",
                    mode="replace",
                    text="void f(int a, int b) {",
                    guard=r"void f\(int a, int b\) \{",
                ),),
            )
            second = FilePatch(
                path="a.cu",
                edits=(Edit(id="wrong", anchor=r"^void f\(int a, int c\) \{$",
                            text="\n    use(c);", guard=r"use\(c\);"),),
            )
            results = apply_all([first, second], tree.root)
            self.assertFalse(all(r.ok for r in results))
            self.assertEqual(tree.read(), before,
                             "a failure anywhere must leave the tree untouched")


class TestAppliesIf(unittest.TestCase):
    def test_not_applicable_is_not_a_failure(self):
        with TempTree("CMakeLists.txt", CMAKE) as tree:
            patch = FilePatch(
                path="CMakeLists.txt",
                edits=(Edit(
                    id="old-shape",
                    anchor=r"^option\(GGML_HIP_EXPORT_METRICS.*$",
                    text="\nINSERTED\n",
                    guard=r"^INSERTED$",
                    applies_if=r"^option\(GGML_SOMETHING_ANCIENT",
                ),),
            )
            result = apply_patch(patch, tree.root)
            self.assertTrue(result.ok)
            self.assertEqual(result.results[0].status, "not-applicable")
            self.assertNotIn("INSERTED", tree.read())


if __name__ == "__main__":
    unittest.main()
