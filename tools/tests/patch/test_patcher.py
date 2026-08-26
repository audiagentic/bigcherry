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
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import csource # noqa: E402
from bigcherry.patcher import (  # noqa: E402
    Edit, FilePatch, PatchError, apply_all, apply_patch, resolve_contained_target,
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

    def test_real_pass_failure_after_a_successful_write_rolls_back(self):
        """PA07 (L1.1): the trial pass proves both patches CAN be placed, but
        a real-pass failure (state drift, I/O error, anything) after the
        first patch's write already landed on disk must not leave that write
        behind -- the whole run is all-or-nothing, not just the trial."""
        import bigcherry.patch.apply as apply_module

        with TempTree("a.cu", "void f(int a) {\n    body();\n}\n") as tree:
            (tree.root / "b.cu").write_text("void g(int a) {\n    body();\n}\n",
                                             encoding="utf-8", newline="")
            first = FilePatch(
                path="a.cu",
                edits=(Edit(id="p1", anchor=r"^void f\(int a\) \{$", mode="replace",
                            text="void f(int a, int b) {",
                            guard=r"void f\(int a, int b\) \{"),),
            )
            second = FilePatch(
                path="b.cu",
                edits=(Edit(id="p2", anchor=r"^void g\(int a\) \{$", mode="replace",
                            text="void g(int a, int b) {",
                            guard=r"void g\(int a, int b\) \{"),),
            )
            before_a = (tree.root / "a.cu").read_text(encoding="utf-8")
            before_b = (tree.root / "b.cu").read_text(encoding="utf-8")

            real_apply_patch = apply_module.apply_patch
            call_count = {"n": 0}

            def flaky_apply_patch(patch, root, *, dry_run=False, texts=None):
                if not dry_run:
                    call_count["n"] += 1
                    if call_count["n"] == 2:
                        raise RuntimeError("simulated I/O failure on second real write")
                return real_apply_patch(patch, root, dry_run=dry_run, texts=texts)

            with mock.patch.object(apply_module, "apply_patch", side_effect=flaky_apply_patch):
                with self.assertRaises(RuntimeError):
                    apply_module.apply_all([first, second], tree.root)

            self.assertEqual((tree.root / "a.cu").read_text(encoding="utf-8"), before_a,
                             "first patch's real write must be rolled back")
            self.assertEqual((tree.root / "b.cu").read_text(encoding="utf-8"), before_b)

    def test_rollback_completes_even_if_a_later_write_makes_a_path_newly_unsafe(self):
        """Adversarial-review follow-up: _restore() used to re-resolve every
        backed-up path via resolve_contained_target() during rollback. If an
        earlier patch's real write left something (e.g. a symlink) that
        makes a LATER file's path resolve unsafely, that second resolution
        would raise and abort the loop with earlier files still unrestored.
        Snapshotting the resolved Path once, before any real write, removes
        the re-resolution entirely -- rollback must fully restore both files
        even when a symlink appears mid-run."""
        import bigcherry.patch.apply as apply_module

        with TempTree("a.cu", "void f(int a) {\n    body();\n}\n") as tree:
            (tree.root / "b.cu").write_text("void g(int a) {\n    body();\n}\n",
                                             encoding="utf-8", newline="")
            before_a = (tree.root / "a.cu").read_text(encoding="utf-8")
            before_b = (tree.root / "b.cu").read_text(encoding="utf-8")

            first = FilePatch(
                path="a.cu",
                edits=(Edit(id="p1", anchor=r"^void f\(int a\) \{$", mode="replace",
                            text="void f(int a, int b) {",
                            guard=r"void f\(int a, int b\) \{"),),
            )
            second = FilePatch(
                path="b.cu",
                edits=(Edit(id="p2", anchor=r"^void g\(int a\) \{$", mode="replace",
                            text="void g(int a, int b) {",
                            guard=r"void g\(int a, int b\) \{"),),
            )

            real_apply_patch = apply_module.apply_patch
            call_count = {"n": 0}

            def malicious_second_write(patch, root, *, dry_run=False, texts=None):
                if not dry_run:
                    call_count["n"] += 1
                    if call_count["n"] == 2:
                        # Simulate the attack: something (this patch's own
                        # write, or an unrelated concurrent mutation) makes
                        # b.cu's path unsafe to re-resolve, THEN fails.
                        try:
                            (tree.root / "b.cu").unlink()
                            (tree.root / "b.cu").symlink_to(tree.root.parent, target_is_directory=True)
                        except OSError:
                            pass  # symlinks unavailable in this environment -- still assert below
                        raise RuntimeError("simulated failure after planting an unsafe path")
                return real_apply_patch(patch, root, dry_run=dry_run, texts=texts)

            with mock.patch.object(apply_module, "apply_patch", side_effect=malicious_second_write):
                with self.assertRaises(RuntimeError):
                    apply_module.apply_all([first, second], tree.root)

            # a.cu (the first, already-real-written file) must still be
            # fully restored regardless of what happened to b.cu's path.
            self.assertEqual((tree.root / "a.cu").read_text(encoding="utf-8"), before_a)
            if (tree.root / "b.cu").is_symlink():
                (tree.root / "b.cu").unlink()
                (tree.root / "b.cu").write_text(before_b, encoding="utf-8", newline="")
            else:
                self.assertEqual((tree.root / "b.cu").read_text(encoding="utf-8"), before_b)

    def test_rollback_refuses_to_follow_a_symlink_to_an_outside_regular_file(self):
        """Adversarial-review follow-up: the prior regression only tested a
        symlink to a DIRECTORY, which write_text() would have failed on
        anyway (IsADirectoryError) even without any fix -- it never proved
        the real attack, a symlink to an outside REGULAR file, which a
        plain write_text() would happily follow and overwrite. _restore()
        must refuse to write through such a symlink (O_NOFOLLOW), not
        silently corrupt the outside file."""
        import bigcherry.patch.apply as apply_module

        with TempTree("a.cu", "void f(int a) {\n    body();\n}\n") as tree:
            (tree.root / "b.cu").write_text("void g(int a) {\n    body();\n}\n",
                                             encoding="utf-8", newline="")
            outside_dir = tree.root.parent / f"outside-{id(tree)}"
            outside_dir.mkdir(exist_ok=True)
            outside_file = outside_dir / "secret.txt"
            outside_file.write_text("do not touch\n", encoding="utf-8")
            self.addCleanup(lambda: outside_file.unlink(missing_ok=True))
            self.addCleanup(lambda: outside_dir.rmdir() if outside_dir.is_dir() and not any(outside_dir.iterdir()) else None)

            first = FilePatch(
                path="a.cu",
                edits=(Edit(id="p1", anchor=r"^void f\(int a\) \{$", mode="replace",
                            text="void f(int a, int b) {",
                            guard=r"void f\(int a, int b\) \{"),),
            )
            second = FilePatch(
                path="b.cu",
                edits=(Edit(id="p2", anchor=r"^void g\(int a\) \{$", mode="replace",
                            text="void g(int a, int b) {",
                            guard=r"void g\(int a, int b\) \{"),),
            )

            real_apply_patch = apply_module.apply_patch
            call_count = {"n": 0}

            def malicious_second_write(patch, root, *, dry_run=False, texts=None):
                if not dry_run:
                    call_count["n"] += 1
                    if call_count["n"] == 2:
                        (tree.root / "b.cu").unlink()
                        try:
                            (tree.root / "b.cu").symlink_to(outside_file)
                        except OSError:
                            self.skipTest("symlink creation not permitted in this environment")
                        raise RuntimeError("simulated failure after planting a file symlink")
                return real_apply_patch(patch, root, dry_run=dry_run, texts=texts)

            with mock.patch.object(apply_module, "apply_patch", side_effect=malicious_second_write):
                with self.assertRaises(RuntimeError):
                    apply_module.apply_all([first, second], tree.root)

            # The real attack: outside_file must NOT have been overwritten
            # with b.cu's restored contents.
            self.assertEqual(outside_file.read_text(encoding="utf-8"), "do not touch\n")

    def test_retry_after_rollback_succeeds_without_manual_cleanup(self):
        with TempTree("a.cu", "void f(int a) {\n    body();\n}\n") as tree:
            patch = FilePatch(
                path="a.cu",
                edits=(Edit(id="p1", anchor=r"^void f\(int a\) \{$", mode="replace",
                            text="void f(int a, int b) {",
                            guard=r"void f\(int a, int b\) \{"),),
            )
            results = apply_all([patch], tree.root)
            self.assertTrue(all(r.ok for r in results))
            self.assertIn("void f(int a, int b) {", tree.read())


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


class TestPatchTargetContainment(unittest.TestCase):
    """PA06 (source/patch identity hardening L0.1): a patch target must be
    contained by the source root -- no absolute path, `..` escape, or
    symlink-mediated redirection outside it."""

    def test_relative_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(PatchError):
                resolve_contained_target(root, "../outside.txt")

    def test_absolute_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(PatchError):
                resolve_contained_target(root, str(root.parent / "outside.txt"))

    def test_nested_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(PatchError):
                resolve_contained_target(root, "a/b/../../../outside.txt")

    def test_intermediate_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as outside_tmp, tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = Path(outside_tmp)
            link = root / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation not permitted in this environment")
            with self.assertRaises(PatchError):
                resolve_contained_target(root, "linked/escape.txt")

    def test_final_symlink_target_rejected(self):
        with tempfile.TemporaryDirectory() as outside_tmp, tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside_file = Path(outside_tmp) / "secret.txt"
            outside_file.write_text("secret", encoding="utf-8")
            link = root / "leaf.txt"
            try:
                link.symlink_to(outside_file)
            except OSError:
                self.skipTest("symlink creation not permitted in this environment")
            with self.assertRaises(PatchError):
                resolve_contained_target(root, "leaf.txt")

    def test_normal_nested_target_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = resolve_contained_target(root, "nested/dir/file.txt")
            self.assertEqual(target, (root.resolve(strict=True) / "nested" / "dir" / "file.txt"))

    def test_rejection_occurs_before_any_mutation(self):
        with TempTree("keep.txt", "original") as tree:
            patch = FilePatch(
                path="../outside.txt",
                edits=(Edit(id="e", anchor=r"^original$", text="\nadded", guard=r"^added$"),),
            )
            with self.assertRaises(PatchError):
                apply_patch(patch, tree.root)
            self.assertEqual(tree.read(), "original")
            self.assertFalse((tree.root.parent / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
