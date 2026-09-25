"""port_diff: generated edits reproduce the target exactly and are idempotent."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import port_diff  # noqa: E402
from bigcherry.patch.apply import FilePatch  # noqa: E402

OLD = """// header comment
static int f(int x) {
    return x + 1; // add one
}

static int g(int x) {
    return x * 2;
}

static int h(int x) {
    return x - 3;
}
"""


class PortDiffTests(unittest.TestCase):
    def _port(self, new: str) -> FilePatch:
        edits = port_diff.generate_edits(OLD, new, path="a.cu", prefix="t")
        patch = FilePatch(path="a.cu", edits=tuple(edits))
        port_diff.verify(OLD, new, patch)
        return patch

    def test_replacement_inside_commented_code(self):
        new = OLD.replace("return x + 1; // add one", "return x + 2; // add two")
        patch = self._port(new)
        self.assertEqual(len(patch.edits), 1)

    def test_insertion_and_deletion(self):
        new = OLD.replace("static int g(int x) {\n    return x * 2;\n}\n\n", "")
        new = new.replace("static int h", "static int k(int x) {\n    return -x;\n}\n\nstatic int h")
        self._port(new)

    def test_pure_deletion_gets_a_guard(self):
        new = OLD.replace("    return x - 3;\n", "")
        patch = self._port(new)
        self.assertTrue(all(edit.guard for edit in patch.edits))

    def test_identical_files_are_rejected(self):
        with self.assertRaises(port_diff.PortDiffError):
            port_diff.generate_edits(OLD, OLD, path="a.cu", prefix="t")

    def test_render_round_trips_as_python(self):
        new = OLD.replace("x * 2", "x * 4")
        patch = self._port(new)
        namespace: dict[str, object] = {}
        exec("from bigcherry.patcher import Edit, FilePatch\n" + port_diff.render(patch, "P"), namespace)
        rendered = namespace["P"]
        self.assertEqual([e.anchor for e in rendered.edits], [e.anchor for e in patch.edits])


class CreatedFileTests(unittest.TestCase):
    NEW = "#pragma once\n// created by the patch\nstatic int created_function(int x) { return x; }\n"

    def _patch(self) -> FilePatch:
        edits = port_diff.generate_edits("", self.NEW, path="sub/new.cuh", prefix="c")
        return FilePatch(path="sub/new.cuh", edits=tuple(edits), create=True)

    def test_created_file_round_trips(self):
        port_diff.verify("", self.NEW, self._patch())

    def test_create_fails_closed_on_a_colliding_upstream_file(self):
        import tempfile

        from bigcherry.patch.apply import apply_patch

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sub" / "new.cuh"
            target.parent.mkdir(parents=True)
            target.write_text("// upstream grew this file\nint other;\n", encoding="utf-8")
            result = apply_patch(self._patch(), Path(tmp))
            self.assertFalse(result.ok)
            self.assertIn("already exists upstream", result.failed[0].detail)

    def test_rollback_removes_a_created_file(self):
        import tempfile

        from bigcherry.patch.apply import Edit, apply_all

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = FilePatch(path="missing.cu", edits=(Edit(id="x", anchor="nothing", text="y", guard="zzz_never"),))
            results = apply_all([self._patch(), broken], root)
            self.assertFalse(all(r.ok for r in results))
            self.assertFalse((root / "sub" / "new.cuh").exists())


if __name__ == "__main__":
    unittest.main()
