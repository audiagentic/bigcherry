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


if __name__ == "__main__":
    unittest.main()
