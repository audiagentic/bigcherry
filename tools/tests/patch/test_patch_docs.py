"""tools.bigcherry.patch.docs -- per-patch SUMMARY.md merge into a release doc."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import docs as patch_docs  # noqa: E402
from bigcherry.patch import patchset  # noqa: E402


def _module(patch_id: str, path: Path, *, order: int = 1, group: str = "core",
            state: str = "validated") -> patchset.PatchModule:
    return patchset.PatchModule(
        patch_id=patch_id, path=path, order=order, group=group, state=state,
        upstream=None, content_hash="deadbeef",
    )


class ReadPatchSummaryTests(unittest.TestCase):
    def test_reads_the_real_summary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            patch_dir = Path(directory) / "0100_x"
            patch_dir.mkdir()
            (patch_dir / "SUMMARY.md").write_text("# 0100_x\n\nreal content\n", encoding="utf-8")
            module = _module("0100_x", patch_dir / "patch.py")
            self.assertIn("real content", patch_docs.read_patch_summary(module))

    def test_missing_summary_renders_a_visible_placeholder_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            patch_dir = Path(directory) / "0200_y"
            patch_dir.mkdir()
            module = _module("0200_y", patch_dir / "patch.py", state="untested")
            rendered = patch_docs.read_patch_summary(module)
            self.assertIn("No SUMMARY.md found", rendered)
            self.assertIn("0200_y", rendered)
            self.assertIn("untested", rendered)


class RenderReleaseDocTests(unittest.TestCase):
    def test_merges_in_order_with_pin_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_dir, second_dir = root / "0200_b", root / "0100_a"
            first_dir.mkdir()
            second_dir.mkdir()
            (first_dir / "SUMMARY.md").write_text("# 0200_b\n\nsecond patch\n", encoding="utf-8")
            (second_dir / "SUMMARY.md").write_text("# 0100_a\n\nfirst patch\n", encoding="utf-8")
            modules = [
                _module("0200_b", first_dir / "patch.py", order=200),
                _module("0100_a", second_dir / "patch.py", order=100),
            ]
            doc = patch_docs.render_release_doc(
                modules=modules,
                pin_info={"llama.cpp revision": "abc123"},
                selection_label="--recipe workstation",
            )
            self.assertIn("Selection: --recipe workstation", doc)
            self.assertIn("llama.cpp revision:** abc123", doc)
            self.assertIn("2 patch(es) included", doc)
            # order=100 (0100_a) must render before order=200 (0200_b)
            self.assertLess(doc.index("first patch"), doc.index("second patch"))


if __name__ == "__main__":
    unittest.main()
