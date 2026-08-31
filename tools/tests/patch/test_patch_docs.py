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


class ParseSummaryHeaderTests(unittest.TestCase):
    def test_extracts_all_three_fields(self):
        header = patch_docs.parse_summary_header(
            "# x\n\n**Status:** validated\n**Group:** core\n**Plan item:** RD20\n\n## What it does\n"
        )
        self.assertEqual(header, {"status": "validated", "group": "core", "plan_item": "RD20"})

    def test_returns_none_when_header_is_missing(self):
        self.assertIsNone(patch_docs.parse_summary_header("# x\n\nno header here\n"))


class CheckSummaryConsistencyTests(unittest.TestCase):
    def _write_patch(self, root: Path, patch_id: str, *, state: str, group: str,
                      plan_item: str | None = None) -> None:
        # Every real production patch is a packaged directory ("<id>/patch.py"
        # + "<id>/patch.toml", patch.toml authoritative for state/group) --
        # matches that shape rather than the legacy flat "<id>.py" one.
        patch_dir = root / patch_id
        patch_dir.mkdir()
        (patch_dir / "patch.py").write_text('PATCHES = []\n', encoding="utf-8")
        (patch_dir / "patch.toml").write_text(
            "schema = 1\n"
            f'id = "{patch_id}"\n'
            "order = 100\n"
            f'group = "{group}"\n'
            f'state = "{state}"\n'
            'kind = "framework"\n'
            'origin = "local"\n'
            'backend = "agnostic"\n'
            + (f'plan-item = "{plan_item}"\n' if plan_item else "")
            + "plan-ids = []\nrequires = []\nconflicts = []\n"
            "requires-options = []\nforbids-options = []\nsubsystems = []\n"
            "hardware = []\nvalidation-architectures = []\nbackends = []\n",
            encoding="utf-8",
        )

    def test_flags_missing_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_patch(root, "0100_x", state="validated", group="core")
            problems = patch_docs.check_summary_consistency(root)
            self.assertEqual(problems, ["0100_x: missing SUMMARY.md"])

    def test_flags_status_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_patch(root, "0100_x", state="superseded", group="core")
            (root / "0100_x" / "SUMMARY.md").write_text(
                "# 0100_x\n\n**Status:** untested\n**Group:** core\n**Plan item:** none\n",
                encoding="utf-8",
            )
            problems = patch_docs.check_summary_consistency(root)
            self.assertEqual(len(problems), 1)
            self.assertIn("Status='untested'", problems[0])
            self.assertIn("STATE='superseded'", problems[0])

    def test_clean_when_everything_agrees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_patch(root, "0100_x", state="validated", group="core")
            (root / "0100_x" / "SUMMARY.md").write_text(
                "# 0100_x\n\n**Status:** validated\n**Group:** core\n**Plan item:** none\n",
                encoding="utf-8",
            )
            self.assertEqual(patch_docs.check_summary_consistency(root), [])


if __name__ == "__main__":
    unittest.main()
