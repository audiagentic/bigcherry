"""PPS03: the tag vocabulary in registry.PATCH_TAGS and the human-facing
table in docs/reference/patches/PATCH_AUTHORING.md must never silently
drift -- same pattern as TOOL_DISPOSITION.md vs
test_tooling_boundaries.py's row-count assertion."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from bigcherry.patch import registry as patch_registry

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = REPO_ROOT / "docs" / "reference" / "patches" / "PATCH_AUTHORING.md"
_TAG_ROW = re.compile(r"^\|\s*`(?P<tag>[a-z0-9.-]+)`(?:\s*/\s*`(?P<tag2>[a-z0-9.-]+)`)*")


def _doc_tags() -> set[str]:
    text = DOC_PATH.read_text(encoding="utf-8")
    start = text.index("## Tags")
    section = text[start:]
    tags: set[str] = set()
    for line in section.splitlines():
        match = _TAG_ROW.match(line)
        if not match:
            continue
        # A row's first column may list several slash-separated tags
        # sharing one description (e.g. "`gfx1100` / `gfx1101` / ...").
        for code in re.findall(r"`([a-z0-9.-]+)`", line.split("|")[1]):
            tags.add(code)
    return tags


class PatchTagsRegistryTests(unittest.TestCase):
    def test_doc_table_matches_code_constant_exactly(self) -> None:
        doc_tags = _doc_tags()
        self.assertTrue(doc_tags, "no tag rows parsed from PATCH_AUTHORING.md's ## Tags section")
        self.assertEqual(
            doc_tags,
            set(patch_registry.PATCH_TAGS),
            "docs/reference/patches/PATCH_AUTHORING.md's ## Tags table and "
            "registry.PATCH_TAGS have drifted -- update both in the same change",
        )

    def test_every_real_patch_tag_is_in_the_vocabulary(self) -> None:
        registry = patch_registry.load_registry(REPO_ROOT / "patches")
        violations = [
            f"{descriptor.patch_id}: {tag!r}"
            for descriptor in registry.descriptors
            for tag in descriptor.tags
            if tag not in patch_registry.PATCH_TAGS
        ]
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
