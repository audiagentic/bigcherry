"""Which release-please package a changed file counts for (release-please-config.json).

A commit counts for the platform package ("." with exclude-paths) when it changes a file outside every excluded path;
a commit's `Release-As:` footer then sets that package's next version. The engines' generated release notes
(docs/releases/notes/bc-<engine>-<version>.md, committed with a `Release-As:` footer by `bigcherry pin-release`) must not
count for the platform line: when they did, the llamacpp notes commit bb2896d9f forced the platform release PR to
"11552.0.0" (PR #562).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONFIG = json.loads((REPO / "release-please-config.json").read_text(encoding="utf-8"))
PACKAGES = CONFIG["packages"]


def packages_for(path: str) -> set[str]:
    """The packages a change to `path` counts for: under the package path and not under one of its exclude-paths."""
    out = set()
    for root, body in PACKAGES.items():
        base = "" if root == "." else root.rstrip("/") + "/"
        if not path.startswith(base):
            continue
        if any(path == ex or path.startswith(ex.rstrip("/") + "/") for ex in body.get("exclude-paths", [])):
            continue
        out.add(body["component"])
    return out


class PlatformAttributionTests(unittest.TestCase):
    def test_engine_release_notes_do_not_count_for_the_platform_line(self) -> None:
        for note in ("docs/releases/notes/bc-llamacpp-11552.0.0.md", "docs/releases/notes/bc-radiance-1160.0.0.md",
                     "docs/releases/notes/bc-11474.0.0.md"):
            with self.subTest(note=note):
                self.assertNotIn("bc-platform", packages_for(note))

    def test_engine_trees_still_do_not_count_for_the_platform_line(self) -> None:
        self.assertEqual(packages_for("engines/radiance/patches/rad0011_plan_memory/patch.py"), {"bc-radiance"})
        self.assertEqual(packages_for("engines/llamacpp/CHANGELOG.md"), {"bc-llamacpp"})

    def test_real_platform_changes_still_count(self) -> None:
        for path in ("tools/bigcherry/cli/main.py", "docs/releases/PLATFORM_CHANGELOG.md", ".github/workflows/release-please.yml",
                     "docs/planning/active/run-serving-configuration/RSC08.md"):
            with self.subTest(path=path):
                self.assertEqual(packages_for(path), {"bc-platform"})

    def test_every_notes_file_in_the_tree_is_an_engine_note(self) -> None:
        # a platform note placed in docs/releases/notes would be silently excluded: the platform's changelog is elsewhere
        for note in (REPO / "docs/releases/notes").glob("*.md"):
            with self.subTest(note=note.name):
                self.assertNotRegex(note.name, r"^bc-platform-")


if __name__ == "__main__":
    unittest.main()
