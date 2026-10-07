"""Release notes generator and the release-please wiring it feeds."""

from __future__ import annotations

import json
import re
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.release import notes  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]


class ReleaseNotesTests(unittest.TestCase):
    def test_config_loads_and_names_releases_bc_prefix(self):
        config = notes.load_config(_REPO)
        self.assertEqual(config.tag_prefix, "bc-")
        self.assertIn("validated-enhancements", config.build_patch_sets)
        self.assertTrue(config.ledger_sections)

    def test_event_time_uses_the_timestamp_and_falls_back_to_the_event_id(self):
        self.assertEqual(notes._event_time({"timestamp-utc": "2026-10-05T08:25:39Z", "event-id": "chg_x"}),
                         datetime(2026, 10, 5, 8, 25, 39, tzinfo=timezone.utc))
        self.assertEqual(notes._event_time({"event-id": "chg_20260813_093846_some-slug_1234"}),
                         datetime(2026, 8, 13, 9, 38, 46, tzinfo=timezone.utc))

    def test_ledger_filter_keeps_only_events_after_the_previous_release(self):
        events = [{"event-id": "chg_20260901_000000_a_1", "change-class": "feature"},
                  {"event-id": "chg_20261001_000000_b_2", "change-class": "feature"}]
        after = datetime(2026, 9, 15, tzinfo=timezone.utc)
        kept = [e for e in events if notes._event_time(e) > after]
        self.assertEqual([e["event-id"] for e in kept], ["chg_20261001_000000_b_2"])


class ReleasePleaseWiringTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((_REPO / "release-please-config.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((_REPO / ".release-please-manifest.json").read_text(encoding="utf-8"))
        self.workflow = (_REPO / ".github/workflows/release-please.yml").read_text(encoding="utf-8")

    def test_release_please_tags_bc_dash_semver_without_v(self):
        package = self.config["packages"]["."]
        self.assertEqual(package["component"], "bc")
        self.assertTrue(self.config["include-component-in-tag"])
        self.assertFalse(self.config["include-v-in-tag"])
        self.assertEqual(self.config["tag-separator"], "-")

    def test_manifest_version_major_is_a_llama_build_number(self):
        self.assertRegex(self.manifest["."], r"^\d{4,6}\.\d+\.\d+$")

    def test_workflow_adds_the_bc_b_build_tag_and_publishes_the_generated_notes(self):
        prefix = notes.load_config(_REPO).tag_prefix
        self.assertIn(f'bc_tag="{prefix}b${{LLAMA_BUILD}}"', self.workflow)
        self.assertIn("steps.release.outputs.major", self.workflow)
        # every release publishes the notes file named after its own tag; only a pin's first release gets bc-b<build>
        self.assertIn('notes="docs/releases/notes/${RP_TAG}.md"', self.workflow)
        self.assertIn(f'if [ "${{RP_TAG}}" = "{prefix}${{LLAMA_BUILD}}.0.0" ]; then', self.workflow)
        self.assertLess(self.workflow.index(f'if [ "${{RP_TAG}}" = "{prefix}${{LLAMA_BUILD}}.0.0" ]; then'),
                        self.workflow.index(f'bc_tag="{prefix}b${{LLAMA_BUILD}}"'))
        self.assertIn('--notes-file "${notes}"', self.workflow)
        self.assertTrue(re.search(r"branches:\s*\[main\]", self.workflow))
        self.assertEqual(notes.load_config(_REPO).notes_dir, "docs/releases/notes")


if __name__ == "__main__":
    unittest.main()
