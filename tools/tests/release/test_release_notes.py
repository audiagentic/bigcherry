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
from bigcherry.release import pin_release  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]


class ReleaseNotesTests(unittest.TestCase):
    def test_config_loads_and_names_the_llamacpp_release_line(self):
        config = notes.load_config(_REPO)
        self.assertEqual(config.tag_prefix, "bc-llamacpp-")
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

    def test_each_release_line_has_its_own_component_and_tag(self):
        packages = self.config["packages"]
        self.assertEqual(sorted(packages), [".", "engines/llamacpp"])
        self.assertEqual(packages["engines/llamacpp"]["component"], "bc-llamacpp")
        self.assertEqual(packages["."]["component"], "bc-platform")
        # an engine-only commit must not also move the platform line
        self.assertEqual(packages["."]["exclude-paths"], ["engines"])
        self.assertTrue(self.config["separate-pull-requests"])
        self.assertTrue(self.config["include-component-in-tag"])
        self.assertFalse(self.config["include-v-in-tag"])
        self.assertEqual(self.config["tag-separator"], "-")
        # the release tooling and release-please must agree on the llama.cpp line's tag
        self.assertEqual(notes.load_config(_REPO).tag_prefix, packages["engines/llamacpp"]["component"] + "-")
        self.assertEqual(pin_release.RELEASE_PACKAGE, "engines/llamacpp")

    def test_manifest_versions(self):
        self.assertEqual(sorted(self.manifest), [".", "engines/llamacpp"])
        # the llama.cpp line's major is a llama build number; the platform line is a plain semantic version
        self.assertRegex(self.manifest["engines/llamacpp"], r"^\d{4,6}\.\d+\.\d+$")
        self.assertRegex(self.manifest["."], r"^\d{1,3}\.\d+\.\d+$")

    def test_workflow_adds_the_readable_tag_and_publishes_the_generated_notes(self):
        prefix = notes.load_config(_REPO).tag_prefix
        self.assertIn("steps.release.outputs['engines/llamacpp--release_created'] == 'true'", self.workflow)
        self.assertIn("steps.release.outputs['engines/llamacpp--major']", self.workflow)
        # readable tag: <prefix>b<build>-r<N>, N = releases at that build before this one
        self.assertIn(f'count=$(git tag --list "{prefix}${{LLAMA_BUILD}}.*" | wc -l)', self.workflow)
        self.assertIn(f'readable="{prefix}b${{LLAMA_BUILD}}-r$((count - 1))"', self.workflow)
        self.assertIn('notes="docs/releases/notes/${RP_TAG}.md"', self.workflow)
        self.assertIn('--title "${readable}" --notes-file "${notes}"', self.workflow)
        self.assertIn("steps.release.outputs['.--release_created'] == 'true'", self.workflow)
        self.assertTrue(re.search(r"branches:\s*\[main\]", self.workflow))
        self.assertEqual(notes.load_config(_REPO).notes_dir, "docs/releases/notes")


if __name__ == "__main__":
    unittest.main()
