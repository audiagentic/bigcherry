"""pin-release: plan naming, the record and notes phases on a scratch repository, and their idempotence."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.release import notes as _notes  # noqa: E402
from bigcherry.release import pin_release as _pr  # noqa: E402

_CONFIG = _notes.ReleaseConfig(
    tag_prefix="bc-", notes_dir="docs/releases/notes", ledger="ledger.ndjson", release_records="releases",
    recipes="config/recipes.toml", build_patch_sets=(), fallback_prefixes=(), ledger_sections=(),
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


class PinReleaseTests(unittest.TestCase):
    def _repo(self, td: str) -> Path:
        root = Path(td)
        _git(root, "init", "-q", "-b", "work")
        _git(root, "config", "user.email", "t@example.invalid")
        _git(root, "config", "user.name", "t")
        (root / "releases").mkdir()
        (root / "releases" / "b11474.json").write_text(json.dumps({"release_tag": "b11474", "notes": ""}), encoding="utf-8")
        (root / "releases" / "pin-transition.json").write_text("{}", encoding="utf-8")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "seed")
        return root

    def _plan(self, root: Path) -> _pr.Plan:
        with mock.patch.object(_notes, "load_config", return_value=_CONFIG):
            return _pr.make_plan(root, "b11474")

    def test_plan_names(self):
        with tempfile.TemporaryDirectory() as td:
            plan = self._plan(self._repo(td))
            self.assertEqual((plan.build, plan.release_tag, plan.version, plan.branch), (11474, "bc-11474.0.0", "11474.0.0", "work"))

    def test_plan_rejects_other_tags_and_main(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            with mock.patch.object(_notes, "load_config", return_value=_CONFIG):
                with self.assertRaises(_pr.PinReleaseError):
                    _pr.make_plan(root, "v1.2.3")
                _git(root, "checkout", "-q", "-b", "main")
                with self.assertRaises(_pr.PinReleaseError):
                    _pr.make_plan(root, "b11474")

    def test_record_needs_evidence_then_records_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            plan = self._plan(root)
            with self.assertRaises(_pr.PinReleaseError):
                _pr.phase_record(plan, "VERDICT consistent", "")
            self.assertTrue((root / "releases" / "pin-transition.json").exists())   # nothing was changed by the refusal
            self.assertTrue(_pr.phase_record(plan, "VERDICT consistent", "build b-x: smoke identical"))
            record = json.loads((root / "releases" / "b11474.json").read_text(encoding="utf-8"))
            self.assertIn("release 11474.0.0", record["notes"])
            self.assertIn("VERDICT consistent", record["notes"])
            self.assertIn("build b-x: smoke identical", record["notes"])
            self.assertFalse((root / "releases" / "pin-transition.json").exists())
            self.assertEqual(_git(root, "status", "--porcelain"), "")
            self.assertIn("bc-11474.0.0 bump complete", _git(root, "log", "-1", "--format=%s"))
            # a second run changes nothing
            head = _git(root, "rev-parse", "HEAD")
            self.assertFalse(_pr.phase_record(plan, "VERDICT consistent", ""))
            self.assertEqual(head, _git(root, "rev-parse", "HEAD"))

    def test_notes_commit_carries_release_as_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            plan = self._plan(root)

            def fake_write(repo_root, config, llama_tag, ref="HEAD", version=None):
                out = Path(repo_root) / "docs" / "releases" / "notes" / f"bc-{version}.md"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("# notes\n", encoding="utf-8")
                return out

            with mock.patch.object(_notes, "load_config", return_value=_CONFIG), mock.patch.object(_notes, "write_notes", side_effect=fake_write):
                self.assertTrue(_pr.phase_notes(plan))
                self.assertEqual(_git(root, "log", "-1", "--format=%s"), "chore: release notes bc-11474.0.0")
                self.assertIn("Release-As: 11474.0.0", _git(root, "log", "-1", "--format=%b"))
                head = _git(root, "rev-parse", "HEAD")
                self.assertFalse(_pr.phase_notes(plan))
                self.assertEqual(head, _git(root, "rev-parse", "HEAD"))

    def test_versions_on_one_pin(self):
        self.assertEqual(_pr.next_version([], 11474, None), "11474.0.0")
        self.assertEqual(_pr.next_version(["11402.0.0", "b11474"], 11474, None), "11474.0.0")
        self.assertEqual(_pr.next_version(["11474.0.0"], 11474, "minor"), "11474.1.0")
        self.assertEqual(_pr.next_version(["11474.0.0", "11474.1.0"], 11474, "patch"), "11474.1.1")
        self.assertEqual(_pr.next_version(["11474.0.0", "11474.1.1"], 11474, "minor"), "11474.2.0")
        with self.assertRaises(_pr.PinReleaseError):   # a released pin needs an explicit bump
            _pr.next_version(["11474.0.0"], 11474, None)
        with self.assertRaises(_pr.PinReleaseError):   # the first release of a pin is always .0.0
            _pr.next_version([], 11474, "minor")
        self.assertEqual(_notes.release_version("b11474"), "11474.0.0")
        self.assertEqual(_notes.release_version("b11474", "11474.2.1"), "11474.2.1")
        with self.assertRaises(_notes.ReleaseNotesError):
            _notes.release_version("b11474", "11402.1.0")

    def test_dry_run_lists_phases_and_changes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            head = _git(root, "rev-parse", "HEAD")
            lines = []
            with mock.patch.object(_notes, "load_config", return_value=_CONFIG):
                self.assertEqual(_pr.run(root, "b11474", "", through="notes", dry_run=True, log=lines.append), 0)
            self.assertEqual([line.split()[-1] for line in lines[1:]], ["gate", "record", "notes"])
            self.assertEqual(head, _git(root, "rev-parse", "HEAD"))


if __name__ == "__main__":
    unittest.main()
