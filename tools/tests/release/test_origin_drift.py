"""pin-triage origin drift (RCS23): the upstream units a component owns whole, against the last integrated commit."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bigcherry.release import pin_triage


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


class OriginDriftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / "work"
        self.work.mkdir()
        git(self.work, "init", "-q", "-b", "main")
        (self.work / "libr4d").mkdir()
        (self.work / "libr4d" / "r4d_unit.hip").write_text("// v1\n" * 40)
        git(self.work, "add", "-A")
        git(self.work, "commit", "-qm", "base")
        self.base = git(self.work, "rev-parse", "HEAD")
        self.component = Path(self.tmp.name) / "comp"
        self.component.mkdir()
        (self.component / "bc_unit.hip").write_text("// owned\n")

    def origins(self, **over):
        unit = {"name": "r4d_unit", "upstream-path": "libr4d/r4d_unit.hip", "owned-source": "bc_unit.hip", "base-commit": self.base,
                "last-integrated": self.base, "strategy": "whole-unit"}
        unit.update(over)
        body = "[[unit]]\n" + "".join(f'{k} = "{v}"\n' for k, v in unit.items())
        (self.component / "origins.toml").write_text(body)

    def drift(self):
        target = git(self.work, "rev-parse", "HEAD")
        loaded = {"bc-test": SimpleNamespace(root=self.component)}
        with mock.patch.object(pin_triage.components, "load_all", return_value=loaded):
            return pin_triage.origin_drift("radiance", self.work / ".git", target, root=Path("/unused"))

    def test_an_unchanged_origin_is_clean(self):
        self.origins()
        (self.work / "other.txt").write_text("x")
        git(self.work, "add", "-A")
        git(self.work, "commit", "-qm", "elsewhere")
        (item,) = self.drift()
        self.assertEqual((item["state"], item["commits"]), ("clean", []))

    def test_a_changed_origin_is_a_candidate_with_its_commits(self):
        self.origins()
        (self.work / "libr4d" / "r4d_unit.hip").write_text("// v2\n" * 40)
        git(self.work, "commit", "-qam", "libr4d: tune the unit")
        (item,) = self.drift()
        self.assertEqual(item["state"], "candidate")
        self.assertIn("libr4d: tune the unit", item["commits"][0])

    def test_a_missing_base_commit_blocks(self):
        self.origins(**{"base-commit": "0" * 40})
        (item,) = self.drift()
        self.assertEqual(item["state"], "blocked")
        self.assertIn("not in the upstream mirror", item["reason"])

    def test_a_wrong_path_blocks(self):
        self.origins(**{"upstream-path": "libr4d/r4d_nothing.hip"})
        (item,) = self.drift()
        self.assertEqual(item["state"], "blocked")
        self.assertIn("wrong path", item["reason"])

    def test_a_rename_blocks_and_names_the_new_path(self):
        self.origins()
        git(self.work, "mv", "libr4d/r4d_unit.hip", "libr4d/r4d_unit_renamed.hip")
        git(self.work, "commit", "-qm", "rename")
        (item,) = self.drift()
        self.assertEqual(item["state"], "blocked")
        self.assertIn("renamed libr4d/r4d_unit.hip to libr4d/r4d_unit_renamed.hip", item["reason"])

    def test_a_missing_owned_source_or_field_blocks(self):
        self.origins(**{"owned-source": "nope.hip"})
        self.assertIn("owned source", self.drift()[0]["reason"])
        self.origins(strategy="")
        self.assertIn("lacks strategy", self.drift()[0]["reason"])

    def test_the_real_origins_file_names_files_that_exist(self):
        from bigcherry.core import components, engines, paths
        comp = components.load_all(paths.REPO_ROOT, engines.RADIANCE)["bc-iqr"]
        import tomllib
        for unit in tomllib.loads((comp.root / "origins.toml").read_text())["unit"]:
            self.assertTrue((comp.root / unit["owned-source"]).is_file())
            self.assertEqual(unit["strategy"], "whole-unit")


if __name__ == "__main__":
    unittest.main()
