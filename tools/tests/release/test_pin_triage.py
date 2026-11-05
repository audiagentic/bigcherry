"""Read-only pin-triage over a real temporary bare upstream mirror."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bigcherry.cli import engine_scope, main
from bigcherry.core import components, paths
from bigcherry.release import pin_triage

REWRITE = paths.REPO_ROOT / "engines" / "radiance" / "components" / "libr3" / "r3_rewrite.py"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, text=True, capture_output=True).stdout.strip()


class PinTriageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "upstream"
        self.source.mkdir()
        git(self.source, "init", "-q")
        self._source("line one\nlaunch(a);\nline three\n")
        self._put("core/engine.cpp", "old core\n")
        self._commit("initial upstream")
        git(self.source, "tag", "v1")
        self._source("line ONE\nlaunch(a);\nline three\n")
        self._put("core/engine.cpp", "new core\n")
        self._commit("runtime: adjust engine")
        git(self.source, "tag", "v2")
        self.mirror = self.root / "upstream.git"
        subprocess.run(["git", "clone", "--quiet", "--bare", str(self.source), str(self.mirror)], check=True)

        self.component_root = self.root / "libr3"
        (self.component_root / "native").mkdir(parents=True)
        shutil.copyfile(REWRITE, self.component_root / "r3_rewrite.py")
        (self.component_root / "rewrites.txt").write_text("launch(a); ==> launch_gfx11(a);\n")
        (self.component_root / "native" / "k.hip.rw").write_text(
            "@@ find 1\nline one\n@@ replace\nline 1 native\n@@ end\n"
            "@@ find 1\nline 1 native\n@@ replace\nline 2 native\n@@ end\n"
        )
        self.component = components.Component(
            name="libfix", engine="radiance", root=self.component_root,
            targets=("gfx1100",), built_against="v1", outputs=("kernels/x.so",), settings={},
        )
        self.recipes = self.root / "recipes.toml"
        self.recipes.write_text('pinned = "v1"\n')
        self.vendor = self.root / "vendor"
        self.vendor.mkdir()
        (self.vendor / "do-not-edit").write_text("untouched")
        self.context = SimpleNamespace(
            engine="radiance", upstream_repo=self.mirror,
            vendor_root=self.vendor, patches_root=self.root / "patches",
        )
        self.descriptor = SimpleNamespace(patch_id="radtest")
        self.patches = [
            patch.object(pin_triage, "recipes_path", return_value=self.recipes),
            patch.object(pin_triage.components, "load_all", return_value={"libfix": self.component}),
            patch.object(pin_triage.patch_registry, "load_registry", return_value=SimpleNamespace(descriptors=[self.descriptor])),
            patch.object(pin_triage.patch_registry, "load_implementation", return_value=[SimpleNamespace(path="core/engine.cpp")]),
        ]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    def _put(self, file: str, content: str) -> None:
        path = self.source / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def _source(self, content: str) -> None:
        self._put("libr4d/k.hip", content)

    def _commit(self, message: str) -> None:
        git(self.source, "add", ".")
        git(self.source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", message)

    def test_changed_anchors_patch_targets_exact_drift_and_read_only(self):
        state = git(self.source, "status", "--porcelain")
        vendor = (self.vendor / "do-not-edit").read_bytes()
        before_source = (self.source / "libr4d" / "k.hip").read_bytes()
        report = pin_triage.run(engine="radiance", target_ref="v2", context=self.context)
        self.assertEqual(report["commit_count"], 1)
        self.assertEqual(report["verdict"], "patches-affected")
        self.assertEqual(report["rewrite_anchors"]["libr4d/k.hip"], ["libfix/native/k.hip.rw", "libfix/rewrites.txt"])
        self.assertEqual(report["patch_anchors"]["core/engine.cpp"], ["radtest"])
        self.assertTrue(report["core_commit_subjects"][0].endswith("runtime: adjust engine"))
        self.assertEqual(report["counts"]["drift_failures"], 2)
        self.assertEqual(list(report["drift_failures"]), ["libfix/native/k.hip.rw"])
        self.assertIn("block 1", report["drift_details"][0]["failure"])
        self.assertIn("block 2", report["drift_details"][1]["failure"])
        self.assertTrue(all(item["chained"] for item in report["drift_details"]))
        self.assertEqual(report["drift_details"][0]["closest"][0]["text"], "line ONE")
        self.assertEqual(git(self.source, "status", "--porcelain"), state)
        self.assertEqual((self.vendor / "do-not-edit").read_bytes(), vendor)
        self.assertEqual((self.source / "libr4d" / "k.hip").read_bytes(), before_source)
        self.assertFalse((self.vendor / "libr4d").exists())

    def test_verdict_drift_only_when_no_patch_target_moves(self):
        with patch.object(pin_triage.patch_registry, "load_registry", return_value=SimpleNamespace(descriptors=[])):
            report = pin_triage.run(engine="radiance", target_ref="v2", context=self.context)
        self.assertEqual(report["verdict"], "drift-only")
        self.assertEqual(report["counts"]["patch_target_files"], 0)
        self.assertEqual(report["counts"]["drift_failures"], 2)

    def test_same_pin_has_clean_verdict(self):
        report = pin_triage.run(engine="radiance", target_ref="v1", context=self.context)
        self.assertEqual(report["verdict"], "clean")
        self.assertEqual(report["counts"]["changed_files"], 0)
        self.assertFalse(report["drift_details"])

    def test_json_cli_parses_and_scope_is_engine(self):
        self.assertEqual(engine_scope.scope("pin-triage"), engine_scope.ENGINE)
        args = main.build_parser().parse_args(["--engine", "radiance", "pin-triage", "v2", "--json"])
        engine_scope.resolve(args, main.build_parser())
        with patch.object(pin_triage.ProjectContext, "resolve", return_value=self.context):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(args.func(args), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["verdict"], "patches-affected")
        self.assertEqual(report["drift_details"][0]["closest"][0]["text"], "line ONE")

    def test_missing_mirror_is_environment_error_not_a_drift_verdict(self):
        context = SimpleNamespace(**{**vars(self.context), "upstream_repo": self.root / "absent"})
        with self.assertRaisesRegex(pin_triage.PinTriageError, "bare mirror"):
            pin_triage.run(engine="radiance", target_ref="v2", context=context)


if __name__ == "__main__":
    unittest.main()
