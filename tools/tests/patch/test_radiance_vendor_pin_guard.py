"""PRQ04: a synthetic vendor checkout may never silently drift from its pin."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign import build as campaign_build
from bigcherry.core import paths
from bigcherry.core.context import ProjectContext
from bigcherry.patch import rebase
from bigcherry.patch.apply import PATCH_APPLICATION_SEMANTICS_VERSION
from bigcherry.source.workspace import (
    SourcePlan,
    VendorPinMismatchError,
    resolve_vendor_checkout_revision,
)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


class VendorPinGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="prq04-pin-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.checkout = self.root / "upstream"
        git(self.root, "init", str(self.checkout))
        git(self.checkout, "config", "user.email", "fixture@example.invalid")
        git(self.checkout, "config", "user.name", "Fixture")
        (self.checkout / "source.txt").write_text("pinned\n", encoding="utf-8")
        git(self.checkout, "add", "source.txt")
        git(self.checkout, "commit", "-m", "pin")
        self.pinned_sha = git(self.checkout, "rev-parse", "HEAD")
        git(self.checkout, "tag", "v1.17.0")
        self.recipes = self.root / "recipes.toml"
        self.recipes.write_text('pinned = "v1.17.0"\n', encoding="utf-8")

        (self.checkout / "source.txt").write_text("new head\n", encoding="utf-8")
        git(self.checkout, "add", "source.txt")
        git(self.checkout, "commit", "-m", "drift")
        self.head_sha = git(self.checkout, "rev-parse", "HEAD")

    def assert_revisions_named(self, error: BaseException) -> None:
        message = str(error)
        self.assertIn(self.head_sha, message)
        self.assertIn(self.pinned_sha, message)
        self.assertIn("v1.17.0", message)

    def test_resolver_rejects_drift_and_accepts_restored_pin(self) -> None:
        with self.assertRaises(VendorPinMismatchError) as caught:
            resolve_vendor_checkout_revision("radiance", self.checkout, self.recipes)
        self.assert_revisions_named(caught.exception)
        git(self.checkout, "checkout", "--detach", "v1.17.0")
        self.assertEqual(
            resolve_vendor_checkout_revision("radiance", self.checkout, self.recipes),
            self.pinned_sha,
        )

    def test_rebase_fresh_and_report_freshness_both_refuse_drift(self) -> None:
        with (
            patch.object(rebase, "_recipes_path", return_value=self.recipes),
            patch.object(
                rebase, "_resolve_selector_snapshot",
                return_value=SimpleNamespace(identity=object()),
            ),
        ):
            with self.assertRaises(rebase.RebaseCheckError) as caught:
                rebase.run_rebase_check("radiance", self.checkout, source_name="radiance")
            self.assert_revisions_named(caught.exception)
            report = {
                "engine": "radiance",
                "schema_version": rebase.REPORT_SCHEMA_VERSION,
                "patch_application_semantics_version": PATCH_APPLICATION_SEMANTICS_VERSION,
            }
            with self.assertRaises(rebase.StaleRebaseReportError) as caught:
                rebase._require_fresh("radiance", report, self.checkout)
            self.assert_revisions_named(caught.exception)

    def test_build_materialization_refuses_drift_before_cache_or_copy(self) -> None:
        context = ProjectContext.resolve(
            engine="radiance",
            project_root=paths.REPO_ROOT,
            config_path=self.recipes,
            work_root=self.root / "work",
            upstream_repo=self.checkout,
        )
        plan = SourcePlan(upstream_revision="v1.17.0", overlay_enabled=False, patch_ids=())
        with patch.object(ProjectContext, "vendor_root", new_callable=PropertyMock) as vendor:
            vendor.return_value = self.checkout
            with self.assertRaises(VendorPinMismatchError) as caught:
                campaign_build.materialize_source(context, plan, allow_dirty_bigcherry=True)
            self.assert_revisions_named(caught.exception)
        self.assertFalse((self.root / "work" / "sources").exists())


if __name__ == "__main__":
    unittest.main()
