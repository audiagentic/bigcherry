"""VA08 (resolved-pin-SHA slice) tests for
cli/patch.py::cmd_patch_verify_evidence(): the pin must be resolved to its
real immutable commit SHA (local-only, no fetch) and threaded into
validation_evidence_statuses(), failing closed with a real CLI error if
the configured pin cannot resolve locally -- rather than silently
skipping resolved-SHA freshness checking.
"""

from __future__ import annotations

import sys
import unittest
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli import patch as cli_patch # noqa: E402
from bigcherry.source.workspace import WorkspaceError # noqa: E402


def _args(**overrides) -> Namespace:
    base = {"patch_id": None, "json": False, "no_legacy_grandfather": False}
    base.update(overrides)
    return Namespace(**base)


class ResolvedPinShaTests(unittest.TestCase):
    def test_campaign_mirror_is_used_without_vendor_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            mirror = Path(temporary) / "upstream.git"
            mirror.mkdir()
            with patch("bigcherry.core.config.load", return_value=Namespace(pinned="pin")), \
                 patch("bigcherry.patch.patchset.catalog", return_value=[]), \
                 patch("bigcherry.core.context.ProjectContext.resolve", return_value=Namespace(upstream_repo=mirror)), \
                 patch("bigcherry.source.workspace.UpstreamRepository") as repository, \
                 patch("bigcherry.patch.catalog.validation_evidence_statuses", return_value={}):
                repository.return_value.resolve_ref.return_value = "a" * 40
                self.assertEqual(cli_patch.cmd_patch_verify_evidence(_args()), 0)
            repository.assert_called_once_with(mirror)

    def test_resolved_sha_is_passed_to_validation_evidence_statuses(self):
        captured = {}

        def fake_statuses(patch_ids, **kwargs):
            captured.update(kwargs)
            return {}

        with patch("bigcherry.core.config.load") as fake_load, \
             patch("bigcherry.patch.patchset.catalog", return_value=[]), \
             patch("bigcherry.source.workspace.UpstreamRepository") as fake_repo_cls, \
             patch("bigcherry.patch.catalog.validation_evidence_statuses", side_effect=fake_statuses):
            fake_load.return_value = Namespace(pinned="b10692")
            fake_repo_cls.return_value.resolve_ref.return_value = "f" * 40
            rc = cli_patch.cmd_patch_verify_evidence(_args())

        self.assertEqual(rc, 0)
        self.assertEqual(captured.get("pinned_ref"), "b10692")
        self.assertEqual(captured.get("resolved_base_revision"), "f" * 40)

    def test_unresolvable_pin_fails_closed_with_cli_error(self):
        with patch("bigcherry.core.config.load") as fake_load, \
             patch("bigcherry.patch.patchset.catalog", return_value=[]), \
             patch("bigcherry.source.workspace.UpstreamRepository") as fake_repo_cls, \
             patch("bigcherry.patch.catalog.validation_evidence_statuses") as fake_statuses:
            fake_load.return_value = Namespace(pinned="b10692")
            fake_repo_cls.return_value.resolve_ref.side_effect = WorkspaceError("no such ref")
            rc = cli_patch.cmd_patch_verify_evidence(_args())

        self.assertEqual(rc, 2)
        fake_statuses.assert_not_called()


if __name__ == "__main__":
    unittest.main()
