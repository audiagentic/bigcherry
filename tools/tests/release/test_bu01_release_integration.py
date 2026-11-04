"""BU01 release integration: pin-release receipt and patch-promote fail closed."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.release import pin_release as pr
from bigcherry.release import patch_promote as pp

ROOT = Path(__file__).resolve().parents[2]


class ReleaseIntegrationTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT.parent / "work" / "bu01-offline-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.repo = Path(self.temp.name)
        (self.repo / "releases").mkdir()
        self.plan = pr.Plan(
            repo_root=self.repo, engine="radiance", package="engines/radiance",
            upstream_name="radiance", llama_tag="v1.16.0",
            build=1160, release_tag="bc-radiance-1160.0.1",
            version="1160.0.1", branch="feat/verified-bundle"
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_existing_release_retries_asset_verification(self):
        receipt = {"schema": "bigcherry-radiance-runtime-v1"}
        record = {"runtime_assets": {self.plan.release_tag: receipt}}
        (self.repo / "releases/v1.16.0.json").write_text(json.dumps(record))
        with mock.patch.object(pr, "_tag_exists", return_value=True), \
             mock.patch.object(pr, "_asset_environment",
                               return_value=(self.repo / "work", self.repo / "sdk")), \
             mock.patch.object(pr._runtime_assets, "publish") as publish:
            self.assertFalse(pr.phase_release(self.plan))
            publish.assert_called_once_with(record, self.repo, self.repo / "work",
                                            self.plan.release_tag)

    def test_no_receipt_refuses_already_tagged_release(self):
        (self.repo / "releases/v1.16.0.json").write_text(json.dumps({}))
        with mock.patch.object(pr, "_tag_exists", return_value=True):
            with self.assertRaisesRegex(pr.PinReleaseError, "no recorded runtime bundle"):
                pr.phase_release(self.plan)

    def test_patch_promote_defers_unbundled_release_before_tagging(self):
        layout = SimpleNamespace(name="radiance", release_package=lambda: "engines/radiance")
        notes = SimpleNamespace(upstream_name="radiance", tag_prefix="bc-radiance-")
        with mock.patch.object(pp, "_release_version",
                               return_value=("v1.16.0", "1160.0.1", 1160)), \
             mock.patch.object(pp.release_notes, "load_config", return_value=notes), \
             mock.patch.object(pp, "_git", return_value="feat/promotion"), \
             mock.patch.object(pp.pin_release, "_release_pr") as pr_lookup, \
             mock.patch.object(pp.pin_release, "phase_release") as publish:
            self.assertFalse(pp._finish_release_if_ready(self.repo, layout, "minor"))
            pr_lookup.assert_not_called()
            publish.assert_not_called()

    def test_release_receipt_survives_record_dataclass_roundtrip(self):
        from dataclasses import asdict
        from bigcherry.release.records import ReleaseRecord
        record = ReleaseRecord(revision="abcdef123456", release_tag="v1.16.0",
                               runtime_assets={self.plan.release_tag: {"sha256": "a" * 64}})
        copy = ReleaseRecord(**json.loads(json.dumps(asdict(record))))
        copy.validate()
        self.assertEqual(copy.runtime_assets[self.plan.release_tag]["sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
