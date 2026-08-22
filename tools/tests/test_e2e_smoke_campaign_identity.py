"""HI82 item 9: campaign-identity-gated resume (design/implementation:
gpt-auto-agent, req_527bff46e32e481c)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.e2e_smoke_campaign import (  # noqa: E402
    Campaign,
    CampaignError,
    CampaignIdentityContext,
)


class CampaignIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        self.model = self.root / "model.gguf"
        self.manifest = self.root / "manifest.json"
        self.tune_server = self.root / "tune-server"
        self.replay_server = self.root / "replay-server"
        self.stock_bench = self.root / "stock-bench"
        self.tune_bench = self.root / "tune-bench"
        self.replay_bench = self.root / "replay-bench"

        self.model.write_bytes(b"model-v1")
        self.manifest.write_text("{}", encoding="utf-8")
        for path in (
            self.tune_server, self.replay_server,
            self.stock_bench, self.tune_bench, self.replay_bench,
        ):
            path.write_bytes(path.name.encode())

    def tearDown(self):
        self.tmp.cleanup()

    def _context(self, *, patch_digest: str = "patch-a") -> CampaignIdentityContext:
        build = {
            "effective_build_id": "effective",
            "compile_verification_id": "compile",
            "runtime_bundle_hash": "bundle",
        }
        return CampaignIdentityContext(
            patch_name="example", patch_digest=patch_digest, patched_source_tree="tree-a",
            gpu_architecture="gfx1100",
            build_identities={"tune": dict(build), "replay": dict(build), "stock": dict(build)},
        )

    def _campaign(self, *, context=None) -> Campaign:
        return Campaign(
            model=self.model, tune_server=self.tune_server, replay_server=self.replay_server,
            manifest=self.manifest, workdir=self.root / "campaign",
            stock_bench=self.stock_bench, tune_bench=self.tune_bench,
            replay_bench=self.replay_bench,
            identity_context=context if context is not None else self._context(),
        )

    def test_identical_identity_can_reopen_workdir(self):
        first = self._campaign()
        first.ensure_campaign_identity()

        second = self._campaign()
        second.ensure_campaign_identity()

        self.assertEqual(first.campaign_identity_digest, second.campaign_identity_digest)

    def test_patch_digest_change_refuses_resume(self):
        self._campaign().ensure_campaign_identity()

        changed = self._campaign(context=self._context(patch_digest="patch-b"))

        with self.assertRaisesRegex(CampaignError, "campaign identity mismatch"):
            changed.ensure_campaign_identity()

    def test_model_replacement_refuses_resume(self):
        self._campaign().ensure_campaign_identity()

        self.model.write_bytes(b"model-v2")

        with self.assertRaisesRegex(CampaignError, "campaign identity mismatch"):
            self._campaign().ensure_campaign_identity()

    def test_legacy_artifacts_without_status_refuse_resume(self):
        workdir = self.root / "campaign"
        workdir.mkdir(parents=True)
        (workdir / "record.jsonl").write_text('{"old":true}\n', encoding="utf-8")

        with self.assertRaisesRegex(CampaignError, "no identity-bound status"):
            self._campaign().ensure_campaign_identity()

    def test_status_identity_digest_is_self_checked(self):
        campaign = self._campaign()
        campaign.ensure_campaign_identity()

        status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
        status["campaign_identity_digest"] = "tampered"
        campaign.status_path.write_text(json.dumps(status), encoding="utf-8")

        with self.assertRaisesRegex(CampaignError, "does not recompute"):
            self._campaign().ensure_campaign_identity()

    def test_standalone_cli_without_identity_context_still_binds(self):
        # identity_context=None (direct e2e_smoke_campaign CLI use, not via
        # patch_validation_campaign.py) must still produce a self-consistent,
        # resumable identity from executable/model/manifest file content alone.
        campaign = Campaign(
            model=self.model, tune_server=self.tune_server, replay_server=self.replay_server,
            manifest=self.manifest, workdir=self.root / "standalone-campaign",
        )
        first = campaign.campaign_identity_digest
        campaign.ensure_campaign_identity()

        second = Campaign(
            model=self.model, tune_server=self.tune_server, replay_server=self.replay_server,
            manifest=self.manifest, workdir=self.root / "standalone-campaign",
        )
        second.ensure_campaign_identity()
        self.assertEqual(first, second.campaign_identity_digest)


if __name__ == "__main__":
    unittest.main()
