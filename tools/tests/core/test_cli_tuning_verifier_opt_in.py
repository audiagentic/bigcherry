"""HI125 close-out step 6: the manual `bigcherry inventory tuning` CLI's
signature-verifier opt-in guards -- an all-or-none binary/vendor-root pair,
and a required --manifest whenever a verifier is requested at all."""

from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli import tuning as cli_tuning  # noqa: E402
from bigcherry.tuning import inventory as cli_tuning_inventory_module  # noqa: E402


def _base_args(**overrides) -> Namespace:
    defaults = dict(
        measurements=None, database=None, manifest=None, signature_source=[],
        signature_verifier_binary=None, signature_verifier_vendor_root=None,
        signature_verifier_seed=1,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class SignatureVerifierOptInGuardTests(unittest.TestCase):
    def test_binary_without_vendor_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            meas_path = Path(directory) / "m.jsonl"
            meas_path.write_text("", encoding="utf-8")
            args = _base_args(measurements=str(meas_path), signature_verifier_binary="/fake/bin")
            self.assertEqual(cli_tuning.cmd_inventory(args, subcmd="tuning"), 2)

    def test_vendor_root_without_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            meas_path = Path(directory) / "m.jsonl"
            meas_path.write_text("", encoding="utf-8")
            args = _base_args(measurements=str(meas_path), signature_verifier_vendor_root="/fake/root")
            self.assertEqual(cli_tuning.cmd_inventory(args, subcmd="tuning"), 2)

    def test_verifier_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            meas_path = Path(directory) / "m.jsonl"
            meas_path.write_text("", encoding="utf-8")
            args = _base_args(
                measurements=str(meas_path),
                signature_verifier_binary="/fake/bin",
                signature_verifier_vendor_root="/fake/root",
            )
            self.assertEqual(cli_tuning.cmd_inventory(args, subcmd="tuning"), 2)

    def test_verifier_with_manifest_constructs_and_forwards_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            meas_path = Path(directory) / "m.jsonl"
            meas_path.write_text("", encoding="utf-8")
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            args = _base_args(
                measurements=str(meas_path), manifest=str(manifest_path),
                signature_verifier_binary="/fake/bin",
                signature_verifier_vendor_root="/fake/root",
            )

            captured = {}

            def fake_load_measurements(*_a, **kwargs):
                captured["signature_digest_verifier"] = kwargs.get("signature_digest_verifier")
                return {"results": 0, "measurements": 0, "candidates": 0}

            fake_verifier = lambda c: "0" * 32  # noqa: E731
            with (
                patch.object(cli_tuning_inventory_module, "load_measurements", side_effect=fake_load_measurements),
                patch(
                    "bigcherry.tuning.signature_digest_verification.make_signature_digest_verifier",
                    return_value=fake_verifier,
                ),
            ):
                rc = cli_tuning.cmd_inventory(args, subcmd="tuning")

            self.assertEqual(rc, 0)
            self.assertIs(captured["signature_digest_verifier"], fake_verifier)

    def test_no_verifier_flags_preserves_current_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            meas_path = Path(directory) / "m.jsonl"
            meas_path.write_text("", encoding="utf-8")
            args = _base_args(measurements=str(meas_path))

            captured = {}

            def fake_load_measurements(*_a, **kwargs):
                captured["signature_digest_verifier"] = kwargs.get("signature_digest_verifier")
                return {"results": 0, "measurements": 0, "candidates": 0}

            with patch.object(cli_tuning_inventory_module, "load_measurements", side_effect=fake_load_measurements):
                rc = cli_tuning.cmd_inventory(args, subcmd="tuning")

            self.assertEqual(rc, 0)
            self.assertIsNone(captured["signature_digest_verifier"])

    def test_verifier_requested_forwards_require_strengthened_ingest(self):
        # adversarial-review follow-up: --manifest is checked with a bare
        # `if args.manifest` here, but load_measurements() itself only
        # loads a manifest that actually exists on disk -- require_
        # strengthened_ingest=True is what actually closes the "manifest
        # path is a typo/does not exist" gap, at the load_measurements()
        # layer, not this CLI's own presence check.
        with tempfile.TemporaryDirectory() as directory:
            meas_path = Path(directory) / "m.jsonl"
            meas_path.write_text("", encoding="utf-8")
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            args = _base_args(
                measurements=str(meas_path), manifest=str(manifest_path),
                signature_verifier_binary="/fake/bin",
                signature_verifier_vendor_root="/fake/root",
            )

            captured = {}

            def fake_load_measurements(*_a, **kwargs):
                captured["require_strengthened_ingest"] = kwargs.get("require_strengthened_ingest")
                return {"results": 0, "measurements": 0, "candidates": 0}

            with (
                patch.object(cli_tuning_inventory_module, "load_measurements", side_effect=fake_load_measurements),
                patch(
                    "bigcherry.tuning.signature_digest_verification.make_signature_digest_verifier",
                    return_value=lambda c: "0" * 32,
                ),
            ):
                rc = cli_tuning.cmd_inventory(args, subcmd="tuning")

            self.assertEqual(rc, 0)
            self.assertTrue(captured["require_strengthened_ingest"])

    def test_nonexistent_manifest_with_verifier_fails_not_silently_unattested(self):
        # Real (unmocked) inv_mod.load_measurements() call.
        with tempfile.TemporaryDirectory() as directory:
            meas_path = Path(directory) / "m.jsonl"
            header = {
                "kind": "header", "artifact_version": 1,
                "source_revision": "a" * 40, "manifest_hash": "deadbeef",
                "variant_set": "workload-max", "build_descriptor_hash": "desc",
                "producer_capabilities": "0000000000000000000000000000001f",
            }
            meas_path.write_text('{"kind": "header", "artifact_version": 1}\n', encoding="utf-8")
            import json as json_module
            meas_path.write_text(json_module.dumps(header) + "\n", encoding="utf-8")
            nonexistent_manifest = Path(directory) / "does-not-exist.json"
            args = _base_args(
                measurements=str(meas_path), manifest=str(nonexistent_manifest),
                signature_verifier_binary="/fake/bin",
                signature_verifier_vendor_root="/fake/root",
            )

            with patch(
                "bigcherry.tuning.signature_digest_verification.make_signature_digest_verifier",
                return_value=lambda c: "0" * 32,
            ):
                rc = cli_tuning.cmd_inventory(args, subcmd="tuning")

            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
