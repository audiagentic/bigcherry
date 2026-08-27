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


if __name__ == "__main__":
    unittest.main()
