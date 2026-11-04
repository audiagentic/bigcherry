"""BU01 offline Radiance runtime bundle packaging and verification tests."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from bigcherry.release import runtime_assets as assets


ROOT = Path(__file__).resolve().parents[2]
TAG = "bc-radiance-1160.0.0"
PLAN = "a" * 32
TREE_ID = {"engine_tree": "b" * 40, "config_tree": "c" * 40}


class RuntimeAssetsTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "work" / "bu01-offline-tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.path = Path(self.temp.name)
        self.work = self.path / "work"
        self.root = self.path / "repo"
        self.root.mkdir()
        self.sdk = self.path / "sdk"
        (self.sdk / ".info").mkdir(parents=True)
        (self.sdk / ".info/version").write_text("7.2.4")
        self.build = self.work / "builds" / "src-01" / PLAN
        (self.build / "bin").mkdir(parents=True)
        (self.build / "radiance_home/helpers").mkdir(parents=True)
        (self.build / "radiance_home/kernels").mkdir(parents=True)
        (self.build / "bin/radiance").write_bytes(b"radiance engine")
        (self.build / "bin/radiance").chmod(0o755)
        (self.build / "radiance_home/helpers/r3_moe_worker").write_bytes(b"worker")
        (self.build / "radiance_home/helpers/r3_moe_worker").chmod(0o755)
        (self.build / "radiance_home/kernels/libr3moe.so").write_bytes(b"plugin")
        binary_sha = assets.digest(self.build / "bin/radiance")
        self.record = {"notes": f"pin-bump lane build: radiance:stock:linux-radiance "
                                f"build_plan_id={PLAN} binary_sha256={binary_sha}"}
        self.patch = mock.patch.object(assets, "stock_identity", return_value=TREE_ID)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def packaged(self):
        asset = assets.pack(self.record, self.root, self.work, self.sdk, TAG)
        self.record["runtime_assets"] = {TAG: asset}
        archive = self.work / "release-assets" / TAG / asset["name"]
        checksum = archive.parent / asset["checksum_name"]
        return asset, archive, checksum

    def test_pack_restore_stock_binary_without_compiler(self):
        asset, archive, checksum = self.packaged()
        restored = assets.restore(archive, checksum, asset, self.root,
                                  self.path / "other-work", self.sdk)
        self.assertEqual(restored.name, PLAN)
        self.assertEqual(assets.digest(restored / "bin/radiance"), asset["binary_sha256"])
        self.assertEqual((restored / "radiance_home/kernels/libr3moe.so").read_bytes(), b"plugin")
        self.assertTrue((restored / "radiance_home/helpers/r3_moe_worker").stat().st_mode & 0o111)
        self.assertEqual((restored / ".lab-source-tree").read_text(),
                         f"{TREE_ID['engine_tree']} {TREE_ID['config_tree']} {self.sdk.resolve()} 7.2.4 ")
        with self.assertRaisesRegex(assets.BundleError, "already exists"):
            assets.restore(archive, checksum, asset, self.root,
                           self.path / "other-work", self.sdk)

    def test_deterministic_pack_and_repack(self):
        asset, archive, _ = self.packaged()
        before = archive.read_bytes()
        self.assertEqual(asset, assets.pack(self.record, self.root, self.work, self.sdk, TAG))
        self.assertEqual(before, archive.read_bytes())

    def test_binary_mismatch_is_rejected_during_pack(self):
        (self.build / "bin/radiance").write_bytes(b"modified")
        with self.assertRaisesRegex(assets.BundleError, "binary missing or SHA-256"):
            self.packaged()

    def test_repack_changed_payload_does_not_clobber_verified_asset(self):
        asset, archive, checksum = self.packaged()
        original_archive, original_sum = archive.read_bytes(), checksum.read_bytes()
        (self.build / "radiance_home/kernels/libr3moe.so").write_bytes(b"changed plugin")
        with self.assertRaisesRegex(assets.BundleError, "asset changed"):
            assets.pack(self.record, self.root, self.work, self.sdk, TAG)
        self.assertEqual(archive.read_bytes(), original_archive)
        self.assertEqual(checksum.read_bytes(), original_sum)

    def test_bad_checksum_refuses_without_installing(self):
        asset, archive, checksum = self.packaged()
        archive.write_bytes(archive.read_bytes() + b"tamper")
        with self.assertRaisesRegex(assets.BundleError, "SHA-256"):
            assets.restore(archive, checksum, asset, self.root, self.path / "other-work", self.sdk)
        self.assertFalse((self.path / "other-work/builds/src-01" / PLAN).exists())

    def test_source_and_sdk_mismatch_refuse_without_installing(self):
        asset, archive, checksum = self.packaged()
        with mock.patch.object(assets, "stock_identity",
                               return_value={"engine_tree": "x" * 40, "config_tree": "c" * 40}):
            with self.assertRaisesRegex(assets.BundleError, "source trees or ROCm"):
                assets.restore(archive, checksum, asset, self.root,
                               self.path / "other-work", self.sdk)
        (self.sdk / ".info/version").write_text("7.1.0")
        with self.assertRaisesRegex(assets.BundleError, "source trees or ROCm"):
            assets.restore(archive, checksum, asset, self.root,
                           self.path / "other-work", self.sdk)

    def test_malicious_tar_member_refused(self):
        asset, archive, checksum = self.packaged()
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo("../outside")
            member.size = 4
            tar.addfile(member, io.BytesIO(b"evil"))
        asset["sha256"] = assets.digest(archive)
        checksum.write_text(f"{asset['sha256']}  {asset['name']}\n")
        with self.assertRaisesRegex(assets.BundleError, "unsafe runtime archive member"):
            assets.restore(archive, checksum, asset, self.root,
                           self.path / "other-work", self.sdk)
        self.assertFalse((self.path / "outside").exists())


if __name__ == "__main__":
    unittest.main()
