"""CI runtime release bundle: pack a lane build, verify it by restoring, and find a release with the same runtime."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from bigcherry.release import ci_bundle as cb
from bigcherry.release import runtime_assets as assets

ROOT = Path(__file__).resolve().parents[2]
TAG = "bc-radiance-1170.12.0"
PLAN = "d" * 32
TREE_ID = {"engine_tree": "b" * 40, "config_tree": "c" * 40}


class CiBundleTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "work" / "ci-bundle-tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.path = Path(self.temp.name)
        self.work, self.root, self.sdk = self.path / "work", self.path / "repo", self.path / "sdk"
        (self.root / ".github/docker/radiance-build").mkdir(parents=True)
        (self.root / cb.DOCKERFILE).write_text("FROM x\nARG ROCM_VERSION=7.2.4\nRUN true\n")
        (self.sdk / ".info").mkdir(parents=True)
        (self.sdk / ".info/version").write_text("7.2.4")
        build = self.work / "builds" / "src-01" / PLAN
        for rel, data in (("bin/radiance", b"engine"), ("radiance_home/helpers/r3_moe_worker", b"worker"),
                          ("radiance_home/kernels/libr3moe.so", b"plugin")):
            (build / rel).parent.mkdir(parents=True, exist_ok=True)
            (build / rel).write_bytes(data)
            (build / rel).chmod(0o755)
        self.build = build
        for target, name, value in ((assets, "stock_identity", TREE_ID), (cb, "build_inputs_tree", "e" * 64)):
            patch = mock.patch.object(target, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.temp.cleanup)

    def test_image_rocm_version_is_read_from_the_dockerfile(self):
        self.assertEqual(cb.image_rocm_version(self.root), "7.2.4")
        (self.root / cb.DOCKERFILE).write_text("FROM x\n")
        with self.assertRaises(cb.CiBundleError):
            cb.image_rocm_version(self.root)

    def test_pack_and_verify_writes_archive_checksum_and_receipt(self):
        receipt = cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)
        out = self.work / "release-assets" / TAG
        self.assertTrue((out / receipt["name"]).is_file() and (out / receipt["checksum_name"]).is_file())
        written = json.loads((out / (receipt["name"][:-7] + cb.RECEIPT_SUFFIX)).read_text())
        self.assertEqual(written, receipt)
        self.assertEqual(receipt["runtime_key"], {**TREE_ID, "build_inputs": "e" * 64, "rocm_version": "7.2.4"})
        self.assertEqual((receipt["release_tag"], receipt["plan_id"]), (TAG, PLAN))
        self.assertEqual(list(out.glob("ci-bundle-verify-*")), [])      # the scratch restore is gone

    def test_the_bundle_is_named_and_marked_as_bigcherry_s_build_not_upstream(self):
        import tarfile
        receipt = cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)
        self.assertTrue(receipt["name"].startswith("bigcherryrad-") and receipt["name"].endswith(".tar.gz"))
        self.assertEqual(receipt["product"], "bigcherryrad")
        with tarfile.open(self.work / "release-assets" / TAG / receipt["name"], "r:gz") as tar:
            manifest = json.loads(tar.extractfile("bundle.json").read())
        self.assertEqual((manifest["product"], manifest["built_from"]["project"]), ("bigcherryrad", "radiance"))
        # bundles a lab pin-release packed under the old name still restore
        self.assertTrue(assets.ARCHIVE_NAME.fullmatch("radiance-" + PLAN + ".tar.gz"))
        self.assertTrue(assets.ARCHIVE_NAME.fullmatch("bigcherryrad-" + PLAN + ".tar.gz"))
        self.assertFalse(assets.CI_ARCHIVE.fullmatch("radiance-" + PLAN + ".tar.gz"))

    def test_a_tampered_archive_is_not_kept_silently(self):
        original = assets.restore
        def broken(*a, **k):
            raise assets.BundleError("runtime file SHA-256 mismatch: bin/radiance")
        with mock.patch.object(assets, "restore", broken), self.assertRaisesRegex(assets.BundleError, "mismatch"):
            cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)
        self.assertIs(assets.restore, original)

    def test_unsafe_tag_and_ambiguous_build_are_refused(self):
        with self.assertRaises(cb.CiBundleError):
            cb.pack_and_verify("v1.0", root=self.root, work_root=self.work, sdk=self.sdk)
        other = self.work / "builds" / "src-02" / ("e" * 32)
        (other / "bin").mkdir(parents=True)
        (other / "bin/radiance").write_bytes(b"x")
        with self.assertRaisesRegex(cb.CiBundleError, "found 2"):
            cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)
        cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk, plan_id=PLAN)

    def test_find_reusable_matches_the_runtime_key_and_skips_the_tag_itself(self):
        key = {**TREE_ID, "build_inputs": "e" * 64, "rocm_version": "7.2.4"}
        receipts = {
            "bc-radiance-1170.11.0": {"schema": assets.SCHEMA, "runtime_key": {**key, "engine_tree": "f" * 40}},
            "bc-radiance-1170.10.0": {"schema": assets.SCHEMA, "runtime_key": key, "plan_id": PLAN},
        }
        def fake_gh(args, root):
            if args[:2] == ["release", "list"]:
                return json.dumps([{"tagName": TAG}, {"tagName": "bc-radiance-1170.11.0"}, {"tagName": "bc-radiance-1170.10.0"},
                                   {"tagName": "bc-platform-1.2.3"}])
            if args[:2] == ["release", "view"]:
                return json.dumps({"assets": [{"name": f"bigcherryrad-{PLAN}.receipt.json"}]})
            if args[:2] == ["release", "download"]:
                (Path(args[args.index("-D") + 1]) / args[args.index("-p") + 1]).write_text(json.dumps(receipts[args[2]]))
                return ""
            raise AssertionError(args)
        with mock.patch.object(cb, "_gh", fake_gh):
            found = cb.find_reusable(TAG, root=self.root, rocm_version="7.2.4")
            self.assertEqual(found[0], "bc-radiance-1170.10.0")
            self.assertIsNone(cb.find_reusable(TAG, root=self.root, rocm_version="7.14"))

    def test_a_build_tooling_change_changes_the_key(self):
        before = cb.runtime_key(self.root, "7.2.4")
        with mock.patch.object(cb, "build_inputs_tree", return_value="f" * 64):
            self.assertNotEqual(before, cb.runtime_key(self.root, "7.2.4"))


class BuildInputsExistTests(unittest.TestCase):
    def test_every_listed_path_exists_in_the_repository(self):
        repo = Path(__file__).resolve().parents[3]
        for rel in cb.BUILD_INPUTS:
            self.assertTrue((repo / rel).exists(), f"{rel}: BUILD_INPUTS must follow the build code")

    def test_the_real_digest_is_computed_from_git(self):
        repo = Path(__file__).resolve().parents[3]
        digest = cb.build_inputs_tree(repo)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, cb.build_inputs_tree(repo))


if __name__ == "__main__":
    unittest.main()
