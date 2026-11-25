"""CI runtime release bundle: pack a lane build, verify it by restoring, and find a release with the same runtime."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.release import ci_bundle as cb
from bigcherry.release import runtime_assets as assets

ACTUAL_BUILD_INPUTS_TREE = cb.build_inputs_tree
ACTUAL_TAGGED_BUILD_INPUTS_TREE = cb.tagged_build_inputs_tree
ACTUAL_RELEASE_TARGET = assets.release_target

ROOT = Path(__file__).resolve().parents[2]
TAG = "bc-radiance-1170.12.0"
PLAN = "d" * 32
TREE_ID = {"engine_tree": "b" * 40, "config_tree": "c" * 40}
NATIVE_PROFILE = "flashnext-4gpu.yaml"


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
        for rel, data in (("bin/radiance", b"engine"), ("radiance_home/bin/bc-serve", b"serve"), ("radiance_home/bin/bc-release", b"release"), ("radiance_home/helpers/r3_moe_worker", b"worker"),
                          ("radiance_home/kernels/libr3moe.so", b"plugin")):
            (build / rel).parent.mkdir(parents=True, exist_ok=True)
            (build / rel).write_bytes(data)
            (build / rel).chmod(0o755)
        for rel in ("radiance_home/architectures/qwen35_fp8.so", "radiance_home/quantizers/libquant.so",
                    "radiance_home/kernels.rkb", "radiance_home/kernels_moe.rkb"):
            target = build / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"runtime data")
        (build / assets.BUILD_METADATA).write_text('{"build_plan_id": "' + PLAN + '"}')
        self.build = build
        configs = self.root / "engines/radiance/configs"
        configs.mkdir(parents=True)
        for name in assets.SHIPPED_HEAT_SEEDS:
            seed = self.root / "tools/lab/radiance/heat" / name
            seed.parent.mkdir(parents=True, exist_ok=True)
            seed.write_bytes(("heat fixture: " + name).encode())
        # Minimal production-schema build identity for the synthetic repository.
        engine_dir = self.root / "engines/radiance"
        components = ("bc-gfx12", "libr2", "libr3", "moe-remote", "ngram-helper", "serving")
        (engine_dir / "engine.toml").write_text(
            'upstream = "https://codeberg.org/StillDeadcode/radiance"\n'
            '[components]\n' +
            "".join(f'{name} = "engines/radiance/components/{name}"\n' for name in components))
        (engine_dir / "recipes.toml").write_text(
            'pinned = "v1.17.0"\npinned-version = "1.17.0"\n')
        for name in components:
            component_dir = engine_dir / "components" / name
            component_dir.mkdir(parents=True)
            (component_dir / "component.toml").write_text('built-against = "v1.17.0"\n')
        committed_heat = mock.patch.object(assets, "check_committed_heat")
        committed_heat.start()
        self.addCleanup(committed_heat.stop)
        committed = mock.patch.object(assets, "check_committed_config")
        committed.start()
        self.addCleanup(committed.stop)
        for target, name, value in ((assets, "stock_identity", TREE_ID), (cb, "build_inputs_tree", "e" * 64)):
            patch = mock.patch.object(target, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)
        tagged_key = mock.patch.object(cb, "tagged_build_inputs_tree", return_value="e" * 64)
        tagged_key.start()
        self.addCleanup(tagged_key.stop)
        self.addCleanup(self.temp.cleanup)
        # Most packaging fixtures model their declared release tags without a Git checkout.
        target = mock.patch.object(assets, "release_target", side_effect=lambda root, tag=None: ACTUAL_RELEASE_TARGET(root))
        target.start()
        self.addCleanup(target.stop)
        self.serving_release()

    def policy(self):
        return cb._payload_key(ACTUAL_RELEASE_TARGET(self.root), assets.NATIVE_RELEASE_PROFILES)

    def serving_release(self):
        """The release inputs every Radiance release has: the committed contract fingerprints, every shipped
        config YAML, and a build whose engine and owner both use edition 3 with those fingerprints."""
        fingerprint = self.root / assets.SERVING_FINGERPRINT
        fingerprint.parent.mkdir(parents=True, exist_ok=True)
        fingerprint.write_text(f"schema sha256:{'1' * 64}\nsemantics sha256:{'2' * 64}\n")
        for name in assets.NATIVE_RELEASE_PROFILES:
            (self.root / "engines/radiance/configs" / name).write_text("cards: [0]\nmodels: {tiny: {model: tiny.rad}}\n")
        owner = self.build / "components/serving"
        for directory, generated in ((self.build, self.build / "core/bc-serving-generated/v3"),
                                     (owner, owner / "bc-serving-generated/v3")):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "CMakeCache.txt").write_text("BC_SERVING_CONTRACT_VERSION:STRING=3\n")
            generated.mkdir(parents=True, exist_ok=True)
            (generated / "contract.fingerprint").write_bytes(fingerprint.read_bytes())
        (owner / "bin").mkdir(exist_ok=True)
        (owner / "bin/bc-serve").write_bytes((self.build / "radiance_home/bin/bc-serve").read_bytes())
        return self.root / "engines/radiance/configs" / NATIVE_PROFILE

    def test_release_target_reads_the_committed_fingerprints_of_a_tag(self):
        def git(*args):
            subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)

        git("init", "-q")
        git("add", assets.RECIPES)
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "recipes")
        git("tag", TAG)
        self.serving_release()
        git("add", assets.SERVING_FINGERPRINT)
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "contract")
        tagged = "bc-radiance-20200.0.0"
        git("tag", tagged)
        with self.assertRaisesRegex(assets.BundleError, "missing .*contract.fingerprint at release tag"):
            ACTUAL_RELEASE_TARGET(self.root, TAG)
        target = ACTUAL_RELEASE_TARGET(self.root, tagged)
        self.assertEqual(target.lane, assets.RELEASE_LANE)
        self.assertEqual(target.profiles, assets.NATIVE_RELEASE_PROFILES)
        self.assertEqual((target.schema_fingerprint, target.semantic_fingerprint), ("sha256:" + "1" * 64, "sha256:" + "2" * 64))

    def test_pack_requires_matched_engine_owner_and_profile(self):
        opts = {"root": self.root, "work_root": self.work, "sdk": self.sdk}
        profile = self.root / "engines/radiance/configs" / NATIVE_PROFILE
        native_tag = "bc-radiance-20200.0.0"
        engine_cache = self.build / "CMakeCache.txt"
        owner_cache = self.build / "components/serving/CMakeCache.txt"
        engine_fingerprint = self.build / "core/bc-serving-generated/v3/contract.fingerprint"
        owner_fingerprint = self.build / "components/serving/bc-serving-generated/v3/contract.fingerprint"
        engine_cache.write_text("BC_SERVING_CONTRACT_VERSION:STRING=2\n")
        owner_cache.write_text("BC_SERVING_CONTRACT_VERSION:STRING=2\n")
        with self.assertRaisesRegex(assets.BundleError, "edition 3 in the engine"):
            cb.pack_and_verify(native_tag, **opts)
        engine_cache.write_text("BC_SERVING_CONTRACT_VERSION:STRING=3\n")
        with self.assertRaisesRegex(assets.BundleError, "edition 3 in the owner"):
            cb.pack_and_verify(native_tag, **opts)
        owner_cache.write_text("BC_SERVING_CONTRACT_VERSION:STRING=3\n")
        good_fingerprint = owner_fingerprint.read_bytes()
        owner_fingerprint.write_text("mismatched fingerprint\n")
        with self.assertRaisesRegex(assets.BundleError, "generated owner serving fingerprint differs"):
            cb.pack_and_verify(native_tag, **opts)
        owner_fingerprint.write_bytes(good_fingerprint)
        linked_owner = self.build / "components/serving/bin/bc-serve"
        linked_owner.write_bytes(b"stale owner")
        with self.assertRaisesRegex(assets.BundleError, "staged bc-serve differs"):
            cb.pack_and_verify(native_tag, **opts)
        linked_owner.write_bytes((self.build / "radiance_home/bin/bc-serve").read_bytes())
        content = profile.read_bytes()
        profile.unlink()
        with self.assertRaisesRegex(assets.BundleError, NATIVE_PROFILE):
            cb.pack_and_verify(native_tag, **opts)
        profile.write_bytes(content)

        receipt = cb.pack_and_verify(native_tag, **opts)
        target = ACTUAL_RELEASE_TARGET(self.root)
        self.assertIn(f"configs/{NATIVE_PROFILE}", receipt["file_names"])
        self.assertEqual(receipt["serving_contract"], {
            "edition": 3, "lane": assets.RELEASE_LANE,
            "schema_fingerprint": target.schema_fingerprint,
            "semantic_fingerprint": target.semantic_fingerprint,
            "engine_sha256": assets.digest(self.build / "bin/radiance"),
            "owner_sha256": assets.digest(self.build / "radiance_home/bin/bc-serve"),
        })
        stage = self.work / "release-assets" / native_tag
        self.assertEqual(self.verify_staged(native_tag, stage), receipt)
        def download_archive(args, root):
            dest = Path(args[args.index("-D") + 1])
            for name in (receipt["name"], receipt["checksum_name"]):
                shutil.copy2(stage / name, dest / name)

        with mock.patch.object(cb, "_gh", side_effect=download_archive), \
             mock.patch.object(cb, "tagged_config", return_value=profile.read_bytes()):
            cb.verify_reusable_archive(native_tag, self.root, receipt)
        reused_tag = "bc-radiance-20200.1.0"
        record = next(stage.glob("*.receipt.json"))
        reused = {**receipt, "release_tag": reused_tag, "reused_from": native_tag}
        record.write_text(json.dumps(reused))
        self.assertEqual(self.verify_staged(reused_tag, stage), reused)
        reused["serving_contract"] = {**receipt["serving_contract"], "owner_sha256": "0" * 64}
        record.write_text(json.dumps(reused))
        with self.assertRaisesRegex(assets.BundleError, "mismatch: serving_contract"):
            self.verify_staged(reused_tag, stage)


    def test_image_rocm_version_is_read_from_the_dockerfile(self):
        self.assertEqual(cb.image_rocm_version(self.root), "7.2.4")
        (self.root / cb.DOCKERFILE).write_text("FROM x\n")
        with self.assertRaises(cb.CiBundleError):
            cb.image_rocm_version(self.root)

    def test_pack_and_verify_writes_archive_checksum_and_receipt(self):
        receipt = cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)
        out = self.work / "release-assets" / TAG
        self.assertTrue((out / receipt["name"]).is_file() and (out / receipt["checksum_name"]).is_file())
        written = json.loads((out / (receipt["name"][:-7] + cb.ra.CI_RECEIPT_SUFFIX)).read_text())
        self.assertEqual(written, receipt)
        self.assertEqual(receipt["runtime_key"], {**TREE_ID, "build_inputs": "e" * 64, "payload_contract": self.policy(), "rocm_version": "7.2.4"})
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
        self.assertEqual(receipt["file_names"], sorted(manifest["files"]))
        # bundles a lab pin-release packed under the old name still restore
        self.assertTrue(assets.ARCHIVE_NAME.fullmatch("radiance-" + PLAN + ".tar.gz"))
        self.assertTrue(assets.ARCHIVE_NAME.fullmatch("bigcherryrad-" + PLAN + ".tar.gz"))
        self.assertFalse(assets.CI_ARCHIVE.fullmatch("radiance-" + PLAN + ".tar.gz"))

    def test_the_build_metadata_travels_in_the_bundle_and_is_required(self):
        import tarfile
        receipt = cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)
        with tarfile.open(self.work / "release-assets" / TAG / receipt["name"], "r:gz") as tar:
            manifest = json.loads(tar.extractfile("bundle.json").read())
            self.assertIn(assets.BUILD_METADATA, tar.getnames())
        self.assertIn(assets.BUILD_METADATA, manifest["files"])
        (self.build / assets.BUILD_METADATA).unlink()
        with self.assertRaisesRegex(assets.BundleError, "missing runtime requirement"):
            cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)

    def verify_staged(self, release_tag: str, stage: Path) -> dict:
        with mock.patch.object(assets, "tag_identity", return_value=TREE_ID), \
             mock.patch.object(cb, "tagged_heat",
                               side_effect=lambda root, tag, name:
                               (self.root / "tools/lab/radiance/heat" / name).read_bytes()), \
             mock.patch.object(cb, "tagged_config",
                               side_effect=lambda root, tag, name:
                               (self.root / "engines/radiance/configs" / name).read_bytes()):
            return cb.verify_publish_candidate(release_tag, stage, root=self.root)

    def test_final_upload_gate_verifies_actual_asset_without_sdk(self):
        receipt = cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)
        stage = self.work / "release-assets" / TAG
        self.assertEqual(self.verify_staged(TAG, stage), receipt)
        self.assertTrue((stage / receipt["name"]).is_file())

    def test_every_release_ships_each_config_and_checks_it_against_the_tag(self):
        config = self.root / "engines/radiance/configs" / NATIVE_PROFILE
        config.write_bytes(b"cards: [0]\r\nmodels: {tiny: {model: tiny.rad}}\r\n")
        opts = {"root": self.root, "work_root": self.work, "sdk": self.sdk}
        member = f"configs/{NATIVE_PROFILE}"
        candidate = "bc-radiance-1170.13.0"
        selected = cb.pack_and_verify(candidate, **opts)
        self.assertTrue({f"configs/{name}" for name in assets.NATIVE_RELEASE_PROFILES} <= set(selected["file_names"]))
        self.assertIn(member, selected["file_names"])
        stage = self.work / "release-assets" / candidate
        self.assertEqual(self.verify_staged(candidate, stage), selected)

        receipt_path = next(stage.glob("*.receipt.json"))
        bad = {**selected, "runtime_key": {**selected["runtime_key"], "payload_contract": "0" * 64}}
        receipt_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(cb.CiBundleError, "runtime key differs"):
            self.verify_staged(candidate, stage)
        receipt_path.write_text(json.dumps(selected))
        config.write_bytes(config.read_bytes().replace(b"tiny.rad", b"other.rad"))
        with self.assertRaisesRegex(cb.CiBundleError, "config differs from tagged source"):
            self.verify_staged(candidate, stage)

    def test_optional_native_profile_refuses_unknown_or_unsafe_files(self):
        config = self.root / "engines/radiance/configs" / NATIVE_PROFILE
        for bad in (("other.yaml",), (NATIVE_PROFILE, NATIVE_PROFILE)):
            with self.assertRaisesRegex(assets.BundleError, "unapproved or repeated native profile"):
                assets.runtime_files(self.build, self.root, native_profiles=bad)
        config.unlink()
        with self.assertRaisesRegex(assets.BundleError, "missing or unsafe RSC12 config"):
            assets.runtime_files(self.build, self.root, native_profiles=(NATIVE_PROFILE,))
        config.symlink_to("flashnext.toml")
        with self.assertRaisesRegex(assets.BundleError, "missing or unsafe RSC12 config"):
            assets.runtime_files(self.build, self.root, native_profiles=(NATIVE_PROFILE,))
        config.unlink()
        for content, error in ((b"cards: [0]\n# python3\n", "interpreter reference"),
                               (b"#\xff\ncards: [0]\n", "non-UTF8"),
                               (b"#\0\ncards: [0]\n", "NUL in native profile")):
            with self.subTest(error=error):
                config.write_bytes(content)
                with self.assertRaisesRegex(assets.BundleError, error):
                    assets.runtime_files(self.build, self.root, native_profiles=(NATIVE_PROFILE,))
        config.write_text("cards: [0]\n")
        config.chmod(0o755)
        with self.assertRaisesRegex(assets.BundleError, "native profile must be non-executable"):
            assets.runtime_files(self.build, self.root, native_profiles=(NATIVE_PROFILE,))

    def test_elf_debug_sidecar_is_outside_upload_staging(self):
        # A real ELF exercises the debug-split branch (unlike the byte fixtures).
        source = self.path / "engine.c"
        source.write_text("int main(void) { return 0; }\n")
        subprocess.run(["cc", "-g", "-o", str(self.build / "bin/radiance"), str(source)],
                       check=True)
        receipt = cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)
        stage = self.work / "release-assets" / TAG
        self.assertEqual({item.name for item in stage.iterdir()},
                         {receipt["name"], receipt["checksum_name"],
                          receipt["name"][:-len(".tar.gz")] + assets.CI_RECEIPT_SUFFIX})
        self.assertTrue((self.work / "release-debug" / TAG / "bin/radiance.debug").is_file())
        self.assertEqual(self.verify_staged(TAG, stage), receipt)

    def test_final_upload_gate_refuses_script_files_checksum_and_tag_confusion(self):
        receipt = cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)
        stage = self.work / "release-assets" / TAG
        record = stage / (receipt["name"][:-len(".tar.gz")] + assets.CI_RECEIPT_SUFFIX)
        archive = stage / receipt["name"]
        checksum = stage / receipt["checksum_name"]
        extra = stage / "setup.py"
        extra.write_text("print('unreviewed')")
        with self.assertRaisesRegex(cb.CiBundleError, "unexpected release files"):
            self.verify_staged(TAG, stage)
        extra.unlink()
        original_receipt = record.read_bytes()
        try:
            record.write_text(json.dumps({**receipt, "release_tag": "bc-radiance-1170.91.0"}))
            with self.assertRaisesRegex(cb.CiBundleError, "schema, scope, tag"):
                self.verify_staged(TAG, stage)
            record.write_bytes(original_receipt)
            record.write_text(json.dumps({**receipt, "runtime_key": {**receipt["runtime_key"],
                                                                    "payload_contract": "0" * 64}}))
            with self.assertRaisesRegex(cb.CiBundleError, "runtime key"):
                self.verify_staged(TAG, stage)
            record.write_text(json.dumps({**receipt, "runtime_key": {**receipt["runtime_key"],
                                                                    "build_inputs": "0" * 64}}))
            with self.assertRaisesRegex(cb.CiBundleError, "runtime key"):
                self.verify_staged(TAG, stage)
            record.write_text(json.dumps({**receipt, "engine_tree": "0" * 40}))
            with self.assertRaisesRegex(cb.CiBundleError, "runtime key"):
                self.verify_staged(TAG, stage)
        finally:
            record.write_bytes(original_receipt)
        original_checksum = checksum.read_bytes()
        try:
            checksum.write_text("0" * 64 + "  " + archive.name + "\n")
            with self.assertRaisesRegex(cb.CiBundleError, "checksum differs"):
                self.verify_staged(TAG, stage)
        finally:
            checksum.write_bytes(original_checksum)
        with archive.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(cb.CiBundleError, "tag-bound digest"):
            self.verify_staged(TAG, stage)

    def test_final_upload_gate_accepts_only_verified_same_tree_reuse(self):
        source = "bc-radiance-1170.10.0"
        receipt = cb.pack_and_verify(source, root=self.root, work_root=self.work, sdk=self.sdk)
        stage = self.work / "release-assets" / source
        new = {**receipt, "release_tag": TAG, "reused_from": source}
        record = stage / (receipt["name"][:-len(".tar.gz")] + assets.CI_RECEIPT_SUFFIX)
        record.write_text(json.dumps(new))
        self.assertEqual(self.verify_staged(TAG, stage), new)
        original = assets.tag_identity
        def different_tree(root, tag):
            return TREE_ID if tag == TAG else {"engine_tree": "a" * 40, "config_tree": "c" * 40}
        with mock.patch.object(assets, "tag_identity", side_effect=different_tree), \
             mock.patch.object(cb, "tagged_heat",
                               side_effect=lambda root, tag, name:
                               (self.root / "tools/lab/radiance/heat" / name).read_bytes()), \
             mock.patch.object(cb, "tagged_config",
                               side_effect=lambda root, tag, name:
                               (self.root / "engines/radiance/configs" / name).read_bytes()), \
             self.assertRaisesRegex(cb.CiBundleError, "source trees"):
            cb.verify_publish_candidate(TAG, stage, root=self.root)
        self.assertIs(assets.tag_identity, original)

    def test_final_upload_gate_rechecks_tagged_heat_bytes(self):
        cb.pack_and_verify(TAG, root=self.root, work_root=self.work, sdk=self.sdk)
        stage = self.work / "release-assets" / TAG
        with mock.patch.object(assets, "tag_identity", return_value=TREE_ID), \
             mock.patch.object(cb, "tagged_config",
                               side_effect=lambda root, tag, name:
                               (self.root / "engines/radiance/configs" / name).read_bytes()), \
             mock.patch.object(cb, "tagged_heat", return_value=b"altered tagged heat"), \
             self.assertRaisesRegex(cb.CiBundleError, "heat seed differs from tagged source"):
            cb.verify_publish_candidate(TAG, stage, root=self.root)

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
        key = {**TREE_ID, "build_inputs": "e" * 64, "payload_contract": self.policy(), "rocm_version": "7.2.4"}
        receipts = {
            "bc-radiance-1170.11.0": {"schema": assets.SCHEMA, "payload_scope": assets.RUN_SCOPE,
                                       "runtime_key": {**key, "engine_tree": "f" * 40}},
            "bc-radiance-1170.10.0": {"schema": assets.SCHEMA, "payload_scope": assets.RUN_SCOPE,
                                       "runtime_key": key, "plan_id": PLAN,
                                       "file_names": sorted(assets.runtime_files(self.build, self.root, native_profiles=assets.NATIVE_RELEASE_PROFILES))},
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
        with mock.patch.object(cb, "_gh", fake_gh), mock.patch.object(cb, "verify_reusable_archive"):
            found = cb.find_reusable(TAG, root=self.root, rocm_version="7.2.4")
            self.assertEqual(found[0], "bc-radiance-1170.10.0")
            self.assertIsNone(cb.find_reusable(TAG, root=self.root, rocm_version="7.14"))
        old_receipt = receipts["bc-radiance-1170.10.0"]
        names = old_receipt["file_names"]
        old_receipt["file_names"] = ["bin/radiance"]
        with mock.patch.object(cb, "_gh", fake_gh), mock.patch.object(cb, "verify_reusable_archive"):
            self.assertIsNone(cb.find_reusable(TAG, root=self.root, rocm_version="7.2.4"))
        old_receipt["file_names"] = names
        # A bundle that lacks one of the shipped configs is not this runtime, whatever its key says.
        old_receipt["file_names"] = [n for n in names if n != f"configs/{NATIVE_PROFILE}"]
        with mock.patch.object(cb, "_gh", fake_gh), mock.patch.object(cb, "verify_reusable_archive"):
            self.assertIsNone(cb.find_reusable(TAG, root=self.root, rocm_version="7.2.4"))
        old_receipt["file_names"] = names
        old_receipt["runtime_key"] = key
        old_receipt.pop("payload_scope")
        with mock.patch.object(cb, "_gh", fake_gh), mock.patch.object(cb, "verify_reusable_archive"):
            self.assertIsNone(cb.find_reusable(TAG, root=self.root, rocm_version="7.2.4"))

    def test_reuse_verifies_archived_files_and_checksum(self):
        old = "bc-radiance-1170.10.0"
        receipt = cb.pack_and_verify(old, root=self.root, work_root=self.work, sdk=self.sdk)
        out = self.work / "release-assets" / old
        receipt_name = receipt["name"][:-7] + assets.CI_RECEIPT_SUFFIX
        def gh(args, root):
            if args[:2] == ["release", "list"]:
                return json.dumps([{"tagName": old}])
            if args[:2] == ["release", "view"]:
                return json.dumps({"assets": [{"name": receipt_name}]})
            if args[:2] == ["release", "download"]:
                dest = Path(args[args.index("-D") + 1])
                for i, arg in enumerate(args):
                    if arg == "-p":
                        shutil.copy2(out / args[i + 1], dest / args[i + 1])
                return ""
            raise AssertionError(args)
        with mock.patch.object(cb, "_gh", gh), \
             mock.patch.object(cb, "tagged_config", return_value=(self.root / "engines/radiance/configs" / NATIVE_PROFILE).read_bytes()):
            self.assertEqual(cb.find_reusable(TAG, root=self.root, rocm_version="7.2.4")[0], old)
            archive = out / receipt["name"]
            archive.write_bytes(archive.read_bytes() + b"corrupt")
            self.assertIsNone(cb.find_reusable(TAG, root=self.root, rocm_version="7.2.4"))

    def test_build_inputs_are_pinned_and_platform_changes_do_not_change_key(self):
        # Deliberate independent oracle: BUILD_INPUTS is the reviewed keying
        # policy, not a list synthesized from imports or from the code under
        # test. Changing its scope requires updating this literal and separately
        # passing the static-import coverage gate.
        expected = (
            ".github/docker/radiance-build", ".github/workflows/radiance-build.yml",
            "tools/bigcherry/build", "tools/bigcherry/campaign/build.py",
            "tools/bigcherry/campaign/source.py", "tools/bigcherry/core/config.py",
            "tools/bigcherry/core/engines.py", "tools/bigcherry/core/canonical.py",
            "tools/bigcherry/engine_adapters/base.py",
            "tools/bigcherry/engine_adapters/radiance.py",
            "tools/bigcherry/engine_adapters/radiance_inventory.py",
            "tools/bigcherry/engine_adapters/radiance_resources.py",
            "tools/bigcherry/source/identity.py", "tools/bigcherry/source/sources.py",
            "tools/bigcherry/source/upstream.py", "tools/bigcherry/source/workspace.py",
            "tools/bigcherry/patch/apply.py", "tools/bigcherry/patch/overlay.py",
            "tools/bigcherry/patch/patchset.py", "tools/bigcherry/patch/selection.py",
            "tools/bigcherry/patch/registry.py", "tools/bigcherry/patch/catalog.py",
            "tools/bigcherry/patch/pinned_source.py", "tools/bigcherry/patch/compose_check.py",
            "tools/bigcherry/patch/rebase.py", "tools/bigcherry/patch/source.py",
            "tools/bigcherry/bin/bc-ccache",
            "tools/bigcherry/campaign/resolution.py",
            "tools/bigcherry/campaign/runtime_bundle.py",
            "tools/bigcherry/core/artifacts.py",
            "tools/bigcherry/core/component_coupling.py",
    "tools/bigcherry/core/components.py",
            "tools/bigcherry/core/context.py",
            "tools/bigcherry/core/csource.py",
            "tools/bigcherry/core/environment.py",
            "tools/bigcherry/core/paths.py",
            "tools/bigcherry/core/process.py",
            "tools/bigcherry/core/provenance.py",
            "tools/bigcherry/engine_adapters/radiance_bump.py",
            "tools/bigcherry/engine_adapters/radiance_layout.py",
            "tools/bigcherry/patch/docs.py",
            "tools/bigcherry/source/audit.py",
        )
        self.assertEqual(cb.BUILD_INPUTS, expected)
        repo = self.path / "key-git-fixture"
        repo.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
               "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}
        def git(*args):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)
        def change(name, data):
            file = repo / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(data)
        git("init", "-q")
        for path in expected:
            change(path + "/fixture" if path in ("tools/bigcherry/build", ".github/docker/radiance-build")
                   else path, "initial")
        platform_file = "tools/bigcherry/jobs/scheduler.py"
        change(platform_file, "platform-v1")
        git("add", ".")
        git("commit", "-qm", "fixture")
        git("tag", TAG)
        before = ACTUAL_BUILD_INPUTS_TREE(repo)
        self.assertEqual(ACTUAL_TAGGED_BUILD_INPUTS_TREE(repo, TAG), before)
        change(platform_file, "platform-v2")
        git("add", ".")
        git("commit", "-qm", "platform only")
        self.assertEqual(ACTUAL_BUILD_INPUTS_TREE(repo), before)
        change("tools/bigcherry/patch/apply.py", "engine patch v2")
        git("add", ".")
        git("commit", "-qm", "engine build change")
        self.assertNotEqual(ACTUAL_BUILD_INPUTS_TREE(repo), before)
        self.assertEqual(ACTUAL_TAGGED_BUILD_INPUTS_TREE(repo, TAG), before)
        after = ACTUAL_BUILD_INPUTS_TREE(repo)
        change(".github/docker/radiance-build/fixture", "image v2")
        git("add", ".")
        git("commit", "-qm", "image change")
        self.assertNotEqual(ACTUAL_BUILD_INPUTS_TREE(repo), after)
        # Every newly discovered compile input independently invalidates the key.
        added = cb.BUILD_INPUTS[len(expected) - 15:]
        self.assertEqual(len(added), 15)
        before = ACTUAL_BUILD_INPUTS_TREE(repo)
        for rel in added:
            change(rel, "changed " + rel)
            git("add", ".")
            git("commit", "-qm", "compiled input mutation")
            updated = ACTUAL_BUILD_INPUTS_TREE(repo)
            self.assertNotEqual(updated, before, rel)
            before = updated

    def test_bc_serve_is_an_engine_requirement_in_runtime_reuse_identity(self):
        server = ("radiance_home/bin/bc-serve", True)
        self.assertIn(server, assets.RUN_MANIFEST)
        original = cb.runtime_key(self.root, "7.2.4")
        with mock.patch.object(assets, "RUN_MANIFEST",
                               tuple(rule for rule in assets.RUN_MANIFEST if rule != server)):
            changed = cb.runtime_key(self.root, "7.2.4")
        self.assertNotEqual(original["payload_contract"], changed["payload_contract"])

    def test_manifest_only_change_invalidates_runtime_key(self):
        before = cb.runtime_key(self.root, "7.2.4")
        with mock.patch.object(assets, "RUN_MANIFEST",
                               assets.RUN_MANIFEST + (("radiance_home/new_runtime/*.so", True),)):
            self.assertNotEqual(before, cb.runtime_key(self.root, "7.2.4"))

    def test_build_import_dependency_closure_is_covered(self):
        repo = Path(__file__).resolve().parents[3]
        missing = {path for path in cb.discover_compile_dependencies(repo)
                   if not any(path == item or path.startswith(item + "/") for item in cb.BUILD_INPUTS)}
        self.assertEqual(missing, set(), f"unkeyed build imports: {sorted(missing)}")

    def test_new_build_import_is_detected_by_the_dependency_gate(self):
        repo = self.path / "imports"
        base = repo / "tools/bigcherry"
        for rel in ("campaign/build.py", "engine_adapters/radiance.py",
                    "patch/apply.py", "core/fresh.py"):
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("VALUE = 1\n")
        key = "tools/bigcherry/core/fresh.py"
        self.assertNotIn(key, cb.discover_compile_dependencies(repo))
        (base / "campaign/build.py").write_text("from ..core import fresh\n")
        self.assertIn(key, cb.discover_compile_dependencies(repo))

    def test_a_build_tooling_change_changes_the_key(self):
        before = cb.runtime_key(self.root, "7.2.4")
        with mock.patch.object(cb, "build_inputs_tree", return_value="f" * 64):
            self.assertNotEqual(before, cb.runtime_key(self.root, "7.2.4"))


class PruneTests(unittest.TestCase):
    def fake(self, releases):
        """releases: {tag: (createdAt, runtime_key or None, body)}; returns a _gh double and the delete log."""
        deleted = []
        def gh(args, root):
            if args[:2] == ["release", "list"]:
                return json.dumps([{"tagName": t, "createdAt": v[0]} for t, v in releases.items()])
            if args[:2] == ["release", "view"]:
                tag = args[2]
                assets = [] if releases[tag][1] is None else [
                    {"name": "bigcherryrad-" + "a" * 32 + ".tar.gz"}, {"name": "bigcherryrad-" + "a" * 32 + ".tar.gz.sha256"},
                    {"name": "bigcherryrad-" + "a" * 32 + cb.ra.CI_RECEIPT_SUFFIX}]
                return json.dumps({"assets": assets, "body": releases[tag][2]})
            if args[:2] == ["release", "download"]:
                (Path(args[args.index("-D") + 1]) / args[args.index("-p") + 1]).write_text(json.dumps({"runtime_key": releases[args[2]][1]}))
                return ""
            if args[:2] == ["release", "delete-asset"]:
                deleted.append((args[2], args[3]))
                return ""
            raise AssertionError(args)
        return gh, deleted

    def test_keeps_the_newest_distinct_runtimes_the_newest_release_and_marked_ones(self):
        releases = {f"bc-radiance-1170.{n}.0": (f"2026-11-{n:02d}T00:00:00Z", {"k": n // 2}, "") for n in range(1, 13)}
        releases["bc-radiance-1170.2.0"] = ("2026-11-02T00:00:00Z", {"k": 1}, "notes\nkeep-bundle\n")
        releases["bc-radiance-1170.13.0"] = ("2026-11-13T00:00:00Z", None, "")          # no bundle: ignored
        gh, deleted = self.fake(releases)
        with mock.patch.object(cb, "_gh", gh):
            plan = cb.prune_plan(root=Path("."), keep=3)
            # keys by release number: 12->6, 11->5, 10->5, 9->4, ..., so the newest three distinct keys are 6, 5, 4 (12, 11, 10, 9, 8)
            self.assertEqual(sorted(plan["kept"]), sorted(["bc-radiance-1170.12.0", "bc-radiance-1170.11.0", "bc-radiance-1170.10.0",
                                                          "bc-radiance-1170.9.0", "bc-radiance-1170.8.0", "bc-radiance-1170.2.0"]))
            self.assertEqual({e["tag"] for e in plan["prune"]},
                             {f"bc-radiance-1170.{n}.0" for n in (7, 6, 5, 4, 3, 1)})
            self.assertTrue(all(len(e["assets"]) == 3 for e in plan["prune"]))
            self.assertEqual(deleted, [])                      # planning deletes nothing
            cb.apply_prune(plan, root=Path("."))
            self.assertEqual(len(deleted), 18)


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
