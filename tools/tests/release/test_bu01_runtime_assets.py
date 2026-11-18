"""BU01 offline Radiance runtime bundle packaging and verification tests."""
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
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
        config_dir = self.root / "engines/radiance/configs"
        config_dir.mkdir(parents=True)
        engine_dir = self.root / "engines/radiance"
        component_names = ("bc-gfx12", "libr2", "libr3", "moe-remote", "ngram-helper", "serving")
        (engine_dir / "engine.toml").write_text(
            'upstream = "https://codeberg.org/StillDeadcode/radiance"\n'
            '[components]\n' +
            "".join(f'{name} = "engines/radiance/components/{name}"\n' for name in component_names))
        (engine_dir / "recipes.toml").write_text('pinned = "v1.17.0"\npinned-version = "1.17.0"\n')
        for name in component_names:
            component = engine_dir / "components" / name
            component.mkdir(parents=True)
            (component / "component.toml").write_text('built-against = "v1.17.0"\n')
        for name in assets.SHIPPED_CONFIGS:
            (config_dir / name).write_text(f"schema = 1\n# {name}\n")
        self.committed_config = mock.patch.object(assets, "check_committed_config")
        self.committed_config.start()
        self.sdk = self.path / "sdk"
        (self.sdk / ".info").mkdir(parents=True)
        (self.sdk / ".info/version").write_text("7.2.4")
        self.build = self.work / "builds" / "src-01" / PLAN
        (self.build / "bin").mkdir(parents=True)
        (self.build / "radiance_home/helpers").mkdir(parents=True)
        (self.build / "radiance_home/kernels").mkdir(parents=True)
        (self.build / "bin/radiance").write_bytes(b"radiance engine")
        (self.build / "bin/radiance").chmod(0o755)
        serving = self.build / "radiance_home/bin/bc-serve"
        serving.parent.mkdir(parents=True)
        serving.write_bytes(b"bc-serve")
        serving.chmod(0o755)
        releaser = self.build / "radiance_home/bin/bc-release"
        releaser.write_bytes(b"bc-release")
        releaser.chmod(0o755)
        (self.build / "radiance_home/helpers/r3_moe_worker").write_bytes(b"worker")
        (self.build / "radiance_home/helpers/r3_moe_worker").chmod(0o755)
        (self.build / "radiance_home/kernels/libr3moe.so").write_bytes(b"plugin")
        for rel in ("radiance_home/architectures/qwen35_fp8.so",
                    "radiance_home/quantizers/libquant.so",
                    "radiance_home/kernels.rkb", "radiance_home/kernels_moe.rkb",
                    assets.BUILD_METADATA):
            target = self.build / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"runtime data")
        binary_sha = assets.digest(self.build / "bin/radiance")
        self.record = {"notes": f"pin-bump lane build: radiance:stock:linux-radiance "
                                f"build_plan_id={PLAN} binary_sha256={binary_sha}"}
        self.patch = mock.patch.object(assets, "stock_identity", return_value=TREE_ID)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.committed_config.stop()
        self.temp.cleanup()

    def packaged(self):
        asset = assets.pack(self.record, self.root, self.work, self.sdk, TAG)
        self.record["runtime_assets"] = {TAG: asset}
        archive = self.work / "release-assets" / TAG / asset["name"]
        checksum = archive.parent / asset["checksum_name"]
        return asset, archive, checksum

    def test_bc_serve_is_required_and_pack_includes_it(self):
        self.assertIn(("radiance_home/bin/bc-serve", True), assets.RUN_MANIFEST)
        asset, archive, checksum = self.packaged()
        self.assertIn("radiance_home/bin/bc-serve", asset["file_names"])
        with tarfile.open(archive, "r:gz") as tar:
            self.assertIn("radiance_home/bin/bc-serve", tar.getnames())
        (self.build / "radiance_home/bin/bc-serve").unlink()
        with self.assertRaisesRegex(assets.BundleError, "missing runtime requirement"):
            assets.pack(self.record, self.root, self.work, self.sdk, TAG)

    def test_runtime_manifest_binds_every_component_to_upstream_pin(self):
        asset, archive, _ = self.packaged()
        with tarfile.open(archive, "r:gz") as tar:
            manifest = json.load(tar.extractfile("bundle.json"))
        identity = manifest["built_from"]
        self.assertEqual(identity["pinned"], "v1.17.0")
        self.assertEqual(identity["pinned_version"], "1.17.0")
        self.assertEqual(identity["components"], {
            name: "v1.17.0" for name in
            ("bc-gfx12", "libr2", "libr3", "moe-remote", "ngram-helper", "serving")
        })
        self.assertEqual(asset["built_from"], identity)

    def test_pin_move_updates_identity_and_stale_component_refuses(self):
        first = assets.built_from(self.root)
        engine_dir = self.root / "engines/radiance"
        (engine_dir / "recipes.toml").write_text('pinned = "v2.1.0"\npinned-version = "2.1.0"\n')
        with self.assertRaisesRegex(assets.BundleError, "built-against"):
            self.packaged()
        for component in (engine_dir / "components").iterdir():
            (component / "component.toml").write_text('built-against = "v2.1.0"\n')
        second = assets.built_from(self.root)
        self.assertNotEqual(first, second)
        self.assertTrue(all(x == "v2.1.0" for x in second["components"].values()))
        asset, archive, _ = self.packaged()
        with tarfile.open(archive, "r:gz") as tar:
            manifest = json.load(tar.extractfile("bundle.json"))
        self.assertEqual(manifest["built_from"], second)
        self.assertEqual(asset["built_from"], second)

    def test_missing_component_identity_refuses(self):
        path = self.root / "engines/radiance/components/libr3/component.toml"
        path.unlink()
        with self.assertRaisesRegex(assets.BundleError, "missing or invalid build identity"):
            self.packaged()

    def test_source_build_sha_is_bound_to_the_packaged_record(self):
        asset, archive, checksum = self.packaged()
        self.assertIn("source_binary_sha256", asset)
        asset["source_binary_sha256"] = "0" * 64
        with self.assertRaisesRegex(assets.BundleError, "source_binary_sha256"):
            assets.restore(archive, checksum, asset, self.root,
                           self.path / "other-work", self.sdk)

    def test_benchmark_catalogs_are_not_runtime_members(self):
        # Both files are present in the build fixture, but never ship.
        for name in ("radiance_home/kernels.rkb", "radiance_home/kernels_moe.rkb"):
            self.assertTrue((self.build / name).is_file())
            self.assertNotIn(name, assets.runtime_files(self.build, self.root))
        asset, archive, _ = self.packaged()
        with tarfile.open(archive, "r:gz") as tar:
            self.assertFalse(any(name.endswith(".rkb") for name in tar.getnames()))
        self.assertFalse(any(name.endswith(".rkb") for name in asset["file_names"]))

    def test_elf_debug_policy_in_runtime_reuse_key(self):
        self.assertIn("split", assets.ELF_DEBUG_POLICY)
        current = assets.payload_contract_key()
        with mock.patch.object(assets, "ELF_DEBUG_POLICY", "do-not-split"):
            self.assertNotEqual(current, assets.payload_contract_key())

    def test_pack_splits_debug_elf_preserves_loader_and_offload(self):
        # A card-free ELF fixture exercises the real pack and checked extract.
        source = self.path / "probe.c"
        source.write_text(
            '__attribute__((used, section(".hip_fatbin"))) '
            'const unsigned char fatbin[] = {0, 17, 34, 255};\n'
            '__attribute__((used, section(".llvm.offloading"))) '
            'const unsigned char offload[] = {11, 22, 33};\n'
            'int main(void) { return fatbin[1] == 17 ? 0 : 1; }\n'
        )
        binary = self.build / "bin/radiance"
        subprocess.run(["cc", "-g", "-O0", "-o", str(binary), str(source)], check=True)
        # The entrypoint, a native helper, and a plugin must all be split.
        serving = self.build / "radiance_home/bin/bc-serve"
        helper = self.build / "radiance_home/helpers/r3_moe_worker"
        plugin = self.build / "radiance_home/kernels/libr3moe.so"
        for other in (serving, helper):
            shutil.copy2(binary, other)
        subprocess.run(["cc", "-g", "-shared", "-fPIC", "-o", str(plugin),
                        str(source)], check=True)
        original_digest = assets.digest(binary)
        self.record["notes"] = (
            f"pin-bump lane build: radiance:stock:linux-radiance "
            f"build_plan_id={PLAN} binary_sha256={original_digest}"
        )
        before_dynamic = subprocess.check_output(["readelf", "-d", str(binary)])
        before_ldd = re.sub(rb"\s*\(0x[0-9a-f]+\)", b"", subprocess.check_output(["ldd", str(binary)]))
        original_sections = {
            section: subprocess.check_output(["readelf", "-x", section, str(binary)])
            for section in (".hip_fatbin", ".llvm.offloading")
        }
        asset, archive, checksum = self.packaged()
        restored = assets.restore(
            archive, checksum, asset, self.root, self.path / "unpacked", self.sdk
        )
        shipped = restored / "bin/radiance"
        self.assertEqual(subprocess.check_output(["readelf", "-d", str(shipped)]),
                         before_dynamic)
        self.assertEqual(re.sub(rb"\s*\(0x[0-9a-f]+\)", b"",
                                subprocess.check_output(["ldd", str(shipped)])), before_ldd)
        for section, content in original_sections.items():
            self.assertEqual(subprocess.check_output(
                ["readelf", "-x", section, str(shipped)]), content)
        sections = subprocess.check_output(["readelf", "-SW", str(shipped)]).decode()
        self.assertNotIn(".debug_info", sections)
        self.assertNotIn(".symtab", sections)
        self.assertNotIn(".strtab", sections)
        self.assertIn(".dynsym", sections)
        self.assertNotEqual(assets.digest(shipped), original_digest)
        self.assertEqual(assets.digest(shipped), asset["binary_sha256"])
        self.assertEqual(asset["source_binary_sha256"], original_digest)
        debug = archive.parent / "debug" / "bin/radiance.debug"
        self.assertTrue(debug.is_file(), debug)
        self.assertIn(".debug_info", subprocess.check_output(
            ["readelf", "-SW", str(debug)]).decode())
        self.assertEqual(subprocess.run([str(shipped)], check=False).returncode, 0)
        with tarfile.open(archive, "r:gz") as tar:
            self.assertFalse(any(name.endswith(".debug") for name in tar.getnames()))
        for name in ("radiance_home/bin/bc-serve",
                     "radiance_home/helpers/r3_moe_worker",
                     "radiance_home/kernels/libr3moe.so"):
            stripped = restored / name
            section_names = subprocess.check_output(
                ["readelf", "-SW", str(stripped)]).decode()
            self.assertNotIn(".debug_info", section_names)
            self.assertNotIn(".symtab", section_names)
            self.assertIn(".dynsym", section_names)
            self.assertTrue((archive.parent / "debug" / (name + ".debug")).is_file())
        self.assertEqual(subprocess.run([str(restored / "radiance_home/bin/bc-serve")],
                                        check=False).returncode, 0)

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

    def test_scoped_manifest_excludes_tool_test_platform_and_build_files(self):
        import tarfile
        include = {
            "radiance_home/helpers/libr3_ngram_helper.so": b"ngram",
            "radiance_home/vectors/qwen4exp-control-vector.gguf": b"vector",
            "runtime-prefix/lib/librad.so": b"shared library",
        }
        exclude = {
            "bin/rad-tune", "bin/rad-kbench", "bin/rad-convert", "bin/rad-gbnf-check",
            "bin/rad-tokcheck", "bin/rad-schemas", "bin/rad-info", "bin/r4d_selftest",
            "radiance_home/testarch/radtest_arch.so",
            "radiance_home/testkernels/radtest_base.so",
            "radiance_home/kernels/rad-avx-bench", "radiance_home/kernels/rad-avx-check",
            "runtime-prefix/bin/rad-convert", "runtime-prefix/bin/radiance",
            "runtime-prefix/lib/cmake/radiance/radianceTargets.cmake",
        }
        for rel, data in include.items():
            path = self.build / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for rel in exclude:
            path = self.build / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"must never ship")
        asset, archive, checksum = self.packaged()
        with tarfile.open(archive, "r:gz") as tar:
            manifest = json.loads(tar.extractfile("bundle.json").read())
            names = set(tar.getnames()) - {"bundle.json"}
        expected = set(assets.runtime_files(self.build, self.root))
        self.assertEqual(names, expected)
        self.assertEqual(set(manifest["files"]), expected)
        self.assertEqual(asset["payload_scope"], assets.RUN_SCOPE)
        self.assertEqual(manifest["payload_scope"], assets.RUN_SCOPE)
        self.assertEqual(asset["file_names"], sorted(expected))
        self.assertTrue(set(include) <= names)
        self.assertIn("radiance_home/bin/bc-serve", names)
        self.assertTrue(exclude.isdisjoint(names))
        restored = assets.restore(archive, checksum, asset, self.root,
                                  self.path / "other-work", self.sdk)
        self.assertTrue(all((restored / p).is_file() for p in include))
        self.assertTrue(all(not (restored / p).exists() for p in exclude))

    def test_runtime_files_rejects_leaf_and_ancestor_symlinks_outside_build(self):
        external = self.path / "external"
        external.mkdir()
        (external / "libforeign.so").write_bytes(b"untrusted")
        dest = self.build / "runtime-prefix/lib"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(assets.BundleError, "symlink escapes"):
            assets.runtime_files(self.build, self.root)
        dest.unlink()
        dest.mkdir()
        (dest / "libforeign.so").symlink_to(external / "libforeign.so")
        with self.assertRaisesRegex(assets.BundleError, "symlink escapes"):
            assets.runtime_files(self.build, self.root)

    def test_only_published_legacy_asset_receipts_are_grandfathered(self):
        published = assets.PUBLISHED_LEGACY_ASSETS["bc-radiance-1170.0.0"]
        asset = {"schema": assets.SCHEMA, "name": published["name"],
                 "sha256": published["sha256"], "manifest_sha256": published["manifest_sha256"]}
        assets.require_published_legacy_asset(asset, "bc-radiance-1170.0.0")
        for tag, changed in (("bc-radiance-1170.1.0", asset),
                             ("bc-radiance-1170.0.0", {**asset, "sha256": "0" * 64})):
            with self.subTest(tag=tag), self.assertRaisesRegex(assets.BundleError, "legacy"):
                assets.require_published_legacy_asset(changed, tag)

    def test_unpublished_unscoped_archive_is_rejected_before_extract(self):
        asset, archive, checksum = self.packaged()
        asset.pop("payload_scope")
        asset.pop("file_names")
        with mock.patch.object(assets, "checked_extract", side_effect=AssertionError("extracted")):
            with self.assertRaisesRegex(assets.BundleError, "legacy"):
                assets.restore(archive, checksum, asset, self.root,
                               self.path / "other-work", self.sdk,
                               tag="bc-radiance-1170.1.0")

    def test_every_required_manifest_entry_fails_closed_if_missing(self):
        for pattern, required in assets.RUN_MANIFEST:
            if not required:
                continue
            if pattern.startswith("configs/"):
                members = [self.root / "engines/radiance" / pattern]
                refusal = "RSC12 config closure changed"
            else:
                members = list(self.build.glob(pattern))
                refusal = "missing runtime requirement"
            self.assertTrue(members, pattern)
            saved = [(item, item.read_bytes()) for item in members]
            try:
                for item, _ in saved:
                    item.unlink()
                with self.subTest(pattern=pattern), self.assertRaisesRegex(
                    assets.BundleError, refusal
                ):
                    assets.runtime_files(self.build, self.root)
            finally:
                for item, data in saved:
                    item.write_bytes(data)

    def test_packaged_rsc12_configs_are_verified_without_checkout(self):
        asset, archive, checksum = self.packaged()
        for name in assets.SHIPPED_CONFIGS:
            self.assertIn(f"configs/{name}", asset["file_names"])
        installed = assets.restore(archive, checksum, asset, self.root,
                                   self.path / "other-work", self.sdk)
        for name in assets.SHIPPED_CONFIGS:
            self.assertEqual((installed / "configs" / name).read_bytes(),
                             (self.root / "engines/radiance/configs" / name).read_bytes())
        # Two source TOMLs, no source tree, scripts or layouts in the payload.
        self.assertFalse((installed / "engines").exists())
        self.assertFalse((installed / "tools").exists())

    def test_changed_or_extra_rsc12_source_config_refuses_pack(self):
        extra = self.root / "engines/radiance/configs/unreviewed.toml"
        extra.write_text("schema = 1\n")
        with self.assertRaisesRegex(assets.BundleError, "RSC12 config closure"):
            self.packaged()
        extra.unlink()
        with mock.patch.object(assets, "check_committed_config",
                               side_effect=assets.BundleError("uncommitted RSC12 config")):
            with self.assertRaisesRegex(assets.BundleError, "uncommitted"):
                self.packaged()

    def test_python_shell_and_interpreter_content_refused_before_pack(self):
        vector_dir = self.build / "radiance_home/vectors"
        vector_dir.mkdir(parents=True)
        for rel, data in (
            ("entry.py", b"print(1)"),
            ("launch.sh", b"echo hello"),
            ("script.dat", b"#!/usr/bin/env python3\npass\n"),
        ):
            target = vector_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            with self.subTest(name=rel), self.assertRaisesRegex(
                assets.BundleError, "interpreter|script|shell"
            ):
                self.packaged()
            target.unlink()
        # A build-side pycache under an optional directory is not an asset
        # member (the manifest selects files, never whole directories).
        cache = vector_dir / "__pycache__/entry.pyc"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"compiled")
        self.assertNotIn("radiance_home/vectors/__pycache__/entry.pyc",
                         assets.runtime_files(self.build, self.root))
        with self.assertRaisesRegex(assets.BundleError, "interpreter or shell"):
            assets.check_run_manifest({"radiance_home/vectors/__pycache__/entry.pyc"})
        cache.unlink()
        metadata = self.build / assets.BUILD_METADATA
        initial = metadata.read_bytes()
        try:
            metadata.write_text('{"runner": "/usr/bin/python3"}')
            with self.assertRaisesRegex(assets.BundleError, "interpreter reference"):
                self.packaged()
        finally:
            metadata.write_bytes(initial)

    def test_restore_refuses_receipt_file_set_disagreeing_with_manifest(self):
        asset, archive, checksum = self.packaged()
        asset["file_names"].append("bin/rad-info")
        with self.assertRaisesRegex(assets.BundleError, "receipt file set"):
            assets.restore(archive, checksum, asset, self.root,
                           self.path / "other-work", self.sdk)

    def test_restore_rejects_extra_files_even_with_self_consistent_digests(self):
        (self.build / "bin/rad-tune").write_bytes(b"not runtime")
        with mock.patch.object(assets, "RUN_MANIFEST",
                               assets.RUN_MANIFEST + (("bin/rad-tune", False),)):
            asset, archive, checksum = self.packaged()
        with self.assertRaisesRegex(assets.BundleError, "non-runtime file"):
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


class PublishedLegacyRecordsTests(unittest.TestCase):
    def test_unscoped_receipts_match_the_explicit_published_allowlist(self):
        repo = Path(__file__).resolve().parents[3]
        found = {}
        for file in (repo / "releases").glob("*.json"):
            record = json.loads(file.read_text())
            for tag, asset in record.get("runtime_assets", {}).items():
                if asset.get("payload_scope") is None:
                    found[tag] = {key: asset[key] for key in
                                  ("name", "sha256", "manifest_sha256")}
        self.assertEqual(found, assets.PUBLISHED_LEGACY_ASSETS)


if __name__ == "__main__":
    unittest.main()
