"""Offline release-install contract: real tags/worktrees, tiny verified tar, fake gh."""

from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.release import install as release_install
from bigcherry.release import runtime_assets

ROOT = Path(__file__).resolve().parents[3]
TAG = "bc-radiance-1170.0.0"
PLAN = "a" * 32
CI_TAG = "bc-radiance-1180.0.0"

FAKE_GH = """#!/usr/bin/env python3
import fnmatch
import json
import os
from pathlib import Path
import shutil
import sys
args = sys.argv[1:]
releases = json.loads(os.environ["FAKE_RELEASES"])
if args[:2] not in (["release", "download"], ["release", "view"]):
    sys.exit("unexpected gh invocation")
files = [Path(f) for f in releases.get(args[2], [])]
if not files:
    sys.exit("release not found")
if args[1] == "view":
    print(json.dumps({"assets": [{"name": f.name} for f in files]}))
    sys.exit(0)
patterns = [args[i + 1] for i, arg in enumerate(args) if arg == "-p"]
dest = Path(args[args.index("-D") + 1])
dest.mkdir(parents=True, exist_ok=True)
for f in files:
    if any(fnmatch.fnmatch(f.name, p) for p in patterns):
        shutil.copy2(f, dest / f.name)
with open(os.environ["FAKE_GH_CALLS"], "a", encoding="utf-8") as f:
    f.write("download\\n")
"""


class ReleaseInstallTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "work" / "release-install-tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.path = Path(self.temp.name)
        self.root = self.path / "repo"
        self.root.mkdir()
        self.pack_work = self.path / "pack"
        self.install_work = self.path / "installed"
        self.sdk = self.path / "sdk"
        (self.sdk / ".info").mkdir(parents=True)
        (self.sdk / ".info/version").write_text("7.2.4")
        self.gh_calls = self.path / "downloads.log"
        fakebin = self.path / "fakebin"
        fakebin.mkdir()
        fake_gh = fakebin / "gh"
        fake_gh.write_text(FAKE_GH)
        fake_gh.chmod(fake_gh.stat().st_mode | stat.S_IXUSR)
        self._git("init", "-q")
        (self.root / "engines/radiance").mkdir(parents=True)
        (self.root / "config").mkdir()
        (self.root / "releases").mkdir()
        (self.root / "engines/radiance/source.txt").write_text("radiance production")
        (self.root / "config/recipes.txt").write_text("pinned")
        (self.root / "releases/index.json").write_text('{"releases": []}')
        self._commit("initial source trees")
        self._git("tag", "bc-radiance-1160.0.0")  # older release, no bundle

        build = self.pack_work / "builds/src-1" / PLAN
        for subdir in ("bin", "radiance_home/helpers", "radiance_home/kernels"):
            (build / subdir).mkdir(parents=True)
        for name, value in (
            ("bin/radiance", b"tiny-radiance"),
            ("radiance_home/helpers/r3_moe_worker", b"tiny-helper"),
            ("radiance_home/kernels/libp.so", b"tiny-plugin"),
        ):
            target = build / name
            target.write_bytes(value)
            target.chmod(0o755)
        notes = (
            f"release asset lane build: radiance:stock:linux-radiance "
            f"build_plan_id={PLAN} binary_sha256={runtime_assets.digest(build / 'bin/radiance')}\n"
            "release 1170.0.0"
        )
        record = {"release_tag": "v1.17.0", "notes": notes}
        self.asset = runtime_assets.pack(
            record, self.root, self.pack_work, self.sdk, TAG
        )
        record["runtime_assets"] = {TAG: self.asset}
        (self.root / "releases/v1.17.0.json").write_text(json.dumps(record))
        self._commit("verified record")
        self._git("tag", TAG)
        self._git("tag", "bc-radiance-1180.0.0")  # newer tag without its own receipt
        asset_dir = self.pack_work / "release-assets" / TAG
        self.archive = asset_dir / self.asset["name"]
        self.checksum = asset_dir / self.asset["checksum_name"]
        self.releases = {TAG: [str(self.archive), str(self.checksum)]}
        self.env = {
            "PATH": f"{fakebin}:{os.environ['PATH']}",
            "FAKE_RELEASES": json.dumps(self.releases),
            "FAKE_GH_CALLS": str(self.gh_calls),
        }

    def tearDown(self):
        self.temp.cleanup()

    def _git(self, *args):
        result = subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def _commit(self, message):
        self._git("add", ".")
        self._git("commit", "-qm", message)

    def _ci_bundle(self, tag, receipt_tag=None, **extra):
        """Pack a bundle for a tag with no record entry and attach it, with its receipt, to the fake release."""
        notes = (
            f"release asset lane build: radiance:stock:linux-radiance build_plan_id={PLAN} "
            f"binary_sha256={self.asset['binary_sha256']}"
        )
        asset = runtime_assets.pack(
            {"notes": notes}, self.root, self.pack_work, self.sdk, tag
        )
        out = self.pack_work / "release-assets" / tag
        archive, checksum = out / asset["name"], out / asset["checksum_name"]
        receipt = out / f"bigcherryrad-{PLAN}.receipt.json"
        key = {k: asset[k] for k in ("engine_tree", "config_tree", "rocm_version")}
        receipt.write_text(
            json.dumps(
                {
                    **asset,
                    "release_tag": receipt_tag or tag,
                    "runtime_key": key,
                    **extra,
                }
            )
        )
        self.releases[receipt_tag or tag] = [str(archive), str(checksum), str(receipt)]
        self.env["FAKE_RELEASES"] = json.dumps(self.releases)
        return archive, checksum

    def _install(self, tag=TAG):
        return release_install.install(
            tag, root=self.root, work_root=self.install_work, sdk=self.sdk
        )

    def test_install_and_verified_reuse_without_redownload(self):
        with mock.patch.dict(os.environ, self.env):
            first = self._install()
            second = self._install()
        self.assertEqual(first, second)
        self.assertEqual(self.gh_calls.read_text().splitlines(), ["download"])
        self.assertEqual(first["plan_id"], PLAN)
        self.assertEqual(first["source"], "record")
        self.assertEqual(first["binary_sha256"], self.asset["binary_sha256"])
        self.assertEqual(first["rocm_version"], "7.2.4")
        self.assertEqual(first["archive_sha256"], self.asset["sha256"])
        self.assertEqual(
            (Path(first["build_dir"]) / "bin/radiance").read_bytes(), b"tiny-radiance"
        )
        self.assertEqual(
            json.loads(
                (self.install_work / "releases" / TAG / "install.json").read_text()
            ),
            first,
        )
        self.assertEqual(
            self._git("rev-parse", f"refs/tags/{TAG}^{{commit}}"),
            subprocess.check_output(
                ["git", "-C", first["worktree"], "rev-parse", "HEAD"], text=True
            ).strip(),
        )
        detached = subprocess.run(
            ["git", "-C", first["worktree"], "symbolic-ref", "-q", "HEAD"], check=False
        )
        self.assertNotEqual(detached.returncode, 0)
        self.assertEqual(
            release_install.launch_command(first),
            f"bash {first['worktree']}/tools/lab/radiance/launch-flashnext-best.sh start --build {first['build_dir']}",
        )

    def test_latest_only_selects_tag_with_record_or_ci_bundle(self):
        with mock.patch.dict(os.environ, self.env):
            self.assertEqual(release_install.resolve_tag(self.root, "latest")[0], TAG)
            self.assertEqual(self._install("latest")["tag"], TAG)
            with self.assertRaisesRegex(
                release_install.ReleaseInstallError, "release not found"
            ):
                self._install("bc-radiance-1180.0.0")

    def test_latest_picks_ci_bundled_tag(self):
        self._ci_bundle(CI_TAG)
        with mock.patch.dict(os.environ, self.env):
            self.assertEqual(
                release_install.resolve_tag(self.root, "latest"), (CI_TAG, None)
            )

    def test_ci_bundle_installs_with_ci_receipt(self):
        self._ci_bundle(CI_TAG)
        with mock.patch.dict(os.environ, self.env):
            first = self._install(CI_TAG)
            second = self._install(CI_TAG)
        self.assertEqual(first, second)
        self.assertEqual(first["source"], "ci")
        self.assertEqual(first["tag"], CI_TAG)
        self.assertEqual(
            (Path(first["build_dir"]) / "bin/radiance").read_bytes(), b"tiny-radiance"
        )
        downloads = self.install_work / "release-assets" / "downloads"
        self.assertEqual(list(downloads.iterdir()), [])

    def test_reused_ci_bundle_needs_reused_from_with_same_trees(self):
        reuse_tag = "bc-radiance-1190.0.0"
        self._git("tag", reuse_tag)  # same source trees as CI_TAG
        self._ci_bundle(CI_TAG, receipt_tag=reuse_tag)
        self._refused(reuse_tag, "bundle.json is for bc-radiance-1180.0.0")
        self._ci_bundle(CI_TAG, receipt_tag=reuse_tag, reused_from=CI_TAG)
        with mock.patch.dict(os.environ, self.env):
            self.assertEqual(self._install(reuse_tag)["source"], "ci")

    def test_ci_receipt_for_another_tag_refused(self):
        self._ci_bundle(CI_TAG)
        self.releases["bc-radiance-1190.0.0"] = self.releases[CI_TAG]
        self.env["FAKE_RELEASES"] = json.dumps(self.releases)
        self._git("tag", "bc-radiance-1190.0.0")
        self._refused(
            "bc-radiance-1190.0.0", "receipt asset is for bc-radiance-1180.0.0"
        )

    def test_record_entry_preferred_over_ci_bundle(self):
        with (
            mock.patch.dict(os.environ, self.env),
            mock.patch.object(
                runtime_assets, "ci_asset", side_effect=AssertionError("CI path used")
            ),
        ):
            receipt = self._install(TAG)
        self.assertEqual(receipt["source"], "record")
        self.assertEqual(receipt["archive_sha256"], self.asset["sha256"])

    def _refused(self, tag, pattern):
        with (
            mock.patch.dict(os.environ, self.env),
            self.assertRaisesRegex(release_install.ReleaseInstallError, pattern),
        ):
            self._install(tag)
        self.assertFalse(
            (self.install_work / "releases" / tag / "install.json").exists()
        )
        self.assertFalse((self.install_work / "releases" / tag / "src").exists())
        self.assertFalse((self.install_work / "builds/src-1" / PLAN).exists())

    def test_ci_archive_checksum_mismatch_refused(self):
        archive, _ = self._ci_bundle(CI_TAG)
        archive.write_bytes(archive.read_bytes() + b"tampered")
        self._refused(CI_TAG, "does not match its .sha256 asset")

    def test_ci_file_digest_mismatch_refused(self):
        archive, checksum = self._ci_bundle(CI_TAG)
        repacked = io.BytesIO()
        with (
            tarfile.open(archive, "r:gz") as src,
            tarfile.open(fileobj=repacked, mode="w:gz") as dst,
        ):
            for member in src:
                data = src.extractfile(member).read()
                if member.name == "bin/radiance":
                    data = b"tiny-radiancX"
                dst.addfile(member, io.BytesIO(data))
        archive.write_bytes(repacked.getvalue())
        sha = runtime_assets.digest(archive)
        checksum.write_text(f"{sha}  {archive.name}\n")
        receipt = archive.parent / f"bigcherryrad-{PLAN}.receipt.json"
        receipt.write_text(
            json.dumps({**json.loads(receipt.read_text()), "sha256": sha})
        )
        self._refused(CI_TAG, "runtime file SHA-256 mismatch: bin/radiance")

    def test_ci_source_tree_mismatch_refused(self):
        tag = "bc-radiance-1190.0.0"
        self._ci_bundle(tag)  # packed from the trees before the commit below
        (self.root / "engines/radiance/source.txt").write_text("radiance changed")
        self._commit("change engine after the bundle")
        self._git("tag", tag)
        self._refused(tag, "bundle source trees differ from tag")

    def test_ci_sdk_mismatch_refused(self):
        self._ci_bundle(CI_TAG)
        (self.sdk / ".info/version").write_text("7.1.0")
        self._refused(CI_TAG, "ROCm SDK mismatch")

    def test_existing_binary_mismatch_is_refused(self):
        with mock.patch.dict(os.environ, self.env):
            saved = self._install()
            Path(saved["build_dir"]).joinpath("bin/radiance").write_bytes(b"tampered")
            with self.assertRaisesRegex(
                release_install.ReleaseInstallError, "binary SHA-256 mismatch"
            ):
                self._install()
        self.assertEqual(self.gh_calls.read_text().splitlines(), ["download"])
        self.assertTrue(
            (self.install_work / "releases" / TAG / "install.json").exists()
        )

    def test_dirty_release_worktree_is_refused(self):
        with mock.patch.dict(os.environ, self.env):
            saved = self._install()
            (Path(saved["worktree"]) / "dirty.txt").write_text("untracked")
            with self.assertRaisesRegex(release_install.ReleaseInstallError, "dirty"):
                self._install()
        self.assertEqual(self.gh_calls.read_text().splitlines(), ["download"])

    def test_sdk_mismatch_fails_before_any_worktree_or_download(self):
        (self.sdk / ".info/version").write_text("7.1.0")
        with (
            mock.patch.dict(os.environ, self.env),
            self.assertRaisesRegex(
                release_install.ReleaseInstallError, "ROCm SDK mismatch"
            ),
        ):
            self._install()
        self.assertFalse((self.install_work / "releases" / TAG / "src").exists())
        self.assertFalse(self.gh_calls.exists())

    def test_corrupted_download_cleans_stages_and_new_worktree(self):
        self.archive.write_bytes(self.archive.read_bytes() + b"invalid")
        with (
            mock.patch.dict(os.environ, self.env),
            self.assertRaisesRegex(release_install.ReleaseInstallError, "SHA-256"),
        ):
            self._install()
        self.assertFalse((self.install_work / "releases" / TAG / "src").exists())
        self.assertFalse(
            (self.install_work / "releases" / TAG / "install.json").exists()
        )
        self.assertFalse((self.install_work / "builds/src-1" / PLAN).exists())
        downloads = self.install_work / "release-assets" / "downloads"
        self.assertEqual(list(downloads.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
